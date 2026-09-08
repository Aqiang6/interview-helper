"""Agent 节点 - LangGraph Graph Nodes

每个节点是面试流程中的一个步骤，通过 LangGraph 的条件边连接：
1. plan_interview：分析简历，规划面试话题（Planning）
2. generate_question：生成面试问题
3. evaluate_response：评估候选人回答（Memory 更新，LLM 深度分析）
4. decide_next：决定下一步动作（追问/换题/结束）
5. generate_evaluation：生成最终评估报告
"""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

from ai_interviewer.agent.state import InterviewState
from ai_interviewer.agent.topic_prioritizer import (
    get_topic_prioritizer,
    make_route_decision,
    prioritize_by_jd,
)

logger = logging.getLogger(__name__)

# 模块级流式回调，供 generate_question 节点内部使用。
# 调用方在 invoke graph 前设置，invoke 完后清除。
_stream_cb = None

# ── 系统提示词 ──

_INTERVIEWER_SYSTEM = """你是一位资深技术面试官，正在针对岗位「{position_name}」进行技术面试。

## 面试策略
1. 根据目标岗位 JD 解析结果决定问题方向和深度：required 技能要深挖原理、源码、线上问题；bonus 技能点到为止即可
2. 当前话题要求强度: {topic_importance_label}（required=必须问到深层、preferred=问到中等深度、bonus=快速扫一遍）
3. 每次只问一个问题，层层递进，深度追问
4. 根据对话历史中候选人的上一轮回答调整追问方向
5. 可调用工具辅助决策

## 项目追问策略（重点）
当话题涉及候选人简历中的项目时，按以下模式追问：
- 技术选型："你在{{project}}这个功能里选了{{tech}}实现，为什么不选{{alt_tech}}？选型依据是什么？"
- 使用场景："为什么在这个场景用{{tech}}，它解决了什么问题？不用行不行？"
- 效果验证："用了{{tech}}之后怎么验证效果？有没有量化指标？上线后有没有踩坑？"
追问时直接引用简历中的具体项目名称和技术栈，不要泛泛而谈。

## 自定义面试指令
{custom_prompt}

## 简历项目详情（用于项目追问参考）
{resume_projects}

## 面试进度
阶段: {phase} | 已提问: {question_count}/{max_questions} | 当前话题: {current_topic} | 本话题已问: {topic_round}/{quota}轮
本话题 JD 要求强度: {topic_importance_label}

输出格式：直接输出面试官的话，不要加前缀。"""


# ── 节点函数 ──

def plan_interview(state: InterviewState) -> dict:
    """节点1：规划面试流程（Planning，JD 驱动）

    主路径：调用 LLM 根据「目标岗位 JD + 简历技能 + 简历摘要」动态生成：
      - ordered_topics（按 JD 要求强度从高到低排序）
      - skill_importance（required / preferred / bonus）
      - suggested_max_questions（按岗位技术密度调整 13-20）
      - topics_per_skill（每档技能该问几轮）
    失败自动降级为旧版 TopicPrioritizer 规则排序。
    如果已有计划，则跳过（幂等）。
    """
    # 已有计划 → 跳过
    if state.get("interview_plan", {}).get("topics"):
        return {}

    api_key = state.get("api_key", "")
    base_url = state.get("base_url", "https://api.openai.com/v1")
    model = state.get("model", "") or "gpt-4o-mini"
    jd_position_name = state.get("jd_position_name", "") or "通用技术岗"
    jd_raw_text = state.get("jd_raw_text", "")
    target_position = state.get("target_position", "")

    # ── JD 驱动 LLM 动态排序主路径（失败自动降级） ──
    jd_result = prioritize_by_jd(
        api_key=api_key,
        base_url=base_url,
        model=model,
        jd_position_name=jd_position_name,
        jd_raw_text=jd_raw_text,
    )
    sorted_topics = jd_result.ordered_topics if jd_result.ordered_topics else ["技术基础"]
    topic_aliases = {canonical: original for original, canonical in jd_result.ordered_aliases}
    if not topic_aliases and sorted_topics:
        topic_aliases = {t: t for t in sorted_topics}
    topic_question_counter = {t: 0 for t in sorted_topics}
    max_questions = jd_result.suggested_max_questions or 15

    # 计算 phase：如果 required 技能很多 → 直接 deep_dive；否则 basics
    req_count = sum(1 for v in jd_result.skill_importance.values() if v == "required")
    phase = "deep_dive" if req_count >= 2 else "basics"

    plan = {
        "topics": sorted_topics,
        "topic_aliases": topic_aliases,
        "current_topic_index": 0,
        "phase": phase,
        "questions_per_topic": 3,  # 遗留字段兜底
        # JD 新增字段
        "skill_importance": dict(jd_result.skill_importance),
        "topics_per_skill": dict(jd_result.topics_per_skill),
        "topic_question_counter": topic_question_counter,
        "position_id": target_position,
        "position_name": jd_position_name,
        "priority_source": jd_result.source,
    }

    logger.info(
        "面试计划生成(source=%s): position=%s max_q=%d required=%d topics=%s",
        jd_result.source, jd_position_name, max_questions, req_count, sorted_topics[:10],
    )

    return {
        "interview_plan": plan,
        "current_topic": plan["topics"][0] if plan["topics"] else "技术基础",
        "max_questions": max_questions,  # 覆盖默认 15
    }


def generate_question(state: InterviewState) -> dict:
    """节点2：生成面试问题（Tool Use）

    调用 LLM 生成面试问题，基于 JD + 简历项目 + 上一轮回答分析。
    """
    plan = state.get("interview_plan", {})
    messages = list(state.get("messages", []))
    question_count = state.get("question_count", 0)
    current_topic = state.get("current_topic", "")

    # 计算当前话题要求强度 + 已经问了几轮（供 system prompt 显示，也供路由决策用）
    importance_map: dict[str, str] = plan.get("skill_importance", {}) or {}
    importance_label = importance_map.get(current_topic)
    if importance_label not in {"required", "preferred", "bonus"}:
        # 模糊匹配
        importance_label = "preferred"
        for k, v in importance_map.items():
            if k and (k in current_topic or current_topic in k) and v in {"required", "preferred", "bonus"}:
                importance_label = v
                break
    topics_per_skill_cfg: dict[str, int] = plan.get("topics_per_skill", {}) or {"required": 4, "preferred": 3, "bonus": 2}
    quota = int(topics_per_skill_cfg.get(importance_label, 3))

    # topic_question_counter：当前话题问了几轮（本题出完要 +1）
    counter: dict[str, int] = plan.get("topic_question_counter", {}) or {}
    topic_round_prev = int(counter.get(current_topic, 0))  # 出这道题之前的轮数

    # 构建系统提示词
    position_name = plan.get("position_name") or state.get("jd_position_name") or "通用技术岗"

    # 简历项目段落
    resume_projects = state.get("resume_projects", "") or ""

    # 用户自定义面试指令（"再来一次"时填写的定向需求）
    custom_prompt = state.get("custom_prompt", "") or "（无）"

    system_content = _INTERVIEWER_SYSTEM.format(
        position_name=position_name,
        resume_projects=resume_projects or "（简历中未识别到项目段落）",
        custom_prompt=custom_prompt,
        phase=plan.get("phase", "intro"),
        question_count=question_count,
        max_questions=state.get("max_questions", 15),
        current_topic=current_topic or "未开始",
        topic_importance_label=importance_label,
        topic_round=topic_round_prev,
        quota=quota,
    )

    # 替换或添加系统消息
    if messages and isinstance(messages[0], SystemMessage):
        messages[0] = SystemMessage(content=system_content)
    else:
        messages.insert(0, SystemMessage(content=system_content))

    # 调用 LLM（原生 openai 库，避免 langchain 消息格式不兼容 bigmodel）
    try:
        openai_messages = _langchain_to_openai_messages(messages)
        ai_content = _call_llm(state, openai_messages, temperature=0.1, max_tokens=1500, stream_callback=_stream_cb)
    except Exception as e:
        logger.error("LLM 调用失败: %s", e)
        ai_content = (
            f"（系统提示：LLM 调用失败，请检查 API Key / Base URL / 模型名称是否正确。"
            f"错误：{type(e).__name__}: {e}）"
        )

    # 更新状态
    messages.append(AIMessage(content=ai_content))
    new_count = question_count + 1
    new_phase = plan.get("phase", "intro")
    if new_count == 1:
        new_phase = "basics"
    elif new_count >= 3:
        new_phase = "deep_dive"

    # 累加当前话题已问轮数
    new_counter = dict(counter)
    if current_topic:
        new_counter[current_topic] = topic_round_prev + 1

    return {
        "messages": messages,
        "question_count": new_count,
        "interview_plan": {
            **plan,
            "phase": new_phase,
            "topic_question_counter": new_counter,
        },
    }


def evaluate_response(state: InterviewState) -> dict:
    """节点3：直通节点（占位）

    早期版本在此调用 LLM 做定性分析并产出 answer_analysis，该链路已删除：
    出题上下文只需简历项目 + JD 解析结果，候选人回答本身就在 messages
    历史里，出题 LLM 自会看到。保留节点以维持图拓扑稳定。
    """
    return {}


def decide_next(state: InterviewState) -> str:
    """节点4：条件路由 - 自然状态转换（已取消 quality_level 评分）

    不再依赖回答质量等级，按配额 + 题数自然转换：
    - 题数到顶 → end
    - 本话题配额满且还有后续话题 → next_topic
    - 已是最后一个话题 → followup（深挖到题数到顶）
    - 剩题不足以覆盖剩余话题 → next_topic（压缩追问）
    - 默认 → followup（自然追问）

    返回值与原兼容："followup" / "next_topic" / "end"
    """
    question_count = state.get("question_count", 0)
    max_questions = state.get("max_questions", 15)
    plan = state.get("interview_plan", {})
    topics = plan.get("topics", [])
    current_idx = plan.get("current_topic_index", 0)
    current_topic = state.get("current_topic", "")

    # 当前话题要求强度（required / preferred / bonus）
    importance_map: dict[str, str] = plan.get("skill_importance", {}) or {}
    importance = importance_map.get(current_topic, "preferred")
    if importance not in {"required", "preferred", "bonus"}:
        for k, v in importance_map.items():
            if k and (k in current_topic or current_topic in k) and v in {"required", "preferred", "bonus"}:
                importance = v
                break
        else:
            importance = "preferred"

    # 当前话题问了几轮（用于配额保护）
    counter: dict[str, int] = plan.get("topic_question_counter", {}) or {}
    questions_on_this_topic = int(counter.get(current_topic, 0))

    # 已取消 quality_level 评分：路由按配额 + 题数自然转换，不再读回答质量
    return make_route_decision(
        importance=importance,
        questions_on_this_topic=questions_on_this_topic,
        questions_per_skill=plan.get("topics_per_skill", {"required": 4, "preferred": 3, "bonus": 2}),
        question_count=question_count,
        max_questions=max_questions,
        current_topic_index=current_idx,
        total_topics=len(topics),
    )


def next_topic(state: InterviewState) -> dict:
    """节点5a：切换到下一个面试话题（同时清 RAG 上下文，避免混用上一个话题的知识库）"""
    plan = dict(state.get("interview_plan", {}))
    topics = plan.get("topics", [])
    current_idx = plan.get("current_topic_index", 0)

    new_idx = min(current_idx + 1, len(topics) - 1) if topics else 0
    new_topic = topics[new_idx] if topics else "技术基础"

    return {
        "interview_plan": {**plan, "current_topic_index": new_idx},
        "current_topic": new_topic,
    }


def followup(state: InterviewState) -> dict:
    """节点5b：追问准备（已取消 quality_level，本节点仅占位，实际追问由 generate_question 完成）"""
    return {}


def generate_evaluation(state: InterviewState) -> dict:
    """节点6：面试结束，标记完成（不再生成对话回顾和补强建议）

    取消原"不足反馈"和"对话回顾"功能，仅标记完成，供前端展示"再来一次"入口。
    """
    return {
        "is_finished": True,
        "last_tool_result": json.dumps({"question_count": state.get("question_count", 0)}, ensure_ascii=False),
    }


def _langchain_to_openai_messages(messages: list) -> list[dict]:
    """langchain 消息对象列表 → 标准 OpenAI 格式（bigmodel 不兼容 langchain 额外字段）"""
    result: list[dict] = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            result.append({"role": "system", "content": msg.content})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def _call_llm(
    state: InterviewState,
    messages: list[dict],
    temperature: float = 0.1,
    max_tokens: int = 4096,
    timeout: int = 60,
    stream_callback=None,
) -> str:
    """用原生 openai 库调 LLM，返回响应文本（失败直接抛异常）

    当 stream_callback 不为 None 时，使用 stream=True 逐 delta 调用回调函数，
    实现真流式输出。回调签名：stream_callback(delta: str) -> None。
    max_tokens=4096 兼容思考模型（如 glm-5.3-flash）：思考过程+输出内容共需较大空间
    """
    import time as _time
    # bigmodel 要求至少一条 user 消息，只有 system 时补一条
    if not any(m.get("role") == "user" for m in messages):
        messages = messages + [{"role": "user", "content": "请开始提问。"}]
    client = OpenAI(
        api_key=state.get("api_key", ""),
        base_url=state.get("base_url", "https://api.openai.com/v1"),
        timeout=timeout,
        max_retries=0,
    )
    t0 = _time.time()

    if stream_callback is not None:
        # 流式模式：逐 delta 推送给回调，同时拼接完整文本
        stream = client.chat.completions.create(
            model=state.get("model", "gpt-4o"),
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        content = ""
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                content += delta
                stream_callback(delta)
        elapsed = _time.time() - t0
        logger.info("LLM 流式调用完成: model=%s 耗时=%.1fs 输出=%d字符", state.get("model", ""), elapsed, len(content))
        return content

    # 非流式模式（原有逻辑不变）
    resp = client.chat.completions.create(
        model=state.get("model", "gpt-4o"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    elapsed = _time.time() - t0
    content = resp.choices[0].message.content or ""
    logger.info("LLM 调用完成: model=%s 耗时=%.1fs 输出=%d字符", state.get("model", ""), elapsed, len(content))
    return content


def _extract_project_details(resume_text: str, max_chars: int = 2000) -> str:
    """从简历原文中提取项目经历段落。

    规则刻意极简：从首个项目段标题开始收集，到下一个非项目段标题为止。
    - 段标题判定：短行（≤20字）且不含冒号 —— 正文叙述行天然不会误判；
    - 找不到项目段标题 → 返回空串（绝不兜底拼姓名/联系方式等无关内容）。
    """
    if not resume_text:
        return ""

    raw_lines = resume_text.split("\n")

    # 项目段落起点标记
    project_start_markers = [
        "项目", "工作经历", "项目经验", "工作经验", "工程经历",
        "实习经历", "职业经历", "职业经验", "相关经历",
        "PROJECT", "EXPERIENCE",
    ]

    # 非项目段标题关键词（短行出现 → 项目段结束）
    non_project_sections = [
        "教育", "技能", "优势", "评价", "证书", "资质",
        "获奖", "奖项", "荣誉", "兴趣", "爱好",
        "自我介绍", "语言能力", "作品集", "其他",
        "References", "Skill", "Tech Stack",
    ]

    def _is_section_header(s: str) -> bool:
        return len(s) <= 20 and "：" not in s and ":" not in s

    start = None
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped or not _is_section_header(stripped):
            continue
        if any(m in stripped for m in project_start_markers):
            start = i
            break
    if start is None:
        return ""

    out: list[str] = []
    for stripped in (ln.strip() for ln in raw_lines[start:]):
        if not stripped:
            continue
        if _is_section_header(stripped) and any(k in stripped for k in non_project_sections):
            break
        out.append(stripped)

    result = "\n".join(out)
    if len(result) > max_chars:
        result = result[:max_chars] + "...（截断）"
    return result


# ── 辅助函数 ──
