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
from ai_interviewer.agent.tools import (
    _QUALITY_EVAL_SYSTEM,
    _parse_analysis_json,
)

logger = logging.getLogger(__name__)

# ── 系统提示词 ──

_INTERVIEWER_SYSTEM = """你是一位资深技术面试官，正在针对岗位「{position_name}」进行技术面试。

## 面试策略
1. 根据目标岗位 JD 解析结果决定问题方向和深度：required 技能要深挖原理、源码、线上问题；bonus 技能点到为止即可
2. 当前话题要求强度: {topic_importance_label}（required=必须问到深层、preferred=问到中等深度、bonus=快速扫一遍）
3. 每次只问一个问题，层层递进，深度追问
4. 根据候选人上一轮回答的定性分析调整追问方向
5. 可调用工具辅助决策

## 项目追问策略（重点）
当话题涉及候选人简历中的项目时，按以下模式追问：
- 技术选型："你在{{project}}这个功能里选了{{tech}}实现，为什么不选{{alt_tech}}？选型依据是什么？"
- 使用场景："为什么在这个场景用{{tech}}，它解决了什么问题？不用行不行？"
- 效果验证："用了{{tech}}之后怎么验证效果？有没有量化指标？上线后有没有踩坑？"
追问时直接引用简历中的具体项目名称和技术栈，不要泛泛而谈。

## 上一轮回答分析（指导本题方向）
{answer_analysis}

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

    skills = state.get("skills", [])
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
        skills=skills,
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

    # 简历项目段落：优先用用户编辑的，没有才自动提取
    resume_projects = state.get("resume_projects", "") or ""
    if not resume_projects:
        resume_raw = state.get("resume_text", "") or ""
        resume_projects = _extract_project_details(resume_raw)

    # 上一轮回答的定性分析（指导本题方向）
    answer_analysis = state.get("answer_analysis", "") or "（首轮提问，无上一轮分析）"

    # 用户自定义面试指令（"再来一次"时填写的定向需求）
    custom_prompt = state.get("custom_prompt", "") or "（无）"

    system_content = _INTERVIEWER_SYSTEM.format(
        position_name=position_name,
        resume_projects=resume_projects or "（简历中未识别到项目段落）",
        answer_analysis=answer_analysis,
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
        ai_content = _call_llm(state, openai_messages, temperature=0.1, max_tokens=1500)
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
    """节点3：定性分析候选人回答（已彻底取消 quality_level 评分）

    通过 LLM 做定性分析：提取已覆盖要点 / 缺失点 / 追问方向。
    不再产生任何质量等级（quality_level）或分数，路由决策完全由配额 + 题数自然转换。
    分析结果注入下一题的系统提示词，保留回答对下次提问的影响。
    任何失败（LLM 不可用 / JSON 解析失败）直接抛出 RuntimeError。
    """
    messages = state.get("messages", [])
    profile = dict(state.get("candidate_profile", {}))

    # 获取最近的候选人回答
    recent_human = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            recent_human = msg.content
            break

    if not recent_human:
        return {}

    current_topic = state.get("current_topic", "unknown")

    # 更新已回答话题
    answered = list(profile.get("answered_topics", []))
    if current_topic not in answered:
        answered.append(current_topic)
    profile["answered_topics"] = answered

    # ── LLM 定性分析（只提取信息，不打分） ──
    eval_data = _llm_evaluate_answer(recent_human, current_topic, state)

    # 构建给下一题参考的分析摘要（已移除"掌握程度"行）
    analysis_parts = []
    if eval_data["covered_well"]:
        analysis_parts.append(f"已覆盖: {'；'.join(eval_data['covered_well'])}")
    if eval_data["gaps"]:
        analysis_parts.append(f"缺失: {'；'.join(eval_data['gaps'])}")
    if eval_data["follow_up_direction"]:
        analysis_parts.append(f"追问方向: {eval_data['follow_up_direction']}")
    answer_analysis = " | ".join(analysis_parts) or "（本轮无明显分析点）"

    logger.info(
        "LLM 回答定性分析: topic=%s gaps=%s 追问方向=%s",
        current_topic, eval_data["gaps"][:3], (eval_data["follow_up_direction"] or "")[:50],
    )

    return {
        "candidate_profile": profile,
        "answer_analysis": answer_analysis,
    }


def _llm_evaluate_answer(answer: str, topic: str, state: InterviewState) -> dict:
    """调用 LLM 做定性分析（不再评分），任何失败直接抛 RuntimeError"""
    from ai_interviewer.config import get_settings
    settings = get_settings()
    user_model = state.get("model") or settings.openai_model or "gpt-4o-mini"
    try:
        prompt = f"""## 当前面试话题
{topic}

## 候选人回答
{answer}

请按照分析要求输出 JSON。"""
        openai_messages = [
            {"role": "system", "content": _QUALITY_EVAL_SYSTEM},
            {"role": "user", "content": prompt},
        ]
        content = _call_llm(
            state, openai_messages,
            temperature=0.1, max_tokens=600, timeout=30,
        )
        eval_data = _parse_analysis_json(content)
        logger.info(
            "LLM 定性分析: topic=%s covered=%s gaps=%s",
            topic, eval_data["covered_well"][:2], eval_data["gaps"][:2],
        )
        return eval_data
    except (RuntimeError, ValueError):
        raise
    except Exception as e:
        err_type = type(e).__name__
        effective_api_key = state.get("api_key") or settings.openai_api_key
        effective_base_url = state.get("base_url") or settings.openai_base_url
        if not effective_api_key:
            masked_key = "(空)"
        elif len(effective_api_key) <= 8:
            masked_key = "***" + (effective_api_key[-4:] if len(effective_api_key) > 4 else "")
        else:
            masked_key = effective_api_key[:8] + "..." + effective_api_key[-4:]
        msg = (
            f"候选人回答分析失败，面试无法继续：[{err_type}] {str(e) or '(无错误消息)'}。"
            f" | topic={topic} | model={user_model} | base_url={effective_base_url}"
            f" | api_key={masked_key}"
        )
        logger.error(msg)
        raise RuntimeError(msg) from e


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


def _extract_project_details(resume_text: str, max_chars: int = 2000) -> str:
    """从简历原文中提取项目相关段落，供面试官追问引用。

    核心原则（用户明确要求：技能优势/核心优势一律不能出现在结果里）：
    1) 起点：首个「工作经历/项目经验/工程经历/实习经历」段标题开始；
    2) 全局截止：遇到「教育/学历/技能/自我评价/证书/奖项/兴趣/校园经历/
       技能优势/核心优势/专业技能/技能清单/Skills/Tech…」这类段标题——
       一律在该标题所在行的「上一行」截断（往上切，技能段一个字都不保留）；
    3) 项目内部的「技术栈：XXX / Tech Stack：XXX」单行锚点，属于单个项目
       的收尾信息，要保留；但如果「技术栈：XXX」之后紧接着就是技能段标题
       或一大串零散技能关键字列表，那么切到「这行技术栈为止」（往下切，技
       能列表不要带进来）；
    4) 兜底返回前，再从后往前扫一遍：遇到技能段标题 / 纯技能关键字散点列
       表，就往上切掉。
    """
    if not resume_text:
        return ""

    raw_lines = resume_text.split("\n")

    # 项目段落触发词（短行标题）
    project_start_markers = [
        "项目", "经历", "工作经历", "项目经验", "工作经验", "工程经历",
        "实习经历", "项目经历", "职业经历", "职业经验", "相关经历",
        "PROJECT", "EXPERIENCE",
    ]

    # 全局段终止关键词（一旦作为短行标题出现，立即 STOP，所有后续内容不要）
    section_stop_keywords = [
        "教育背景", "教育经历", "学历", "学校", "毕业院校", "校园经历",
        "技能", "技能特长", "专业技能", "核心技能", "技能清单", "技能优势",
        "核心优势", "擅长技术", "掌握技能", "个人技能", "IT技能",
        "自我评价", "自我评估", "个人评价", "总结", "自我介绍",
        "证书", "资质", "获奖", "奖项", "荣誉",
        "兴趣爱好", "兴趣", "爱好",
        "语言能力", "作品集", "其他",
        "References", "REFERENCES",
        "Skill", "Skills", "SKILL", "SKILLS",
        "Tech Stack", "TECH STACK",  # 单独出现在短行 → 作段标题处理
    ]

    # 项目段内"单行技术栈标注"锚点（后跟冒号/破折号，且不是长列表开头）
    per_project_tech_anchors = [
        "技术栈", "Tech Stack", "TECH STACK", "核心技术", "使用技术",
        "开发环境", "技术要点", "技术方案", "技术选型", "涉及技术",
    ]

    # 纯「技能散点关键词」行（行内全是这种词，没有主谓句 → 判定为技能列表开头）
    skill_bullet_keywords = [
        "Java", "Python", "Go", "Golang", "C++", "Rust", "TypeScript", "JavaScript",
        "Spring", "SpringBoot", "Spring Boot", "Spring Cloud", "MyBatis", "Dubbo",
        "MySQL", "Redis", "Kafka", "RocketMQ", "RabbitMQ", "Elasticsearch", "ES",
        "MongoDB", "PostgreSQL", "SQLServer", "Oracle",
        "Linux", "Docker", "Kubernetes", "K8s", "Nginx",
        "Git", "Maven", "Gradle", "Jenkins", "CI/CD",
        "Vue", "React", "Node.js", "HTML", "CSS",
        "Hadoop", "Spark", "Flink", "HBase", "Hive",
        "LangChain", "RAG", "向量数据库", "Milvus", "pgvector", "FAISS",
        "大模型", "LLM", "Prompt", "Agent",
        "JVM", "JUC", "并发", "多线程", "锁", "事务", "MVCC", "索引",
        "TCP/IP", "HTTP", "HTTPS", "RPC", "Netty",
        "设计模式", "分布式", "微服务", "高并发", "高可用", "CAP", "BASE",
        "RESTful", "GraphQL", "OAuth", "JWT",
        "熟练", "熟悉", "了解", "掌握", "精通",
    ]

    def _is_short_title(s: str) -> bool:
        return len(s) <= 60

    def _looks_like_inline_tech_anchor(stripped: str) -> bool:
        """项目内部的「技术栈：xxx / 技术栈-xxx / 技术栈 | xxx」式单行标注。"""
        if not any(a in stripped for a in per_project_tech_anchors):
            return False
        # 含常见分隔符 → 说明是「锚点 + 内容」形式，不是段标题单独成行
        return (
            "：" in stripped or ":" in stripped
            or "—" in stripped or "-" in stripped
            or "|" in stripped or "｜" in stripped
            or "·" in stripped  # 技术栈 · Java / MySQL
        )

    def _looks_like_pure_skill_bullets(stripped: str) -> bool:
        """行内容基本是「技能散点」拼接 → 非项目内容，技能列表起点。"""
        s = stripped
        # 去掉常见分隔符与序号
        compact = re.sub(r"^[\s\-•·\d\.\)、】]+", "", s)
        if not compact:
            return False
        # 按常见分隔符切分，若每一段都能匹配到 skill keyword，判定为技能散点行
        parts = re.split(r"[、,，/;；|·\s]+", compact)
        parts = [p for p in parts if p]
        if not parts:
            return False
        # 全部片段或高比例命中技能关键词（大小写不敏感，子串匹配）
        hits = 0
        for p in parts:
            pl = p.lower()
            if any(kw.lower() in pl for kw in skill_bullet_keywords):
                hits += 1
        # 只要命中比例高 且 没有「主谓结构」的长句，就认作技能散点
        ratio = hits / max(len(parts), 1)
        return (ratio >= 0.5 and hits >= 1) or (
            ratio >= 0.35 and hits >= 3 and len(s) <= 120
        )

    def _is_stop_line(stripped: str, *, allow_inline_anchor: bool) -> bool:
        if not stripped:
            return False
        short = _is_short_title(stripped)
        # 项目开始词不算 stop
        if any(m in stripped for m in project_start_markers):
            return False
        # ⭐ 关键修复：无论 allow_inline_anchor 是 True 还是 False，只要该行是
        #   「技术栈：xxx / Tech Stack: xxx / 技术栈 | Java · MySQL」这种锚点+内容形式，
        #   就**绝不**是 STOP 段标题（它是项目内部的收尾标注，必须保留）。
        #   只有当「Tech Stack / 技术栈」作为**独立短标题**（没有冒号、|、· 等内容分隔符）
        #   出现时，才视为新的技能段开始。
        if _looks_like_inline_tech_anchor(stripped):
            return False
        if short and any(kw in stripped for kw in section_stop_keywords):
            return True
        return False

    # ---------- 阶段 A：全局硬切（进入项目段后，一旦遇到段 STOP 标题，立即在该行之前 cut） ----------
    global_cut_line = len(raw_lines)
    entered_project = False
    for i, line in enumerate(raw_lines):
        stripped = line.strip()
        if not stripped:
            continue
        if not entered_project:
            if _is_short_title(stripped) and any(m in stripped for m in project_start_markers):
                entered_project = True
            continue
        # 进入项目段之后：stop 标题 → 立即在该行上一刀切断（技能段本身不保留）
        if _is_stop_line(stripped, allow_inline_anchor=True):
            global_cut_line = i
            break
        # 遇到纯技能散点列表（一行里全是 Java/MySQL/Redis 这类散点关键词）也立刻停
        if _looks_like_pure_skill_bullets(stripped):
            # 技能段本身不保留
            global_cut_line = i
            break
    trimmed_lines = raw_lines[:global_cut_line]

    # ---------- 阶段 B：提取出项目段落（保留「最后一行技术栈：xxx」作为项目收尾） ----------
    project_lines: list[str] = []
    in_project_section = False

    for line in trimmed_lines:
        stripped = line.strip()
        if not stripped:
            continue

        if _is_short_title(stripped) and any(marker in stripped for marker in project_start_markers):
            in_project_section = True
            project_lines.append(stripped)
            continue

        if in_project_section and _is_stop_line(stripped, allow_inline_anchor=True):
            in_project_section = False
            continue

        # 项目段内遇到纯技能散点 → 这一项不加入，并且出段（防止技术栈关键字列表泄漏）
        if in_project_section and _looks_like_pure_skill_bullets(stripped):
            # 如果上一行恰好是「技术栈：xxx」这种锚点，就保留它；技能散点本身跳过
            # 并且立刻出段
            in_project_section = False
            continue

        if in_project_section:
            project_lines.append(stripped)

    # ---------- 阶段 C：从尾往前做「按最后一行技术栈往下切 / 技能往上切」双保险 ----------
    def _rtrim_projects(lines: list[str]) -> list[str]:
        # 倒扫：
        # 1) 遇到 STOP 短标题（含技能/技术栈单独成段）→ 直接切到该标题前
        # 2) 遇到技能散点行（Java/MySQL/Redis…）→ 往上切掉，直到遇到一行主谓语或
        #    「技术栈：xxx」锚点，锚点本身保留（即用户要求的「最后一行技术栈往下
        #    切就切到这里为止」）
        cut_at = len(lines)  # 默认不切
        last_tech_anchor = -1
        for j in range(len(lines) - 1, -1, -1):
            ln = lines[j]
            if not ln:
                continue
            # STOP 标题（不带冒号的那种单独标题）→ 直接在它之前截断
            if _is_stop_line(ln, allow_inline_anchor=False):
                cut_at = j
                continue  # 再往前看有没有更紧的锚点（其实不需要了，已经整体 cut）
            # 「技术栈：xxx」类单行锚点：项目收尾
            if _looks_like_inline_tech_anchor(ln):
                last_tech_anchor = j
                break  # 保留到这条行为止（再往前就是项目正文了）
            # 技能散点：在到达锚点之前遇到，继续往上抹
            if _looks_like_pure_skill_bullets(ln):
                cut_at = j  # 至少要切掉这行及以下（会被循环继续向前推）
                continue
            # 普通正文句：有主谓，不是散点 → 说明已经到项目内容区了，停止倒扫
            # （若此前发现 cut_at 停在某技能散点，则最终 cut_at 为该值）
            if len(ln) >= 6 and not _is_short_title(ln):
                break
        # 若发现了明确的「最后一行技术栈」，优先以它为最终收尾（+1 保留该行本身）
        if last_tech_anchor >= 0:
            final_end = last_tech_anchor + 1
            if final_end < cut_at:
                return lines[:final_end]
        if cut_at < len(lines):
            return lines[:cut_at]
        return lines

    project_lines = _rtrim_projects(project_lines)

    # 再去掉尾部孤立的空行 / 空标题
    def _strip_tail_whitespace(lines: list[str]) -> list[str]:
        while lines and not lines[-1].strip():
            lines.pop()
        return lines

    project_lines = _strip_tail_whitespace(project_lines)

    # ---------- 阶段 D：长度太短时兜底（兜底前仍然保证「技能优势：往上一刀切」） ----------
    if len(project_lines) < 4:
        fallback_lines = [ln.strip() for ln in trimmed_lines if ln.strip()]
        # 倒扫一次：技能 stop 标题或技能散点行 → 立刻往上切
        last_safe = len(fallback_lines)
        seen_content = False
        for j in range(len(fallback_lines) - 1, -1, -1):
            ln = fallback_lines[j]
            if seen_content and (
                _is_stop_line(ln, allow_inline_anchor=False)
                or _looks_like_pure_skill_bullets(ln)
            ):
                last_safe = j
                break
            seen_content = True
        fallback_lines = _rtrim_projects(fallback_lines[:last_safe])
        fallback_lines = _strip_tail_whitespace(fallback_lines)
        # 兜底仍然保留项目 start markers 之后的内容
        fallback_src = "\n".join(fallback_lines)
        result = fallback_src[:max_chars]
    else:
        result = "\n".join(project_lines)

    if len(result) > max_chars:
        result = result[:max_chars] + "...（截断）"

    return result


# ── 辅助函数 ──

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
) -> str:
    """用原生 openai 库调 LLM，返回响应文本（失败直接抛异常）
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
