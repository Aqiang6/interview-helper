"""项目面试：每项目 3 个主问题，每个主问题 1 次首问和 2 次追问。"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

from ai_interviewer.agent.project_question import parse_projects
from ai_interviewer.agent.prompts import INTERVIEW_SYSTEM_PROMPT, PROJECT_EXTRACTION_PROMPT
from ai_interviewer.agent.state import InterviewState

logger = logging.getLogger(__name__)
question_stream_callback = ContextVar("question_stream_callback", default=None)

MAIN_QUESTIONS_PER_PROJECT = 3
ROUNDS_PER_QUESTION = 3
QUESTIONS_PER_PROJECT = MAIN_QUESTIONS_PER_PROJECT * ROUNDS_PER_QUESTION
FINISH_MESSAGE = "本次面试已完成，感谢你的回答。你可以回看对话，或再来一次。"


def interview_progress(state: InterviewState) -> dict:
    """返回显式游标指向的下一题位置；总题数仅用于展示。"""
    projects = state.get("projects", [])
    project_index = state["project_index"]
    return {
        "project_index": project_index,
        "project_name": projects[project_index] if project_index < len(projects) else "",
        "main_question_number": state["main_question_number"],
        "round_number": state["round_number"],
        "total_questions": len(projects) * QUESTIONS_PER_PROJECT,
    }


def route_interview(state: InterviewState) -> str:
    if not state["projects"]:
        return "identify_projects"
    if state["project_index"] >= len(state["projects"]):
        return "finish_interview"
    return "ask_main_question" if state["round_number"] == 1 else "ask_follow_up"


def identify_projects(state: InterviewState) -> dict:
    """首次调用独立确定项目边界，后续节点只接收当前项目的原文。"""
    content = _call_llm(state, [
        {"role": "system", "content": PROJECT_EXTRACTION_PROMPT},
        {"role": "user", "content": state["resume_projects"]},
    ])
    projects, details = parse_projects(content, state["resume_projects"])
    return {"projects": projects, "project_details": details}


def _current_project_messages(state: InterviewState) -> list:
    records = [record for record in state["question_history"] if record["project_index"] == state["project_index"]]
    if not records:
        # 换项目时，刚收到的回答仍属于上个项目，不带进新项目。
        return []
    return state["messages"][records[0]["message_index"]:]


def _generate_question(state: InterviewState, *, follow_up: bool) -> dict:
    project_index = state["project_index"]
    project_name = state["projects"][project_index]
    action = (
        f"围绕当前主问题和上一轮回答，进行第 {state['round_number'] - 1}/2 次追问。"
        if follow_up else "提出一个新的主问题，与本项目已有主问题采用不同的技术切入点。"
    )
    system_content = INTERVIEW_SYSTEM_PROMPT.format(
        jd_raw_text=state.get("jd_raw_text", "") or "（未提供 JD）",
        project_name=project_name,
        project_details=state["project_details"][project_index],
        custom_prompt=state.get("custom_prompt", "") or "（无）",
        project_number=project_index + 1,
        project_total=len(state["projects"]),
        main_question_number=state["main_question_number"],
        round_number=state["round_number"],
        action=action,
        main_question=state["main_question"] if follow_up else "（本轮提出新的主问题）",
    )
    project_messages = _current_project_messages(state)
    messages = [SystemMessage(content=system_content), *project_messages]
    stream_callback = question_stream_callback.get()
    intro = ""
    if project_index > 0 and not project_messages:
        intro = f"接下来聊聊「{project_name}」。"
        if stream_callback is not None:
            stream_callback(intro)
    content = _call_llm(
        state, _langchain_to_openai_messages(messages),
        temperature=0.7,
        max_tokens=4096,
        stream_callback=stream_callback,
    )
    if not content.strip():
        raise ValueError("模型未返回问题，请重试本题")
    return {"pending_question": intro + content}


def ask_main_question(state: InterviewState) -> dict:
    return _generate_question(state, follow_up=False)


def ask_follow_up(state: InterviewState) -> dict:
    return _generate_question(state, follow_up=True)


def advance_progress(state: InterviewState) -> dict:
    """出题完成后提交记录，按轮次 → 主问题 → 项目逐层推进。"""
    question = state["pending_question"]
    project_index = state["project_index"]
    main_number, round_number = state["main_question_number"], state["round_number"]
    record = {
        "project_index": project_index,
        "project_name": state["projects"][project_index],
        "main_question_number": main_number,
        "round_number": round_number,
        "question": question,
        "message_index": len(state["messages"]),
    }
    main_question = question if round_number == 1 else state["main_question"]
    if round_number < ROUNDS_PER_QUESTION:
        round_number += 1
    else:
        round_number = 1
        main_question = ""
        if main_number < MAIN_QUESTIONS_PER_PROJECT:
            main_number += 1
        else:
            main_number = 1
            project_index += 1
    return {
        "messages": [*state["messages"], AIMessage(content=question)],
        "question_history": [*state["question_history"], record],
        "question_count": state["question_count"] + 1,
        "project_index": project_index,
        "main_question_number": main_number,
        "round_number": round_number,
        "main_question": main_question,
        "pending_question": "",
    }


def finish_interview(state: InterviewState) -> dict:
    callback = question_stream_callback.get()
    if callback is not None:
        callback(FINISH_MESSAGE)
    return {
        "is_finished": True,
        "messages": [*state.get("messages", []), AIMessage(content=FINISH_MESSAGE)],
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
