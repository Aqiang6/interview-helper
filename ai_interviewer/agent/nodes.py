"""项目面试：每项目 3 个主问题，每个主问题 1 次首问和 2 次追问。"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

from ai_interviewer.agent.project_question import parse_projects
from ai_interviewer.agent.prompts import (
    INTERVIEW_PROJECT_CONTEXT, INTERVIEW_SYSTEM_PROMPT, PROJECT_EXTRACTION_PROMPT,
)
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


def _generate_question(state: InterviewState, *, follow_up: bool) -> dict:
    project_index = state["project_index"]
    project_name = state["projects"][project_index]
    is_project_opening = state["main_question_number"] == 1 and state["round_number"] == 1
    intro = ""
    if is_project_opening:
        action = "根据当前项目经历，提出该项目的第一个主问题。"
        opening = "先聊聊" if project_index == 0 else "接下来聊聊"
        intro = f"{opening}「{project_name}」。"
    elif follow_up:
        action = "围绕这组对话的主问题和候选人的上一条回答，追问一个还没展开的细节。"
    else:
        action = (
            "根据项目经历提出一个新的主问题，优先选还没作为主问题的技术点。"
            "项目资料较少时，可以从同一技术的另一个角度切入，比如设计取舍或验证方法。"
        )
    project_context = INTERVIEW_PROJECT_CONTEXT.format(
        project_name=project_name,
        project_details=state["project_details"][project_index],
        jd_raw_text=state.get("jd_raw_text", "") or "（未提供 JD）",
        custom_prompt=state.get("custom_prompt", "") or "（无）",
    )
    main_records = [
        record for record in state["question_history"]
        if record["project_index"] == project_index and record["round_number"] == 1
    ]
    topic_messages = []
    if follow_up:
        # 追问所需的对话从当前主问题开始。
        topic_messages = state["messages"][main_records[-1]["message_index"]:]
    else:
        if main_records:
            previous_questions = "\n".join(f"- {record['question']}" for record in main_records)
            project_context += f"\n\n<previous_main_questions>\n{previous_questions}\n</previous_main_questions>"
        project_context += f"\n\n{action}"
    messages = [
        SystemMessage(content=INTERVIEW_SYSTEM_PROMPT.format(action=action)),
        HumanMessage(content=project_context),
        *topic_messages,
    ]

    stream_callback = question_stream_callback.get()
    pending_intro = intro

    def stream_question(delta: str) -> None:
        nonlocal pending_intro
        if pending_intro:
            delta = delta.lstrip()
            if not delta:
                return
            stream_callback(pending_intro)
            pending_intro = ""
        stream_callback(delta)

    extra_body = None
    if (
        not follow_up
        and state["model"].casefold() == "deepseek-v4.1-flash-1017"
        and state["base_url"] == "https://api.modelarts-maas.com/openai/v1"
    ):
        # 华为云该模型默认深度思考，主问题使用直接出题模式。
        extra_body = {"chat_template_kwargs": {"thinking": False}}
    content = _call_llm(
        state, _langchain_to_openai_messages(messages),
        temperature=0.4,
        stream_callback=stream_question if stream_callback is not None else None,
        extra_body=extra_body,
    )
    if not content.strip():
        raise ValueError(f"模型未返回问题（项目「{project_name}」）")
    return {"pending_question": intro + content.lstrip() if intro else content}


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
    extra_body: dict | None = None,
) -> str:
    """用原生 openai 库调 LLM，返回响应文本（失败直接抛异常）

    当 stream_callback 不为 None 时，使用 stream=True 逐 delta 调用回调函数，
    实现真流式输出。回调签名：stream_callback(delta: str) -> None。
    """
    import time as _time
    client = OpenAI(
        api_key=state.get("api_key", ""),
        base_url=state.get("base_url", "https://api.openai.com/v1"),
        timeout=timeout,
        max_retries=0,
    )
    t0 = _time.time()
    model = state.get("model", "gpt-4o")
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream_callback is not None,
        extra_body=extra_body,
    )
    finish_reason = None
    has_reasoning = False
    content = ""
    if stream_callback is not None:
        for chunk in response:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            finish_reason = choice.finish_reason or finish_reason
            has_reasoning = has_reasoning or bool(getattr(choice.delta, "reasoning_content", None))
            delta = getattr(choice.delta, "content", None)
            if delta:
                content += delta
                stream_callback(delta)
    elif response.choices:
        choice = response.choices[0]
        finish_reason = choice.finish_reason
        has_reasoning = bool(getattr(choice.message, "reasoning_content", None))
        content = choice.message.content or ""
    log = logger.info if content.strip() else logger.warning
    log(
        "LLM 调用完成 model=%s project=%d main=%d round=%d stream=%s "
        "正文=%d字符 finish_reason=%s has_reasoning=%s 耗时=%.1fs",
        model, state["project_index"] + 1, state["main_question_number"], state["round_number"],
        stream_callback is not None, len(content), finish_reason, has_reasoning, _time.time() - t0,
    )
    return content
