"""项目面试：每项目 3 个主问题，每个主问题 1 次首问和 2 次追问。"""

from __future__ import annotations

import logging
from contextvars import ContextVar

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from openai import OpenAI

from ai_interviewer.agent.project_question import ProjectQuestion
from ai_interviewer.agent.prompts import FIRST_QUESTION_OUTPUT, INTERVIEW_SYSTEM_PROMPT, QUESTION_OUTPUT
from ai_interviewer.agent.state import InterviewState

logger = logging.getLogger(__name__)
question_stream_callback = ContextVar("question_stream_callback", default=None)

MAIN_QUESTIONS_PER_PROJECT = 3
ROUNDS_PER_QUESTION = 3
QUESTIONS_PER_PROJECT = MAIN_QUESTIONS_PER_PROJECT * ROUNDS_PER_QUESTION
FINISH_MESSAGE = "本次面试已完成，感谢你的回答。你可以回看对话，或再来一次。"


def interview_progress(state: InterviewState) -> dict:
    """根据成功发出的题数计算下一题的位置，不由 LLM 决定轮数。"""
    count = state.get("question_count", 0)
    projects = state.get("projects", [])
    project_index, project_count = divmod(count, QUESTIONS_PER_PROJECT)
    main_index, round_index = divmod(project_count, ROUNDS_PER_QUESTION)
    return {
        "project_index": project_index,
        "project_name": projects[project_index] if project_index < len(projects) else "",
        "main_question_number": main_index + 1,
        "round_number": round_index + 1,
        "total_questions": len(projects) * QUESTIONS_PER_PROJECT,
    }


def route_interview(state: InterviewState) -> str:
    total = len(state.get("projects", [])) * QUESTIONS_PER_PROJECT
    return "finish_interview" if total and state.get("question_count", 0) >= total else "generate_question"


def generate_question(state: InterviewState) -> dict:
    projects = state.get("projects", [])
    count = state.get("question_count", 0)
    progress = interview_progress(state)
    if not projects:
        progress_text = "首次提问：识别项目后，提出第 1 个项目的第 1 个主问题（第 1/3 轮）。"
    else:
        action = "提出新的主问题" if progress["round_number"] == 1 else f"进行第 {progress['round_number'] - 1}/2 次追问"
        progress_text = (
            f"第 {progress['project_index'] + 1}/{len(projects)} 个项目：{progress['project_name']}；"
            f"第 {progress['main_question_number']}/3 个主问题；"
            f"第 {progress['round_number']}/3 轮。现在应{action}。"
        )
    main_question = state.get("main_question", "") if progress["round_number"] > 1 else "（本轮提出新的主问题）"
    system_content = INTERVIEW_SYSTEM_PROMPT.format(
        jd_raw_text=state.get("jd_raw_text", "") or "（未提供 JD）",
        resume_projects=state["resume_projects"],
        custom_prompt=state.get("custom_prompt", "") or "（无）",
        progress=progress_text,
        main_question=main_question,
        output_format=QUESTION_OUTPUT if projects else FIRST_QUESTION_OUTPUT,
    )
    messages = [SystemMessage(content=system_content)] + [
        message for message in state.get("messages", []) if not isinstance(message, SystemMessage)
    ]
    callback = question_stream_callback.get()
    selection = ProjectQuestion(callback) if not projects else None
    stream_callback = selection.feed if selection is not None and callback is not None else callback
    content = _call_llm(
        state, _langchain_to_openai_messages(messages),
        temperature=0.1, max_tokens=4096, stream_callback=stream_callback,
    )
    if selection is not None:
        if callback is None:
            selection.feed(content)
        selection.finish()
        projects, content = selection.projects, selection.text
    if not content.strip():
        raise ValueError("模型未返回问题，请重试本题")
    messages.append(AIMessage(content=content))
    return {
        "messages": messages,
        "projects": projects,
        "question_count": count + 1,
        "main_question": content if progress["round_number"] == 1 else state["main_question"],
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
