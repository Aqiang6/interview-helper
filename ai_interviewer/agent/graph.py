"""LangGraph 项目面试流程：识别项目、首问/追问、显式推进、结束。"""

from __future__ import annotations

import asyncio

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ai_interviewer.agent.nodes import (
    FINISH_MESSAGE, advance_progress, ask_follow_up, ask_main_question,
    finish_interview, identify_projects, question_stream_callback, route_interview,
)
from ai_interviewer.agent.state import InterviewState
from ai_interviewer.resume_parser import extract_project_details


def _build_graph():
    workflow = StateGraph(InterviewState)
    workflow.add_node("identify_projects", identify_projects)
    workflow.add_node("ask_main_question", ask_main_question)
    workflow.add_node("ask_follow_up", ask_follow_up)
    workflow.add_node("advance_progress", advance_progress)
    workflow.add_node("finish_interview", finish_interview)
    workflow.add_conditional_edges(START, route_interview, {
        "identify_projects": "identify_projects", "ask_main_question": "ask_main_question",
        "ask_follow_up": "ask_follow_up", "finish_interview": "finish_interview",
    })
    workflow.add_edge("identify_projects", "ask_main_question")
    workflow.add_edge("ask_main_question", "advance_progress")
    workflow.add_edge("ask_follow_up", "advance_progress")
    workflow.add_edge("advance_progress", END)
    workflow.add_edge("finish_interview", END)
    return workflow.compile()


class InterviewGraphAgent:
    def __init__(self, api_key: str = "", base_url: str = "https://api.openai.com/v1", model: str = "gpt-4o"):
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._graph = _build_graph()
        self._sessions: dict[str, dict] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}

    def create_session(
        self, session_id: str, resume_text: str, candidate_name: str = "", *,
        target_position: str = "", jd_position_name: str = "", jd_raw_text: str = "",
        custom_prompt: str = "", resume_projects: str = "",
    ) -> dict:
        projects_text = resume_projects.strip() or extract_project_details(resume_text)
        if not projects_text:
            raise ValueError("未识别到项目经历，请在项目段落中补充内容后再开始面试")
        state = {
            "messages": [], "resume_projects": projects_text, "candidate_name": candidate_name,
            "target_position": target_position, "jd_position_name": jd_position_name,
            "jd_raw_text": jd_raw_text, "custom_prompt": custom_prompt,
            "projects": [], "project_details": [], "project_index": 0,
            "main_question_number": 1, "round_number": 1, "main_question": "",
            "question_history": [], "question_count": 0, "pending_question": "", "is_finished": False,
            "api_key": self._api_key, "base_url": self._base_url, "model": self._model,
        }
        self._sessions[session_id] = state
        self._session_locks[session_id] = asyncio.Lock()
        return state

    def _invoke(self, state: dict, callback) -> dict:
        token = question_stream_callback.set(callback)
        try:
            return self._graph.invoke(state)
        finally:
            question_stream_callback.reset(token)

    async def _run(self, session_id: str, state: dict, stream_callback=None) -> str:
        result = await asyncio.to_thread(self._invoke, state, stream_callback)
        self._sessions[session_id] = {**state, **result}
        return next(message.content for message in reversed(result["messages"]) if isinstance(message, AIMessage))

    async def get_first_question(self, session_id: str, stream_callback=None) -> str:
        if session_id not in self._sessions:
            raise ValueError("面试会话不存在")
        async with self._session_locks[session_id]:
            state = self._sessions[session_id]
            if state["question_count"]:
                content = next(m.content for m in state["messages"] if isinstance(m, AIMessage))
                if stream_callback is not None:
                    stream_callback(content)
                return content
            return await self._run(session_id, state, stream_callback)

    async def answer_and_ask(self, session_id: str, candidate_answer: str, stream_callback=None) -> str:
        if session_id not in self._sessions:
            raise ValueError("面试会话不存在")
        async with self._session_locks[session_id]:
            state = self._sessions[session_id]
            if state["is_finished"]:
                if stream_callback is not None:
                    stream_callback(FINISH_MESSAGE)
                return FINISH_MESSAGE
            next_state = {**state, "messages": [*state["messages"], HumanMessage(content=candidate_answer)]}
            return await self._run(session_id, next_state, stream_callback)

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    def get_messages(self, session_id: str) -> list:
        state = self._sessions.get(session_id) or {}
        return [m for m in state.get("messages", []) if not isinstance(m, SystemMessage)]

    def reset(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._session_locks.pop(session_id, None)
