"""LangGraph 面试工作流 - Graph Definition

使用 LangGraph StateGraph 构建面试 Agent 的状态机：

    START → plan_interview → generate_question → END (第一个问题)
                                        ↓ (候选人回答后)
                                  evaluate_response → decide_next
                                        ↓              ↓        ↓
                                   followup       next_topic    end
                                        ↓              ↓        ↓
                                  generate_question    generate_question → generate_evaluation → END
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from ai_interviewer.agent.nodes import (
    decide_next,
    evaluate_response,
    followup,
    generate_evaluation,
    generate_question,
    next_topic,
    plan_interview,
)
from ai_interviewer.agent.state import CandidateProfile, InterviewState
from ai_interviewer.config import get_settings

logger = logging.getLogger(__name__)


def _route_after_plan(state: InterviewState) -> str:
    """条件路由：plan_interview 之后的路径选择

    - 首次调用（无消息）→ generate_question
    - 后续调用（有候选人回答）→ evaluate_response
    """
    messages = state.get("messages", [])
    # 如果有 HumanMessage（候选人已回答），走评估流程
    has_human = any(isinstance(m, HumanMessage) for m in messages)
    if has_human:
        return "evaluate_response"
    return "generate_question"


def _build_graph():
    """构建 LangGraph 面试工作流"""
    workflow = StateGraph(InterviewState)

    # 添加节点
    workflow.add_node("plan_interview", plan_interview)
    workflow.add_node("generate_question", generate_question)
    workflow.add_node("evaluate_response", evaluate_response)
    workflow.add_node("decide_next", lambda state: {})  # 路由节点，不修改状态
    workflow.add_node("next_topic", next_topic)
    workflow.add_node("followup", followup)
    workflow.add_node("generate_evaluation", generate_evaluation)

    # 设置入口
    workflow.set_entry_point("plan_interview")

    # 条件路由：plan_interview 之后根据是否有候选人回答决定走哪条路
    workflow.add_conditional_edges(
        "plan_interview",
        _route_after_plan,
        {
            "generate_question": "generate_question",
            "evaluate_response": "evaluate_response",
        },
    )

    # evaluate_response → decide_next（条件路由）
    workflow.add_edge("evaluate_response", "decide_next")

    # decide_next 返回的字符串决定走哪条边
    workflow.add_conditional_edges(
        "decide_next",
        decide_next,
        {
            "followup": "followup",
            "next_topic": "next_topic",
            "end": "generate_evaluation",
        },
    )

    # next_topic → generate_question
    workflow.add_edge("next_topic", "generate_question")

    # followup → generate_question
    workflow.add_edge("followup", "generate_question")

    # generate_evaluation → END
    workflow.add_edge("generate_evaluation", END)

    # generate_question → END（输出问题后暂停，等待候选人回答）
    workflow.add_edge("generate_question", END)

    return workflow.compile()


# ── 对外接口 ──

class InterviewGraphAgent:
    """LangGraph 面试 Agent - 对外统一接口

    封装 LangGraph 工作流，提供与原 InterviewAgent 兼容的接口。
    """

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o",
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._graph = _build_graph()
        self._sessions: dict[str, dict] = {}

    def create_session(
        self,
        session_id: str,
        resume_text: str,
        candidate_name: str = "",
        skills: list[str] | None = None,
        *,
        target_position: str = "",
        jd_position_name: str = "",
        jd_raw_text: str = "",
        custom_prompt: str = "",
        resume_projects: str = "",
    ) -> dict:
        """创建面试会话，初始化 Agent 状态

        JD 相关字段为可选：
        - target_position: 岗位 ID（比如 agent_dev_engineer），前端选择后端后直接传入
        - jd_position_name: 岗位中文名（比如「Agent 开发工程师」）
        - jd_raw_text: 岗位 JD 原文（要求+加分项原文，plan_interview 里传给 LLM 做动态优先级）
        - custom_prompt: 用户"再来一次"时填写的定向调整需求，注入面试官系统提示词
        """
        self._sessions[session_id] = {
            "messages": [],
            "resume_text": resume_text,
            "resume_projects": resume_projects,
            "candidate_name": candidate_name,
            "skills": skills or [],
            "candidate_profile": {
                "name": candidate_name,
                "skills": skills or [],
                "answered_topics": [],
                "difficulty_level": "mid",
            },
            "interview_plan": {},
            "question_count": 0,
            "max_questions": 15,
            "current_topic": "",
            "is_finished": False,
            "last_tool_result": "",
            "answer_analysis": "",
            "custom_prompt": custom_prompt,
            "api_key": self._api_key,
            "base_url": self._base_url,
            "model": self._model,
            # JD 字段（plan_interview 节点读取后决定 topic 优先级和配额）
            "target_position": target_position,
            "jd_position_name": jd_position_name,
            "jd_raw_text": jd_raw_text,
        }
        return self._sessions[session_id]

    async def get_first_question(self, session_id: str) -> str:
        """获取第一个面试问题：plan_interview → generate_question"""
        state = self._sessions.get(session_id)
        if not state:
            return "面试会话不存在"

        # 运行 graph：START → plan_interview → generate_question → END
        result = self._graph.invoke(state)

        # 更新状态
        self._sessions[session_id] = {**state, **result}

        # 提取最后一条 AI 消息
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                return msg.content

        return "你好！让我们开始面试吧，请先介绍一下你自己。"

    async def answer_and_ask(self, session_id: str, candidate_answer: str) -> str:
        """候选人回答后，Agent 评估并生成追问/下一个问题"""
        state = self._sessions.get(session_id)
        if not state:
            return "面试会话不存在"
        if state.get("is_finished"):
            return "面试已结束"

        # 添加候选人回答到消息历史
        messages = list(state.get("messages", []))
        messages.append(HumanMessage(content=candidate_answer))
        state["messages"] = messages

        # 检查是否达到最大问题数
        if state.get("question_count", 0) >= state.get("max_questions", 15):
            # 直接生成评估
            result = self._graph.invoke(
                state,
                config={"recursion_limit": 10},
            )
            self._sessions[session_id] = {**state, **result}
            return "面试到此结束，感谢你的参与！请查看评估报告。"

        # 运行 graph：evaluate_response → decide_next → (followup/next_topic) → generate_question → END
        # 需要跳过 plan_interview，直接从 evaluate_response 开始
        result = self._graph.invoke(
            state,
            config={"recursion_limit": 10},
        )

        self._sessions[session_id] = {**state, **result}

        # 提取最后一条 AI 消息
        result_messages = result.get("messages", [])
        for msg in reversed(result_messages):
            if isinstance(msg, AIMessage):
                return msg.content

        return "好的，请继续。能详细说说吗？"

    async def get_evaluation(self, session_id: str) -> dict:
        """生成面试评估报告

        直接调用 generate_evaluation 节点，避免 graph 路由在未达到 max_questions 时
        走到 generate_question 分支导致评估无法生成。
        """
        import json

        state = self._sessions.get(session_id)
        if not state:
            return {"error": "面试会话不存在"}

        # 如果已经有评估结果
        if state.get("last_tool_result"):
            try:
                return json.loads(state["last_tool_result"])
            except (json.JSONDecodeError, TypeError):
                pass

        # 直接调用 generate_evaluation 节点，跳过 graph 路由
        result = generate_evaluation(state)
        self._sessions[session_id].update(result)

        try:
            return json.loads(result.get("last_tool_result", "{}"))
        except (json.JSONDecodeError, TypeError):
            return {
                "conversation": [],
                "answered_topics": [],
                "question_count": 0,
                "comment": "评估生成失败",
            }

    def get_session(self, session_id: str) -> dict | None:
        """获取会话状态"""
        return self._sessions.get(session_id)

    def get_messages(self, session_id: str) -> list:
        """获取会话消息（不含 system）"""
        state = self._sessions.get(session_id)
        if not state:
            return []
        messages = state.get("messages", [])
        return [m for m in messages if not isinstance(m, SystemMessage)]

    def reset(self, session_id: str) -> None:
        """重置会话"""
        self._sessions.pop(session_id, None)
