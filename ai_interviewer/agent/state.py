"""项目面试状态：只保存原始输入、对话和固定轮次所需的进度。"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class InterviewState(TypedDict):
    messages: list[BaseMessage]
    resume_projects: str
    candidate_name: str
    target_position: str
    jd_position_name: str
    jd_raw_text: str
    projects: list[str]  # 首题从项目经历识别出的项目名称，按原文顺序
    main_question: str  # 当前主问题，供两次追问保持同一个方向
    question_count: int
    is_finished: bool
    api_key: str
    base_url: str
    model: str
    custom_prompt: str
