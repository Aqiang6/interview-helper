"""面试状态：显式游标、当前生成的问题和已经提交的问题记录。"""

from typing import TypedDict

from langchain_core.messages import BaseMessage


class QuestionRecord(TypedDict):
    project_index: int
    project_name: str
    main_question_number: int
    round_number: int
    question: str
    message_index: int


class InterviewState(TypedDict):
    messages: list[BaseMessage]
    resume_projects: str
    candidate_name: str
    target_position: str
    jd_position_name: str
    jd_raw_text: str
    projects: list[str]
    project_details: list[str]
    project_index: int  # 下一道待提交问题的项目，0 起始
    main_question_number: int  # 1..3
    round_number: int  # 1..3，首问 + 两次追问
    main_question: str
    question_history: list[QuestionRecord]
    question_count: int  # 仅计已经成功生成并提交的问题
    pending_question: str
    is_finished: bool
    api_key: str
    base_url: str
    model: str
    custom_prompt: str
