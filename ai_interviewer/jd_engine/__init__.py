"""加载岗位信息，保留 JD 原文供面试提问。"""

from ai_interviewer.jd_engine.jd_loader import (
    JobPosition,
    get_position,
    list_positions,
    load_jobs,
)

__all__ = [
    "JobPosition",
    "load_jobs",
    "list_positions",
    "get_position",
]
