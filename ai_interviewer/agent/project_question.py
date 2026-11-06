"""首题响应中只保留问题正文，项目名称用于后端固定计数。"""

from __future__ import annotations

import json
from collections.abc import Callable


class ProjectQuestion:
    def __init__(self, callback: Callable[[str], None] | None = None):
        self.callback = callback
        self.pending = ""
        self.projects: list[str] = []
        self.text = ""

    def feed(self, delta: str) -> None:
        if not self.projects:
            self.pending += delta
            content = self.pending.lstrip()
            if "<projects>".startswith(content):
                return
            if not content.startswith("<projects>"):
                raise ValueError("模型未返回项目名称，请重试首题")
            end = content.find("</projects>")
            if end < 0:
                if len(content) > 8192:
                    raise ValueError("模型返回的项目名称过长，请重试首题")
                return
            projects = json.loads(content[len("<projects>"):end])
            if not isinstance(projects, list) or not projects or any(
                not isinstance(name, str) or not name.strip() for name in projects
            ):
                raise ValueError("模型未识别到有效项目，请检查项目经历内容")
            self.projects = [name.strip() for name in projects]
            delta = content[end + len("</projects>"):]
            self.pending = ""
        if not self.text:
            delta = delta.lstrip()
        if delta:
            self.text += delta
            if self.callback is not None:
                self.callback(delta)

    def finish(self) -> None:
        if not self.projects or not self.text.strip():
            raise ValueError("模型未完整返回项目名称和问题，请重试首题")
