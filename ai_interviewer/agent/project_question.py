"""模型结构化响应的解析与项目原文边界校验。"""

from __future__ import annotations

import json


def parse_object(content: str) -> dict:
    text = content.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.partition("\n")[2].rsplit("```", 1)[0].strip()
    try:
        value = json.loads(text)
    except ValueError as exc:
        raise ValueError("模型未返回有效 JSON 对象") from exc
    if not isinstance(value, dict):
        raise ValueError("模型未返回有效 JSON 对象")
    return value


def parse_projects(content: str, source: str) -> tuple[list[str], list[str]]:
    projects = parse_object(content).get("projects")
    if not isinstance(projects, list) or not projects:
        raise ValueError("模型未识别到有效项目，请检查项目经历内容")
    names, starts = [], []
    search_start = 0
    for project in projects:
        if not isinstance(project, dict):
            raise ValueError("项目必须包含名称和项目原文")
        name, marker = project.get("name"), project.get("start")
        if not isinstance(name, str) or not name.strip() or not isinstance(marker, str) or not marker.strip():
            raise ValueError("项目必须包含名称和原文起始标记")
        name, marker = name.strip(), marker.strip()
        start = source.find(marker, search_start)
        if start < 0 or name.casefold() in {item.casefold() for item in names}:
            raise ValueError("项目名称重复，或项目起始标记被改写、重叠、顺序不正确")
        names.append(name)
        starts.append(start)
        search_start = start + len(marker)
    # 按原文位置切片，避免让模型复制长经历导致截断或改写。
    ends = [*starts[1:], len(source)]
    details = [source[start:end].strip() for start, end in zip(starts, ends)]
    return names, details
