"""Agent 工具集

注意：已彻底取消回答评分机制（quality_level / 四维度分数 / 总分）。
回答对下一题的影响改由 evaluate_response 节点做定性分析（提取已覆盖要点 /
缺失点 / 追问方向）注入下一题系统提示词实现，路由决策不再依赖任何质量等级，
改为按配额和题数自然转换（默认追问 → 配额满换题 → 题数到顶结束）。

RAG 知识库检索已移除（24 条公开技术题，LLM 自身能力足够覆盖，向量检索属过度设计）。
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)


# ── 回答定性分析 LLM 系统提示词（已移除 quality_level 评分） ──
_QUALITY_EVAL_SYSTEM = """你是一位资深技术面试官，正在分析候选人的最新回答，为下一个问题做准备。

## 分析任务
请分析候选人的回答，提取关键信息用于指导下一个问题。不要对回答打分或评级，只提取"下一个问题该往哪个方向走"的信息。

## 输出要求
严格输出 JSON 格式，不要任何额外文字或 markdown 标记：
{{
  "covered_well": ["候选人回答中扎实的点1", "候选人回答中扎实的点2"],
  "gaps": ["候选人回答中缺失或薄弱的点1", "候选人回答中缺失或薄弱的点2"],
  "follow_up_direction": "如果需要追问，应该往哪个方向追问（一句话描述）"
}}

## 字段说明
- covered_well：回答中讲得清楚、可深入的点（用于判断哪些已不用再问）
- gaps：回答中讲不清、缺失、薄弱、含糊的点（用于决定追问方向）
- follow_up_direction：基于 gaps 给出下一步追问的具体方向（一句话）

注意：不要输出 quality_level、score、grade 或任何等级 / 分数判断，只做定性信息提取。"""


def _parse_analysis_json(content: str) -> dict:
    """解析 LLM 返回的定性分析 JSON，容忍 markdown 代码块和多余文本"""
    content = (content or "").strip()

    # 去除 markdown 代码块
    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 2:
            inner = parts[1].strip()
            if inner.startswith("json"):
                inner = inner[4:].strip()
            content = inner

    # 直接解析
    try:
        return _normalize_analysis_data(json.loads(content))
    except json.JSONDecodeError:
        pass

    # 兜底：截取最外层 JSON
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return _normalize_analysis_data(json.loads(content[start:end + 1]))
        except json.JSONDecodeError:
            pass

    msg = (
        "候选人回答分析失败：LLM 返回内容无法解析为合法 JSON 结构。"
        f"原始内容预览：{repr(content[:200]) if content else '(空)'}"
    )
    logger.error(msg)
    raise ValueError(msg)


def _normalize_analysis_data(data: dict) -> dict:
    """规范化 LLM 返回的定性分析数据，保证字段完整（已移除 quality_level）"""
    return {
        "covered_well": data.get("covered_well", []) or [],
        "gaps": data.get("gaps", []) or [],
        "follow_up_direction": data.get("follow_up_direction", "") or "",
    }
