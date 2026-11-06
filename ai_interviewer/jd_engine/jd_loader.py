"""岗位 JD 加载与解析引擎 — 基于腾讯校招 JD 文本。

结构化数据供：
1. 前端选岗位的下拉列表（/api/positions）
2. 将 JD 原文直接提供给项目面试 LLM
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class JobPosition:
    """单个岗位 JD 的结构化视图。"""

    position_id: str                      # 英文 id，如 "agent_dev_engineer"
    name: str                             # 中文岗位名，如 "Agent开发工程师"
    level: str                            # 职级（应届毕业生/社招等）
    # 前端下拉展示需要的展示字段（腾讯校招 JD 原文没拆分，做合理默认值；如需自定义可在 parse 时覆盖）
    group: str = "TEG AI 平台部"            # 所属 BG/部门
    location: str = "深圳 / 北京"            # 工作地点
    description: list[str] = field(default_factory=list)   # 岗位描述（分条）
    requirements: list[str] = field(default_factory=list)  # 岗位要求（分条）
    bonuses: list[str] = field(default_factory=list)       # 加分项（分条）
    raw_text: str = ""                                        # JD 完整原文（给 LLM 读）

    @property
    def id(self) -> str:
        """对外接口统一用 id 字段（position_id 的别名，前端习惯取 id）。"""
        return self.position_id

    @property
    def short_tagline(self) -> str:
        """前端下拉列表展示的简短描述（取岗位要求第1条）。"""
        if self.requirements:
            head = self.requirements[0]
            return (head[:36] + "…") if len(head) > 38 else head
        return self.level

    def to_dict(self) -> dict:
        return {
            "position_id": self.position_id,
            "name": self.name,
            "level": self.level,
            "short_tagline": self.short_tagline,
            "sections": {
                "description": self.description,
                "requirements": self.requirements,
                "bonuses": self.bonuses,
            },
        }


# JD 文件路径（与本模块同目录或 project 根）
_DEFAULT_JD_PATH = Path(__file__).resolve().parent.parent / "腾讯jd.txt"

# 岗位名 → position_id 映射（解析时用）
# 注意：对外接口前端用的 id 保持语义清晰；同时保留别名避免传错。
_NAME_TO_ID = {
    "ai全栈工程师": "ai_fullstack_engineer",
    "全栈ai工程师": "ai_fullstack_engineer",
    "agent开发工程师": "agent_dev_engineer",
    "agent工程师": "agent_dev_engineer",
    "ai应用工程师": "ai_application_engineer",
    "后台开发": "backend_engineer",
    "后端开发": "backend_engineer",
    "后台开发工程师": "backend_engineer",
}


def _slug_to_id(name: str) -> str:
    key = name.strip().lower()
    if key in _NAME_TO_ID:
        return _NAME_TO_ID[key]
    # 兜底：中文拼音式 slug
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip()).strip("_").lower()
    return safe or "unknown_position"


def _parse_jd_text(text: str) -> list[JobPosition]:
    """解析 JD 原文 → 结构化岗位列表。"""
    if not text:
        return []
    lines = [ln.rstrip() for ln in text.splitlines()]

    # 定位每个岗位的起始行：下一行是 "技术"，再下一行是"应届毕业生" → 这就是岗位头
    positions: list[JobPosition] = []
    starts: list[int] = []
    for i in range(len(lines) - 2):
        if (
            lines[i].strip()
            and lines[i + 1].strip() == "技术"
            and "届毕业生" in lines[i + 2]
        ):
            starts.append(i)

    # 每段切片
    for seg_idx, start in enumerate(starts):
        end = starts[seg_idx + 1] if seg_idx + 1 < len(starts) else len(lines)
        block = lines[start:end]
        positions.append(_parse_single_block(block))

    return [p for p in positions if p.name]


def _parse_single_block(block: list[str]) -> JobPosition:
    """解析单个岗位的文本块。"""
    pos = JobPosition(position_id="", name="", level="")
    pos.name = block[0].strip()
    # block[1] = "技术"
    pos.level = block[2].strip() if len(block) > 2 else "应届毕业生"
    pos.position_id = _slug_to_id(pos.name)

    # 分 section：岗位描述 / 岗位要求 / 加分项或注意事项
    section_name = None
    numbered_re = re.compile(r"^\s*\d+[、.．]\s*(.+)$")

    for line in block[3:]:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "岗位描述":
            section_name = "description"
            continue
        if stripped == "岗位要求":
            section_name = "requirements"
            continue
        if "加分项" in stripped or "注意事项" in stripped:
            section_name = "bonuses"
            continue
        m = numbered_re.match(stripped)
        if m and section_name:
            item = m.group(1).strip()
            if section_name == "description":
                pos.description.append(item)
            elif section_name == "requirements":
                pos.requirements.append(item)
            elif section_name == "bonuses":
                pos.bonuses.append(item)
        elif section_name:
            # 没有编号但属于当前 section 的头行（岗位描述里的"你是腾讯产品背后的英雄"这种就是）
            target = {
                "description": pos.description,
                "requirements": pos.requirements,
                "bonuses": pos.bonuses,
            }[section_name]
            if target:
                # 追加到上一条末尾（作为解释性语句）
                target[-1] = target[-1] + " " + stripped
            else:
                target.append(stripped)

    # raw_text：岗位块完整原文（留着 LLM 深度消费）
    pos.raw_text = "\n".join(b for b in block if b.strip())
    return pos


# ═══════════════════════════════════════════
#  全局 Loader（单例 + 懒加载）
# ═══════════════════════════════════════════

_positions_cache: list[JobPosition] | None = None


def load_jobs(path: str | Path | None = None, force_reload: bool = False) -> list[JobPosition]:
    """加载并解析 JD 文件。默认读 ai_interviewer/腾讯jd.txt。"""
    global _positions_cache
    if _positions_cache is None or force_reload:
        jd_path = Path(path) if path else _DEFAULT_JD_PATH
        if not jd_path.exists():
            logger.warning("JD 文件不存在: %s", jd_path)
            _positions_cache = []
            return []
        text = jd_path.read_text(encoding="utf-8", errors="ignore")
        _positions_cache = _parse_jd_text(text)
        logger.info("加载 JD 成功: %d 个岗位 [%s]", len(_positions_cache),
                    ", ".join(p.name for p in _positions_cache))
    return _positions_cache


def list_positions() -> list[JobPosition]:
    """返回岗位对象列表（前端选岗位 / 后端按 id 查找都统一用 JobPosition）。"""
    return list(load_jobs())


def get_position(position_id_or_name: str) -> JobPosition | None:
    """按 position_id 或中文名模糊匹配单个岗位。"""
    if not position_id_or_name:
        return None
    jobs = load_jobs()
    key = position_id_or_name.strip().lower()
    for j in jobs:
        if j.position_id.lower() == key or j.name.lower() == key:
            return j
    # 模糊：id/name 包含关键词
    for j in jobs:
        if key in j.position_id.lower() or key in j.name.lower():
            logger.info("模糊匹配岗位: %s -> %s", position_id_or_name, j.name)
            return j
    return None


def create_custom_position(name: str, jd_text: str) -> JobPosition:
    """根据用户自定义输入创建临时岗位（不入库，仅用于当前面试）

    Args:
        name: 岗位名称（如"资深 Java 后端工程师"）
        jd_text: JD 原文（岗位描述 + 要求 + 加分项，纯文本即可）

    Returns:
        JobPosition 对象，position_id 以 custom_ 前缀标识
    """
    position_id = "custom_" + _slug_to_id(name)
    # 尝试简单分段：按常见标题词切分，不强制结构化
    lines = [ln.strip() for ln in jd_text.splitlines() if ln.strip()]
    description: list[str] = []
    requirements: list[str] = []
    bonuses: list[str] = []
    section = None
    for line in lines:
        if any(kw in line for kw in ["岗位描述", "工作内容", "职责描述", "Job Description"]):
            section = "description"
            continue
        if any(kw in line for kw in ["岗位要求", "任职要求", "任职资格", "Requirements", "任职条件"]):
            section = "requirements"
            continue
        if any(kw in line for kw in ["加分项", "优先条件", "Bonus", "Preferred"]):
            section = "bonuses"
            continue
        if section == "description":
            description.append(line)
        elif section == "requirements":
            requirements.append(line)
        elif section == "bonuses":
            bonuses.append(line)
        else:
            # 没有明确段落标记 → 全归到 requirements
            requirements.append(line)

    # 如果没有任何分段命中，把整段 jd_text 作为 requirements
    if not description and not requirements and not bonuses:
        requirements = [jd_text.strip()]

    return JobPosition(
        position_id=position_id,
        name=name.strip() or "自定义岗位",
        level="自定义",
        group="自定义岗位",
        location="不限",
        description=description,
        requirements=requirements,
        bonuses=bonuses,
        raw_text=jd_text,
    )
