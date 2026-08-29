"""简历缓存服务 - 避免重复解析

上传同一份 PDF 时直接从缓存读取解析结果，无需每次重新解析。
缓存键为 PDF 文件内容的 SHA256 哈希值，存储位置为 ai_interviewer/.resume_cache/。

注意：已取消简历向量化（原 .npy 向量缓存已移除）——经核查全代码库无任何
模块消费 CachedResume.vector，简历向量属冗余链路，删除后只缓存文本解析结果。
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).resolve().parent / ".resume_cache"


@dataclass
class CachedResume:
    """缓存的简历数据（解析结果，纯文本，不含向量）"""
    raw_text: str = ""
    name: str = ""
    skills: list[str] = field(default_factory=list)
    projects: list[str] = field(default_factory=list)
    experience_years: str = ""
    education: str = ""
    summary: str = ""

    def to_dict(self) -> dict:
        return {
            "raw_text": self.raw_text,
            "name": self.name,
            "skills": self.skills,
            "projects": self.projects,
            "experience_years": self.experience_years,
            "education": self.education,
            "summary": self.summary,
        }


def _ensure_cache_dir() -> Path:
    if not _CACHE_DIR.exists():
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _CACHE_DIR


def _pdf_hash(file_bytes: bytes) -> str:
    """计算 PDF 文件内容的 SHA256 哈希作为缓存键"""
    return hashlib.sha256(file_bytes).hexdigest()


def _meta_path(hash_key: str) -> Path:
    return _ensure_cache_dir() / f"{hash_key}.json"


def get_cached(file_bytes: bytes) -> CachedResume | None:
    """查找缓存中是否存在该 PDF 的解析结果，命中则返回 CachedResume"""
    hash_key = _pdf_hash(file_bytes)
    meta_file = _meta_path(hash_key)
    if not meta_file.exists():
        return None

    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        cached = CachedResume(
            raw_text=data.get("raw_text", ""),
            name=data.get("name", ""),
            skills=data.get("skills", []),
            projects=data.get("projects", []),
            experience_years=data.get("experience_years", ""),
            education=data.get("education", ""),
            summary=data.get("summary", ""),
        )
        logger.info("简历缓存命中: hash=%s... skills=%d", hash_key[:12], len(cached.skills))
        return cached
    except Exception as e:
        logger.warning("读取简历缓存失败，将重新解析: %s", e)
        return None


def list_cached_resumes() -> list[dict]:
    """列出所有已缓存的简历摘要（不含 raw_text，供前端「已保存简历」界面展示选择）"""
    if not _CACHE_DIR.exists():
        return []
    items: list[dict] = []
    for meta_file in _CACHE_DIR.glob("*.json"):
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            hash_key = meta_file.stem
            stat = meta_file.stat()
            items.append({
                "hash": hash_key,
                "name": data.get("name", "") or "未命名简历",
                "skills": data.get("skills", []) or [],
                "projects": data.get("projects", []) or [],
                "experience_years": data.get("experience_years", "") or "",
                "education": data.get("education", "") or "",
                "summary": data.get("summary", "") or "",
                "cached_at": int(stat.st_mtime),
            })
        except Exception as e:
            logger.warning("读取缓存简历摘要失败 %s: %s", meta_file.name, e)
    # 按缓存时间倒序（最新在前）
    items.sort(key=lambda x: x.get("cached_at", 0), reverse=True)
    return items


def get_cached_by_hash(hash_key: str) -> CachedResume | None:
    """根据 hash 获取完整缓存简历（含 raw_text），供前端选中后直接开始面试"""
    meta_file = _meta_path(hash_key)
    if not meta_file.exists():
        return None
    try:
        data = json.loads(meta_file.read_text(encoding="utf-8"))
        cached = CachedResume(
            raw_text=data.get("raw_text", ""),
            name=data.get("name", ""),
            skills=data.get("skills", []),
            projects=data.get("projects", []),
            experience_years=data.get("experience_years", ""),
            education=data.get("education", ""),
            summary=data.get("summary", ""),
        )
        logger.info("按 hash 读取缓存简历: hash=%s... skills=%d", hash_key[:12], len(cached.skills))
        return cached
    except Exception as e:
        logger.warning("读取缓存简历失败 %s: %s", hash_key, e)
        return None


def save_cached(file_bytes: bytes, resume_data) -> None:
    """保存解析结果到缓存

    Args:
        file_bytes: 原始 PDF 字节
        resume_data: ResumeData 对象（有 raw_text/name/skills 等字段）
    """
    hash_key = _pdf_hash(file_bytes)
    meta_file = _meta_path(hash_key)

    try:
        meta_data = {
            "raw_text": resume_data.raw_text,
            "name": resume_data.name,
            "skills": resume_data.skills,
            "projects": resume_data.projects,
            "experience_years": resume_data.experience_years,
            "education": resume_data.education,
            "summary": resume_data.summary,
        }
        meta_file.write_text(json.dumps(meta_data, ensure_ascii=False), encoding="utf-8")

        logger.info("简历缓存已保存: hash=%s... skills=%d", hash_key[:12], len(resume_data.skills))
    except Exception as e:
        logger.warning("保存简历缓存失败（不影响当前解析）: %s", e)
