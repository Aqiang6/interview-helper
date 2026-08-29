"""全局配置管理 - 通过环境变量和 .env 文件加载所有配置项"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# 先把 .env 里的 HF_ENDPOINT / 代理类变量显式写入 os.environ，避免 pydantic_settings 的
# env_file 只喂给 BaseSettings fields，第三方库（huggingface_hub、sentence_transformers）
# 又直接读 os.environ 拿不到的问题。
try:
    from dotenv import dotenv_values  # type: ignore
except Exception:  # pragma: no cover
    dotenv_values = None

if dotenv_values is not None:
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.is_file():
        for _k, _v in dotenv_values(_ENV_PATH, encoding="utf-8").items():
            if _v is None:
                continue
            # 仅覆盖当前进程未显式传入的值：这样命令行 export 仍可覆盖 .env。
            os.environ.setdefault(_k, _v)


class AppSettings(BaseSettings):
    """应用全局配置，所有敏感信息均从环境变量加载，禁止硬编码"""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM Provider 配置 ──
    openai_api_key: str = Field(default="", description="主模型 API Key")
    openai_base_url: str = Field(default="https://api.openai.com/v1", description="主模型 Base URL")
    openai_model: str = Field(default="gpt-4o", description="主模型名称")

    fallback_api_key: str = Field(default="", description="备用模型 API Key")
    fallback_base_url: str = Field(default="https://api.openai.com/v1", description="备用模型 Base URL")
    fallback_model: str = Field(default="gpt-3.5-turbo", description="备用模型名称")

    # ── 加密密钥 ──
    encryption_key: str = Field(default="", description="Fernet 对称加密密钥，为空时自动生成")

    # ── LangGraph Agent ──
    agent_max_questions: int = Field(default=15, ge=3, le=50, description="单次面试最大问题数")
    agent_recursion_limit: int = Field(default=10, ge=3, le=25, description="LangGraph 递归限制")
    agent_enable_tools: bool = Field(default=True, description="是否启用 Tool Use")

    # ── OpenTelemetry ──
    otel_exporter_otlp_endpoint: str = Field(default="http://localhost:4317")
    otel_service_name: str = Field(default="ai-agent-interviewer")

    # ── 服务配置 ──
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000, ge=1024, le=65535)

    # ── 刷题助手：PostgreSQL + pgvector 知识库 ──
    postgres_dsn: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/interview_helper",
        description="PostgreSQL 连接串（SQLAlchemy 风格），用户需先 CREATE EXTENSION vector",
    )
    quiz_embedding_backend: str = Field(
        default="sentence_transformers",
        description="刷题助手向量后端：sentence_transformers（本地模型，免 Key）或 openai（走 OPENAI_API_KEY）",
    )
    quiz_sentence_transformer_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence_transformers 本地模型名；默认 all-MiniLM-L6-v2 输出 384 维，中文题可用 sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    quiz_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="仅 quiz_embedding_backend=openai 时使用：刷题助手向量检索 embedding 模型（走 openai_base_url / openai_api_key）",
    )
    quiz_embedding_dimensions: int = Field(
        default=384,
        description="向量维度：all-MiniLM-L6-v2=384 / paraphrase-multilingual-MiniLM-L12-v2=384 / text-embedding-3-small 默认 1536；schema 首次初始化时生效",
    )
    quiz_top_k: int = Field(default=10, ge=1, le=200, description="自定义主题向量检索返回题数")
    quiz_min_similarity: float = Field(
        default=0.3, ge=0.0, le=1.0, description="自定义主题向量检索最低相似度（余弦），低于阈值的题会被过滤"
    )
    quiz_crawler_timeout: float = Field(default=30.0, ge=5.0, description="爬取单页超时秒数")
    quiz_crawler_concurrency: int = Field(default=5, ge=1, le=32, description="爬取并发数")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """获取全局配置单例"""
    return AppSettings()
