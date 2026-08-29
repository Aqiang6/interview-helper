"""刷题助手 - 数据访问层。

表结构（方案 A：一题一行，答案整块保存，只对题干元信息做向量化）::

    quiz_questions
    ├── id              BIGSERIAL PK
    ├── big_topic       TEXT NOT NULL          (H1，用于 GROUP BY / WHERE 精准筛选)
    ├── mid_topic       TEXT NOT NULL          (H2，子章节，精准定位)
    ├── question        TEXT NOT NULL          (题干，原样保留 ⭐️ / emoji 等)
    ├── answer_md       TEXT NOT NULL          (答案 Markdown，含链接/图片/表格/代码)
    ├── source_url      TEXT NOT NULL          (来源页 URL)
    ├── answer_anchor   TEXT NOT NULL          (页内锚点，用来拼回 #anchor)
    ├── source_hash     TEXT UNIQUE NOT NULL   (sha1(source_url + '#' + answer_anchor)，唯一去重键)
    ├── ordinal         INTEGER NOT NULL DEFAULT 0 (同一页面题目的顺序，用于保留原题序)
    ├── embedding       vector(N) NOT NULL     (对 big_topic + "|" + mid_topic + "|" + question 做的向量)
    ├── created_at      TIMESTAMPTZ DEFAULT NOW()
    ├── updated_at      TIMESTAMPTZ DEFAULT NOW()

索引::
    - UNIQUE (source_hash)  增量去重
    - (big_topic, mid_topic)  按主题拉题
    - GIN (to_tsvector('simple', question))  可选，后续加关键词搜索用
    - HNSW on embedding (vector_cosine_ops)  自定义主题时的向量 ANN

设计说明：
- 全部用原生 SQL + psycopg 文本参数，向量类型用 pgvector 自带的 ``vector``
- 不把 pgvector 的 Python ORM 扩展（VectorType）强绑定进来，方便未装扩展时也能 import；
  在写入/检索时通过 SQL 字面量拼接 vector(N) 字符串即可。
"""

from __future__ import annotations

import hashlib
import logging
import re
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, Optional, Sequence

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    Integer,
    String,
    Table,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.engine import Engine, create_engine
from sqlalchemy.orm import Session, registry, sessionmaker

from ai_interviewer.config import get_settings

logger = logging.getLogger(__name__)

mapper_registry = registry()
metadata = mapper_registry.metadata


# ═══════════════════════════════════════════
#  ORM Model
# ═══════════════════════════════════════════

@mapper_registry.mapped
@dataclass
class QuizQuestion:
    """一道面试题（方案 A：一题=一行）。"""

    __table__ = Table(
        "quiz_questions",
        metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("big_topic", Text, nullable=False, comment="H1 大标题 / 主题名，如 Java 基础 / 大模型基础"),
        Column("mid_topic", Text, nullable=False, comment="H2 中标题 / 子章节名，如 基础概念与常识 / LLM 运行机制"),
        Column("question", Text, nullable=False, comment="题干，原样保留 ⭐️ / 标点 / emoji"),
        Column("answer_md", Text, nullable=False, comment="完整答案 Markdown，含链接/图片/表格/代码块"),
        Column("source_url", Text, nullable=False, comment="来源页面 URL（不含 #anchor）"),
        Column("answer_anchor", Text, nullable=False, comment="页内 anchor，如 jvm-vs-jdk-vs-jre"),
        Column("source_hash", String(64), nullable=False, comment="sha1(source_url + '#' + answer_anchor) 去重键"),
        Column("ordinal", Integer, nullable=False, default=0, comment="同页内题目的先后顺序"),
        Column(
            # 延迟到运行时再 CREATE COLUMN，因为需要读 QUIZ_EMBEDDING_DIMENSIONS
            "embedding",
            Text,
            nullable=False,
            comment=f"embedding 向量字面量，形如 '[0.1, 0.2, ...]'，真实列类型为 vector(N)",
        ),
        Column("created_at", DateTime(timezone=True), server_default=text("NOW()")),
        Column("updated_at", DateTime(timezone=True), server_default=text("NOW()")),
        UniqueConstraint("source_hash", name="uq_quiz_questions_source_hash"),
        Index(
            "ix_quiz_questions_big_mid_topic",
            "big_topic",
            "mid_topic",
        ),
        Index(
            "ix_quiz_questions_big_topic",
            "big_topic",
        ),
        # NOTE: 向量 HNSW 索引在 ensure_schema() 中按实际维度 DDL 创建，不写进 ORM metadata
    )

    id: int
    big_topic: str
    mid_topic: str
    question: str
    answer_md: str
    source_url: str
    answer_anchor: str
    source_hash: str
    ordinal: int
    embedding: str
    created_at: datetime
    updated_at: datetime


# ═══════════════════════════════════════════
#  Engine / Session 工厂
# ═══════════════════════════════════════════

_engine: Optional[Engine] = None
_session_factory: Optional[sessionmaker] = None


def get_engine() -> Engine:
    """懒加载单例 Engine（psycopg3 同步连接）。"""
    global _engine, _session_factory
    if _engine is None:
        settings = get_settings()
        dsn = settings.postgres_dsn
        if not dsn:
            raise RuntimeError("未配置 POSTGRES_DSN，请在 .env 中设置后再使用刷题助手")
        # psycopg3：pipeline 可减少往返；pool_pre_ping 保活；pool_recycle 避免连接被 PG 侧超时断开
        _engine = create_engine(
            dsn,
            future=True,
            pool_pre_ping=True,
            pool_recycle=1800,
            pool_size=10,
            max_overflow=20,
            connect_args={
                # psycopg[binary] kwargs：自动解码 bytes->str 等默认即可
            },
        )
        _session_factory = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def session_factory() -> sessionmaker:
    get_engine()
    assert _session_factory is not None  # pragma: no cover - get_engine 后必初始化
    return _session_factory


@contextmanager
def session_scope() -> Iterator[Session]:
    """事务会话上下文：commit/rollback/close 全包。"""
    factory = session_factory()
    sess = factory()
    try:
        yield sess
        sess.commit()
    except Exception:
        sess.rollback()
        raise
    finally:
        sess.close()


# ═══════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════

def make_source_hash(source_url: str, anchor: str) -> str:
    """去重键 = sha1(source_url + '#' + anchor)。"""
    raw = f"{source_url.rstrip('#/')}#{anchor}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


_VECTOR_LITERAL_RE = re.compile(r"^\s*\[.*\]\s*$", re.S)


def validate_embedding_literal(value: object) -> str:
    """把任意合法输入统一转成安全的 pgvector 字面量字符串 ``'[x,y,...]'::vector``。

    支持：``list[float]`` / ``tuple[float]`` / ``np.ndarray`` / 已形如 ``[x,...]`` 的字符串。
    输出**带外层单引号**，方便调用方直接拼进 SQL（避免 SQLAlchemy text() 的冒号参数解析误匹配）。
    """
    if value is None:
        raise ValueError("embedding 不能为空")
    # numpy.ndarray -> list[float]
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        nums = [float(x) for x in value]
        # 紧凑格式：保留 7 位小数足够保留 MiniLM / text-embedding-3 的精度，又不会把 SQL 撑太长。
        body = ",".join(format(x, ".7f") for x in nums)
        return "'[" + body + "]'::vector"
    if isinstance(value, (bytes, bytearray)):
        s = value.decode("utf-8", errors="replace")
    elif isinstance(value, str):
        s = value
    else:
        raise TypeError(f"embedding 类型错误: {type(value).__name__}")
    s = s.strip()
    # 如果用户/上层已经写了引号或 ::vector，先剥掉再重建，避免重复。
    s_clean = s
    if s_clean.startswith("'") and s_clean.endswith("'::vector"):
        s_clean = s_clean[1:-9].rstrip("'")
    if s_clean.startswith("'") and s_clean.endswith("'"):
        s_clean = s_clean[1:-1]
    if not _VECTOR_LITERAL_RE.match(s_clean):
        raise ValueError("embedding 字符串必须形如 '[0.1, 0.2, ...]'")
    # 防止 SQL 注入：字符串里不应出现单引号（字面量数字列表里不可能有），保守起见转义一次。
    safe = s_clean.replace("'", "''")
    return "'" + safe + "'::vector"


# ═══════════════════════════════════════════
#  建表 / 建索引 / 改列类型
# ═══════════════════════════════════════════

def ensure_schema() -> None:
    """幂等地：CREATE EXTENSION + 建表 + 把 embedding TEXT 列 ALTER 成 vector(DIM) + HNSW 索引。

    只在服务启动或 ``POST /api/quiz/ingest`` 前调用一次即可；可重复调用，不会报错。
    """
    settings = get_settings()
    dim = int(settings.quiz_embedding_dimensions)
    if dim <= 0:
        raise ValueError("QUIZ_EMBEDDING_DIMENSIONS 必须为正整数")

    engine = get_engine()
    with engine.connect() as conn:
        # 1) 创建 vector 扩展（需要 superuser，失败时给出清晰错误）
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        except Exception as e:  # pragma: no cover - 纯环境依赖
            logger.exception("CREATE EXTENSION vector 失败，请确认 PostgreSQL 已安装 pgvector 扩展")
            raise RuntimeError(
                "PostgreSQL 未启用 pgvector。请在数据库服务器上安装与主版本匹配的预编译包（见"
                " https://github.com/pgvector/pgvector#installation-notes ），然后在目标库里执行 "
                f"`CREATE EXTENSION vector;`。原始错误：{e}"
            ) from e

        # 2) 先 metadata.create_all：会把 embedding 建成 TEXT 列
        metadata.create_all(conn, checkfirst=True)

        # 3) 把 embedding TEXT → vector(DIM)；如果已经是 vector(?) 则跳过
        #    注：USING 用强转。若已有旧数据不合法（非向量字面量），会抛错让用户清库。
        column_is_vector = conn.execute(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name='quiz_questions' AND column_name='embedding'"
            )
        ).scalar()
        if column_is_vector == "USER-DEFINED":
            # 已经是 vector/其它自定义类型，检查类型名是否为 vector，以及维度
            udt_name = conn.execute(
                text(
                    "SELECT udt_name FROM information_schema.columns "
                    "WHERE table_name='quiz_questions' AND column_name='embedding'"
                )
            ).scalar()
            if udt_name != "vector":  # pragma: no cover - 极端场景
                conn.execute(
                    text(
                        "ALTER TABLE quiz_questions "
                        f"ALTER COLUMN embedding TYPE vector({dim}) "
                        "USING embedding::text::vector"
                    )
                )
        else:
            conn.execute(
                text(
                    "ALTER TABLE quiz_questions "
                    f"ALTER COLUMN embedding TYPE vector({dim}) "
                    "USING embedding::vector"
                )
            )

        # 4) HNSW 余弦相似度索引（若已存在则不做）
        index_exists = conn.execute(
            text(
                "SELECT 1 FROM pg_indexes WHERE tablename='quiz_questions' "
                "AND indexname='ix_quiz_questions_embedding_hnsw'"
            )
        ).scalar()
        if not index_exists:
            # ivfflat 需要数据积累后再 build，HNSW 可以直接建且查询更快；m=16 / ef_construction=64 是常用默认
            conn.execute(
                text(
                    "CREATE INDEX ix_quiz_questions_embedding_hnsw ON quiz_questions "
                    f"USING hnsw (embedding vector_cosine_ops) WITH (m=16, ef_construction=64)"
                )
            )

        # 5) updated_at 自动更新触发器（轻量，避免每次写 ORM 都要手动 set）
        conn.execute(text(
            """
            CREATE OR REPLACE FUNCTION quiz_set_updated_at()
            RETURNS TRIGGER AS $$
            BEGIN
                NEW.updated_at := NOW();
                RETURN NEW;
            END;
            $$ LANGUAGE plpgsql;
            """
        ))
        trigger_exists = conn.execute(
            text(
                "SELECT 1 FROM pg_trigger WHERE tgname='quiz_questions_updated_at_trg' "
                "AND tgrelid='quiz_questions'::regclass"
            )
        ).scalar()
        if not trigger_exists:
            conn.execute(text(
                """
                CREATE TRIGGER quiz_questions_updated_at_trg
                BEFORE UPDATE ON quiz_questions
                FOR EACH ROW EXECUTE FUNCTION quiz_set_updated_at();
                """
            ))

        conn.commit()
    logger.info("[quiz.db] schema ready, dim=%s", dim)


# ═══════════════════════════════════════════
#  批量 upsert（按 source_hash 去重）
# ═══════════════════════════════════════════

@dataclass
class UpsertRow:
    """待入库的一行（embedding 已算好）。"""

    big_topic: str
    mid_topic: str
    question: str
    answer_md: str
    source_url: str
    answer_anchor: str
    ordinal: int
    embedding: str  # vector 字面量字符串 '[...]'


def bulk_upsert(rows: Sequence[UpsertRow]) -> dict:
    """批量 upsert，冲突时更新非 embedding 字段（题目文本/答案可能被网站修订）。

    返回统计信息：``{"inserted": n, "updated": m, "skipped": k, "total": n+m+k}``。
    """
    if not rows:
        return {"inserted": 0, "updated": 0, "skipped": 0, "total": 0}

    inserted = 0
    updated = 0
    skipped = 0

    with session_scope() as sess:
        # 为避免 psycopg 把数组当 Python list 展开、同时保证 vector 类型正确，
        # 我们直接用 SQL 字面量拼接（rows 来自可信内部 pipeline，不是用户输入），
        # 但大标题/题干/答案等字符串仍用参数绑定，避免 SQL 注入。
        for r in rows:
            source_hash = make_source_hash(r.source_url, r.answer_anchor)
            emb_lit = validate_embedding_literal(r.embedding)
            # 先查询是否存在，存在则只更新可变字段（不含 embedding，避免重算开销）
            existing = sess.execute(
                text(
                    "SELECT id, question, answer_md, big_topic, mid_topic "
                    "FROM quiz_questions WHERE source_hash=:h FOR UPDATE"
                ),
                {"h": source_hash},
            ).mappings().first()

            if existing is None:
                # 注意：emb_lit 已经是完整的 ``'[x,y,...]'::vector`` 字面量（含引号+cast），
                # 这里不再重复追加 "::vector"，避免类型转换重复。
                sess.execute(
                    text(
                        "INSERT INTO quiz_questions "
                        "(big_topic, mid_topic, question, answer_md, source_url, "
                        " answer_anchor, source_hash, ordinal, embedding) "
                        "VALUES (:bt, :mt, :q, :a, :su, :an, :sh, :ord, " + emb_lit + ")"
                    ),
                    {
                        "bt": r.big_topic,
                        "mt": r.mid_topic,
                        "q": r.question,
                        "a": r.answer_md,
                        "su": r.source_url,
                        "an": r.answer_anchor,
                        "sh": source_hash,
                        "ord": int(r.ordinal),
                    },
                )
                inserted += 1
            else:
                # 仅当内容变化时才 UPDATE（减少 UPDATE 次数 / 避免 updated_at 抖动）
                if (
                    existing["question"] != r.question
                    or existing["answer_md"] != r.answer_md
                    or existing["big_topic"] != r.big_topic
                    or existing["mid_topic"] != r.mid_topic
                ):
                    sess.execute(
                        text(
                            "UPDATE quiz_questions SET "
                            "  big_topic=:bt, mid_topic=:mt, question=:q, answer_md=:a, "
                            "  source_url=:su, answer_anchor=:an, ordinal=:ord "
                            "WHERE source_hash=:sh"
                        ),
                        {
                            "bt": r.big_topic,
                            "mt": r.mid_topic,
                            "q": r.question,
                            "a": r.answer_md,
                            "su": r.source_url,
                            "an": r.answer_anchor,
                            "ord": int(r.ordinal),
                            "sh": source_hash,
                        },
                    )
                    updated += 1
                else:
                    skipped += 1

    total = inserted + updated + skipped
    logger.info("[quiz.db] bulk_upsert done: inserted=%d updated=%d skipped=%d total=%d",
                inserted, updated, skipped, total)
    return {"inserted": inserted, "updated": updated, "skipped": skipped, "total": total}


def chunked(it: Iterable, size: int) -> Iterator[list]:
    """通用迭代分块工具（给上层 embedding 批处理 / upsert 分批用）。"""
    buf: list = []
    for x in it:
        buf.append(x)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf
