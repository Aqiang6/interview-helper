"""刷题助手 - 出题检索层。

硬需求对应关系（方案 A）：
  - "根据大标题检索对应主题的所有题目" → :func:`pick_by_topic` 只传 big_topic
  - "中标题能精准对应主题下的每一道题"   → :func:`pick_by_topic` 传 (big_topic, mid_topic)
  - "用户要求显示答案时，把具体问题对应的答案显示出来，包括其中的链接" → :func:`get_answer`

三种出题模式：
  1. 随机出题                → :func:`pick_random`
  2. 用户自选主题出题        → :func:`pick_by_topic`
  3. 用户自定义主题出题      → :func:`pick_by_custom`（向量 cosine 相似度 topK + 阈值过滤）

注意：出题返回的结果里 **默认不带 answer_md**（只给题干 + 元信息），用户点击"显示答案"
按钮时才显式调 ``get_answer`` 取答案，符合刷题的交互。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from sqlalchemy import text

from ai_interviewer.config import get_settings
from ai_interviewer.quiz.db import session_scope, validate_embedding_literal
from ai_interviewer.quiz.ingest import _build_embeddings, _text_for_embedding
from ai_interviewer.quiz.splitter import QuestionChunk, absolutize_source_url_anchor

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  Pydantic-free 轻量数据结构（app.py 会再包一层 dict 返回）
# ═══════════════════════════════════════════

@dataclass
class QuizQuestionOut:
    """出题接口返回的题干卡片（不含答案）。"""

    id: int
    big_topic: str
    mid_topic: str
    question: str
    source_url: str              # 原页面 URL（不带 #anchor）
    source_url_with_anchor: str  # 带 #anchor，前端可直接"跳原题"
    ordinal: int                 # 原页顺序
    similarity: Optional[float]  # 自定义主题模式下的余弦相似度（其余模式为 None）

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "big_topic": self.big_topic,
            "mid_topic": self.mid_topic,
            "question": self.question,
            "source_url": self.source_url,
            "source_url_with_anchor": self.source_url_with_anchor,
            "ordinal": self.ordinal,
            "similarity": self.similarity,
        }


@dataclass
class TopicMidNode:
    mid_topic: str
    count: int

    def as_dict(self) -> dict:
        return {"mid_topic": self.mid_topic, "count": self.count}


@dataclass
class TopicNode:
    big_topic: str
    count: int
    mid_topics: list[TopicMidNode]

    def as_dict(self) -> dict:
        return {
            "big_topic": self.big_topic,
            "count": self.count,
            "mid_topics": [m.as_dict() for m in self.mid_topics],
        }


# ═══════════════════════════════════════════
#  Row → 输出对象
# ═══════════════════════════════════════════

def _row_to_out(row: Any, similarity: Optional[float] = None) -> QuizQuestionOut:
    src = row["source_url"] or ""
    anchor = row["answer_anchor"] or ""
    return QuizQuestionOut(
        id=int(row["id"]),
        big_topic=row["big_topic"],
        mid_topic=row["mid_topic"],
        question=row["question"],
        source_url=src,
        source_url_with_anchor=absolutize_source_url_anchor(src, anchor),
        ordinal=int(row["ordinal"]),
        similarity=float(similarity) if similarity is not None else None,
    )


# ═══════════════════════════════════════════
#  1. 主题树（大标题 -> 中标题）
# ═══════════════════════════════════════════

def list_topics() -> list[TopicNode]:
    """返回所有"大标题 → 中标题列表 → 数量"的结构，供前端"自选主题"下拉用。"""
    with session_scope() as sess:
        rows = sess.execute(text(
            "SELECT big_topic, mid_topic, COUNT(*) AS cnt "
            "FROM quiz_questions "
            "GROUP BY big_topic, mid_topic "
            "ORDER BY big_topic ASC, mid_topic ASC"
        )).mappings().all()

    tree: dict[str, TopicNode] = {}
    for r in rows:
        big = r["big_topic"]
        node = tree.get(big)
        if node is None:
            node = TopicNode(big_topic=big, count=0, mid_topics=[])
            tree[big] = node
        cnt = int(r["cnt"])
        node.count += cnt
        node.mid_topics.append(TopicMidNode(mid_topic=r["mid_topic"], count=cnt))
    return list(tree.values())


def stats() -> dict:
    """返回总题目数 / 主题数 / 中章节数 等基础统计。"""
    with session_scope() as sess:
        r = sess.execute(text(
            "SELECT "
            "  COUNT(*) AS total,"
            "  COUNT(DISTINCT big_topic) AS big_count,"
            "  COUNT(DISTINCT (big_topic, mid_topic)) AS mid_count,"
            "  COUNT(DISTINCT source_url) AS page_count "
            "FROM quiz_questions"
        )).mappings().first()
    return {
        "total_questions": int(r["total"]) if r else 0,
        "big_topics": int(r["big_count"]) if r else 0,
        "mid_topics": int(r["mid_count"]) if r else 0,
        "pages": int(r["page_count"]) if r else 0,
    }


# ═══════════════════════════════════════════
#  2. 随机出题
# ═══════════════════════════════════════════

def pick_random(count: int = 1, *, exclude_ids: Optional[Iterable[int]] = None) -> list[QuizQuestionOut]:
    """从整个题库随机抽 ``count`` 道题（PG 原生 ORDER BY RANDOM()，题量万级内直接 OK）。

    exclude_ids 可用于"同一会话里不出已经做过的题"。
    """
    count = max(1, int(count))
    exclude = list(int(x) for x in (exclude_ids or []))
    sql = (
        "SELECT id, big_topic, mid_topic, question, source_url, answer_anchor, ordinal "
        "FROM quiz_questions "
    )
    params: dict = {}
    if exclude:
        # ⚠️ PostgreSQL 的 $n 占位符只能绑定单值，不能直接展开成 IN (a,b,c)。
        #    所以这里用数组匹配语法：id = ANY(CAST($1 AS bigint[])) 等价于 IN (...)。
        #    psycopg3 会把 Python list 直接适配成 PG 数组类型。
        sql += "WHERE NOT (id = ANY(CAST(:excl AS bigint[]))) "
        params["excl"] = exclude
    sql += "ORDER BY RANDOM() LIMIT :lim"
    params["lim"] = count

    with session_scope() as sess:
        rows = sess.execute(text(sql), params).mappings().all()
    return [_row_to_out(r) for r in rows]


# ═══════════════════════════════════════════
#  3. 自选主题出题（精准筛选）
# ═══════════════════════════════════════════

def pick_by_topic(
    *,
    big_topic: str,
    mid_topic: Optional[str] = None,
    count: Optional[int] = None,
    shuffle: bool = False,
) -> list[QuizQuestionOut]:
    """按"大标题"或"大标题 + 中标题"精准拉题。

    - ``big_topic`` 必填：精确匹配（大小写全角半角敏感，跟库中一致）。如果前端想做模糊，
      可以先调 ``list_topics`` 让用户选 exact 值。
    - ``mid_topic`` 可选：传了就按章节精准过滤。
    - ``count`` 不传 = 返回该主题所有题；传了就最多取 count 条。
    - ``shuffle`` = True 时内部 ORDER BY RANDOM()；否则按 (source_url, ordinal) 保持原顺序。
    """
    clauses = ["big_topic = :bt"]
    params: dict = {"bt": big_topic}
    if mid_topic:
        clauses.append("mid_topic = :mt")
        params["mt"] = mid_topic
    order_sql = "ORDER BY RANDOM()" if shuffle else "ORDER BY source_url ASC, ordinal ASC"
    limit_sql = ""
    if count and int(count) > 0:
        limit_sql = f"LIMIT {int(count)}"
    sql = (
        "SELECT id, big_topic, mid_topic, question, source_url, answer_anchor, ordinal "
        "FROM quiz_questions WHERE " + " AND ".join(clauses) + f" {order_sql} {limit_sql}"
    )
    with session_scope() as sess:
        rows = sess.execute(text(sql), params).mappings().all()
    return [_row_to_out(r) for r in rows]


# ═══════════════════════════════════════════
#  4. 自定义主题出题（向量 cosine 检索）
# ═══════════════════════════════════════════

def pick_by_custom(
    custom_topic: str,
    *,
    count: Optional[int] = None,
    min_similarity: Optional[float] = None,
) -> list[QuizQuestionOut]:
    """用户说"我想练 XX"，把这句话 embedding 后做 pgvector cosine 相似度检索。

    返回的 ``similarity`` ∈ [-1, 1]，值越大越相似；低于 min_similarity 的行会被丢弃。
    """
    if not custom_topic or not custom_topic.strip():
        raise ValueError("自定义主题不能为空")

    s = get_settings()
    limit = max(1, min(int(count or s.quiz_top_k), 500))
    threshold = float(s.quiz_min_similarity if min_similarity is None else min_similarity)

    # 构造查询向量（和入库时完全一致："大｜中｜题干"的拼接文本）
    # 这里把 custom_topic 直接伪装成一个"假 chunk"走同一个文本函数，保证后续改拼接逻辑时
    # 不用同时改两边。
    pseudo = QuestionChunk(
        big_topic=custom_topic,
        mid_topic=custom_topic,
        question=custom_topic,
        answer_md="",
        answer_anchor="",
    )
    query_text = _text_for_embedding(pseudo)

    # 1) 调 embedding API
    emb_client = _build_embeddings()
    # 单条：用 embed_query（避免 embed_documents 的批量逻辑引入不同默认参数）
    query_vec_list = emb_client.embed_query(query_text)
    # 维度校验（避免配错 DIM 后直接进 SQL 报 vector 错）
    expected_dim = int(s.quiz_embedding_dimensions)
    if len(query_vec_list) != expected_dim:
        raise RuntimeError(
            f"Embedding 维度不匹配：配置 QUIZ_EMBEDDING_DIMENSIONS={expected_dim}，"
            f"但模型 {s.quiz_embedding_model!r} 实际返回 {len(query_vec_list)} 维。"
            "请修正 .env 后重新跑一次 POST /api/quiz/ingest（需要把旧数据清掉重建 vector 列）。"
        )
    vec_lit = validate_embedding_literal(query_vec_list)

    # 2) pgvector 余弦相似度：<=> 操作符 = 余弦距离，所以相似度 = 1 - distance
    #    注意：vec_lit 已经是带类型标注的合法 SQL 字面量（例如 '[-0.1,...]'::vector），
    #    这里通过字符串直接插值（值已通过 validate_embedding_literal 校验）；
    #    threshold / limit 走 SQLAlchemy bindparam 防注入。
    sql = text(
        "SELECT "
        "  id, big_topic, mid_topic, question, source_url, answer_anchor, ordinal, "
        f" (1 - (embedding <=> {vec_lit})) AS similarity "
        "FROM quiz_questions "
        f"WHERE (1 - (embedding <=> {vec_lit})) >= :thr "
        f"ORDER BY embedding <=> {vec_lit} "
        "LIMIT :lim"
    ).bindparams(thr=threshold, lim=limit)

    with session_scope() as sess:
        rows = sess.execute(sql).mappings().all()
    out: list[QuizQuestionOut] = []
    for r in rows:
        sim = float(r["similarity"]) if r["similarity"] is not None else None
        out.append(_row_to_out(r, similarity=sim))
    logger.info(
        "[quiz.retriever] 自定义主题=%r 返回 %d 条（阈值=%s limit=%s）",
        custom_topic, len(out), threshold, limit,
    )
    return out


# ═══════════════════════════════════════════
#  5. 显示答案
# ═══════════════════════════════════════════

def get_answer(question_id: int) -> Optional[dict]:
    """取单道题的答案（含完整 Markdown 文本、所有链接）。

    返回：``None`` 表示题不存在；否则 dict 字段 = 出题卡片 + answer_md。
    """
    with session_scope() as sess:
        r = sess.execute(
            text(
                "SELECT id, big_topic, mid_topic, question, answer_md, source_url, answer_anchor, ordinal "
                "FROM quiz_questions WHERE id=:qid"
            ),
            {"qid": int(question_id)},
        ).mappings().first()
        if r is None:
            return None
    out = _row_to_out(r).as_dict()
    out["answer_md"] = r["answer_md"]
    return out
