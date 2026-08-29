"""刷题助手 - 入库 Pipeline（爬取 → 切片 → Embedding → 批量 upsert）。

Embedding 策略（方案 A，二选一后端）
-----------------------------------
对每道题只做 1 次 embedding，输入文本 = ``{大标题}|{中标题}|{题干}``：

- 大/中标题作为"分类约束"，向量天然会把同主题的题聚在一起；
- 题干包含考点关键词，是"自定义主题语义检索"时的主要匹配依据；
- 答案整块不切、不 embedding（因为方案 A 的定位就是"显示答案时原样整段返回"）。

Embedding 后端通过 ``QUIZ_EMBEDDING_BACKEND`` 选择：

* ``sentence_transformers``（默认，免 Key）：本地跑 Hugging Face 的 Sentence-BERT。
  - 默认模型：``all-MiniLM-L6-v2``，输出 384 维；中文题可切到
    ``sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2``（同样 384 维）。
  - 首次运行会自动下载模型权重（~80MB）。
* ``openai``：走 ``langchain-openai`` 包，复用 ``OPENAI_API_KEY / OPENAI_BASE_URL`` 凭据。
  - 用户之前偏好：**未配 Key 或 Key 错误直接抛错，不静默 fallback**。本模块严格遵守。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ai_interviewer.config import get_settings
from ai_interviewer.quiz.crawler import CrawlResult, crawl_default_file, crawl_urls, load_url_list
from ai_interviewer.quiz.db import UpsertRow, bulk_upsert, chunked, ensure_schema
from ai_interviewer.quiz.splitter import QuestionChunk

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    """统一两套后端接口：给一批 texts -> 一批等长 float 向量。"""

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量；retriever 自定义语义检索使用。"""
        ...


# ═══════════════════════════════════════════
#  返回值结构
# ═══════════════════════════════════════════

@dataclass
class IngestReport:
    """一次入库的完整统计。"""

    pages_total: int = 0
    pages_ok: int = 0
    pages_failed: int = 0
    questions_ingested: int = 0  # = inserted + updated
    questions_inserted: int = 0
    questions_updated: int = 0
    questions_skipped: int = 0  # 内容未变未更新
    failed_pages: list[dict] = None  # [{"url": ..., "error": ...}]

    def as_dict(self) -> dict:
        return {
            "pages": {
                "total": self.pages_total,
                "ok": self.pages_ok,
                "failed": self.pages_failed,
                "failed_details": self.failed_pages or [],
            },
            "questions": {
                "ingested": self.questions_ingested,
                "inserted": self.questions_inserted,
                "updated": self.questions_updated,
                "skipped": self.questions_skipped,
            },
        }


# ═══════════════════════════════════════════
#  后端 1：sentence-transformers（本地，免 Key）
# ═══════════════════════════════════════════

class SentenceTransformerEmbedder:
    """薄封装：懒加载 SentenceTransformer 单例并批量 encode。

    注意：SentenceTransformer 已经支持批量 encode，内部会自动拆分 mini-batch 到 GPU/CPU，
    不需要我们在上层做 tenacity 重试（本地模型几乎不存在"网络抖动"的重试语义）。
    """

    _MODEL_CACHE: dict[tuple[str, str], object] = {}

    def __init__(self, model_name: str, expected_dimensions: int, *, normalize: bool = True) -> None:
        self.model_name = model_name
        self.expected_dimensions = expected_dimensions
        self.normalize = normalize

    def _load(self):
        key = (self.model_name, str(self.expected_dimensions))
        cached = __class__._MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "quiz_embedding_backend=sentence_transformers 但缺少 sentence-transformers 依赖，\n"
                "请在运行主程序的 Python 环境中安装（首次会下载模型权重 ~80MB）：\n"
                "  python -m pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu\n"
                "或（默认 pip 源，GPU 用户）：\n"
                "  python -m pip install sentence-transformers"
            ) from e
        logger.info("[quiz.ingest] Loading SentenceTransformer %s ...", self.model_name)
        model = SentenceTransformer(self.model_name)
        # 推断实际输出维度；若与 settings.quiz_embedding_dimensions 不一致，给出明确错误，
        # 避免 DB 侧 vector(N) 列与实际维度 mismatch 导致插入失败且难排查。
        probe = model.encode(["probe text"], convert_to_numpy=True, normalize_embeddings=self.normalize)
        actual_dim = int(getattr(probe[0], "shape", (0,))[0]) if hasattr(probe[0], "shape") else len(probe[0])
        if actual_dim != self.expected_dimensions:
            raise RuntimeError(
                f"SentenceTransformer 模型 {self.model_name!r} 实际输出维度={actual_dim}，"
                f"但 settings.quiz_embedding_dimensions={self.expected_dimensions} 不匹配。\n"
                f"请在 .env 中把 QUIZ_EMBEDDING_DIMENSIONS 改成 {actual_dim} 或换一个维度一致的模型。"
            )
        __class__._MODEL_CACHE[key] = model
        logger.info("[quiz.ingest] SentenceTransformer %s loaded, dim=%d", self.model_name, actual_dim)
        return model

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        encode_kwargs = dict(convert_to_numpy=True, normalize_embeddings=self.normalize, show_progress_bar=False)
        # SentenceTransformer.encode 默认 batch_size=32；传大一点加速本地 CPU 批量
        vecs = model.encode(texts, batch_size=64, **encode_kwargs)
        if len(vecs) != len(texts):
            raise RuntimeError(
                f"SentenceTransformer.encode 返回长度 {len(vecs)} != 输入 {len(texts)}"
            )
        # convert list[np.ndarray[float]] -> list[list[float]]
        out: list[list[float]] = []
        for v in vecs:
            if hasattr(v, "tolist"):
                out.append([float(x) for x in v.tolist()])
            else:
                out.append([float(x) for x in v])
        return out

    def embed_query(self, text: str) -> list[float]:
        """单条查询向量。复用 embed_batch([text]) 避免和 batch 路径产生参数漂移。"""
        if text is None:
            raise ValueError("embed_query: text 不能为空")
        batch = self.embed_batch([str(text)])
        if not batch:
            raise RuntimeError("SentenceTransformerEmbedder.embed_query 返回空向量")
        return batch[0]


# ═══════════════════════════════════════════
#  后端 2：OpenAI (langchain-openai)
# ═══════════════════════════════════════════

class OpenAIAPIEmbedder:
    def __init__(self) -> None:
        from langchain_openai import OpenAIEmbeddings

        s = get_settings()
        api_key = (s.openai_api_key or "").strip()
        if not api_key:
            raise RuntimeError(
                "未配置 OPENAI_API_KEY，无法生成 embedding 入库。"
                "请在 .env 中设置 OPENAI_API_KEY / OPENAI_BASE_URL（对应 embedding 模型服务）。"
                "如果不想用 API，也可以切回本地：QUIZ_EMBEDDING_BACKEND=sentence_transformers。"
            )
        extra_kwargs = {}
        # text-embedding-3-small/large 支持 dimensions 参数；ada-002 传了会报错
        if "3" in (s.quiz_embedding_model or ""):
            extra_kwargs["dimensions"] = s.quiz_embedding_dimensions
        self._impl = OpenAIEmbeddings(
            model=s.quiz_embedding_model,
            api_key=api_key,
            base_url=s.openai_base_url or "https://api.openai.com/v1",
            max_retries=0,  # 4xx 一律不重试，符合用户偏好：API 失败直接抛
            request_timeout=30.0,
            **extra_kwargs,
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=0.4, max=3),
    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # OpenAI 接口单次最多 2048 条；上层按 batch_size=32 切，这里不再拆。
        return list(self._impl.embed_documents(texts))

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_random_exponential(min=0.4, max=3),
    )
    def embed_query(self, text: str) -> list[float]:
        return list(self._impl.embed_query(str(text)))


# ═══════════════════════════════════════════
#  Embedding 工厂 & 文本构造
# ═══════════════════════════════════════════

def _build_embeddings() -> Embedder:
    """根据 settings 构造统一接口的 Embedder。关键前置条件不满足立刻抛错，不降级。"""
    s = get_settings()
    backend = (s.quiz_embedding_backend or "sentence_transformers").lower()
    if backend == "sentence_transformers":
        return SentenceTransformerEmbedder(
            model_name=s.quiz_sentence_transformer_model,
            expected_dimensions=s.quiz_embedding_dimensions,
            normalize=True,
        )
    if backend == "openai":
        return OpenAIAPIEmbedder()
    raise ValueError(
        f"未知 QUIZ_EMBEDDING_BACKEND={s.quiz_embedding_backend!r}，可选: sentence_transformers | openai"
    )


def _text_for_embedding(c: QuestionChunk) -> str:
    """方案 A：把 大/中/题干 拼接成一个短文本用于 embedding；不塞答案，避免向量被答案里的
    高频词稀释。字段分隔符用全角｜避免和中文/英文标点混。
    """
    return f"{c.big_topic}｜{c.mid_topic}｜{c.question}"


# ═══════════════════════════════════════════
#  核心入库逻辑
# ═══════════════════════════════════════════

def _flatten_chunks(results: Sequence[CrawlResult]) -> list[tuple[str, QuestionChunk]]:
    """把多页的 CrawlResult 展开，并给每页内的题加上 ordinal（页内顺序编号）。

    返回 [(source_url, chunk_with_ordinal_set)]；chunk 对象会被原地修改 ordinal。
    """
    flat: list[tuple[str, QuestionChunk]] = []
    for r in results:
        if not r.ok:
            continue
        for ordinal, c in enumerate(r.chunks, start=1):
            c.ordinal = getattr(c, "ordinal", ordinal)  # type: ignore[attr-defined]
            flat.append((r.source_url, c))
    return flat


def embed_and_upsert(chunks: Sequence[QuestionChunk], *, batch_size: int = 64) -> dict:
    """对 QuestionChunk 列表批量 embedding 并 upsert。

    Args:
        chunks: 已经带有 source_url/answer_anchor/ordinal 信息的切片数组。
                若缺 source_url，会报错（爬虫流程会在 ingest_from_results 里填）。
        batch_size: 本地 sentence-transformers 默认 64；走 OpenAI 时建议 32~64，视 Token 预算。
    """
    if not chunks:
        return {"inserted": 0, "updated": 0, "skipped": 0, "total": 0}

    emb = _build_embeddings()
    upsert_rows: list[UpsertRow] = []

    for idx, batch in enumerate(chunked(chunks, batch_size), start=1):
        texts = [_text_for_embedding(c) for c in batch]
        vectors = emb.embed_batch(texts)
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding 返回长度 {len(vectors)} != 输入 {len(batch)}，请检查后端响应"
            )
        expected_dim = get_settings().quiz_embedding_dimensions
        for c, vec in zip(batch, vectors):
            if len(vec) != expected_dim:
                raise RuntimeError(
                    f"向量维度不匹配：实际 {len(vec)} != 配置 QUIZ_EMBEDDING_DIMENSIONS={expected_dim}"
                )
            src = getattr(c, "source_url", None)
            if not src:
                raise ValueError(f"题缺少 source_url 字段: big={c.big_topic!r} q={c.question!r}")
            upsert_rows.append(
                UpsertRow(
                    big_topic=c.big_topic,
                    mid_topic=c.mid_topic,
                    question=c.question,
                    answer_md=c.answer_md,
                    source_url=src,
                    answer_anchor=c.answer_anchor,
                    ordinal=int(getattr(c, "ordinal", 0) or 0),
                    embedding=vec,  # db.validate_embedding_literal 会把 list[float] 转成字符串
                )
            )
        logger.info(
            "[quiz.ingest] encoded batch #%d: %d / %d chunks",
            idx, len(upsert_rows), len(chunks),
        )

    return bulk_upsert(upsert_rows)


# ═══════════════════════════════════════════
#  上层入口
# ═══════════════════════════════════════════

async def ingest_from_results(results: Sequence[CrawlResult]) -> IngestReport:
    """把爬取结果入库（先 ensure_schema）。"""
    ensure_schema()

    report = IngestReport()
    report.pages_total = len(results)
    report.pages_ok = sum(1 for r in results if r.ok)
    report.pages_failed = report.pages_total - report.pages_ok
    report.failed_pages = [
        {"url": r.source_url, "error": r.error} for r in results if not r.ok
    ]

    flat = _flatten_chunks(results)
    # 给 chunk 补 source_url 属性（splitter 层不关心 URL，只负责切）
    chunks: list[QuestionChunk] = []
    for src, c in flat:
        if not hasattr(c, "source_url") or not c.source_url:  # type: ignore[attr-defined]
            c.source_url = src  # type: ignore[attr-defined]
        chunks.append(c)

    if not chunks:
        logger.warning("[quiz.ingest] 没有任何可供入库的题目 chunks。")
        return report

    stats = embed_and_upsert(chunks)
    report.questions_inserted = int(stats.get("inserted", 0))
    report.questions_updated = int(stats.get("updated", 0))
    report.questions_skipped = int(stats.get("skipped", 0))
    report.questions_ingested = report.questions_inserted + report.questions_updated
    return report


async def ingest_from_default_file(
    *,
    urls: Optional[Sequence[str]] = None,
) -> IngestReport:
    """跑一次完整流程：读爬虫.txt（或自定义 urls） → 爬 → 切 → embedding → 入库。"""
    use_urls = list(urls) if urls is not None else load_url_list()
    if not use_urls:
        rep = IngestReport()
        return rep
    results = await crawl_urls(use_urls)
    return await ingest_from_results(results)


if __name__ == "__main__":  # pragma: no cover
    # CLI: python -m ai_interviewer.quiz.ingest
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    rep = asyncio.run(ingest_from_default_file())
    import json as _json
    print(_json.dumps(rep.as_dict(), ensure_ascii=False, indent=2))
