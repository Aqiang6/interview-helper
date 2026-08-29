"""刷题助手 - 爬虫。

职责：
1. 从 ``ai_interviewer/爬虫.txt`` 中读取 URL 列表（每行一个，空行/#注释忽略）。
2. 对每个 URL 发起 HTTP GET（带 UA、重试、超时），取 HTML。
3. 用 BeautifulSoup4 + html2text 把 HTML 文章正文转成规范化 Markdown（保留标题层级 / 表格 /
   代码 / 链接 / 图片）。
4. 调用 :mod:`splitter` 切成 ``QuestionChunk`` 列表，返回给上层 ingest。

设计说明：
- 正文选择器：JavaGuide 用的是 VitePress / docsify 风格，主内容在 ``<main>`` 或
  ``<article>`` / ``div.theme-default-content`` / ``.page`` 等常见容器里。我们按优先级
  抓最大的可读容器，并去掉 header/footer/nav/侧边栏/TLE 等噪音。
- 并发：用 ``asyncio.Semaphore`` 控制并发数（配置 ``QUIZ_CRAWLER_CONCURRENCY``）。
- 失败策略：单页失败不影响其它页，最终把错误页在返回值中标注，方便前端提示。
- 幂等：抓出来的 (source_url, anchor) 由 ingest 层做去重，crawler 层本身不关心去重。
"""

from __future__ import annotations

import asyncio
import difflib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from bs4 import BeautifulSoup
from html2text import HTML2Text
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from ai_interviewer.config import get_settings
from ai_interviewer.quiz.splitter import QuestionChunk, absolutize_links, split_markdown

logger = logging.getLogger(__name__)

CRAWLER_DIR = Path(__file__).resolve().parent.parent  # ai_interviewer/
DEFAULT_URLS_FILE = CRAWLER_DIR / "爬虫.txt"


# ═══════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════

@dataclass
class CrawlResult:
    """一页的抓取结果。"""

    source_url: str
    ok: bool
    chunks: list[QuestionChunk] = field(default_factory=list)
    error: Optional[str] = None
    # 用于日志：返回的 Markdown 大小（字符数），失败时为 0
    chars: int = 0


# ═══════════════════════════════════════════
#  URL 列表解析
# ═══════════════════════════════════════════

def load_url_list(path: Optional[Path] = None) -> list[str]:
    """从 ``爬虫.txt`` 读取 URL 列表，去重 + 过滤空行/注释。

    Args:
        path: 可选自定义路径；默认读项目内置的 ``ai_interviewer/爬虫.txt``。
    """
    p = Path(path) if path else DEFAULT_URLS_FILE
    if not p.exists():
        raise FileNotFoundError(f"未找到 URL 列表文件: {p}")
    urls: list[str] = []
    seen: set[str] = set()
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        # 如果一行里多个以空白分隔也可以拆开（兜底）
        for part in re.split(r"\s+", s):
            part = part.strip().rstrip(",;")
            if not part:
                continue
            if not part.startswith(("http://", "https://")):
                logger.warning("[quiz.crawler] 忽略非 URL 行: %s", part)
                continue
            if part in seen:
                continue
            seen.add(part)
            urls.append(part)
    logger.info("[quiz.crawler] 从 %s 加载到 %d 个去重 URL", p, len(urls))
    return urls


# ═══════════════════════════════════════════
#  HTML → Markdown
# ═══════════════════════════════════════════

# 这些容器优先级从高到低；选择器命中第一个"含有足够文本量"的节点作为正文
_ARTICLE_SELECTORS = [
    "article",
    "main",
    'div[class*="content"]',
    'main[class*="content"]',
    ".page",
    ".theme-default-content",
    ".theme-doc-markdown",
    "div.post-content",
    "div.md-content__inner",
    "#content",
]

# 要从正文里剥离的噪音节点（常见于文档站）
_NOISE_SELECTORS = [
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    'div[class*="ads"]',
    'div[class*="advert"]',
    'div[class*="comment"]',
    'div[class*="toc"]',
    'div[class*="sidebar"]',
    'div[class*="breadcrumbs"]',
    'div[class*="prev-next"]',
    ".page-edit",
    ".page-nav",
    ".edit-this-page",
    ".contributors",
    ".last-updated",
    'a[aria-label*="Edit this page"]',
    ".DocSearch-Button",
]


def _extract_main_html(soup: BeautifulSoup) -> str:
    """从 BS soup 中挑出最佳正文容器并返回其 inner HTML。找不到时回退到 <body>。"""
    candidates = []
    for sel in _ARTICLE_SELECTORS:
        candidates.extend(soup.select(sel))
    if not candidates:
        body = soup.body or soup
        return str(body)
    # 取"最长文本长度"的候选（避免取到只有导航的 div）
    best = max(candidates, key=lambda el: len(el.get_text(" ", strip=True)))
    # 剥离噪音
    for ns in _NOISE_SELECTORS:
        for el in best.select(ns):
            el.decompose()
    return str(best)


def _build_html2text(base_url: str) -> HTML2Text:
    h = HTML2Text(baseurl=base_url)
    h.unicode_snob = True            # 保留 Unicode 字符（中文/⭐️）
    h.body_width = 0                 # 不自动折行，避免把代码/表格折烂
    # 注意：protect_links=True 会把链接输出成 [text](<url>) 形式，后续 URL 处理
    # （相对链接拼绝对路径、标题锚点解析）都容易被 <...> 干扰，这里必须关闭。
    h.protect_links = False
    h.inline_links = True            # [text](url) 形式，不是参考链接
    h.wrap_links = False             # 不单独给链接换行
    h.ignore_images = False
    h.ignore_emphasis = False
    h.mark_code = True               # 代码加围栏
    h.skip_internal_links = False    # 保留页内 #anchor 链接（对显示答案有用）
    return h


def html_to_markdown(html: str, source_url: str) -> str:
    """把 HTML 正文转成结构化 Markdown。"""
    soup = BeautifulSoup(html, "lxml")
    main_html = _extract_main_html(soup)
    h2t = _build_html2text(source_url)
    md = h2t.handle(main_html)
    # html2text 在中文/英文混排时偶有多余空格，这里不动，让 splitter 自己 strip
    return md


# ═══════════════════════════════════════════
#  抓取单页（带重试）
# ═══════════════════════════════════════════

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36 "
    "InterviewHelperQuizBot/1.0"
)


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_random_exponential(min=0.5, max=4),
    retry=retry_if_exception_type((httpx.TransportError, httpx.HTTPStatusError, asyncio.TimeoutError)),
)
async def _fetch_one(client: httpx.AsyncClient, url: str, timeout: float) -> str:
    resp = await client.get(url, timeout=timeout, follow_redirects=True)
    # 4xx/5xx 都走重试（tenacity 条件），这里显式 raise
    resp.raise_for_status()
    # 深层链接被 302 到站点首页：拿到的是首页/404 兜底内容而非目标文章，按失败处理
    if resp.history:
        path = urlparse(str(resp.url)).path
        if path in ("", "/"):
            raise RuntimeError(f"重定向到站点首页（疑似 404 兜底页）: {url} -> {resp.url}")
    return resp.text


# ═══════════════════════════════════════════
#  二层爬取：种子页 → 跟进详情/相关文章链接
# ═══════════════════════════════════════════

_RELATED_LINE_RE = re.compile(r"相关(?:内容|文章)[:：]")
_MD_ANY_LINK_RE = re.compile(r"!?\[([^\]]*)\]\(\s*([^)\s]+)(?:\s+\"[^\"]*\")?\s*\)")
_ASSET_EXT_RE = re.compile(r"\.(png|jpe?g|gif|webp|svg|ico|pdf|zip|xml|json|atom)$", re.I)
_MAX_DETAIL_PAGES = 30  # 单个种子页最多跟进的详情链接数，防止失控


def _same_site(a: str, b: str) -> bool:
    """判断两个 URL 是否同站（主域相同即可，覆盖 javaguide.cn / interview.javaguide.cn）。"""
    pa, pb = urlparse(a), urlparse(b)
    if not pa.netloc or not pb.netloc:
        return False
    if pa.netloc == pb.netloc:
        return True
    da = ".".join(pa.netloc.lower().split(":")[0].split(".")[-2:])
    db = ".".join(pb.netloc.lower().split(":")[0].split(".")[-2:])
    return da == db


def _extract_detail_links(md_text: str, page_url: str) -> list[str]:
    """提取本页所有可跟进的同站文章链接（"相关内容/相关文章：" 行优先，其余正文链接兜底）。

    爬虫.txt 只是种子方向，页面里的详细链接都要进去爬，所以这里不再局限于
    "相关内容：" 行，正文里出现的同站文章链接全部收集。
    """
    related: list[str] = []
    other: list[str] = []
    seen: set[str] = set()
    page_path = urlparse(page_url).path
    for line in md_text.splitlines():
        is_related = bool(_RELATED_LINE_RE.search(line))
        for m in _MD_ANY_LINK_RE.finditer(line):
            url = m.group(2).strip().strip("<>").strip()
            if not url or url.startswith("#"):
                continue
            url = urljoin(page_url, url)
            p = urlparse(url)
            if p.scheme not in ("http", "https"):
                continue
            if p.path == page_path:  # 指回本页自身的锚点链接
                continue
            if _ASSET_EXT_RE.search(p.path):
                continue
            if not _same_site(url, page_url):
                continue
            # 去掉 fragment 后去重：同一篇文章带不同锚点只算一次
            base = urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
            if base in seen:
                continue
            seen.add(base)
            (related if is_related else other).append(base)
    return (related + other)[:_MAX_DETAIL_PAGES]


def _norm_q(q: str) -> str:
    """题目匹配用的归一化：去空白/标点/装饰，转小写。"""
    q = unicodedata.normalize("NFKC", q or "").lower()
    return re.sub(r"[\s,，。.、;；:：?？!！'\"“”‘’()（）\[\]【】《》<>·\-—_~～*`]", "", q)


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


def _similarity(a: str, b: str) -> float:
    """题干相似度：精确 -> 包含 -> max(difflib, bigram-Jaccard)。

    面试列表里的题干常常是详情页题干的改写（加/减"是什么""区别"等后缀），
    单用 difflib 0.75 阈值会漏配，这里取两种度量的较大值来兜底改写场景。
    """
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    if len(a) >= 6 and len(b) >= 6 and (a in b or b in a):
        return min(len(a), len(b)) / max(len(a), len(b))
    r = difflib.SequenceMatcher(None, a, b).ratio()
    ba, bb = _bigrams(a), _bigrams(b)
    j = (len(ba & bb) / len(ba | bb)) if ba and bb else 0.0
    return max(r, j)


_MATCH_THRESHOLD = 0.62
# 0.45~0.62：详情页有相关小节但题干写法差异过大，不能确定对应关系（用于归因细分）
_RELATED_THRESHOLD = 0.45


# 疑问词停用：不参与内容词匹配（避免"什么是/怎么"这类通用词拉高覆盖率）
_QSTOP_RE = re.compile(
    r"什么是|为什么|为啥|怎么样|怎么|如何|介绍|一下|说说|谈谈|理解|解释|哪些|常见|面试|请问|总结"
)


def _content_tokens(s: str) -> set[str]:
    """内容词集合：英文/数字词整体保留，中文按连续段切 bigram。

    用于"内容覆盖度"匹配：题干被改写时题干相似度不够，但详情小节正文里
    会出现题干的关键内容词。
    """
    s = _QSTOP_RE.sub(" ", s.lower())
    toks: set[str] = set()
    for m in re.finditer(r"[a-z][a-z0-9_\-]+", s):
        toks.add(m.group(0))
    for m in re.finditer(r"[\u4e00-\u9fff]+", s):
        seg = m.group(0)
        if len(seg) == 1:
            toks.add(seg)
        else:
            toks.update(seg[i:i + 2] for i in range(len(seg) - 1))
    return toks


def _find_best_match(
    norm_q: str,
    q_toks: set[str],
    index: Sequence[tuple[str, QuestionChunk, str, frozenset[str], frozenset[str]]],
) -> Optional[tuple[QuestionChunk, str, float]]:
    """在详情页题目索引里找最匹配的一题。

    双信号取最大值：题干相似度 + 内容词覆盖率（题干关键词出现在详情小节
    正文中的比例，对改写/换问法鲁棒）。正文覆盖率有"标题门控"：候选小节
    标题必须与题干共享至少一个内容词，否则正文覆盖率不可信（防止把题干
    关键词偶然出现在无关小节正文里当成匹配）。先精确，再算分。
    返回 (chunk, 来源链接, 综合分数)；索引为空返回 None。
    阈值判断（_MATCH_THRESHOLD / _RELATED_THRESHOLD）由调用方按分数归因。
    """
    best: Optional[tuple[QuestionChunk, str, float]] = None
    best_score = 0.0
    for k, dc, link, t_toks, b_toks in index:
        if k == norm_q:
            return dc, link, 1.0
        score = _similarity(norm_q, k)
        if q_toks and q_toks & t_toks:
            cover = len(q_toks & b_toks) / len(q_toks)
            score = max(score, cover)
        if score > best_score:
            best, best_score = (dc, link, score), score
    return best


async def _fetch_page_chunks(
    client: httpx.AsyncClient, url: str, timeout: float
) -> tuple[str, list[QuestionChunk]]:
    """抓单页 → Markdown → 切片；链接已在 absolutize 阶段绝对化。失败抛异常。"""
    html = await _fetch_one(client, url, timeout)
    md = html_to_markdown(html, url)
    md = absolutize_links(md, url)
    chunks = split_markdown(md, source_url=url)
    return md, chunks


# 无答案原因常量（写入 QuestionChunk.no_answer_reason，最终展示给用户）
_REASON_NO_LINK = "原页面未提供可跟随的详情/相关文章链接"
_REASON_FETCH_FAILED = "详情页抓取失败（网络或页面不可达），无法补齐答案"
_REASON_NO_MATCH = "详情页已抓取，但其中没有与该题对应的内容（疑似原站就没有该题答案）"
_REASON_WEAK_MATCH = "详情页存在相似小节，但题干写法差异过大，无法确定对应关系，未强行填充"
_REASON_NO_RETRY = "二层爬取不递归详情页，该题在其来源页也未给出答案"


def _apply_placeholder(c: QuestionChunk, fallback_url: str) -> None:
    """为确实没抓到答案的题生成占位文案：原因 + 跳转链接 / 此题无答案。"""
    reason = c.no_answer_reason or "原页面未提供该题答案"
    jump = getattr(c, "source_url", "") or fallback_url
    if c.no_answer_reason == _REASON_NO_MATCH:
        # 详情页都翻过了确实没有 -> 明确告知"此题无答案"
        c.answer_md = (
            "**此题无答案**。\n\n"
            f"- 原因：{reason}\n"
            f"- 原文跳转：[点击查看原文]({jump})"
        )
    else:
        # 其它原因（无链接/网络失败/不递归）属于"暂时没抓到"，给出可重试的说明
        c.answer_md = (
            "**暂无答案**。\n\n"
            f"- 原因：{reason}\n"
            f"- 原文跳转：[点击查看原文]({jump})\n"
            "- 提示：可重新执行爬取任务重试。"
        )


async def crawl_single_url(
    client: httpx.AsyncClient,
    url: str,
    timeout: float,
    *,
    sem: Optional[asyncio.Semaphore] = None,
    detail_cache: Optional[dict] = None,
) -> list[CrawlResult]:
    """二层爬取一个种子页：抓种子页 + 跟进其全部详情/相关文章链接。

    返回值：[种子页 CrawlResult, 各详情页 CrawlResult...]。
    - 种子页上"只有题干"的题会从详情页切片中匹配补答案，并记录无答案原因。
    - 详情页自身也切出题目（也是题库的一部分）。
    - detail_cache 用于跨种子页共享详情页抓取结果，避免同一详情页重复请求。
    - 失败策略：详情页失败不影响种子页；种子页失败则只返回一个失败结果。
    """
    # ── 1) 种子页 ──
    try:
        md, chunks = await _fetch_page_chunks(client, url, timeout)
    except Exception as e:  # tenacity 3 次后仍失败 / 切片失败等
        msg = f"{type(e).__name__}: {e}"
        logger.error("[quiz.crawler] 抓页失败 %s -> %s", url, msg)
        return [CrawlResult(source_url=url, ok=False, error=msg, chars=0)]

    # ── 2) 收集详情链接（爬虫.txt 只是方向，页面里的详细链接都要进去爬）──
    detail_links = _extract_detail_links(md, url)

    # ── 3) 并发抓详情页（带跨种子页缓存）──
    detail_ok: dict[str, tuple[str, list[QuestionChunk]]] = {}
    detail_err: dict[str, str] = {}
    if detail_links:
        cache = detail_cache if detail_cache is not None else {}

        async def _one(link: str):
            if link in cache:
                return cache[link]
            if sem is not None:
                async with sem:
                    if link in cache:  # 双重检查：等锁期间可能已被其它任务填好
                        return cache[link]
                    val = await _fetch_page_chunks(client, link, timeout)
            else:
                val = await _fetch_page_chunks(client, link, timeout)
            cache[link] = val
            return val

        outcomes = await asyncio.gather(
            *(_one(l) for l in detail_links), return_exceptions=True
        )
        for link, out in zip(detail_links, outcomes):
            if isinstance(out, BaseException):
                msg = f"{type(out).__name__}: {out}"
                detail_err[link] = msg
                logger.warning("[quiz.crawler] 详情页抓取失败 %s -> %s", link, msg)
            else:
                detail_ok[link] = out

        # 过滤垃圾详情页（404 兜底/目录页）：绝大多数题干下面没有正文的页面
        junk = [
            link for link, (_dmd, dchunks) in detail_ok.items()
            if dchunks
            and sum(1 for c in dchunks if not c.answer_missing) / len(dchunks) < 0.3
        ]
        for link in junk:
            n_missing = sum(1 for c in detail_ok[link][1] if c.answer_missing)
            logger.info("[quiz.crawler] 丢弃疑似目录/404兜底页 %s（%d 题中 %d 题无正文）",
                        link, len(detail_ok[link][1]), n_missing)
            del detail_ok[link]

    # ── 4) 用详情页切片为种子页"仅题干"的题补答案，并归因 ──
    missing = [c for c in chunks if c.answer_missing]
    filled = 0
    if missing:
        if not detail_links:
            for c in missing:
                c.no_answer_reason = _REASON_NO_LINK
        elif not detail_ok:
            for c in missing:
                c.no_answer_reason = _REASON_FETCH_FAILED
        else:
            index: list[tuple[str, QuestionChunk, str, frozenset[str], frozenset[str]]] = []
            for link, (_dmd, dchunks) in detail_ok.items():
                for dc in dchunks:
                    t_toks = frozenset(_content_tokens(dc.question))
                    b_toks = frozenset(
                        _content_tokens(dc.question + "\n" + dc.answer_md[:2000])
                    )
                    index.append((_norm_q(dc.question), dc, link, t_toks, b_toks))
            for c in missing:
                hit = _find_best_match(
                    _norm_q(c.question), _content_tokens(c.question), index
                )
                if hit is None:
                    c.no_answer_reason = _REASON_NO_MATCH
                    continue
                dc, link, score = hit
                if score >= _MATCH_THRESHOLD:
                    c.answer_md = dc.answer_md
                    c.answer_anchor = dc.answer_anchor or c.answer_anchor
                    c.answer_missing = False
                    c.source_url = link  # 指向真正含答案的详情页（跳转/去重键都用它）
                    filled += 1
                elif score >= _RELATED_THRESHOLD:
                    # 0.45~0.62：有相似小节但写法差异过大，不强行张冠李戴
                    c.no_answer_reason = _REASON_WEAK_MATCH
                else:
                    c.no_answer_reason = _REASON_NO_MATCH
            logger.info("[quiz.crawler] %s 通过详情页补齐 %d/%d 道题的答案",
                        url, filled, len(missing))

    # ── 5) 占位文案：原因 + 跳转链接 / 此题无答案 ──
    for c in chunks:
        if c.answer_missing and not c.answer_md.strip():
            _apply_placeholder(c, url)

    results = [CrawlResult(source_url=url, ok=True, chunks=chunks, chars=len(md))]

    # ── 6) 详情页切片也作为题库的一部分返回（但详情页里的"仅题干"题不再递归跟进）──
    for link, (dmd, dchunks) in detail_ok.items():
        for c in dchunks:
            if c.answer_missing and not c.answer_md.strip():
                c.no_answer_reason = _REASON_NO_RETRY
                _apply_placeholder(c, link)
        results.append(CrawlResult(source_url=link, ok=True, chunks=dchunks, chars=len(dmd)))

    return results


def _chunk_answer_score(c: QuestionChunk) -> int:
    """答案质量评分：占位/无答案 = 0；有答案按内容长度。"""
    if c.answer_missing or not c.answer_md.strip():
        return 0
    return len(c.answer_md)


def dedup_results(results: list[CrawlResult]) -> list[CrawlResult]:
    """跨页查重清洗：同一道题（归一化题干相同）只保留答案最全的一条。

    - 多个种子页/详情页互相引用时同一题会重复出现，这里全局去重。
    - 保留规则：优先"有答案"的；答案相同时保留先出现的（题干/主题更贴近种子页）。
    - 去重是原地从各 CrawlResult.chunks 里移除被淘汰的题。
    """
    best: dict[str, tuple[CrawlResult, QuestionChunk]] = {}
    dup_removed = 0
    for r in results:
        if not r.ok:
            continue
        kept: list[QuestionChunk] = []
        for c in r.chunks:
            key = _norm_q(c.question)
            if not key:
                kept.append(c)
                continue
            prev = best.get(key)
            if prev is None:
                best[key] = (r, c)
                kept.append(c)
                continue
            prev_r, prev_c = prev
            if _chunk_answer_score(c) > _chunk_answer_score(prev_c):
                # 新的更全：旧的从其所在页移除，新的保留
                try:
                    prev_r.chunks.remove(prev_c)
                except ValueError:
                    pass
                best[key] = (r, c)
                kept.append(c)
                dup_removed += 1
            else:
                dup_removed += 1  # 丢弃当前重复的，保留旧的
        r.chunks = kept
    if dup_removed:
        logger.info("[quiz.crawler] 查重清洗：移除重复题 %d 道", dup_removed)
    return results


# ═══════════════════════════════════════════
#  对外入口：批量并发抓
# ═══════════════════════════════════════════

async def crawl_urls(
    urls: Sequence[str],
    *,
    concurrency: Optional[int] = None,
    timeout: Optional[float] = None,
) -> list[CrawlResult]:
    """并发抓取一批种子 URL（每个都二层跟进详情页），返回扁平化的结果列表。"""
    settings = get_settings()
    conc = int(concurrency if concurrency is not None else settings.quiz_crawler_concurrency)
    to = float(timeout if timeout is not None else settings.quiz_crawler_timeout)

    sem = asyncio.Semaphore(max(1, conc))
    headers = {
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    }
    limits = httpx.Limits(max_keepalive_connections=max(4, conc), keepalive_expiry=15.0)

    client = httpx.AsyncClient(headers=headers, limits=limits, follow_redirects=True, timeout=to * 2)
    try:
        detail_cache: dict = {}  # 跨种子页共享的详情页缓存

        async def _task(seed: str) -> list[CrawlResult]:
            return await crawl_single_url(client, seed, to, sem=sem, detail_cache=detail_cache)

        per_seed = await asyncio.gather(*(_task(u) for u in urls))

        # 扁平化 + 详情页 URL 全局去重（多个种子页引用同一详情页时只保留一份结果）
        seed_set = set(urls)
        flat: list[CrawlResult] = []
        seen_detail: set[str] = set()
        for group in per_seed:
            for r in group:
                if not r.ok:
                    flat.append(r)
                    continue
                if r.source_url in seed_set:
                    flat.append(r)  # 种子页本身始终保留
                    continue
                if r.source_url in seen_detail:
                    continue
                seen_detail.add(r.source_url)
                flat.append(r)

        # 全局查重清洗
        return dedup_results(flat)
    finally:
        await client.aclose()


async def crawl_default_file() -> list[CrawlResult]:
    """便捷入口：直接抓内置爬虫.txt 的全部 URL。"""
    urls = load_url_list()
    if not urls:
        return []
    return await crawl_urls(urls)
