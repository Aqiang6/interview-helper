"""结构化标题切片器（方案 A 的核心）。

输入：爬虫返回的 **完整 Markdown 文本**（已由 html2text 转换，保留了 ``#/##/###`` 标题、
链接、图片、表格、代码围栏），以及来源页 URL。

输出：``QuestionChunk`` 列表，每一项对应 "一道题"（含大标题/中标题/题干/答案 Markdown/锚点）。

识别规则（两层兼容，覆盖 JavaGuide interview 站 + javaguide.cn AI 专题两种版式）：

1. **大标题**：文档首个 ``# H1`` 行（去前后空白、去 ``⭐️`` 等装饰前缀保留原样作为显示）。
   特殊情况：若文档内有多个 ``#``，只取第一个，后续的 ``#`` 一律视为坏格式忽略。

2. **中标题**：``## H2`` 行，例如 "基础概念与常识" / "LLM 运行机制"。
   中标题会"持有"随后直到下一个 H2 之前的所有内容。

3. **题目（两种写法都能识别，会自动兜底）**：
   a) **H3 版（Java 基础页）**：中标题区内的 ``###`` 行 = 题干，答案 = 该 H3 之后直到
      下一个同级 H3 / H2 / H1 之前的所有 Markdown 段落。页内 anchor 来自该 H3 的 id 链接。
   b) **列表版（LLM 面试页）**：H2 下以"常见面试题："/"常见问题："为前缀的小节中，
      ``- 题干文本?/？/：`` 开头的无序列表项 = 题干，答案 = 该列表项之后到下一个同级列表
      项 / H2 之前的段落（若段落为空则用占位答案说明）。

4. **兜底合并**：H2 区域内既没有 H3 也没有 "- 题干" 列表，则把整段 H2 视作"章节型题目"
   （H2 作为题干，整个章节内容作为答案），避免漏题。

全部实现都是纯文本 + 状态机扫描，不依赖 LLM / NLP，可重现、可测试。
"""

from __future__ import annotations

import logging
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional
from urllib.parse import urldefrag, urljoin, urlparse, urlunparse

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════
#  数据结构
# ═══════════════════════════════════════════

@dataclass
class QuestionChunk:
    """切片后的一道题（还未向量化，embedding 在 ingest 层做）。"""

    big_topic: str
    mid_topic: str
    question: str
    answer_md: str
    answer_anchor: str  # 纯 anchor 文本（不含 #），用于 source_hash 和前端跳转
    # 本页只有题干没有答案时置 True，由 crawler 层跟随详情链接补齐答案
    answer_missing: bool = False
    # 无答案归因（crawler 层填写）：空串表示有答案；否则记录为什么没抓到答案
    no_answer_reason: str = ""


# ═══════════════════════════════════════════
#  通用小工具
# ═══════════════════════════════════════════

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.M)
# Markdown 链接形式 ``[文本](url "title")`` -> 拆文本，作为 anchor（html2text 会对 H2/H3 包成链接）
_MD_LINK_RE = re.compile(r"^\[(.+?)\]\(([^)]+)\)\s*$")
# list item: `- xxx?` 开头
_LIST_ITEM_RE = re.compile(r"^\s*[-*+]\s+(.+?)\s*$", re.S)
# markdown 代码围栏
_FENCE_RE = re.compile(r"^\s*```")
# "常见面试题：" / "常见问题：" 这类提示，说明后续无序列表是题目列表
_INTRO_LINE_RE = re.compile(r"常见(?:面试)?题[:：]")
# 噪声标题（精确匹配）："写在最后" / "参考资料" 等对题库没有价值
_NOISE_TITLE_EXACT_RE = re.compile(
    r"^(?:写在最后|写在最后的话|结语|参考文献|参考资料|参考链接|参考文章|参考|"
    r"延伸阅读|推荐阅读|相关阅读|推荐文章|相关推荐|转载声明|免责声明|声明|关于作者|"
    r"更新记录|版本记录|references?|reference links?|further reading)$",
    re.I,
)
# 噪声标题（子串兜底）：覆盖 "参考资料与延伸阅读" 这类复合标题
_NOISE_TITLE_SUBSTR = ("写在最后", "参考文献", "参考链接", "参考文章")
# 纯链接列表项（如参考资料里的 `- [xxx](url)`），不能当作题干
_PURE_LINK_ITEM_RE = re.compile(r"^\s*!?\[[^\]]*\]\([^)]*\)\s*[。.，,、；;：:]?\s*$")
# 导航链接项：加粗包裹的纯链接，或"链接 + 一段括号描述"（如前言里的
# `**[Java 基础常见面试题总结(上)](url)**（Java 语言的基本概念...）`）。
# 这类项是页面导航，不是题目；其指向的页面会被二层爬取单独切题。
_NAV_LINK_ITEM_RE = re.compile(
    r"^\s*\*{0,2}\s*!?\[[^\]]*\]\([^)]*\)\s*\*{0,2}\s*(?:[（(][^）)]*[）)])?\s*$"
)


def _is_noise_title(title: str) -> bool:
    """判断 H2/H3 标题是否为噪声章节（写在最后/参考资料等），命中则整节丢弃。"""
    t = _strip_decorators(title)
    # 去掉标题开头的 ⭐️ 等装饰字符后再比对
    t = re.sub(r"^[\W_]+", "", t).strip()
    if not t or _NOISE_TITLE_EXACT_RE.match(t):
        return True
    return any(s in t for s in _NOISE_TITLE_SUBSTR)


def _strip_decorators(s: str) -> str:
    """去掉标题前后的 ⭐️ 等装饰 emoji（保留题干中内部的装饰用于显示，只删两端）。"""
    s = s.strip()
    return s.strip(" \t\u3000").strip("*·-—•")


def _extract_heading_text_and_anchor(line: str) -> tuple[str, str]:
    """把一行（``### [⭐️JVM vs JDK vs JRE](url#anchor)`` / ``### ⭐️JVM vs JDK vs JRE``）
    解析成 ``(显示文本, anchor)``，anchor 找不到时用 slug 生成。
    """
    m = _MD_LINK_RE.match(line.strip())
    if m:
        text = m.group(1).strip()
        target = m.group(2).strip()
        # 解析 fragment
        _, frag = urldefrag(target)
        anchor = frag or _slugify(text)
        return text.strip(), anchor
    return line.strip(), _slugify(line.strip())


def _slugify(s: str) -> str:
    """兼容中文的 slug（直接把空白替换，不做 transliteration；中文可直接当 anchor）。"""
    s = s.strip()
    # 去掉 markdown 格式残留
    s = re.sub(r"[`*_#]", "", s)
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"-+", "-", s).strip("-")
    return unicodedata.normalize("NFC", s)


def _is_heading(line: str, level: Optional[int] = None) -> Optional[tuple[int, str]]:
    """若 line 是 Markdown 标题，返回 (层级, 去掉 # 后的内容)；否则 None。"""
    m = _HEADING_RE.match(line)
    if not m:
        return None
    lvl = len(m.group(1))
    if level is not None and lvl != level:
        return None
    return lvl, m.group(2).strip()


def _in_code_fence(fence_open: bool, line: str) -> bool:
    """返回更新后的 fence 状态。"""
    if _FENCE_RE.match(line):
        return not fence_open
    return fence_open


# ═══════════════════════════════════════════
#  对外主入口
# ═══════════════════════════════════════════

def split_markdown(md_text: str, source_url: str = "") -> list[QuestionChunk]:
    """把一篇 Markdown 切成一组 QuestionChunk。

    Args:
        md_text: 页面正文（含 #/##/### 标题结构）
        source_url: 仅用于打日志/报错定位，不影响切片结果。
    """
    if not md_text or not md_text.strip():
        return []

    lines = md_text.splitlines()
    # ── Step 1: 抽出 H1（大标题） ──
    big_topic: Optional[str] = None
    for line in lines:
        h = _is_heading(line, 1)
        if h:
            big_topic = _strip_decorators(_extract_heading_text_and_anchor(h[1])[0])
            break
    if not big_topic:
        # 没有 H1 就从 URL 猜一个，避免整页被丢
        fallback = urlparse(source_url or "").path.rstrip("/").rsplit("/", 1)[-1] or "未命名专题"
        big_topic = fallback.replace("-", " ").replace("_", " ").strip() or "未命名专题"
        logger.warning("[quiz.splitter] 未找到 H1，从 URL 回退大标题为 %s (src=%s)", big_topic, source_url)

    # ── Step 2: 按 H2 分块，得到 mid_sections: list[(mid_title, mid_anchor, body_lines)] ──
    mid_sections: list[tuple[str, str, list[str]]] = []
    cur_title: Optional[str] = None
    cur_anchor: str = ""
    cur_lines: list[str] = []

    # 先跳过 H1 之前的所有内容（TOC / 站点元信息）
    before_h1_done = False
    for line in lines:
        h = _is_heading(line)
        if not before_h1_done:
            if h and h[0] == 1:
                before_h1_done = True
            continue

        if h and h[0] == 2:
            # 提交上一节
            if cur_title is not None:
                mid_sections.append((cur_title, cur_anchor, cur_lines))
            text, anchor = _extract_heading_text_and_anchor(h[1])
            if _is_noise_title(text):
                # "写在最后"/"参考资料" 等噪声章节：连同正文一并丢弃
                cur_title = None
                cur_anchor = ""
                cur_lines = []
                continue
            cur_title = _strip_decorators(text) or "(未命名章节)"
            cur_anchor = anchor or _slugify(cur_title)
            cur_lines = []
            continue

        if cur_title is None:
            # 暂未进入任何 H2（通常是 H1 → 前言部分），丢掉；前言/介绍一般不是题目
            continue
        cur_lines.append(line)

    if cur_title is not None:
        mid_sections.append((cur_title, cur_anchor, cur_lines))

    # 整页没有 H2 的兜底：把正文整体当一个 mid_topic
    if not mid_sections:
        mid_sections.append((big_topic, _slugify(big_topic), lines))

    # ── Step 3: 每个 H2 块内解析 "题 -> 答案" ──
    chunks: list[QuestionChunk] = []
    for mid_title, mid_anchor, body_lines in mid_sections:
        body_chunks = _parse_mid_section(
            big_topic=big_topic,
            mid_title=mid_title,
            mid_anchor=mid_anchor,
            body_lines=body_lines,
        )
        chunks.extend(body_chunks)

    # 极端兜底：整页没有切出任何题目，就把整页当一题（避免整页浪费）
    if not chunks:
        chunks.append(
            QuestionChunk(
                big_topic=big_topic,
                mid_topic=big_topic,
                question=f"[{big_topic}] 整节概览",
                answer_md=_lines_to_md(lines),
                answer_anchor=_slugify(big_topic) or "overview",
            )
        )

    logger.info("[quiz.splitter] %s 切出 %d 道题（章节数 %d）",
                big_topic, len(chunks), len(mid_sections))
    return chunks


# ═══════════════════════════════════════════
#  Section 级解析
# ═══════════════════════════════════════════

def _parse_mid_section(
    *,
    big_topic: str,
    mid_title: str,
    mid_anchor: str,
    body_lines: list[str],
) -> list[QuestionChunk]:
    """在一个 H2 区块内解析出 1~N 道题。

    策略优先级：
    1) 如果 section.body 里出现 ``###`` → 使用 H3 模式（每题=H3 + 其后内容到下一个同级 H3/H2）
    2) 否则如果出现 "常见面试题：" / "常见问题：" 引导语且其后有 ``- xxx`` 列表 → 列表模式
    3) 否则如果直接就是一连串独立的 ``- xxx?`` 列表项 → 列表模式
    4) 否则整节作为章节型题目
    """
    # 判断优先级：扫描 body 头部看是否有 H3
    has_h3 = any(_is_heading(line, 3) is not None for line in body_lines)
    if has_h3:
        return _parse_mid_section_h3(big_topic, mid_title, mid_anchor, body_lines)

    # 列表模式：尝试抽 "- 题干" 序列；至少要两道才算有效，否则可能是普通列表
    intro_line_idx: Optional[int] = None
    for idx, line in enumerate(body_lines):
        if _INTRO_LINE_RE.search(line):
            intro_line_idx = idx
            break
    scan_from = (intro_line_idx + 1) if intro_line_idx is not None else 0
    list_items = _extract_list_items(body_lines[scan_from:])
    # 提取出 >=1 项且至少一项带问号/冒号（面试题特征）就认为是题库结构
    if list_items and any(
        ("?" in t or "？" in t or ":" in t or "：" in t) for t, _ in list_items
    ):
        chunks = _parse_mid_section_list(
            big_topic, mid_title, mid_anchor, body_lines, scan_from, list_items
        )
        if chunks:
            return chunks

    # 章节型兜底：H2 本身作为题的标题，答案是整节 body
    answer = _lines_to_md(body_lines).strip()
    if not answer:
        # 连正文都没有，就不给空题，直接跳过
        return []
    return [
        QuestionChunk(
            big_topic=big_topic,
            mid_topic=mid_title,
            question=f"{mid_title}（章节考点）",
            answer_md=answer,
            answer_anchor=mid_anchor or _slugify(mid_title),
        )
    ]


# ═══════════════════════════════════════════
#  模式 1：H3 题干
# ═══════════════════════════════════════════

def _parse_mid_section_h3(
    big_topic: str,
    mid_title: str,
    mid_anchor: str,
    body_lines: list[str],
) -> list[QuestionChunk]:
    """按 H3 切分，每个 H3 对应一题，题头 H2 下方直到第一个 H3 的文本作为章节引言并入 mid_title
    章节题（可选，如果文本够多）。
    """
    chunks: list[QuestionChunk] = []

    intro_buf: list[str] = []
    cur_q: Optional[str] = None
    cur_anchor: str = ""
    cur_ans: list[str] = []
    fence_open = False

    def _flush_question():
        if cur_q is None:
            return
        # 题干清理：markdown 链接只留显示文本（"什么是 [Redis](url)？" -> "什么是 Redis？"）
        q = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", cur_q).strip()
        q = _strip_decorators(q)
        # 纯链接题干（如"前言"里的导航链接 "- [xx(上)](url)"）不是题目，跳过；
        # 其指向的页面会被二层爬取单独切题
        if not q or _PURE_LINK_ITEM_RE.match(cur_q.strip()):
            return
        ans = _lines_to_md(cur_ans).strip()
        if not ans:
            # 只有题干没有答案：打标记，由 crawler 层跟随详情链接补齐
            chunks.append(
                QuestionChunk(
                    big_topic=big_topic,
                    mid_topic=mid_title,
                    question=q,
                    answer_md="",
                    answer_anchor=cur_anchor or _slugify(q),
                    answer_missing=True,
                )
            )
            return
        chunks.append(
            QuestionChunk(
                big_topic=big_topic,
                mid_topic=mid_title,
                question=q,
                answer_md=ans,
                answer_anchor=cur_anchor or _slugify(q),
            )
        )

    # 噪声 H3（如"### 写在最后"）：其下内容丢弃，直到下一个正常 H3
    dropping = False
    for line in body_lines:
        fence_open = _in_code_fence(fence_open, line)
        # 代码围栏里的内容即使看起来像标题也不处理
        if not fence_open:
            h = _is_heading(line)
            if h and h[0] == 3:
                _flush_question()
                q_text, q_anchor = _extract_heading_text_and_anchor(h[1])
                if _is_noise_title(q_text):
                    cur_q = None
                    cur_anchor = ""
                    cur_ans = []
                    dropping = True
                    continue
                dropping = False
                cur_q = q_text
                cur_anchor = q_anchor
                cur_ans = []
                continue
            if h and h[0] < 3:  # 遇到 H2/H1，本 section 结束
                break

        if dropping:
            continue
        if cur_q is None:
            # 第一题出现前的引言
            intro_buf.append(line)
        else:
            cur_ans.append(line)

    _flush_question()

    # 如果引言较长 (> 120 字符)，说明 H2 本身带了一些考点介绍，也作为 1 道章节型题目入题库
    intro_md = _lines_to_md(intro_buf).strip()
    if len(intro_md) >= 120:
        chunks.insert(
            0,
            QuestionChunk(
                big_topic=big_topic,
                mid_topic=mid_title,
                question=f"{mid_title} - 章节引言/考点总览",
                answer_md=intro_md,
                answer_anchor=mid_anchor or _slugify(mid_title),
            ),
        )
    return chunks


# ═══════════════════════════════════════════
#  模式 2：列表题干（LLM 面试页风格）
# ═══════════════════════════════════════════

def _extract_list_items(lines: list[str]) -> list[tuple[str, int]]:
    """返回 ``[(text, idx_in_lines)]``，idx 是原列表行在 lines 中的位置。

    注意：只抓顶级 `-` 列表（行首空白不超过 3 个），避免抓嵌套 bullet（题目里可能带
    "1. xxx / - subpoint" 这样的答案项）。
    """
    items: list[tuple[str, int]] = []
    in_code = False
    for idx, line in enumerate(lines):
        in_code = _in_code_fence(in_code, line)
        if in_code:
            continue
        # 顶级 bullet：缩进 <= 3（4 及以上属于嵌套列表/代码块延续）
        indent_match = re.match(r"^(\s*)[-*+]\s+", line)
        if indent_match and len(indent_match.group(1)) <= 3:
            item_text = line[indent_match.end():].strip()
            if not item_text:
                continue
            # 导航/纯链接项（**[xxx](url)**、[xxx](url)（描述））不是题干，跳过
            if _NAV_LINK_ITEM_RE.match(item_text) or _PURE_LINK_ITEM_RE.match(item_text):
                continue
            items.append((item_text, idx))
    return items


def _parse_mid_section_list(
    big_topic: str,
    mid_title: str,
    mid_anchor: str,
    body_lines: list[str],
    scan_from: int,
    list_items: list[tuple[str, int]],
) -> list[QuestionChunk]:
    """列表模式：``- 题干`` 作为题，答案 = 该 bullet 后直到下一题 bullet / H2 / 文件结尾之间
    的所有正文。如果某题后正文为空（常见于 AI 专题把"答案"放在对应外链文章），给占位说明。
    """
    chunks: list[QuestionChunk] = []

    # 在 body_lines 中的绝对位置 = scan_from + item[1]
    abs_positions = [scan_from + rel for _, rel in list_items]
    for i, (q_text, _rel) in enumerate(list_items):
        start = abs_positions[i] + 1  # 跳过题目行本身
        end = abs_positions[i + 1] if i + 1 < len(abs_positions) else len(body_lines)
        answer_lines = body_lines[start:end]
        # 结尾遇到 H2/H3 提前终止
        trimmed_answer: list[str] = []
        in_fence = False
        for ln in answer_lines:
            in_fence = _in_code_fence(in_fence, ln)
            if not in_fence:
                h = _is_heading(ln)
                if h and h[0] <= 3:
                    break
            trimmed_answer.append(ln)
        ans_md = _lines_to_md(trimmed_answer).strip()
        # 只有题干没有答案：置空并打标记，由 crawler 层跟随详情链接补齐
        answer_missing = not ans_md
        # 题干清理：剥掉加粗与 markdown 链接外壳；导航项/剥完为空则不是题目
        if _NAV_LINK_ITEM_RE.match(q_text) or _PURE_LINK_ITEM_RE.match(q_text):
            continue
        q_clean = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", q_text).strip()
        q_clean = _strip_decorators(q_clean)
        if not q_clean:
            continue
        anchor = _slugify(q_clean)
        chunks.append(
            QuestionChunk(
                big_topic=big_topic,
                mid_topic=mid_title,
                question=q_clean,
                answer_md=ans_md,
                answer_anchor=anchor or (f"{mid_anchor}-q{i}" if mid_anchor else f"q{i}"),
                answer_missing=answer_missing,
            )
        )
    return chunks


def _lines_to_md(lines: Iterable[str]) -> str:
    """把按行切分的 Markdown 合并回字符串，并去掉首尾多余空行。"""
    text = "\n".join(lines)
    # 首尾连续空行压缩
    return text.strip("\n")


# ═══════════════════════════════════════════
#  辅助：修正答案 Markdown 中的相对链接
# ═══════════════════════════════════════════

def absolutize_links(md_text: str, base_url: str) -> str:
    """把 ``[x](./a.html)`` / ``[y](/a/b.md)`` / ``![img](./x.png)`` 这类相对链接
    改写成绝对 URL，保证用户在答案中看到的链接可直接点击。

    纯正则实现，不覆盖全部极端 Markdown 场景，但足以处理 html2text 的规范化输出。
    """
    if not md_text or not base_url:
        return md_text

    def _sub(m: re.Match) -> str:
        prefix = m.group(1)  # '[' or '!['
        inner = m.group(2)
        # html2text 的 protect_links 会把 URL 包成 <...>，先剥掉再判断，
        # 否则 "<https://x>" 会被当成相对路径拼上页面 URL 导致链接损坏
        url = m.group(3).strip().strip("<>").strip()
        title = m.group(4) or ""
        # 跳过已绝对的、mailto:、#anchor、协议开头的
        if (
            not url
            or url.startswith(("http://", "https://", "mailto:", "data:", "//", "#"))
        ):
            return m.group(0)
        try:
            url = urljoin(base_url, url)
        except Exception:  # pragma: no cover
            return m.group(0)
        if title:
            return f"{prefix}{inner}]({url} {title})"
        return f"{prefix}{inner}]({url})"

    # [text](url "title") / ![alt](url)
    pattern = re.compile(
        r"(!?\[)((?:[^\[\]\\]|\\.)*)\]\(\s*([^)\s]+)(\s+\"[^\"]*\")?\s*\)"
    )
    return pattern.sub(_sub, md_text)


def absolutize_source_url_anchor(source_url: str, anchor: str) -> str:
    """把 ``source_url + anchor`` 拼成带 #fragment 的可点击跳转 URL。"""
    if not source_url:
        return ""
    parsed = list(urlparse(source_url))
    # [4] = query, [5] = fragment
    parsed[5] = anchor or ""
    return urlunparse(parsed)
