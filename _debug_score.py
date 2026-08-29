"""临时诊断：打印列表页题目与详情页题目的最佳匹配分数分布。"""
import asyncio
import difflib
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

from ai_interviewer.quiz.crawler import (
    _UA, _extract_related_detail_links, _norm_q,
    _fetch_one, html_to_markdown,
)
from ai_interviewer.quiz.splitter import absolutize_links, split_markdown

PAGE = "https://javaguide.cn/ai/interview-questions/agent-interview-questions.html"


def bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else {s}


def jaccard(a: str, b: str) -> float:
    A, B = bigrams(a), bigrams(b)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


async def main() -> None:
    async with httpx.AsyncClient(headers={"User-Agent": _UA}, follow_redirects=True, timeout=60) as client:
        html = await _fetch_one(client, PAGE, 30.0)
        md = absolutize_links(html_to_markdown(html, PAGE), PAGE)
        links = _extract_related_detail_links(md, PAGE)

        page_chunks = split_markdown(md, source_url=PAGE)
        missing = [c for c in page_chunks if c.answer_missing]

        index = []  # (norm, question_display, link, is_q_marked)
        for link in links:
            dhtml = await _fetch_one(client, link, 30.0)
            dmd = absolutize_links(html_to_markdown(dhtml, link), link)
            dchunks = split_markdown(dmd, source_url=link)
            for dc in dchunks:
                index.append((_norm_q(dc.question), dc.question, link, dc.answer_missing))

        print(f"列表缺答案 {len(missing)} 题，索引 {len(index)} 条\n")
        n_match = 0
        for c in missing:
            nq = _norm_q(c.question)
            best, bs, bq = None, 0.0, ""
            for k, disp, link, miss in index:
                if not k:
                    continue
                s1 = difflib.SequenceMatcher(None, nq, k).ratio()
                s2 = jaccard(nq, k)
                s = max(s1, s2)
                if s > bs:
                    best, bs, bq = link, s, disp
            mark = "✓" if bs >= 0.75 else ("~" if bs >= 0.55 else "✗")
            if mark == "✓":
                n_match += 1
            print(f"{mark} {bs:.2f}  {c.question[:44]}")
            if mark != "✓":
                print(f"      best[{bs:.2f}] {bq[:44]} ({best.rsplit('/',1)[-1]})")
        print(f"\n>=0.75 匹配: {n_match}/{len(missing)}")


asyncio.run(main())
