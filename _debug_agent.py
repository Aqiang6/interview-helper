"""诊断：Agent 种子页未匹配题 vs agent-basis 详情页题干的相似度。"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

from ai_interviewer.quiz.crawler import (
    _UA, _fetch_page_chunks, _norm_q, _similarity,
)

SEED = "https://javaguide.cn/ai/interview-questions/agent-interview-questions.html"
DETAIL = "https://javaguide.cn/ai/agent/agent-basis.html"


async def main() -> None:
    async with httpx.AsyncClient(
        headers={"User-Agent": _UA, "Accept-Language": "zh-CN,zh;q=0.9"},
        follow_redirects=True, timeout=60,
    ) as client:
        _, seed_chunks = await _fetch_page_chunks(client, SEED, 30.0)
        _, detail_chunks = await _fetch_page_chunks(client, DETAIL, 30.0)

    print("详情页切出题目数:", len(detail_chunks))
    for i, dc in enumerate(detail_chunks):
        print(f"  [{i}] {dc.question[:70]}")

    missing = [c for c in seed_chunks if c.answer_missing]
    print("\n种子页未补到答案的题数:", len(missing))
    for c in missing:
        nq = _norm_q(c.question)
        scored = sorted(
            ((dc, _similarity(nq, _norm_q(dc.question))) for dc in detail_chunks),
            key=lambda t: t[1], reverse=True,
        )[:3]
        print("-" * 60)
        print("  未匹配题:", c.question[:70])
        for dc, s in scored:
            print(f"    {s:.3f}  <- {dc.question[:60]}")


asyncio.run(main())
