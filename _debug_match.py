"""临时诊断：Agent 页详情链接提取 + 题目匹配情况。"""
import asyncio
import sys

sys.stdout.reconfigure(encoding="utf-8")

import httpx

from ai_interviewer.quiz.crawler import (
    _UA, _extract_related_detail_links, _norm_q,
    _fetch_one, html_to_markdown,
)
from ai_interviewer.quiz.splitter import absolutize_links, split_markdown

PAGE = "https://javaguide.cn/ai/interview-questions/agent-interview-questions.html"


async def main() -> None:
    async with httpx.AsyncClient(headers={"User-Agent": _UA}, follow_redirects=True, timeout=60) as client:
        html = await _fetch_one(client, PAGE, 30.0)
        md = absolutize_links(html_to_markdown(html, PAGE), PAGE)
        links = _extract_related_detail_links(md, PAGE)
        print("提取到详情链接:", len(links))
        for l in links:
            print("  ", l)

        page_chunks = split_markdown(md, source_url=PAGE)
        page_missing = [c for c in page_chunks if c.answer_missing]
        print("\n列表页题数:", len(page_chunks), " 缺答案:", len(page_missing))

        for link in links:
            dhtml = await _fetch_one(client, link, 30.0)
            dmd = absolutize_links(html_to_markdown(dhtml, link), link)
            dchunks = split_markdown(dmd, source_url=link)
            print("\n详情页:", link, " 切出", len(dchunks), "题")
            for dc in dchunks[:6]:
                print("   -", dc.question[:60])


asyncio.run(main())
