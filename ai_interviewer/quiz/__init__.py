"""刷题助手 - 基于 PostgreSQL + pgvector 的题库知识库（方案 A：结构化标题切片）。

对外暴露的主要模块：
- db:         数据访问层（建表 DDL、Engine/Session、ORM Model）
- splitter:   结构化切片器（H1 大标题 / H2 中标题 / 题目 / 答案 Markdown）
- crawler:    从 ``ai_interviewer/爬虫.txt`` 读取 URL 列表并抓页面产出结构化 Markdown
- ingest:     爬取 -> 切片 -> Embedding -> 批量 upsert 入库
- retriever:  出题检索（随机 / 按主题 / 自定义主题语义检索 / 获取答案）
"""

__all__ = ["db", "splitter", "crawler", "ingest", "retriever"]
