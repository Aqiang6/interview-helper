# Interview_Helper — AI 模拟面试 + 刷题助手 + AI 问答助手

一个本地运行的面试准备工具，由三个模块组成：

* **AI 模拟面试**：上传简历 + 选择/粘贴 JD，由 LangGraph Agent 扮演面试官进行多轮对话式提问（按 JD 要求强度决定追问深度），结束后输出评估。
* **刷题助手**：自动爬取面试题网站，切题入库到 PostgreSQL + pgvector，支持随机 / 按主题 / 自定义主题（向量语义检索）三种出题方式，答案可展开、可跳转原文。
* **AI 问答助手**：基于刷题知识库做语义检索 + LLM 综合回答，回答下方附参考例题与原文链接，可"试试"指定题目的等值命中模式。

前端为单页应用（`frontend/index.html`，Vue 3 + 原生组件），后端 FastAPI（`ai_interviewer/app.py`），接口走 HTTP / SSE。

***

## 一、项目功能

### 1. AI 模拟面试

| 功能 | 说明 |
| --- | --- |
| 简历解析 | 上传 PDF 或粘贴文本，提取姓名 / 技能列表 / 项目经历段落；PDF 按文件 SHA256 缓存，二次上传秒读 |
| JD 驱动的提问规划 | 预置腾讯校招岗位或粘贴自定义 JD，LLM 输出 `ordered_topics + skill_importance(required/preferred/bonus) + suggested_max_questions + topics_per_skill`，预置 JD 首次生成后缓存 |
| 多轮对话面试 | 出题 → 听回答 → 自然状态路由（追问 / 换话题 / 结束）→ 下一题，全程 SSE 真流式逐 token 推送 |
| 长期记忆 | Agent 全程维护 `InterviewState`：对话历史 + 面试计划 + 当前话题进度，每轮出题 LLM 都能看到完整上下文 |
| 面试评估 | 题数到顶自动结束，`/api/evaluation` 返回题数统计；前端展示"再来一次"入口 |
| 自定义面试指令 | "再来一次"时填写定向调整需求，注入面试官 System Prompt |

### 2. 刷题助手

| 功能 | 说明 |
| --- | --- |
| 一键构建题库 | 从 `ai_interviewer/爬虫.txt` 的种子 URL 出发做两层爬取，自动 HTML→Markdown→切题→向量化→入库 |
| 三种出题模式 | 随机（PG `ORDER BY RANDOM()`）/ 按"大标题 + 中标题"精准 WHERE 过滤 / 输入一句话自定义主题（pgvector 余弦相似度 topK + 阈值） |
| 答案按需展开 | 出题卡片默认只给题干 + 元信息，点"显示答案"时前端再单独请求完整 Markdown（含链接、代码、表格） |
| 原题跳转 | 每道题带 `原文URL#锚点`，可直接跳到来源页面出题位置 |
| 无答案归因 | 抓不到答案的题不静默丢弃：显示"此题无答案" + 原因 + 原文链接 |
| 主题树浏览 | `大标题 → 中标题 → 题目数` 的树状接口，供前端下拉选择 |
| 题库统计 | 总题数 / 大标题数 / 中标题数 / 来源页数 |

### 3. AI 问答助手

| 功能 | 说明 |
| --- | --- |
| 语义检索 + LLM 综合 | 用问题向量做 pgvector 余弦相似度召回 topK，再把参考片段拼进 LLM System Prompt 由模型综合作答 |
| 参考例题展示 | 回答下方单独列出参考例题（含 `source_url_with_anchor`），不在回答正文里堆引用 |
| SSE 流式回答 | 先推 status（加载模型 / 检索 / 生成中）→ 逐 delta 推 answer → 最后推 references + done |
| Top-K 分差过滤 | 召回结果以最高相似度为锚，只保留分差 ≤0.05 的题（最多 3 条），避免堆砌 |
| "试试"模式 | 指定 `(mid_topic, question)` 时跳过向量化与语义检索，只参考等值命中的那道题作答，命中失败 LLM 仍可凭自身知识作答 |
| 模型预热 | FastAPI lifespan 启动时后台 daemon 线程预热 sentence-transformers，避免首次问问题卡 Loading weights |

***

## 二、部署与使用

### 环境要求

* Python 3.10+
* PostgreSQL 14+，安装 **pgvector 扩展**
* 一个 OpenAI 兼容的 LLM API（面试 Agent 和问答助手用；刷题模块默认用本地 embedding 模型，不需要 Key）

### 1. 安装依赖

```bash
cd Interview_Helper
pip install -e .
# 刷题使用本地 embedding 时额外装（CPU 版 PyTorch + sentence-transformers）：
python -m pip install sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
```

### 2. 配置 `.env`

在项目根目录创建 `.env`（字段名见 `ai_interviewer/config.py`），必配项：

```ini
# 面试 Agent / 问答助手的 LLM（OpenAI 兼容接口均可）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 刷题知识库：PostgreSQL 连接串（需先装好 pgvector 扩展）
#   建库：CREATE DATABASE interview_helper;
#   进入库后：CREATE EXTENSION IF NOT EXISTS vector;
POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/interview_helper

# 刷题 embedding：本地模型，免 Key
QUIZ_EMBEDDING_BACKEND=sentence_transformers
QUIZ_SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
QUIZ_EMBEDDING_DIMENSIONS=384
```

可选调优项（均有默认值）：

| 字段 | 默认 | 说明 |
| --- | --- | --- |
| `AGENT_MAX_QUESTIONS` | 15 | 单场面试最大题数（LLM 会按 JD 密度在 8–25 内调整） |
| `QUIZ_TOP_K` | 10 | 自定义主题 / 问答助手召回题数 |
| `QUIZ_MIN_SIMILARITY` | 0.3 | 语义检索余弦相似度阈值 |
| `QUIZ_CRAWLER_CONCURRENCY` | 5 | 爬虫并发数 |
| `QUIZ_INDEX_K` | 12 | pgvector HNSW 候选池大小 |
| `HF_ENDPOINT` | `https://hf-mirror.com` | Hugging Face 模型下载镜像（国内推荐） |

### 3. 初始化刷题知识库

```bash
python -m ai_interviewer.quiz.ingest
```

自动完成：建表 / 建 HNSW 索引 → 读取种子 URL → 两层爬取 → 切题 → 向量化 → 入库。幂等可重复执行；首次运行需下载 embedding 模型权重并爬全量页面，耗时较长。

### 4. 启动应用

```bash
# 方式一：Windows 双击（自动拉起本地 PostgreSQL + 启动 Web 应用 + 打开浏览器）
start_app.bat

# 方式二：命令行
python -m ai_interviewer.app
```

启动后访问 <http://localhost:8000>（FastAPI 同时在 `/docs` 提供 OpenAPI 交互文档）。

### 5. 使用步骤

**AI 模拟面试：**

1. 顶部切换到"面试助手"页，在"模型设置"填写 `API Key / Base URL / 模型名`；
2. 上传简历 PDF（或粘贴文本）→ 自动解析候选人画像，已上传过的 PDF 直接从缓存列表选用；
3. 选择预置岗位（`/api/positions`）或粘贴自定义 JD → 后端 LLM 生成话题规划；
4. 点击"开始面试"→ Agent 给出第一题（SSE 流式）；
5. 在对话框逐题作答，Agent 按配额 + 题数自然追问或切换话题；
6. 达到题数上限自动结束，可查看评估或点"再来一次"填定向调整需求重开。

**刷题助手：**

1. 顶部切换到"刷题助手"页，查看题库统计和主题树；
2. 三种出题方式任选：随机 / 选"大标题(+中标题)" / 输入一句话自定义主题；
3. 点"显示答案"展开完整解析，或点"查看原题"跳转到来源页面锚点位置；
4. 无答案的题会显示"此题无答案" + 原因 + 原文链接。

**AI 问答助手：**

1. 顶部切换到"AI 问答助手"页（首页已随机展示示例问题可点击）；
2. 在输入框输入技术问题，点"发送"；
3. SSE 流式返回：先看到 status 进度（向量化问题 / 已检索 N 条 / 生成中），随后逐字输出回答；
4. 回答下方自动列出参考例题（含原题链接），可点"试试"用该题做等值命中检索后再问一次。

### 6. 常用 API 一览

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/api/positions` | 列出预置岗位（腾讯校招 JD） |
| POST | `/api/parse-resume` / `/api/parse-pdf` | 解析简历文本 / PDF |
| GET / DELETE | `/api/resumes` / `/api/resumes/{hash}` | 已缓存简历列表 / 取一条 / 删一条 |
| POST | `/api/start-interview` | 创建面试会话（简历 + JD + 岗位） |
| POST | `/api/first-question-stream` | 获取第一题（SSE 真流式） |
| POST | `/api/chat` / `/api/chat-stream` | 提交回答（后者 SSE 流式） |
| GET | `/api/evaluation` | 获取面试评估 |
| GET | `/api/agent-state` | 查看 LangGraph Agent 内部状态（计划 + 进度） |
| GET | `/api/quiz/stats` / `/api/quiz/topics` | 题库统计 / 主题树 |
| POST | `/api/quiz/ingest` | 触发一次知识库重建入库 |
| POST | `/api/quiz/question` | 出题（`mode=random/by_topic/custom`） |
| GET | `/api/quiz/question/{id}/answer` | 查看某题答案 |
| POST | `/api/qa/ask` / `/api/qa/ask-stream` | AI 问答助手（后者 SSE 流式，支持 `try_question` 试试模式） |
| GET | `/api/health` | 健康检查 |

***

## 三、底层实现简述

### 1. AI 模拟面试：LangGraph 状态机（`ai_interviewer/agent/`）

7 节点共享 `InterviewState`：`messages`（Memory）/ `interview_plan`（Planning）/ 简历与 JD（Context）/ 题数与当前话题（Progress）。

```
plan_interview（JD+简历规划话题与配额）
  ├─ 无 HumanMessage → generate_question（出第一题）
  └─ 有 HumanMessage → evaluate_response（占位） → decide_next（纯规则路由）
                                              ├─ followup（追问）  ┐
                                              ├─ next_topic（换话题）┴→ generate_question → END
                                              └─ end → generate_evaluation → END
```

* **JD 驱动话题规划**：`prioritize_by_jd`（`topic_prioritizer.py`）把 JD 原文 + 岗位名喂给 LLM，要求输出 JSON `{ordered_topics, skill_importance, suggested_max_questions, topics_per_skill}`；预置 JD 按 `sha256(jd_raw_text)` 缓存到 `.jd_cache/`，下次秒读；任何失败直接抛错（不降级），错误消息对 `api_key` 脱敏后回传前端。
* **多因子规则评分（备用规划器 `TopicPrioritizer`）**：4 层可插拔因子 — Category 基础权重（AI_AGENT_CORE > 分布式 > 数据层 > 语言框架 > 工具）+ 简历命中深度加成 + 知识库覆盖加成 + 同类聚类连续性；通过 `register_rule()` 扩展无需改核心。
* **路由决策（`make_route_decision`）**：纯规则、不看回答质量 — 题数到顶 → end；本话题配额满且还有后续 → next_topic；末话题 → followup 深挖；剩题不足以覆盖剩余话题 → next_topic 压缩追问；默认 followup。
* **出题上下文**：System Prompt 拼装 = 面试官人设 + JD 要求强度（required/preferred/bonus）+ 当前话题配额进度 + 简历项目段落 + 自定义面试指令；调用原生 `openai` 库（不用 langchain 的 LLM 抽象，避免 bigmodel 兼容问题）。
* **SSE 真流式**：模块级 `_stream_cb` 回调机制 — invoke graph 前设置回调，graph 内 `_call_llm` 用 `stream=True` 逐 delta 调回调，回调把 delta 放入 `queue.Queue`，SSE generator 从队列读取推送；graph 在独立线程跑同步 `invoke`，避免阻塞 async event loop。
* **简历解析**：`resume_parser.py` 用关键词词库 + 正则模式提取技能 / 项目 / 学历 / 年限；`_extract_project_details` 用极简状态机切项目段落（短行 + 无冒号 = 段标题），不调 LLM；PDF 按文件 SHA256 缓存解析结果到 `.resume_cache/`，避免重复解析。

### 2. 刷题助手：爬取 → 切片 → 向量化 → 入库（`ai_interviewer/quiz/`）

```
爬虫.txt (种子 URL)
  → crawler.crawl_urls (httpx + tenacity 重试 + asyncio.Semaphore 并发)
  → BeautifulSoup 选最大正文容器 + html2text 转 Markdown (保留 H1/H2/H3/链接/表格/代码)
  → splitter.split_markdown (纯状态机切题，不依赖 LLM)
  → ingest._build_embeddings (sentence-transformers 批量 encode)
  → db.bulk_upsert (sha1(source_url + '#' + anchor) 去重 upsert + HNSW 索引)
```

* **两层爬取**：种子页当目录（抽出详情链接），详情页才是题目；`asyncio.Semaphore(QUIZ_CRAWLER_CONCURRENCY)` 控并发，单页失败不影响其它页。
* **状态机切片**：H1 = 大主题、H2 = 中主题，H2 内按"每个 H3 一题"或"`- 题干?` 列表项一题"切分；噪声章节（写在最后 / 参考资料 / 延伸阅读等，正则 + 子串双重匹配）丢弃；只有题干的题跟随详情链接回填答案，补不到记 `no_answer_reason`；H2 区域既无 H3 也无列表时兜底把整个 H2 当作章节型题目。
* **embedding 文本** = `大主题｜中主题｜题干`（不含答案）—— 大/中标题作分类约束让同主题向量聚拢，题干含考点关键词作语义匹配依据。
* **去重 + 索引**：`source_hash = sha1(source_url + '#' + answer_anchor)` UNIQUE 约束做增量 upsert；`embedding` 列类型 `vector(N)`，建 HNSW 索引（`vector_cosine_ops`）+ `(big_topic, mid_topic)` 复合索引 + `to_tsvector('simple', question)` GIN 索引（备用关键词搜索）。
* **三模式检索**：随机 `ORDER BY RANDOM()`；按主题 `(big_topic, mid_topic)` 精确 WHERE + `ordinal` 排序保原题序；自定义主题把输入文本用同一 `Embedder.embed_query` 做向量，pgvector `<=>` 余弦距离排序 + 阈值过滤取 topK。出题接口一律**不带答案**，点"显示答案"时前端再单独请求。
* **embedding 后端二选一**：`sentence_transformers`（默认，本地 Sentence-BERT，首次下载权重 ~80MB，启动时后台 daemon 线程预热）；`openai`（复用 `OPENAI_API_KEY` 走 `text-embedding-3-small` 等）。维度由 `QUIZ_EMBEDDING_DIMENSIONS` 显式声明，校验失败直接抛错（不静默 fallback）。

### 3. AI 问答助手：知识库语义检索 + LLM 综合（`ai_interviewer/app.py` + `quiz/retriever.py`）

```
用户问题
  → retriever.search_for_qa (问题文本 → embedding → pgvector <=> 召回 topK + 阈值)
  → _filter_qa_items (以 top1 为锚，只留分差 ≤0.05 的，最多 3 条)
  → _build_qa_context (拼成"【片段 N】主题/问题/答案"喂 LLM)
  → _qa_llm_stream (openai stream=True 逐 delta 产出)
  → SSE 推 status → answer 逐 delta → references → done
```

* **检索策略**：QA 检索只用问题原文做 embedding，不复制三遍塞进 `big_topic｜mid_topic｜question` 三段（避免停用词三倍加权导致"句式像但实体不匹配"的题得分偏高，例如问 LangChain 反而召回 HNSW）。SQL 与刷题自定义主题模式同构，但多取一列 `answer_md` 用于拼上下文。
* **LLM Prompt**：System Prompt 模板要求"优先采用你的专业知识，拿到知识库的主题不一定准确，符合思路就参考，不符合就不参考"+ 知识库参考片段；明确禁止 LLM 在回答末尾自列引用（参考例题由系统在回答下方单独展示）。
* **"试试"模式**：`try_question={mid_topic, question}` 时跳过向量化与常规检索，走 `pick_exact_for_qa` 等值精准命中单题；LLM user 消息同时带上中标题与问题；命中失败返回空上下文 + 空引用，LLM 仍可凭自身知识作答。
* **流式编排**：SSE generator 在线程池逐块拉同步生成器（`asyncio.to_thread(next, gen, None)`），首个 answer chunk 前先推一个空 status 清空前端进度条；同步生成器 + 异步推送的混合模型让真 LLM delta 流式可见，不靠"假流式"分块返回。
* **复用与解耦**：问答助手与刷题助手共用 `quiz_questions` 表与 `Embedder`，但检索时维度校验、阈值、`_text_for_embedding` 拼接逻辑都走 quiz 模块同一份代码，改一处两边生效。
