# Interview_Helper — AI 模拟面试 + 刷题助手 + AI 问答助手

一个本地运行的面试准备工具，由三个模块组成：

* **AI 模拟面试**：上传简历 + 选择/粘贴 JD，由 LangGraph Agent 扮演面试官进行多轮对话式提问，结合简历项目和回答历史追问。
* **刷题助手**：自动爬取面试题网站，切题入库到 PostgreSQL + pgvector，支持随机 / 按主题 / 自定义主题（向量语义检索）三种出题方式，答案可展开、可跳转原文。
* **AI 问答助手**：基于刷题知识库做语义检索 + LLM 综合回答，回答下方附参考例题与原文链接，可"试试"指定题目的等值命中模式。

面试练习侧重提问、追问和暴露知识盲点。回答得怎么样、哪些地方需要补强，由使用者自行判断和复盘；流程中不设置 AI 回答评分，也不生成综合评价报告。

前端为单页应用（`frontend/index.html`，Vue 3 + Element Plus），后端 FastAPI（`ai_interviewer/app.py`），接口走 HTTP / SSE。

***

## 一、底层实现简述

### 1. AI 模拟面试：LangGraph 状态机（`ai_interviewer/agent/`）

图节点共享 `InterviewState`：对话历史、面试计划、简历与 JD、题数与当前话题。会话保存在 Python 进程内存中，重启服务后不会恢复。

```text
/api/start-interview → 按岗位名 + JD 规划话题与配额（优先读缓存）

每轮运行图：plan_interview（已有计划则跳过）
  ├─ 无 HumanMessage → generate_question（出第一题）
  └─ 有 HumanMessage → evaluate_response（直通占位） → decide_next（按配额与题数路由）
                                              ├─ followup（追问）  ┐
                                              ├─ next_topic（换话题）┴→ generate_question → END
                                              └─ end → generate_evaluation（仅标记结束）→ END
```

* **JD 驱动话题规划**：`prioritize_by_jd`（`topic_prioritizer.py`）把 JD 原文 + 岗位名传给 LLM，要求输出 JSON `{ordered_topics, skill_importance, suggested_max_questions, topics_per_skill}`。当前规划调用没有传入简历技能或简历摘要；简历项目用于后续出题。预置和自定义 JD 都按 JD 原文哈希缓存到 `ai_interviewer/.jd_cache/`；规划调用失败会中止启动，不切换到旧的规则排序器。
* **路由决策（`make_route_decision`）**：纯规则、不看回答质量 — 题数到顶 → end；本话题配额满且还有后续 → next_topic；末话题 → followup 深挖；剩题不足以覆盖剩余话题 → next_topic 压缩追问；默认 followup。
* **回答与结束处理**：候选人回答直接追加到 `messages`，供下一轮出题参考；`evaluate_response` 返回空更新，不调用 LLM 分析或打分。`generate_evaluation` 沿用旧名称，实际只设置 `is_finished` 并记录 `question_count`，前端结束页提供“再来一次”入口。
* **出题上下文**：System Prompt 拼装 = 面试官人设 + JD 要求强度（required/preferred/bonus）+ 当前话题配额进度 + 简历项目段落 + 自定义面试指令；调用原生 `openai` 库（不用 langchain 的 LLM 抽象，避免 bigmodel 兼容问题）。
* **SSE 真流式**：模块级 `_stream_cb` 回调机制 — invoke graph 前设置回调，graph 内 `_call_llm` 用 `stream=True` 逐 delta 调回调，回调把 delta 放入 `queue.Queue`，SSE generator 从队列读取推送；graph 在独立线程跑同步 `invoke`，避免阻塞 async event loop。
* **简历解析**：`resume_parser.py` 用关键词词库 + 正则模式提取技能 / 项目 / 学历 / 年限；`_extract_project_details` 用段标题规则提取项目经历，不调 LLM，提取结果可在前端编辑。PDF 先用 Tika 提取文本，文本有效性过低时尝试 OCR；按文件 SHA256 缓存文本解析结果到 `ai_interviewer/.resume_cache/`，不对简历做向量化。

### 2. 刷题助手：爬取 → 切片 → 向量化 → 入库（`ai_interviewer/quiz/`）

**爬虫方案的由来**：一开始不知道 JavaGuide 的[开源仓库](https://github.com/Snailclimb/JavaGuide)中已经提供了 Markdown 源文档，因此选择了爬取网页、再转为 Markdown 的方案。后来发现可直接获取仓库里的 `.md` 文档，再做切题、向量化和入库，其实无需爬虫；当前项目保留的是下述爬虫实现。

```
爬虫.txt (种子 URL)
  → crawler.crawl_urls (httpx + tenacity 重试 + asyncio.Semaphore 并发)
  → BeautifulSoup 选最大正文容器 + html2text 转 Markdown (保留 H1/H2/H3/链接/表格/代码)
  → splitter.split_markdown (纯状态机切题，不依赖 LLM)
  → ingest.embed_and_upsert (sentence-transformers 批量 encode)
  → db.bulk_upsert (按来源 URL + 答案锚点去重入库)
```

* **两层爬取**：种子页和跟进的详情页都会切题；种子页上只有题干的题，尝试从详情页匹配并补齐答案，详情页不再递归跟进。详情页请求通过 `asyncio.Semaphore(QUIZ_CRAWLER_CONCURRENCY)` 控并发，同次任务共享抓取缓存；单页失败记录到报告中。
* **状态机切片**：H1 = 大主题、H2 = 中主题，H2 内按“每个 H3 一题”或带题目特征的列表项切分；噪声章节（写在最后 / 参考资料 / 延伸阅读等）丢弃。没有 H3 或有效题目列表的章节直接跳过，已移除“整节作为一道题”的兜底。补不到答案时，将原因和原文链接写入答案占位文案。
* **embedding 文本** = `大主题｜中主题｜题干`（不含答案）—— 大/中标题作分类约束让同主题向量聚拢，题干含考点关键词作语义匹配依据。
* **去重 + 索引**：爬取后先按归一化题干跨页去重，优先保留有答案且内容更长的版本。入库按来源 URL + 答案锚点的 SHA1 值去重，已有记录在文本内容变化时更新；`embedding` 列类型为 `vector(N)`，建 HNSW 余弦索引及主题索引。
* **三模式检索**：随机 `ORDER BY RANDOM()`；按主题 `(big_topic, mid_topic)` 精确 WHERE + `ordinal` 排序保原题序；自定义主题把输入文本用同一 `Embedder.embed_query` 做向量，pgvector `<=>` 余弦距离排序 + 阈值过滤取 topK。出题接口一律**不带答案**，点"显示答案"时前端再单独请求。
* **embedding 后端**：使用 `sentence_transformers` 在本地运行 Sentence-BERT，无需 API Key，启动时后台 daemon 线程预热。向量维度由 `QUIZ_EMBEDDING_DIMENSIONS` 显式声明，校验失败直接抛错（不静默 fallback）。

### 3. AI 问答助手：知识库语义检索 + LLM 综合（`ai_interviewer/app.py` + `quiz/retriever.py`）

```
用户问题
  → retriever.search_for_qa (问题文本 → embedding → pgvector <=> 召回 topK + 阈值)
  → _build_qa_context (拼成"【片段 N】主题/问题/答案"喂 LLM)
  → _qa_llm_stream (openai stream=True 逐 delta 产出)
  → SSE 推 status → answer 逐 delta → references → done
```

* **检索策略**：QA 检索只用问题原文做 embedding，按余弦相似度阈值和 topK 召回，同时取出 `answer_md` 拼上下文。上图是前端使用的 `/api/qa/ask-stream` 流程；非流式 `/api/qa/ask` 另有 `_filter_qa_items`，只保留与最高相似度分差 ≤0.05 的结果，最多 3 条，流式接口未调用该过滤函数。
* **LLM Prompt**：System Prompt 模板要求"优先采用你的专业知识，拿到知识库的主题不一定准确，符合思路就参考，不符合就不参考"+ 知识库参考片段；明确禁止 LLM 在回答末尾自列引用（参考例题由系统在回答下方单独展示）。
* **"试试"模式**：`try_question={mid_topic, question}` 时跳过向量化与常规检索，走 `pick_exact_for_qa` 等值精准命中单题；LLM user 消息同时带上中标题与问题；命中失败返回空上下文 + 空引用，LLM 仍可凭自身知识作答。
* **流式编排**：SSE generator 在线程池逐块拉同步生成器（`asyncio.to_thread(next, gen, None)`），首个 answer chunk 前先推一个空 status 清空前端进度条；同步生成器 + 异步推送的混合模型让真 LLM delta 流式可见，不靠"假流式"分块返回。
* **共用题库与模型**：问答和刷题共用 `quiz_questions` 表与 embedding 实现。入库文本为 `大主题｜中主题｜题干`；刷题自定义主题查询把输入填入这三个位置，问答查询则直接使用问题原文。

***

## 二、项目功能

### 1. AI 模拟面试

| 功能 | 说明 |
| --- | --- |
| 简历解析 | 上传 PDF 或粘贴文本，提取姓名 / 技能列表 / 项目经历段落；PDF 按文件 SHA256 缓存，重复上传复用解析结果 |
| JD 驱动的提问规划 | 预置腾讯校招岗位或粘贴自定义 JD，LLM 输出 `ordered_topics + skill_importance(required/preferred/bonus) + suggested_max_questions + topics_per_skill`，两类 JD 都缓存规划结果 |
| 多轮对话面试 | 出题 → 接收回答 → 按话题配额与题数追问 / 换话题 / 结束，问题通过 SSE 流式推送 |
| 会话内上下文 | 维护本轮对话历史 + 面试计划 + 当前话题进度，出题时传入历史消息；服务重启后会话不保留 |
| 面试结束与重练 | 答完最后一道题后标记结束，前端展示“再来一次”入口；回答质量由使用者自行复盘 |
| 自定义面试指令 | "再来一次"时填写定向调整需求，注入面试官 System Prompt |

### 2. 刷题助手

| 功能 | 说明 |
| --- | --- |
| 题库构建 | 通过命令行或入库 API，从 `ai_interviewer/爬虫.txt` 的种子 URL 出发做两层爬取，完成 HTML→Markdown→切题→向量化→入库 |
| 三种出题模式 | 随机（PG `ORDER BY RANDOM()`）/ 按"大标题 + 中标题"精准 WHERE 过滤 / 输入一句话自定义主题（pgvector 余弦相似度 topK + 阈值） |
| 答案按需展开 | 出题卡片默认只给题干 + 元信息，点"显示答案"时前端再单独请求完整 Markdown（含链接、代码、表格） |
| 原题跳转 | 每道题带 `原文URL#锚点`，可直接跳到来源页面出题位置 |
| 无答案说明 | 抓不到答案的题保留占位内容，区分“此题无答案”和“暂无答案”，附原因与原文链接 |
| 主题树浏览 | `大标题 → 中标题 → 题目数` 的树状接口，供前端下拉选择 |
| 题库统计 | 总题数 / 大标题数 / 中标题数 / 来源页数 |

### 3. AI 问答助手

| 功能 | 说明 |
| --- | --- |
| 语义检索 + LLM 综合 | 用问题向量做 pgvector 余弦相似度召回 topK，再把参考片段拼进 LLM System Prompt 由模型综合作答 |
| 参考例题展示 | 回答下方单独列出参考例题（含 `source_url_with_anchor`），不在回答正文里堆引用 |
| SSE 流式回答 | 先推 status（加载模型 / 检索 / 生成中）→ 逐 delta 推 answer → 最后推 references + done |
| 召回控制 | 按 topK 和最低余弦相似度筛选参考题；当前流式接口不做额外的分差过滤 |
| "试试"模式 | 指定 `(mid_topic, question)` 时跳过向量化与语义检索，只参考等值命中的那道题作答，命中失败 LLM 仍可凭自身知识作答 |
| 模型预热 | FastAPI lifespan 启动时后台 daemon 线程加载本地 sentence-transformers 模型，减少首次检索等待；预热失败会记录警告 |

***

## 三、部署与使用

### 环境要求

* Python 3.10+
* PostgreSQL 14+，安装 **pgvector 扩展**
* 一个 OpenAI 兼容的 LLM API（面试 Agent 和问答助手用；刷题模块用本地 sentence-transformers embedding 模型，不需要 Key）
* PDF 简历解析需要 Java 运行环境供 Tika 使用；扫描件 OCR 还需 Tesseract（含中英文语言包）和 Poppler。也可以直接粘贴简历文本。

### 1. 安装依赖

```bash
cd Interview_Helper
pip install -e .
# 刷题模块额外安装本地 embedding 依赖（CPU 版 PyTorch + sentence-transformers）：
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install sentence-transformers
```

### 2. 配置 `.env`

在项目根目录创建 `.env`，配置 PostgreSQL 和本地 embedding（字段名见 `ai_interviewer/config.py`）。面试和问答的 LLM API Key / Base URL / 模型名在前端“模型设置”中填写，并随请求传给后端；页面不会自动读取 `.env` 中的 LLM 配置。

```ini
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
| `QUIZ_TOP_K` | 3 | 自定义主题 / 问答助手召回题数，请求中传 count / top_k 可覆盖 |
| `QUIZ_MIN_SIMILARITY` | 0.3 | 语义检索余弦相似度阈值 |
| `QUIZ_CRAWLER_CONCURRENCY` | 5 | 爬虫并发数 |
| `QUIZ_CRAWLER_TIMEOUT` | 30.0 | 单页请求超时秒数 |

面试总题数来自 JD 规划结果，LLM 新生成的建议值会限制在 8–25 题。

当前 embedding 加载代码会启用 Hugging Face 离线模式，需要先单独下载并缓存与 `.env` 一致的模型。以上面的多语言模型为例，在独立 Python 进程中执行：

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"
```

### 3. 初始化刷题知识库

```bash
python -m ai_interviewer.quiz.ingest
```

执行过程包括读取种子 URL、两层爬取、切题、建表 / 建 HNSW 索引、向量化和入库。可重复执行，已有记录按去重键处理；模型需事先缓存，首次全量爬取和向量化耗时较长。

### 4. 启动应用

```bash
# 方式一：Windows 双击（尝试启动本地 PostgreSQL + 启动 Web 应用 + 打开浏览器）
start_app.bat

# 方式二：命令行
python -m ai_interviewer.app
```

启动后访问 <http://localhost:8000>（FastAPI 同时在 `/docs` 提供 OpenAPI 交互文档）。

`start_app.bat` 中的 `PG_HOME`、`PG_DATA` 当前写的是本机路径，使用前需按自己的 PostgreSQL 安装目录和数据目录调整，或自行启动数据库。

### 5. 使用步骤

**AI 模拟面试：**

1. 顶部切换到"面试助手"页，在"模型设置"填写 `API Key / Base URL / 模型名`；
2. 上传简历 PDF（或粘贴文本）→ 自动解析候选人画像，已上传过的 PDF 直接从缓存列表选用；
3. 选择预置岗位（`/api/positions`）或粘贴自定义 JD，确认并按需编辑简历项目段落；
4. 点击“开始面试”→ 后端读取缓存或调用 LLM 生成话题规划，再给出第一题（SSE 流式）；
5. 在对话框逐题作答，Agent 按配额 + 题数自然追问或切换话题；
6. 答完最后一道题后结束，自行复盘本轮回答；可点“再来一次”填定向调整需求重开。

**刷题助手：**

1. 顶部切换到"刷题助手"页，查看题库统计和主题树；
2. 三种出题方式任选：随机 / 选"大标题(+中标题)" / 输入一句话自定义主题；
3. 点"显示答案"展开完整解析，或点"查看原题"跳转到来源页面锚点位置；
4. 未获取到答案的题会显示“此题无答案”或“暂无答案” + 原因 + 原文链接。

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
| GET | `/api/resumes` / `/api/resumes/{hash}` | 已缓存简历列表 / 读取一条 |
| DELETE | `/api/resumes/{hash}` | 删除一条缓存简历 |
| POST | `/api/start-interview` | 创建面试会话（简历 + JD + 岗位） |
| POST | `/api/first-question-stream` | 获取第一题（SSE 真流式） |
| POST | `/api/chat` / `/api/chat-stream` | 提交回答（后者 SSE 流式） |
| GET | `/api/evaluation` | 兼容保留：标记会话结束并返回题数；不生成评分或评价报告，当前前端不调用 |
| GET | `/api/agent-state` | 查看 LangGraph Agent 内部状态（计划 + 进度） |
| GET | `/api/quiz/stats` / `/api/quiz/topics` | 题库统计 / 主题树 |
| POST | `/api/quiz/ingest` | 触发一次爬取与增量入库 |
| POST | `/api/quiz/question` | 出题（`mode=random/by_topic/custom`） |
| GET | `/api/quiz/question/{id}/answer` | 查看某题答案 |
| POST | `/api/qa/ask` / `/api/qa/ask-stream` | AI 问答助手（后者 SSE 流式，支持 `try_question` 试试模式） |
| GET | `/api/health` | 健康检查 |
