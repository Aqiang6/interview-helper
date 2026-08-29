# Interview\_Helper — AI 模拟面试 + 刷题助手

一个本地运行的面试准备工具，包含两个模块：

* **AI 模拟面试**：上传简历 + 粘贴 JD，由 LangGraph Agent 扮演面试官进行多轮对话式提问（会追问、会换话题），结束后输出评估。

* **刷题助手**：自动爬取面试题网站，切题入库到 PostgreSQL + pgvector，支持随机 / 按主题 / 自定义主题（向量语义检索）三种出题方式，答案可展开、可跳转原文。

前端为单页应用（`frontend/index.html`），后端 FastAPI（`ai_interviewer/app.py`），接口走 HTTP/SSE。

***

## 一、项目功能

### 1. AI 模拟面试

| 功能         | 说明                                         |
| ---------- | ------------------------------------------ |
| 简历解析       | 上传 PDF 或粘贴文本，提取候选人姓名 / 技能列表 / 项目经历         |
| JD 驱动的提问规划 | 粘贴目标岗位 JD，自动调整话题优先级、每个话题的提问配额、最大题数         |
| 多轮对话面试     | 出题 → 听回答 → 定性评估 → 追问或换话题，全程流式输出（SSE）       |
| 长期记忆       | Agent 全程维护候选人画像（掌握度、已考话题、整体印象），后续提问参考前面的回答 |
| 面试评估       | 达到题数上限后结束会话，可拉取整场面试的评估结果                   |

### 2. 刷题助手

| 功能     | 说明                                                    |
| ------ | ----------------------------------------------------- |
| 一键构建题库 | 从 `ai_interviewer/爬虫.txt` 的种子 URL 出发做两层爬取，自动切题、向量化、入库 |
| 三种出题模式 | 随机出题 / 按"大标题 + 中标题"精准出题 / 输入一句话自定义主题（向量检索近似题）         |
| 答案按需展开 | 出题卡片默认只给题干，点"显示答案"才返回完整 Markdown（含链接、代码、表格）           |
| 原题跳转   | 每道题都带 `原文URL#锚点`，可直接跳到来源页面出题位置                        |
| 无答案归因  | 抓不到答案的题不静默丢弃：显示"此题无答案"+ 原因 + 原文链接，方便人工核查              |
| 主题树浏览  | `大标题 → 中标题 → 题目数` 的树状接口，供前端下拉选择                       |

***

## 二、部署与使用

### 环境要求

* Python 3.10+

* PostgreSQL 14+，安装 **pgvector 扩展**

* 一个 OpenAI 兼容的 LLM API（面试 Agent 用；刷题模块默认用本地 embedding 模型，不需要 Key）

### 1. 安装依赖

```bash
cd Interview_Helper
pip install -e .
```

### 2. 配置 `.env`

在项目根目录创建 `.env`（字段名见 `config.py`），必配项：

```ini
# 面试 Agent 的 LLM（OpenAI 兼容接口均可）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 刷题知识库：PostgreSQL 连接串（需先装好 pgvector 扩展）
POSTGRES_DSN=postgresql+psycopg://postgres:postgres@localhost:5432/interview_helper

# 刷题 embedding：本地模型，免 Key
QUIZ_EMBEDDING_BACKEND=sentence_transformers
QUIZ_SENTENCE_TRANSFORMER_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
QUIZ_EMBEDDING_DIMENSIONS=384
```

可选调优项（均有默认值）：`AGENT_MAX_QUESTIONS`（单场面试最大题数，默认 15）、`QUIZ_TOP_K`（自定义主题返回题数，默认 10）、`QUIZ_MIN_SIMILARITY`（语义检索阈值，默认 0.3）、`QUIZ_CRAWLER_CONCURRENCY`（爬虫并发，默认 5）。

### 3. 初始化知识库

```bash
python -m ai_interviewer.quiz.ingest
```

自动完成：建表/建索引 → 读取种子 URL → 两层爬取 → 切题 → 向量化 → 入库。幂等可重复执行，首次运行需下载 embedding 模型并爬全量页面，耗时较长。

### 4. 启动应用

```bash
# 方式一：Windows 双击
start_app.bat

# 方式二：命令行
python -m ai_interviewer.app
```

启动后访问 <http://localhost:8000>（FastAPI 同时在 `/docs` 提供 OpenAPI 交互文档）。

### 5. 使用步骤

**AI 模拟面试：**

1. 首页上传简历 PDF（或粘贴文本）→ 自动解析候选人画像；
2. 粘贴目标岗位 JD → 后端生成话题规划；
3. 点击"开始面试"→ Agent 给出第一题；
4. 在对话框逐题作答，Agent 根据回答追问或切换话题（SSE 流式显示）；
5. 达到题数上限后自动结束，可查看整场评估。

**刷题助手：**

1. 打开刷题页，查看题库统计和主题树；
2. 三种出题方式任选：随机 / 选"大标题(+中标题)" / 输入一句话自定义主题；
3. 点"显示答案"展开完整解析，或点"查看原题"跳转到来源页面锚点位置；
4. 无答案的题会显示"此题无答案"+ 原因 + 原文链接。

### 6. 常用 API 一览

| 方法   | 路径                                     | 用途                                |
| ---- | -------------------------------------- | --------------------------------- |
| POST | `/api/parse-resume` / `/api/parse-pdf` | 解析简历文本 / PDF                      |
| POST | `/api/start-interview`                 | 创建面试会话（简历 + JD + 岗位）              |
| POST | `/api/chat` / `/api/chat-stream`       | 提交回答（后者 SSE 流式）                   |
| GET  | `/api/evaluation`                      | 获取面试评估                            |
| GET  | `/api/quiz/stats` / `/api/quiz/topics` | 题库统计 / 主题树                        |
| POST | `/api/quiz/ingest`                     | 触发一次知识库重建入库                       |
| POST | `/api/quiz/question`                   | 出题（`mode=random/by_topic/custom`） |
| GET  | `/api/quiz/question/{id}/answer`       | 查看某题答案                            |

***

## 三、底层实现简述

### 1. 面试 Agent：LangGraph 状态机（`agent/` 目录）

7 个节点组成一张状态图，共享状态 `InterviewState` 里放着对话历史、候选人画像（长期记忆）和面试计划（话题顺序、每话题配额）：

```
plan_interview（JD+简历规划话题与配额）
  ├─ 用户还没回答 → generate_question（出第一题）
  └─ 用户已回答   → evaluate_response（定性分析本轮回答，更新画像）
                     → decide_next（纯规则路由）：
                         配额未满 → 追问；话题问完 → 换话题；总题数到 → 结束
```

每轮出题时，LLM 的上下文由 System Prompt 拼装：面试官人设 + 当前话题/配额进度 + 上一轮回答的分析 + 简历项目经历 + 完整对话历史。会话按 `session_id` 存在内存字典里，流式接口用 SSE 推 token。

### 2. 刷题知识库：爬取 → 切片 → 向量化（`quiz/` 目录）

* **爬取**：从种子 URL 出发二层爬取（种子页当目录，详情页才是题目），正文转 Markdown；

* **切片**：纯状态机不用 LLM——H1 = 大主题、H2 = 中主题，H2 内按"每个 H3 一题"或"`- 题干?` 列表项一题"切分，答案取到下一个同级标题为止；噪声章节（参考资料等）丢弃，只有题干的题跟随详情链接回填答案，补不到则记原因；

* **入库**：embedding 文本 = `大主题｜中主题｜题干`（不含答案），以 `sha1(来源URL+锚点)` 去重 upsert，向量存 PostgreSQL pgvector 并建 HNSW 索引，默认本地模型 `paraphrase-multilingual-MiniLM-L12-v2`（384 维）。

### 3. 检索上下文

* **随机**：SQL `ORDER BY RANDOM()`；**按主题**：大/中标题精确 WHERE 过滤；**自定义主题**：把输入文本用与入库完全相同的拼接方式做 embedding，pgvector 余弦相似度排序 + 阈值过滤取 topK；

* 出题接口一律**不带答案**，点"显示答案"时前端再单独请求完整 Markdown；

* 面试侧上下文来自简历项目提取（规则化裁剪，不调 LLM）、JD 规划产物和对话历史。

