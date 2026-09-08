"""AI 模拟面试 Web 应用入口

启动命令：
    python -m ai_interviewer.app

访问地址：
    http://localhost:8000/
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from ai_interviewer.agent import InterviewGraphAgent
from ai_interviewer.agent.topic_prioritizer import prioritize_by_jd
from ai_interviewer.jd_engine.jd_loader import create_custom_position, get_position, list_positions
from ai_interviewer.resume_parser import parse_pdf, parse_text

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════
#  全局面试 Agent 实例（API Key 运行时从请求中传入）
# ═══════════════════════════════════════════

_agent = InterviewGraphAgent(api_key="")
_sessions: dict[str, InterviewGraphAgent] = {}

# ══════════════════════════════════════════
#  FastAPI 应用
# ═══════════════════════════════════════════

app = FastAPI(title="AI 模拟面试", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def _warmup_embedding_model_on_startup():
    """后台线程预热 sentence-transformers 模型，避免用户第一次问问题时卡在 Loading weights
    进度条上（第一次 ~80MB 权重加载会显示进度条，看起来像"向量化慢"）。
    仅 QUIZ_EMBEDDING_BACKEND=sentence_transformers 时执行；OpenAI 后端无需预热。
    预热在 daemon 线程里跑，不阻塞 FastAPI 启动；失败只打 warning，不影响服务可用性。
    """
    import threading
    try:
        from ai_interviewer.config import get_settings
        s = get_settings()
        if (s.quiz_embedding_backend or "sentence_transformers").lower() != "sentence_transformers":
            return
        def _warm():
            try:
                from ai_interviewer.quiz.ingest import warmup_embedding_model
                warmup_embedding_model()
            except Exception as e:
                logger.warning("[startup] embedding 模型预热失败: %s", e)
        threading.Thread(target=_warm, daemon=True, name="embedding-warmup").start()
    except Exception as e:
        logger.warning("[startup] embedding 预热钩子异常: %s", e)


# ═══════════════════════════════════════════
#  工具函数
# ═══════════════════════════════════════════

def _get_agent(session_id: str) -> InterviewGraphAgent:
    """获取或创建绑定到 session 的 Agent"""
    if session_id not in _sessions:
        _sessions[session_id] = InterviewGraphAgent(api_key="")
    return _sessions[session_id]


# ═══════════════════════════════════════════
#  面试 API
# ═══════════════════════════════════════════

@app.get("/")
async def index():
    """返回前端页面"""
    frontend_path = Path(__file__).resolve().parent.parent / "frontend" / "index.html"
    if frontend_path.exists():
        return FileResponse(str(frontend_path))
    return {"message": "AI 模拟面试服务运行中"}


@app.get("/api/positions")
async def api_list_positions():
    """返回前端可选的岗位列表（腾讯 JD）

    示例返回：
    [{
      "id": "agent_dev_engineer",
      "name": "Agent 开发工程师",
      "group": "TEG AI 平台部",
      "location": "深圳/北京",
      "summary": "负责 AI Agent 框架与推理服务开发..."
    }]
    """
    try:
        positions = list_positions()
    except Exception as e:
        logger.exception("加载岗位列表失败")
        raise HTTPException(status_code=500, detail=f"加载岗位列表失败: {str(e)}")

    def _desc_text(p) -> str:
        if p.description:
            return "；".join(p.description)
        if p.requirements:
            return "；".join(p.requirements)
        return p.level

    return [
        {
            "id": p.id,
            "name": p.name,
            "group": p.group,
            "location": p.location,
            "summary": _desc_text(p)[:180],
            "requirements_count": len(p.requirements),
            "bonus_count": len(p.bonuses),
        }
        for p in positions
    ]


@app.post("/api/parse-resume")
async def api_parse_resume(request: Request):
    """解析简历（文本格式）"""
    data = await request.json()
    text = data.get("text", "")
    rd = parse_text(text)
    from ai_interviewer.agent.nodes import _extract_project_details
    llm_projects = _extract_project_details(rd.raw_text)
    return {
        "raw_text": rd.raw_text,
        "name": rd.name,
        "skills": rd.skills,
        "projects": rd.projects,
        "experience_years": rd.experience_years,
        "education": rd.education,
        "summary": rd.summary,
        "llm_projects": llm_projects,
    }


@app.post("/api/parse-pdf")
async def api_parse_pdf(request: Request):
    """解析 PDF 简历"""
    data = await request.json()
    pdf_base64 = data.get("pdf_data", "")
    if not pdf_base64:
        raise HTTPException(status_code=400, detail="PDF 数据不能为空")
    
    try:
        import base64
        pdf_bytes = base64.b64decode(pdf_base64)
        rd = parse_pdf(pdf_bytes)
        logger.info(f"PDF 解析结果 - 姓名: {rd.name}, 学历: {rd.education}, 年限: {rd.experience_years}, 技能数: {len(rd.skills)}")
        raw_sample = rd.raw_text[:500]
        logger.info(f"PDF 原始文本长度: {len(rd.raw_text)}")
        logger.info(f"PDF 原始文本repr(前200字符): {repr(raw_sample[:200])}")
        from ai_interviewer.agent.nodes import _extract_project_details
        llm_projects = _extract_project_details(rd.raw_text)
        return {
            "raw_text": rd.raw_text,
            "name": rd.name,
            "skills": rd.skills,
            "projects": rd.projects,
            "experience_years": rd.experience_years,
            "education": rd.education,
            "summary": rd.summary,
            "llm_projects": llm_projects,
        }
    except Exception as e:
        logger.error("PDF 解析失败: %s", e)
        raise HTTPException(status_code=500, detail=f"PDF 解析失败: {str(e)}")


@app.get("/api/resumes")
async def api_list_saved_resumes():
    """返回已保存的简历缓存列表（前端「已保存简历」界面直接选用，免重复解析）"""
    from ai_interviewer.resume_cache import list_cached_resumes
    try:
        return {"resumes": list_cached_resumes()}
    except Exception as e:
        logger.exception("加载已保存简历列表失败")
        raise HTTPException(status_code=500, detail=f"加载已保存简历失败: {str(e)}")


@app.get("/api/resumes/{hash}")
async def api_get_saved_resume(hash: str):
    """获取单条已保存简历的完整数据（含 raw_text），供前端选中后直接开始面试"""
    from ai_interviewer.agent.nodes import _extract_project_details
    from ai_interviewer.resume_cache import get_cached_by_hash
    cached = get_cached_by_hash(hash)
    if cached is None:
        raise HTTPException(status_code=404, detail="简历不存在或缓存已被清除")
    result = cached.to_dict()
    result["llm_projects"] = _extract_project_details(cached.raw_text)
    return result


@app.delete("/api/resumes/{hash}")
async def api_delete_saved_resume(hash: str):
    """删除一条已保存的简历缓存"""
    from ai_interviewer.resume_cache import delete_cached_by_hash
    deleted = delete_cached_by_hash(hash)
    if not deleted:
        raise HTTPException(status_code=404, detail="简历不存在或已被删除")
    return {"ok": True}


@app.post("/api/start-interview")
async def api_start_interview(request: Request):
    """开始面试（支持前端选择岗位或自定义 JD，后端自动注入）

    参数说明：
    - target_position: 预置岗位 ID（从 /api/positions 获取）
    - custom_jd_name + custom_jd_text: 自定义岗位（用户直接填写 JD 文本）
    - custom_prompt: "再来一次"时的定向调整需求，注入面试官系统提示词
    - resume_text: 简历原文（用于项目追问参考）
    """
    data = await request.json()
    api_key = data.get("api_key", "")
    base_url = data.get("base_url", "https://api.openai.com/v1")
    model = data.get("model", "gpt-4o")
    resume_text = data.get("resume_text", "")
    candidate_name = data.get("candidate_name", "")
    target_position = (data.get("target_position") or "").strip()
    custom_jd_name = (data.get("custom_jd_name") or "").strip()
    custom_jd_text = (data.get("custom_jd_text") or "").strip()
    custom_prompt = (data.get("custom_prompt") or "").strip()
    resume_projects = (data.get("resume_projects") or "").strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="请先配置 API Key")
    if not resume_text:
        raise HTTPException(status_code=400, detail="请先上传简历")

    # ── JD 解析：预置岗位 / 自定义 JD ──
    jd_position_name: str = ""
    jd_raw_text: str = ""
    if custom_jd_name and custom_jd_text:
        # 自定义 JD 优先
        try:
            pos = create_custom_position(custom_jd_name, custom_jd_text)
            jd_position_name = pos.name
            jd_raw_text = pos.raw_text
            target_position = pos.position_id
        except Exception as e:
            logger.exception("自定义 JD 解析失败")
            raise HTTPException(status_code=500, detail=f"自定义 JD 解析失败: {str(e)}")
    elif target_position:
        try:
            pos = get_position(target_position)
            if pos is None:
                raise HTTPException(status_code=400, detail=f"岗位「{target_position}」不存在，请先调用 GET /api/positions 查看可选岗位")
            jd_position_name = pos.name
            jd_raw_text = pos.raw_text
        except HTTPException:
            raise
        except Exception as e:
            logger.exception("JD 解析失败")
            raise HTTPException(status_code=500, detail=f"JD 解析失败: {str(e)}")

    session_id = data.get("session_id") or "session_1"
    agent = InterviewGraphAgent(api_key=api_key, base_url=base_url, model=model)
    _sessions[session_id] = agent

    agent.create_session(
        session_id=session_id,
        resume_text=resume_text,
        candidate_name=candidate_name,
        target_position=target_position,
        jd_position_name=jd_position_name,
        jd_raw_text=jd_raw_text,
        custom_prompt=custom_prompt,
        resume_projects=resume_projects,
    )

    # ── 生成面试计划（在线程池跑同步 openai 调用，避免阻塞 async event loop）──
    try:
        jd_result = await asyncio.to_thread(
            prioritize_by_jd,
            api_key=api_key,
            base_url=base_url,
            model=model,
            jd_position_name=jd_position_name or "通用技术岗",
            jd_raw_text=jd_raw_text,
        )
        sorted_topics = list(jd_result.ordered_topics) or ["技术基础"]
        topic_aliases = {canonical: original for original, canonical in jd_result.ordered_aliases}
        if not topic_aliases and sorted_topics:
            topic_aliases = {t: t for t in sorted_topics}
        topic_question_counter = {t: 0 for t in sorted_topics}
        req_count = sum(1 for v in (jd_result.skill_importance or {}).values() if v == "required")

        interview_plan = {
            "topics": sorted_topics,
            "topic_aliases": topic_aliases,
            "current_topic_index": 0,
            "phase": "deep_dive" if req_count >= 2 else "basics",
            "questions_per_topic": 3,
            "skill_importance": dict(jd_result.skill_importance or {}),
            "topics_per_skill": dict(jd_result.topics_per_skill or {}),
            "topic_question_counter": topic_question_counter,
            "position_id": target_position,
            "position_name": jd_position_name,
            "priority_source": jd_result.source,
        }
        max_questions_final = int(jd_result.suggested_max_questions or 15)
        current_topic_final = sorted_topics[0] if sorted_topics else "技术基础"
        agent._sessions[session_id].update({
            "interview_plan": interview_plan,
            "max_questions": max_questions_final,
            "current_topic": current_topic_final,
        })
        logger.info(
            "[start-interview] 面试计划生成完成(source=%s): position=%s max_q=%d required=%d topics=%s",
            jd_result.source, jd_position_name or "通用", max_questions_final, req_count, sorted_topics[:8]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[start-interview] 面试计划生成失败")
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "session_id": session_id,
        "status": "ready",
        "plan_source": getattr(jd_result, "source", "llm"),
        "position": {
            "id": target_position,
            "name": jd_position_name,
        } if target_position else None,
        "max_questions": max_questions_final,
        "topics_count": len(sorted_topics),
        "required_count": req_count,
    }


@app.post("/api/chat")
async def api_chat(request: Request):
    """面试对话（非流式）"""
    data = await request.json()
    session_id = data.get("session_id", "session_1")
    answer = data.get("answer", "")

    agent = _get_agent(session_id)
    if not agent.get_session(session_id):
        raise HTTPException(status_code=404, detail="面试会话不存在，请先调用 /api/start-interview")

    try:
        response = await agent.answer_and_ask(session_id, answer)
    except Exception as e:
        logger.exception("面试对话失败")
        response = f"（系统提示：生成回复时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"

    session = agent.get_session(session_id)
    return {
        "response": response,
        "finished": session.get("is_finished", False) if session else False,
        "question_count": session.get("question_count", 0) if session else 0,
    }


@app.post("/api/first-question")
async def api_first_question(request: Request):
    """获取第一个面试问题

    面试计划已在 start-interview 同步生成，直接跑 graph 出第一题。
    """
    data = await request.json()
    session_id = data.get("session_id", "session_1")

    agent = _get_agent(session_id)

    try:
        response = await agent.get_first_question(session_id)
    except RuntimeError as e:
        msg = str(e) or ""
        # JD 话题排序失败、LLM 连不上等"启动前置条件不满足"一律 400 抛给前端弹窗
        if "JD 话题排序失败" in msg or "面试无法开始" in msg:
            logger.error("首题启动前置失败: %s", msg)
            raise HTTPException(status_code=400, detail=msg) from e
        # 其他 RuntimeError 走兜底字符串（理论上不会走到）
        logger.exception("获取首题 RuntimeError")
        response = f"（系统提示：生成问题时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"
    except Exception as e:
        logger.exception("获取首题失败")
        response = f"（系统提示：生成问题时出错，请检查 API 配置。错误：{type(e).__name__}: {e}）"

    session = agent.get_session(session_id)
    return {
        "response": response,
        "finished": session.get("is_finished", False) if session else False,
        "question_count": session.get("question_count", 0) if session else 0,
    }


@app.post("/api/first-question-stream")
async def api_first_question_stream(request: Request):
    """获取第一个面试问题（SSE 真流式，逐 delta 推送）

    通过模块级 _stream_cb 回调实现：在调 get_first_question 前设置回调，
    generate_question 节点内部的 _call_llm 会用 stream=True 逐 delta 调回调，
    回调把 delta 放入 queue，SSE generator 从 queue 读取推送。
    """
    import queue as _queue
    from ai_interviewer.agent import nodes as _nodes

    data = await request.json()
    session_id = data.get("session_id", "session_1")

    agent = _get_agent(session_id)

    q: _queue.Queue = _queue.Queue()
    _sentinel = object()

    def _cb(delta: str):
        q.put(delta)

    _nodes._stream_cb = _cb

    async def _stream():
        try:
            # 在线程里跑 get_first_question（同步阻塞），回调逐 delta 入队
            def _run():
                try:
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(agent.get_first_question(session_id))
                    loop.close()
                    return result
                except Exception as e:
                    q.put(e)
                    return None
                finally:
                    q.put(_sentinel)

            import threading
            t = threading.Thread(target=_run, daemon=True)
            t.start()

            while True:
                item = await asyncio.to_thread(q.get)
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    yield f"data: {json.dumps({'type': 'error', 'error': f'生成问题失败: {type(item).__name__}: {item}'}, ensure_ascii=False)}\n\n"
                    return
                yield f"data: {json.dumps({'type': 'answer', 'content': item}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("首题流式生成失败")
            yield f"data: {json.dumps({'type': 'error', 'error': f'生成问题失败: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
            return
        finally:
            _nodes._stream_cb = None
        session = agent.get_session(session_id)
        finished = session.get("is_finished", False) if session else False
        yield f"data: {json.dumps({'type': 'done', 'finished': finished}, ensure_ascii=False)}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


def _stream_response(agent: InterviewGraphAgent, session_id: str, answer: str):
    """流式返回面试官回复（真 LLM 流式，逐 delta 推送）

    普通函数：调用时同步设置 _stream_cb 并返回 _stream() async generator，
    供 StreamingResponse 直接迭代；避免 async def 返回 coroutine 导致 500。
    """
    import queue as _queue
    from ai_interviewer.agent import nodes as _nodes

    q: _queue.Queue = _queue.Queue()
    _sentinel = object()

    def _cb(delta: str):
        q.put(delta)

    _nodes._stream_cb = _cb

    async def _stream():
        try:
            def _run():
                try:
                    loop = asyncio.new_event_loop()
                    result = loop.run_until_complete(agent.answer_and_ask(session_id, answer))
                    loop.close()
                    return result
                except Exception as e:
                    q.put(e)
                    return None
                finally:
                    q.put(_sentinel)

            import threading
            t = threading.Thread(target=_run, daemon=True)
            t.start()

            while True:
                item = await asyncio.to_thread(q.get)
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    yield f"data: {json.dumps({'type': 'error', 'error': f'生成回复失败: {type(item).__name__}: {item}'}, ensure_ascii=False)}\n\n"
                    return
                yield f"data: {json.dumps({'type': 'answer', 'content': item}, ensure_ascii=False)}\n\n"
        except Exception as e:
            logger.exception("流式生成回复失败")
            yield f"data: {json.dumps({'type': 'error', 'error': f'生成回复失败: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
            return
        finally:
            _nodes._stream_cb = None
        session = agent.get_session(session_id)
        finished = session.get("is_finished", False) if session else False
        yield f"data: {json.dumps({'type': 'done', 'finished': finished}, ensure_ascii=False)}\n\n"

    return _stream()


@app.post("/api/chat-stream")
async def api_chat_stream(request: Request):
    """面试对话（SSE 流式）"""
    data = await request.json()
    session_id = data.get("session_id", "session_1")
    answer = data.get("answer", "")

    agent = _get_agent(session_id)
    if not agent.get_session(session_id):
        raise HTTPException(status_code=404, detail="面试会话不存在")

    return StreamingResponse(
        _stream_response(agent, session_id, answer),
        media_type="text/event-stream",
    )


@app.get("/api/evaluation")
async def api_evaluation(session_id: str = "session_1"):
    """获取面试评估"""
    agent = _get_agent(session_id)
    evaluation = await agent.get_evaluation(session_id)
    return evaluation


# ═══════════════════════════════════════════
#  刷题助手 API（PostgreSQL + pgvector 知识库）
# ═══════════════════════════════════════════

def _quiz_available() -> None:
    """检查刷题助手是否已配置（未配置时给出清晰提示，不使用 fallback）。"""
    from ai_interviewer.config import get_settings
    s = get_settings()
    if not s.postgres_dsn:
        raise HTTPException(status_code=503, detail="未配置 POSTGRES_DSN，刷题助手未启用。请在 .env 中完成 PostgreSQL 配置后重试。")
    backend = (getattr(s, "quiz_embedding_backend", "") or "").lower()
    if backend == "openai" and not s.openai_api_key:
        raise HTTPException(status_code=503, detail="未配置 OPENAI_API_KEY，无法使用 embedding/自定义主题检索。请在 .env 中设置后重试。")


@app.get("/api/quiz/stats")
async def api_quiz_stats():
    """题库统计：总题数 / 大标题数 / 中标题数 / 来源页数。"""
    _quiz_available()
    try:
        from ai_interviewer.quiz import retriever
        return retriever.stats()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("查询题库统计失败")
        raise HTTPException(status_code=500, detail=f"查询题库统计失败: {type(e).__name__}: {e}")


@app.get("/api/quiz/topics")
async def api_quiz_topics():
    """返回主题树：大标题 → [中标题+数量]，用于"自选主题"界面。"""
    _quiz_available()
    try:
        from ai_interviewer.quiz import retriever
        nodes = retriever.list_topics()
        return {"topics": [n.as_dict() for n in nodes]}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("加载主题树失败")
        raise HTTPException(status_code=500, detail=f"加载主题树失败: {type(e).__name__}: {e}")


@app.post("/api/quiz/ingest")
async def api_quiz_ingest(request: Request):
    """触发爬取入库流程。

    body 可选字段：
    - urls: list[str]  自定义 URL 列表（为空/不传则使用项目内置 爬虫.txt）
    - ensure_schema: bool = true  是否执行 ensure_schema（建表+vector列+HNSW索引）

    返回：IngestReport.as_dict()，包含 pages 成功/失败明细 与 questions 新增/更新/跳过数量。
    """
    _quiz_available()
    data = await request.json() or {}
    urls = data.get("urls") or None
    do_schema = bool(data.get("ensure_schema", True))
    try:
        from ai_interviewer.quiz import ingest as quiz_ingest
        if do_schema:
            from ai_interviewer.quiz.db import ensure_schema
            ensure_schema()
        report = await quiz_ingest.ingest_from_default_file(urls=urls)
        return report.as_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("题库入库失败")
        raise HTTPException(status_code=500, detail=f"题库入库失败: {type(e).__name__}: {e}")


@app.post("/api/quiz/question")
async def api_quiz_pick(request: Request):
    """刷题出题入口，三模式合一。

    body::

        {
          "mode": "random" | "by_topic" | "custom",   // 必填
          "count": 10,                                 // 可选，默认 1 / by_topic 不传=整章节
          // -- mode=by_topic --
          "big_topic": "大模型基础面试题总结",          // 必填
          "mid_topic": "LLM 运行机制",                 // 可选
          "shuffle": false,                            // 可选，是否打乱原题序
          // -- mode=custom --
          "custom_topic": "我想练JVM内存模型和GC",      // 必填
          "min_similarity": 0.3,                       // 可选，覆盖 .env QUIZ_MIN_SIMILARITY
          // -- 所有模式可选 --
          "exclude_ids": [1,2,3]                       // 已做过题 id 列表，random 模式会排除
        }
    """
    _quiz_available()
    data = await request.json() or {}
    mode = (data.get("mode") or "").strip()
    if mode not in {"random", "by_topic", "custom"}:
        raise HTTPException(status_code=400, detail="mode 必须为 random / by_topic / custom 之一")

    count_raw = data.get("count")
    count = int(count_raw) if count_raw is not None and int(count_raw) > 0 else None

    try:
        from ai_interviewer.quiz import retriever
        if mode == "random":
            items = retriever.pick_random(
                count=(count or 1),
                exclude_ids=data.get("exclude_ids") or None,
            )
        elif mode == "by_topic":
            big = (data.get("big_topic") or "").strip()
            if not big:
                raise HTTPException(status_code=400, detail="by_topic 模式下 big_topic 必填")
            mid = (data.get("mid_topic") or "").strip() or None
            items = retriever.pick_by_topic(
                big_topic=big,
                mid_topic=mid,
                count=count,
                shuffle=bool(data.get("shuffle", False)),
            )
        else:  # custom
            topic = (data.get("custom_topic") or "").strip()
            if not topic:
                raise HTTPException(status_code=400, detail="custom 模式下 custom_topic 必填")
            min_sim_raw = data.get("min_similarity")
            min_sim = float(min_sim_raw) if min_sim_raw is not None else None
            items = retriever.pick_by_custom(
                custom_topic=topic,
                count=count,
                min_similarity=min_sim,
            )
        return {
            "mode": mode,
            "count": len(items),
            "questions": [it.as_dict() for it in items],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("出题失败")
        raise HTTPException(status_code=500, detail=f"出题失败: {type(e).__name__}: {e}")


@app.get("/api/quiz/question/{question_id}/answer")
async def api_quiz_answer(question_id: int):
    """点击"显示答案"：返回某题的完整答案 Markdown（含链接/图片/表格/代码）。"""
    _quiz_available()
    try:
        from ai_interviewer.quiz import retriever
        ans = retriever.get_answer(question_id)
        if ans is None:
            raise HTTPException(status_code=404, detail=f"题目 id={question_id} 不存在")
        return ans
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("取答案失败 id=%s", question_id)
        raise HTTPException(status_code=500, detail=f"取答案失败: {type(e).__name__}: {e}")


# ═══════════════════════════════════════════
#  AI 问答助手 API（基于 PostgreSQL 知识库 + LLM 综合回答）
# ═══════════════════════════════════════════

_QA_SYSTEM_PROMPT = """你是一个 AI 技术问答助手。用户会提出技术问题，你会拿到一组「知识库参考片段」，请综合这些片段与你的专业知识作答。

## 回答要求
1. 优先采用知识库片段中的准确内容，同时可自然融合你的专业知识补充完善。
2. 回答使用 Markdown 格式，善用标题、列表、代码块、表格让结构清晰。
3. 回答要准确、专业、有深度，适合面试准备与技术学习场景。
4. **不要**在回答末尾自己列出参考题目或引用链接——参考例题会由系统在回答下方单独展示。

## 知识库参考片段
{context}
"""


def _filter_qa_items(items):
    """尽量一个提问匹配一题：以最高相似度为锚，只保留与其分差 ≤0.05 的题（SQL 已卡绝对阈值 0.55）。"""
    if not items:
        return items
    top1 = items[0].similarity or 0.0
    return [it for it in items if (it.similarity or 0.0) >= top1 - 0.05][:3]


def _build_qa_context(items) -> tuple[str, list[dict]]:
    """把检索到的题目拼成给 LLM 的上下文文本，同时返回给前端的参考例题列表。"""
    refs: list[dict] = []
    parts: list[str] = []
    for i, it in enumerate(items, 1):
        parts.append(
            f"【片段 {i}】\n"
            f"主题: {it.big_topic} / {it.mid_topic}\n"
            f"问题: {it.question}\n"
            f"答案:\n{it.answer_md}"
        )
        refs.append({
            "id": it.id,
            "big_topic": it.big_topic,
            "mid_topic": it.mid_topic,
            "question": it.question,
            "source_url": it.source_url,
            "source_url_with_anchor": it.source_url_with_anchor,
            "similarity": it.similarity,
        })
    return "\n\n---\n\n".join(parts), refs


def _qa_llm_stream(*, api_key: str, base_url: str, model: str, question: str, context: str):
    """同步流式调用 LLM，逐 delta 产出回答文本。失败直接抛错，不做降级。"""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
    system = _QA_SYSTEM_PROMPT.format(context=context or "（知识库未检索到相关内容）")
    stream = client.chat.completions.create(
        model=model,
        temperature=0.3,
        max_tokens=4096,
        stream=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": question},
        ],
    )
    produced = False
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content or ""
        if delta:
            produced = True
            yield delta
    if not produced:
        raise RuntimeError("问答助手 LLM 返回空内容")


@app.post("/api/qa/ask")
async def api_qa_ask(request: Request):
    """AI 问答助手：基于知识库语义检索 + LLM 综合回答。

    body::

        {
          "question": "Redis 的 RDB 和 AOF 有什么区别？",
          "api_key": "...",
          "base_url": "https://api.openai.com/v1",
          "model": "gpt-4o",
          "top_k": 3,
          "min_similarity": 0.3
        }

    返回::

        {
          "answer": "...(Markdown)...",
          "references": [ {id, big_topic, mid_topic, question, source_url, source_url_with_anchor, similarity} ]
        }
    """
    _quiz_available()
    data = await request.json() or {}
    question = (data.get("question") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    base_url = (data.get("base_url") or "https://api.openai.com/v1").strip()
    model = (data.get("model") or "gpt-4o").strip()
    top_k_raw = data.get("top_k")
    min_sim_raw = data.get("min_similarity")

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not api_key:
        raise HTTPException(status_code=400, detail="请先在「面试助手 → 模型设置」中配置 API Key")

    top_k = int(top_k_raw) if top_k_raw is not None and int(top_k_raw) > 0 else None
    min_sim = float(min_sim_raw) if min_sim_raw is not None else None

    # 1) 语义检索知识库
    try:
        from ai_interviewer.quiz import retriever
        items = retriever.search_for_qa(
            query=question,
            count=top_k,
            min_similarity=min_sim,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("问答助手检索失败")
        raise HTTPException(status_code=500, detail=f"知识库检索失败: {type(e).__name__}: {e}")

    items = _filter_qa_items(items)
    context, refs = _build_qa_context(items)

    # 2) LLM 综合回答（在线程池跑同步调用，避免阻塞 async event loop）
    try:
        gen = _qa_llm_stream(
            api_key=api_key,
            base_url=base_url,
            model=model,
            question=question,
            context=context,
        )
        answer = await asyncio.to_thread(lambda: "".join(gen))
    except Exception as e:
        err_type = type(e).__name__
        logger.exception("问答助手 LLM 调用失败")
        raise HTTPException(status_code=500, detail=f"生成回答失败: [{err_type}] {e}")

    return {"answer": answer, "references": refs}


async def _qa_stream(question: str, api_key: str, base_url: str, model: str, top_k: int, min_sim):
    """SSE 流式返回问答结果：先推 status 进度，LLM 真流式逐 delta 推 answer，流完最后推 references。"""
    from ai_interviewer.quiz import retriever
    from ai_interviewer.quiz.ingest import is_model_loaded

    # 模型未加载时显示"加载模型"文案，避免文案不准——首次调用实际是模型加载（带 Loading
    # weights 进度条），不是向量化本身慢。已加载（含预热完成）则显示"向量化问题"。
    if is_model_loaded():
        yield f"data: {json.dumps({'type': 'status', 'content': '正在向量化问题…'}, ensure_ascii=False)}\n\n"
    else:
        yield f"data: {json.dumps({'type': 'status', 'content': '正在加载 embedding 模型（首次较慢）…'}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0)

    items = retriever.search_for_qa(query=question, count=top_k, min_similarity=min_sim)

    yield f"data: {json.dumps({'type': 'status', 'content': f'已检索到 {len(items)} 条相关内容，正在生成答案…'}, ensure_ascii=False)}\n\n"
    await asyncio.sleep(0)

    context, refs = _build_qa_context(items)

    try:
        gen = _qa_llm_stream(
            api_key=api_key,
            base_url=base_url,
            model=model,
            question=question,
            context=context,
        )
        # 同步生成器放线程池逐块拉取，真实时产出
        first_chunk = True
        while True:
            delta = await asyncio.to_thread(next, gen, None)
            if delta is None:
                break
            if first_chunk:
                # 首个 answer chunk 前推一个清空状态的信号
                yield f"data: {json.dumps({'type': 'status', 'content': ''}, ensure_ascii=False)}\n\n"
                first_chunk = False
            yield f"data: {json.dumps({'type': 'answer', 'content': delta}, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.exception("问答助手流式 LLM 调用失败")
        yield f"data: {json.dumps({'type': 'error', 'error': f'生成回答失败: {type(e).__name__}: {e}'}, ensure_ascii=False)}\n\n"
        return

    # 回答流式输出完毕后，最后带参考例题列表
    yield f"data: {json.dumps({'type': 'references', 'references': refs}, ensure_ascii=False)}\n\n"
    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"


@app.post("/api/qa/ask-stream")
async def api_qa_ask_stream(request: Request):
    """AI 问答助手（SSE 流式）。事件类型：references / answer / error / done。"""
    _quiz_available()
    data = await request.json() or {}
    question = (data.get("question") or "").strip()
    api_key = (data.get("api_key") or "").strip()
    base_url = (data.get("base_url") or "https://api.openai.com/v1").strip()
    model = (data.get("model") or "gpt-4o").strip()
    top_k_raw = data.get("top_k")
    min_sim_raw = data.get("min_similarity")

    if not question:
        raise HTTPException(status_code=400, detail="问题不能为空")
    if not api_key:
        raise HTTPException(status_code=400, detail="请先在「面试助手 → 模型设置」中配置 API Key")

    top_k = int(top_k_raw) if top_k_raw is not None and int(top_k_raw) > 0 else None
    min_sim = float(min_sim_raw) if min_sim_raw is not None else None

    return StreamingResponse(
        _qa_stream(question, api_key, base_url, model, top_k, min_sim),
        media_type="text/event-stream",
    )


@app.get("/api/health")
async def api_health():
    """健康检查"""
    return {"status": "ok"}


@app.get("/api/agent-state")
async def api_agent_state(session_id: str = "session_1"):
    """查看 LangGraph Agent 内部状态（Planning + Memory）"""
    agent = _get_agent(session_id)
    state = agent.get_session(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="会话不存在")
    return {
        "interview_plan": state.get("interview_plan", {}),
        "current_topic": state.get("current_topic", ""),
        "question_count": state.get("question_count", 0),
        "max_questions": state.get("max_questions", 15),
        "is_finished": state.get("is_finished", False),
    }


# ═══════════════════════════════════════════
#  静态文件挂载
# ═══════════════════════════════════════════

frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")


# ═══════════════════════════════════════════
#  启动入口
# ═══════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    host = "0.0.0.0"
    port = 8000
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass
    uvicorn.run(app, host=host, port=port)
