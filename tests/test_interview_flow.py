import asyncio
import json
import threading
import unittest
from unittest.mock import patch

import httpx

from ai_interviewer import app as web
from ai_interviewer.agent import InterviewGraphAgent
from ai_interviewer.agent import nodes
from ai_interviewer.agent.project_question import ProjectQuestion
from ai_interviewer.jd_engine.jd_loader import list_positions
from ai_interviewer.resume_parser import extract_project_details


PROJECTS = "项目经历\n知识库系统：实现文档检索。\n订单服务：实现订单处理。"
NAMES = ["知识库系统", "订单服务"]
JD = "  岗位要求\n熟悉检索系统。\n\n职责：负责订单服务。  \n"


def fake_llm(state, messages, *, stream_callback=None, **kwargs):
    text = f"问题{state['question_count'] + 1}？"
    if not state["projects"]:
        text = f"<projects>{json.dumps(NAMES, ensure_ascii=False)}</projects>\n" + text
    if stream_callback is not None:
        for char in text:
            stream_callback(char)
    return text


class ProjectStreamTests(unittest.TestCase):
    def test_arbitrary_chunk_boundaries_hide_metadata(self):
        raw = '<projects>["知识库系统", "订单服务"]</projects>\n为什么选择这个方案？'
        for split in range(len(raw) + 1):
            chunks = []
            response = ProjectQuestion(chunks.append)
            response.feed(raw[:split])
            response.feed(raw[split:])
            response.finish()
            self.assertEqual(response.projects, NAMES)
            self.assertEqual("".join(chunks), "为什么选择这个方案？")

    def test_incomplete_or_empty_projects_do_not_emit_a_question(self):
        for raw in ['<projects>[]</projects>问题', '<projects>["项目"]', '普通问题']:
            chunks = []
            response = ProjectQuestion(chunks.append)
            with self.assertRaises(ValueError):
                response.feed(raw)
                response.finish()
            self.assertEqual(chunks, [])

    def test_project_text_is_not_truncated_before_later_projects(self):
        text = "项目经历\n项目一：" + "负责系统实现。" * 400 + "\n项目二：订单服务\n教育背景\n本科"
        extracted = extract_project_details(text)
        self.assertIn("项目二：订单服务", extracted)
        self.assertNotIn("本科", extracted)


class InterviewFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_projects_have_exactly_nine_questions_each(self):
        agent = InterviewGraphAgent(api_key="test")
        agent.create_session("test", PROJECTS, jd_raw_text=JD, resume_projects=PROJECTS)
        expected = [
            (project, main, turn)
            for project in NAMES for main in (1, 2, 3) for turn in (1, 2, 3)
        ]
        with patch.object(nodes, "_call_llm", side_effect=fake_llm) as llm:
            for index, (project, main, turn) in enumerate(expected):
                if index == 0:
                    answer = await agent.get_first_question("test")
                else:
                    answer = await agent.answer_and_ask("test", f"回答{index}")
                self.assertEqual(answer, f"问题{index + 1}？")
                state = agent.get_session("test")
                self.assertEqual(state["question_count"], index + 1)
                self.assertFalse(state["is_finished"])
                prompt = llm.call_args.args[1][0]["content"]
                self.assertIn(JD, prompt)
                self.assertIn(PROJECTS, prompt)
                if index:
                    self.assertIn(project, prompt)
                    self.assertIn(f"第 {main}/3 个主问题", prompt)
                    self.assertIn(f"第 {turn}/3 轮", prompt)
                root_number = index - (turn - 1) + 1
                self.assertEqual(state["main_question"], f"问题{root_number}？")
            finished = []
            await agent.answer_and_ask("test", "最后一题的回答", finished.append)
            self.assertEqual(finished, [nodes.FINISH_MESSAGE])
            self.assertTrue(agent.get_session("test")["is_finished"])
            self.assertEqual(llm.call_count, 18)
            await agent.answer_and_ask("test", "结束后继续发送")
            self.assertEqual(llm.call_count, 18)

    async def test_failed_question_does_not_advance_or_duplicate_answer(self):
        agent = InterviewGraphAgent(api_key="test")
        agent.create_session("test", PROJECTS, resume_projects=PROJECTS)
        with patch.object(nodes, "_call_llm", side_effect=fake_llm):
            await agent.get_first_question("test")
        before = list(agent.get_session("test")["messages"])
        with patch.object(nodes, "_call_llm", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                await agent.answer_and_ask("test", "尚未提交的回答")
        self.assertEqual(agent.get_session("test")["question_count"], 1)
        self.assertEqual(agent.get_session("test")["messages"], before)
        with patch.object(nodes, "_call_llm", side_effect=fake_llm):
            self.assertEqual(await agent.answer_and_ask("test", "重试的回答"), "问题2？")


class InterviewApiTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=web.app), base_url="http://test")
        self.addAsyncCleanup(self.client.aclose)
        self.ids = []

    async def asyncTearDown(self):
        for sid in self.ids:
            web._sessions.pop(sid, None)

    async def start(self, **kwargs):
        sid = f"regression-{id(self)}-{len(self.ids)}"
        self.ids.append(sid)
        response = await self.client.post("/api/start-interview", json={
            "session_id": sid, "api_key": "test", "resume_text": PROJECTS,
            "resume_projects": PROJECTS, **kwargs,
        })
        self.assertEqual(response.status_code, 200, response.text)
        return sid, response

    async def test_preset_and_custom_jds_start_without_any_model_call(self):
        with patch.object(nodes, "OpenAI", side_effect=AssertionError("unexpected model call")):
            for position in list_positions():
                sid, response = await self.start(target_position=position.id)
                self.assertEqual(response.json()["questions_per_project"], 9)
                self.assertEqual(web._sessions[sid].get_session(sid)["jd_raw_text"], position.raw_text)
            sid, _ = await self.start(custom_jd_name="自定义岗位", custom_jd_text=JD)
            self.assertEqual(web._sessions[sid].get_session(sid)["jd_raw_text"], JD)

    async def test_custom_jd_and_projects_reach_sse_question_prompt_verbatim(self):
        sid, _ = await self.start(custom_jd_name="自定义岗位", custom_jd_text=JD)
        with patch.object(nodes, "_call_llm", side_effect=fake_llm) as llm:
            response = await self.client.post("/api/first-question-stream", json={"session_id": sid})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertEqual("".join(e["content"] for e in events if e["type"] == "answer"), "问题1？")
            self.assertEqual(events[-1], {"type": "done", "finished": False})
            self.assertNotIn("<projects>", response.text)
            prompt = llm.call_args.args[1][0]["content"]
            self.assertIn(JD, prompt)
            self.assertIn(PROJECTS, prompt)
            self.assertEqual(llm.call_count, 1)
            response = await self.client.post("/api/chat-stream", json={"session_id": sid, "answer": "用了混合检索"})
            self.assertEqual(response.status_code, 200)
            self.assertIn("第 2/3 轮", llm.call_args.args[1][0]["content"])

    async def test_blocked_quiz_query_does_not_block_health_or_start(self):
        entered, release = threading.Event(), threading.Event()

        def slow_stats():
            entered.set()
            release.wait(3)
            return {"total_questions": 1}

        with patch.object(web, "_quiz_available"), patch("ai_interviewer.quiz.retriever.stats", side_effect=slow_stats):
            query = asyncio.create_task(self.client.get("/api/quiz/stats?_=0"))
            try:
                self.assertTrue(await asyncio.to_thread(entered.wait, 1))
                health = await asyncio.wait_for(self.client.get("/api/health"), 0.75)
                self.assertEqual(health.status_code, 200)
                await asyncio.wait_for(self.start(custom_jd_name="自定义岗位", custom_jd_text=JD), 0.75)
                self.assertFalse(query.done())
            finally:
                release.set()
                await query


if __name__ == "__main__":
    unittest.main()
