import asyncio
from collections import Counter
import json
import threading
import unittest
from unittest.mock import patch

import httpx

from ai_interviewer import app as web
from ai_interviewer.agent import InterviewGraphAgent
from ai_interviewer.agent import nodes
from ai_interviewer.agent.project_question import parse_projects
from ai_interviewer.jd_engine.jd_loader import list_positions
from ai_interviewer.resume_parser import extract_project_details


PROJECTS = "项目经历\n知识库系统：实现文档检索。\n订单服务：实现订单处理。"
NAMES = ["知识库系统", "订单服务"]
DETAILS = ["知识库系统：实现文档检索。", "订单服务：实现订单处理。"]
JD = "  岗位要求\n熟悉检索系统。\n\n职责：负责订单服务。  \n"
MAAS_BASE_URL = "https://api.modelarts-maas.com/openai/v1"
MAAS_MODEL = "deepseek-v4.1-flash-1017"


def fake_llm(state, messages, *, stream_callback=None, **kwargs):
    if not state["projects"]:
        return json.dumps({"projects": [
            {"name": name, "start": name} for name in NAMES
        ]}, ensure_ascii=False)
    text = f"问题{state['question_count'] + 1}？"
    if stream_callback is not None:
        for char in text:
            stream_callback(char)
    return text


class ProjectBoundaryTests(unittest.TestCase):
    def test_project_details_are_verbatim_and_in_source_order(self):
        raw = fake_llm({"projects": []}, [])
        self.assertEqual(parse_projects(raw, PROJECTS), (NAMES, DETAILS))

    def test_rewritten_out_of_order_or_duplicate_project_boundaries_are_rejected(self):
        for projects in [
            [],
            [{"name": "知识库系统", "start": "虚构项目内容"}],
            [{"name": name, "start": name} for name in NAMES[::-1]],
            [{"name": "重复", "start": name} for name in NAMES],
        ]:
            with self.subTest(projects=projects), self.assertRaises(ValueError):
                parse_projects(json.dumps({"projects": projects}), PROJECTS)

    def test_project_text_is_not_truncated_before_later_projects(self):
        text = "项目经历\n项目一：" + "负责系统实现。" * 400 + "\n项目二：订单服务\n教育背景\n本科"
        extracted = extract_project_details(text)
        self.assertIn("项目二：订单服务", extracted)
        self.assertNotIn("本科", extracted)
        raw = json.dumps({"projects": [
            {"name": name, "start": name} for name in ["项目一", "项目二"]
        ]})
        names, details = parse_projects(raw, extracted)
        self.assertEqual(names, ["项目一", "项目二"])
        self.assertEqual(details, ["项目一：" + "负责系统实现。" * 400, "项目二：订单服务"])


class InterviewFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_projects_have_exactly_nine_questions_each(self):
        agent = InterviewGraphAgent(api_key="test")
        agent.create_session("test", PROJECTS, jd_raw_text=JD, resume_projects=PROJECTS)
        expected = [
            (project, main, turn)
            for project in NAMES for main in (1, 2, 3) for turn in (1, 2, 3)
        ]
        with patch.object(nodes, "_call_llm", side_effect=fake_llm) as llm:
            main_question = ""
            previous_main_questions = []
            for index, (project, main, turn) in enumerate(expected):
                if index == 0:
                    answer = await agent.get_first_question("test")
                else:
                    answer = await agent.answer_and_ask("test", f"回答{index}")
                expected_question = f"问题{index + 1}？"
                if index == 0:
                    expected_question = f"先聊聊「{project}」。{expected_question}"
                elif index == 9:
                    expected_question = f"接下来聊聊「{project}」。{expected_question}"
                self.assertEqual(answer, expected_question)
                self.assertEqual(llm.call_args.kwargs["temperature"], 0.4)
                self.assertIsNone(llm.call_args.kwargs["extra_body"])
                state = agent.get_session("test")
                self.assertEqual(state["question_count"], index + 1)
                self.assertFalse(state["is_finished"])
                messages = llm.call_args.args[1]
                prompt, context = messages[0]["content"], messages[1]["content"]
                self.assertIn(JD, context)
                self.assertIn(project, context)
                self.assertIn(DETAILS[index // 9], context)
                self.assertNotIn(DETAILS[index // 9], prompt)
                self.assertNotIn(DETAILS[1 - index // 9], context)
                record = state["question_history"][-1]
                self.assertEqual((record["project_name"], record["main_question_number"], record["round_number"]), (project, main, turn))
                self.assertEqual(record["question"], expected_question)
                roles = [message["role"] for message in messages]
                self.assertEqual(roles, ["system", "user"] + ["assistant", "user"] * (turn - 1))
                if turn == 1:
                    if main == 1:
                        previous_main_questions.clear()
                        self.assertIn("第一个主问题", context)
                        self.assertNotIn("<previous_main_questions>", context)
                    else:
                        self.assertIn("新的主问题", context)
                        self.assertIn("同一技术的另一个角度", context)
                        self.assertIn("\n".join(f"- {question}" for question in previous_main_questions), context)
                        self.assertNotIn(f"问题{index}？", context)
                        self.assertNotIn(f"回答{index}", context)
                    previous_main_questions.append(expected_question)
                    main_question = expected_question
                else:
                    self.assertIn("上一条回答", prompt)
                    self.assertEqual(messages[2]["content"], main_question)
                    self.assertEqual(messages[-1]["content"], f"回答{index}")
                self.assertEqual(state["main_question"], main_question if turn < 3 else "")
            finished = []
            await agent.answer_and_ask("test", "最后一题的回答", finished.append)
            self.assertEqual(finished, [nodes.FINISH_MESSAGE])
            self.assertTrue(agent.get_session("test")["is_finished"])
            self.assertEqual(llm.call_count, 19)  # 独立项目识别 1 次 + 18 道问题
            await agent.answer_and_ask("test", "结束后继续发送")
            self.assertEqual(llm.call_count, 19)

    async def test_three_projects_use_separate_contexts_and_explicit_graph_nodes(self):
        names = ["12306", "知识库系统", "订单服务"]
        details = ["12306：实现余票查询。", *DETAILS]
        projects = "项目经历\n" + "\n".join(details)
        agent = InterviewGraphAgent(api_key="test", base_url=MAAS_BASE_URL, model=MAAS_MODEL)
        agent.create_session("three", projects, resume_projects=projects)
        question_prompts = []
        visited = []
        original_invoke = agent._invoke

        def record_nodes(state, callback):
            # 实际执行图的路由与节点，而非用期望值替代出题状态。
            visited.append(nodes.route_interview(state))
            return original_invoke(state, callback)

        def scoped_llm(state, messages, *, stream_callback=None, **kwargs):
            if not state["projects"]:
                return json.dumps({"projects": [
                    {"name": name, "start": name} for name in names
                ]}, ensure_ascii=False)
            question_prompts.append(messages)
            project_index = state["project_index"]
            context = messages[1]["content"]
            self.assertIn(details[project_index], context)
            for index, detail in enumerate(details):
                if index != project_index:
                    self.assertNotIn(detail, context)
            for message in messages[1:]:
                for index, name in enumerate(names):
                    if index != project_index:
                        self.assertNotIn(name, message["content"])
            text = f"{names[project_index]}的第{state['main_question_number']}题第{state['round_number']}轮怎么实现？"
            if stream_callback:
                for char in text:
                    stream_callback(char)
            return text

        with patch.object(nodes, "_call_llm", side_effect=scoped_llm) as llm, patch.object(agent, "_invoke", side_effect=record_nodes):
            for index in range(27):
                chunks = []
                if not index:
                    question = await agent.get_first_question("three", chunks.append)
                else:
                    question = await agent.answer_and_ask("three", f"{names[(index - 1) // 9]}的回答", chunks.append)
                self.assertEqual("".join(chunks), question)
                self.assertIn(names[index // 9], question)
                if index % 9 == 0:
                    self.assertEqual([message["role"] for message in question_prompts[-1]], ["system", "user"])
                    self.assertIn("第一个主问题", question_prompts[-1][1]["content"])
                    opening = "先聊聊" if index == 0 else "接下来聊聊"
                    self.assertEqual(chunks[0], f"{opening}「{names[index // 9]}」。")
                if index % 3 == 0:
                    self.assertEqual(llm.call_args.kwargs["extra_body"], {"chat_template_kwargs": {"thinking": False}})
                else:
                    self.assertIsNone(llm.call_args.kwargs["extra_body"])
            self.assertFalse(agent.get_session("three")["is_finished"])
            await agent.answer_and_ask("three", "订单服务的最后回答")
            self.assertEqual(llm.call_count, 28)
        state = agent.get_session("three")
        self.assertTrue(state["is_finished"])
        self.assertEqual(state["question_count"], 27)
        self.assertEqual(Counter(record["project_name"] for record in state["question_history"]), dict.fromkeys(names, 9))
        self.assertEqual(visited, ["identify_projects", *[
            "ask_main_question" if index % 3 == 0 else "ask_follow_up" for index in range(1, 27)
        ], "finish_interview"])

    async def test_concurrent_requests_do_not_generate_from_the_same_cursor(self):
        agent = InterviewGraphAgent(api_key="test")
        agent.create_session("concurrent", PROJECTS, resume_projects=PROJECTS)
        with patch.object(nodes, "_call_llm", side_effect=fake_llm) as llm:
            firsts = await asyncio.gather(*[agent.get_first_question("concurrent") for _ in range(2)])
            self.assertEqual(firsts, ["先聊聊「知识库系统」。问题1？"] * 2)
            self.assertEqual(llm.call_count, 2)
            questions = await asyncio.gather(*[agent.answer_and_ask("concurrent", f"回答{i}") for i in range(2)])
        self.assertEqual(questions, ["问题2？", "问题3？"])
        self.assertEqual(agent.get_session("concurrent")["question_count"], 3)

    async def test_failed_question_does_not_advance_or_duplicate_answer(self):
        agent = InterviewGraphAgent(api_key="test")
        agent.create_session("test", PROJECTS, resume_projects=PROJECTS)
        with patch.object(nodes, "_call_llm", side_effect=fake_llm):
            await agent.get_first_question("test")
        before = list(agent.get_session("test")["messages"])
        progress = nodes.interview_progress(agent.get_session("test"))
        with patch.object(nodes, "_call_llm", side_effect=RuntimeError("offline")):
            with self.assertRaises(RuntimeError):
                await agent.answer_and_ask("test", "尚未提交的回答")
        self.assertEqual(agent.get_session("test")["question_count"], 1)
        self.assertEqual(agent.get_session("test")["messages"], before)
        self.assertEqual(nodes.interview_progress(agent.get_session("test")), progress)
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

    async def test_sse_scopes_projects_and_reports_committed_question_history(self):
        sid, _ = await self.start(custom_jd_name="自定义岗位", custom_jd_text=JD)
        with patch.object(nodes, "_call_llm", side_effect=fake_llm) as llm:
            response = await self.client.post("/api/first-question-stream", json={"session_id": sid})
            self.assertEqual(response.status_code, 200)
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            self.assertEqual("".join(e["content"] for e in events if e["type"] == "answer"), "先聊聊「知识库系统」。问题1？")
            self.assertEqual(events[-1], {"type": "done", "finished": False})
            self.assertNotIn('"start"', response.text)
            context = llm.call_args.args[1][1]["content"]
            self.assertIn(JD, context)
            self.assertIn(DETAILS[0], context)
            self.assertNotIn(DETAILS[1], context)
            self.assertEqual(llm.call_count, 2)
            response = await self.client.post("/api/chat-stream", json={"session_id": sid, "answer": "用了混合检索"})
            self.assertEqual(response.status_code, 200)
            messages = llm.call_args.args[1]
            self.assertEqual([message["role"] for message in messages], ["system", "user", "assistant", "user"])
            self.assertEqual(messages[-1]["content"], "用了混合检索")
            snapshot = (await self.client.get(f"/api/agent-state?session_id={sid}")).json()
            self.assertEqual(snapshot["question_count"], 2)
            self.assertEqual(snapshot["progress"]["round_number"], 3)
            self.assertEqual([q["round_number"] for q in snapshot["question_history"]], [1, 2])

    async def test_new_main_questions_use_direct_maas_requests(self):
        third_project = "面试助手"
        third_details = f"{third_project}：用 LangGraph 控制面试流程。"

        def known_projects_llm(state, messages, **kwargs):
            if not state["projects"]:
                return json.dumps({"projects": [
                    {"name": name, "start": name} for name in [*NAMES, third_project]
                ]}, ensure_ascii=False)
            return fake_llm(state, messages, **kwargs)

        cases = [
            (4, DETAILS[0], "新的主问题", "", "你如何验证文档检索的效果？"),
            (19, third_details, "第一个主问题", f"接下来聊聊「{third_project}」。", "你如何用 LangGraph 管理面试进度？"),
        ]
        for number, details, task, intro, question in cases:
            with self.subTest(question_number=number):
                sid, _ = await self.start(
                    resume_projects=f"{PROJECTS}\n{third_details}", base_url=MAAS_BASE_URL, model=MAAS_MODEL,
                )
                agent = web._sessions[sid]
                with patch.object(nodes, "_call_llm", side_effect=known_projects_llm):
                    await agent.get_first_question(sid)
                    for index in range(1, number - 1):
                        await agent.answer_and_ask(sid, f"回答{index}")

                requests = []

                def model_response(request):
                    body = json.loads(request.content)
                    requests.append(body)
                    # 用只返回推理字段的模拟响应检查主问题请求是否显式启用了直接出题。
                    direct = body.get("chat_template_kwargs") == {"thinking": False}
                    delta = {"content": question} if direct else {"reasoning_content": "模型思考内容"}
                    chunk = {
                        "id": "test", "object": "chat.completion.chunk", "created": 0, "model": MAAS_MODEL,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
                    }
                    return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")

                last_answer = "已结束话题的最后回答"
                transport = httpx.MockTransport(model_response)
                with nodes.OpenAI(api_key="test", base_url=MAAS_BASE_URL, http_client=httpx.Client(transport=transport)) as client:
                    with patch.object(nodes, "OpenAI", return_value=client):
                        response = await self.client.post("/api/chat-stream", json={"session_id": sid, "answer": last_answer})

                self.assertEqual(len(requests), 1)
                messages = requests[0]["messages"]
                self.assertEqual([message["role"] for message in messages], ["system", "user"])
                self.assertIn(details, messages[1]["content"])
                self.assertIn(task, messages[1]["content"])
                self.assertNotIn(details, messages[0]["content"])
                self.assertNotIn(last_answer, messages[1]["content"])
                events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
                expected_question = intro + question
                self.assertEqual("".join(e["content"] for e in events if e["type"] == "answer"), expected_question)
                self.assertEqual(events[-1], {"type": "done", "finished": False})
                session = agent.get_session(sid)
                self.assertEqual(session["question_count"], number)
                self.assertEqual(session["question_history"][-1]["question"], expected_question)
                self.assertEqual(agent.get_messages(sid)[-2].content, last_answer)

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
