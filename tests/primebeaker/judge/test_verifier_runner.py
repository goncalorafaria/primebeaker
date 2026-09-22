import unittest
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from primebeaker.judge.judge_server import JudgeRequest
from primebeaker.judge.verifier_runner import (
    GROUPED_JUDGMENTS_TOOL_NAME,
    build_grouped_judgments_tool,
    build_grouped_rubric_messages,
    build_judge_record,
    collect_grouped_judgments_with_retries,
    collect_judgments_with_retries,
    run_grouped_rubric_judge,
    parse_grouped_judgments_tool_call,
    serialize_grouped_judgments,
    serialize_complete_judgments,
)


class FakeRubric:
    def __init__(self, label, *, route="terminal", stop_condition="stop"):
        self.label = label
        self.feedback = "feedback"
        self.workflow_metadata = {"route": route, "stop_condition": stop_condition}

    def to_dict(self):
        return {"label": self.label, "feedback": self.feedback}


def fake_result(rubric_index, label, **kwargs):
    return {
        "final_row": SimpleNamespace(
            rubric_index=rubric_index,
            rubric=FakeRubric(label, **kwargs),
        )
    }


class VerifierRunnerTest(unittest.TestCase):
    def test_request_becomes_one_record_and_one_row_per_rubric(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement", "Second requirement"],
            model="judge-model",
        )

        record = build_judge_record(request)
        rows = record.to_rows()

        self.assertEqual(record.input, request.input)
        self.assertEqual(record.output, request.output)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].rubric.criteria, "First requirement")
        self.assertEqual(rows[1].rubric.criteria, "Second requirement")
        self.assertEqual(rows[0].rubric.scores["pass"], "Passes all of the requirements.")

    def test_complete_judgments_are_normalized_and_ordered(self):
        results = [
            {"final_row": SimpleNamespace(rubric_index=1, rubric=FakeRubric("FAIL"))},
            {"final_row": SimpleNamespace(rubric_index=0, rubric=FakeRubric(" pass "))},
        ]
        judgments = serialize_complete_judgments(results, expected_count=2)
        self.assertEqual([item["rubric_index"] for item in judgments], [0, 1])
        self.assertEqual([item["label"] for item in judgments], ["pass", "fail"])

    def test_null_label_is_a_judge_failure(self):
        results = [{"final_row": SimpleNamespace(
            rubric_index=0,
            rubric=FakeRubric(None, route="max_iters_exhausted", stop_condition="max_turns_reached"),
        )}]
        with self.assertRaisesRegex(
            RuntimeError,
            "unfinished judgment.*max_iters_exhausted.*max_turns_reached",
        ):
            serialize_complete_judgments(results, expected_count=1)


    def test_grouped_prompt_contains_every_indexed_rubric(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement", "Second requirement"],
            model="judge-model",
        )
        record = build_judge_record(request)
        template = (
            Path(__import__("primebeaker").__file__).parent
            / "resources/judge_templates/recordset-batch-rubric-judge.json"
        )

        messages = build_grouped_rubric_messages(record, str(template))

        self.assertEqual(len(messages), 2)
        user_prompt = messages[1]["content"]
        self.assertIn('rubric_id="0"', user_prompt)
        self.assertIn('rubric_id="1"', user_prompt)
        self.assertIn("First requirement", user_prompt)
        self.assertIn("Second requirement", user_prompt)
        self.assertIn("submit_grouped_judgments", user_prompt)
        self.assertIn("Do not answer in text", user_prompt)

    def test_grouped_response_tool_schema_is_restricted_to_requested_ids(self):
        tool = build_grouped_judgments_tool([1, 3])
        function = tool["function"]
        judgments = function["parameters"]["properties"]["judgments"]
        item_properties = judgments["items"]["properties"]

        self.assertEqual(function["name"], GROUPED_JUDGMENTS_TOOL_NAME)
        self.assertEqual(judgments["minItems"], 2)
        self.assertEqual(judgments["maxItems"], 2)
        self.assertEqual(item_properties["rubric_id"]["enum"], [1, 3])
        self.assertEqual(item_properties["label"]["enum"], ["pass", "fail"])

    def test_grouped_response_parses_final_tool_call_arguments(self):
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    type="function",
                    function=SimpleNamespace(
                        name=GROUPED_JUDGMENTS_TOOL_NAME,
                        arguments=(
                            '{"judgments":[{"rubric_id":0,"label":"pass",'
                            '"feedback":"Present."}]}'
                        ),
                    ),
                )
            ]
        )

        payload = parse_grouped_judgments_tool_call(message)

        self.assertEqual(payload["judgments"][0]["rubric_id"], 0)
        self.assertEqual(payload["judgments"][0]["label"], "pass")

    def test_grouped_response_accepts_literal_control_characters(self):
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    type="function",
                    function=SimpleNamespace(
                        name=GROUPED_JUDGMENTS_TOOL_NAME,
                        arguments=(
                            '{"judgments":[{"rubric_id":0,"label":"pass",'
                            '"feedback":"Uses $p_i=\thbar k_i$."}]}'
                        ),
                    ),
                )
            ]
        )

        payload = parse_grouped_judgments_tool_call(message)

        self.assertEqual(
            payload["judgments"][0]["feedback"],
            "Uses $p_i=\thbar k_i$.",
        )

    def test_grouped_response_still_rejects_structurally_invalid_json(self):
        message = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    type="function",
                    function=SimpleNamespace(
                        name=GROUPED_JUDGMENTS_TOOL_NAME,
                        arguments='{"judgments": [}',
                    ),
                )
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "not valid JSON"):
            parse_grouped_judgments_tool_call(message)

    def test_grouped_response_requires_named_tool_call(self):
        with self.assertRaisesRegex(RuntimeError, "did not call"):
            parse_grouped_judgments_tool_call(SimpleNamespace(tool_calls=None))

        wrong_call = SimpleNamespace(
            tool_calls=[
                SimpleNamespace(
                    type="function",
                    function=SimpleNamespace(name="other_tool", arguments="{}"),
                )
            ]
        )
        with self.assertRaisesRegex(RuntimeError, "expected"):
            parse_grouped_judgments_tool_call(wrong_call)


    def test_grouped_response_maps_strict_json_to_api_schema(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement", "Second requirement"],
            model="judge-model",
        )
        record = build_judge_record(request)
        judgments = serialize_grouped_judgments(
            '{"judgments":['
            '{"rubric_id":1,"label":"fail","feedback":"Missing."},'
            '{"rubric_id":0,"label":"pass","feedback":"Present."}]}',
            record,
            trace={"rubric_mode": "grouped_rubrics"},
        )

        self.assertEqual([item["rubric_index"] for item in judgments], [0, 1])
        self.assertEqual([item["label"] for item in judgments], ["pass", "fail"])
        self.assertEqual(judgments[1]["trace"]["rubric_index"], 1)

    def test_grouped_response_rejects_missing_rubric(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement", "Second requirement"],
            model="judge-model",
        )
        record = build_judge_record(request)
        with self.assertRaisesRegex(RuntimeError, "do not exactly match"):
            serialize_grouped_judgments(
                '{"judgments":['
                '{"rubric_id":0,"label":"pass","feedback":"Present."}]}',
                record,
                trace={},
            )


class VerifierRetryTest(unittest.IsolatedAsyncioTestCase):
    async def test_grouped_request_forces_final_tool_call(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement", "Second requirement"],
            model="Qwen/Qwen3.5-4B",
        )
        record = build_judge_record(request)
        template = (
            Path(__import__("primebeaker").__file__).parent
            / "resources/judge_templates/recordset-batch-rubric-judge.json"
        )
        profile = SimpleNamespace(
            tools=[],
            max_turns=1,
            max_tool_calls=0,
            timeout=300,
            max_retries=0,
            max_tokens=8192,
            temperature=0,
            prompt_template_path=str(template),
            rubric_max_retries=0,
            evaluation_timeout=500,
        )
        message = SimpleNamespace(
            content=None,
            reasoning_content=None,
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    type="function",
                    function=SimpleNamespace(
                        name=GROUPED_JUDGMENTS_TOOL_NAME,
                        arguments=(
                            '{"judgments":['
                            '{"rubric_id":0,"label":"pass","feedback":"Present."},'
                            '{"rubric_id":1,"label":"fail","feedback":"Missing."}'
                            "]}"
                        ),
                    ),
                )
            ],
        )
        response = SimpleNamespace(
            id="response-1",
            model=request.model,
            usage=None,
            choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )

        class FakeOpenAIContext:
            async def __aenter__(self):
                return client

            async def __aexit__(self, _exc_type, _exc, _traceback):
                return False

        with patch("primebeaker.judge.verifier_runner.AsyncOpenAI", return_value=FakeOpenAIContext()):
            judgments = await run_grouped_rubric_judge(
                record,
                request,
                model_server="http://judge.example/v1",
                profile=profile,
            )

        kwargs = create.await_args.kwargs
        self.assertNotIn("response_format", kwargs)
        self.assertEqual(
            kwargs["tool_choice"]["function"]["name"],
            GROUPED_JUDGMENTS_TOOL_NAME,
        )
        self.assertEqual(
            kwargs["tools"][0]["function"]["name"],
            GROUPED_JUDGMENTS_TOOL_NAME,
        )
        self.assertFalse(
            kwargs["extra_body"]["chat_template_kwargs"]["enable_thinking"]
        )
        self.assertEqual([item["label"] for item in judgments], ["pass", "fail"])

    async def test_only_unfinished_rubric_is_retried(self):
        rows = [SimpleNamespace(rubric_index=index) for index in range(3)]
        attempted_indices = []

        async def run_rows(selected_rows):
            indices = [row.rubric_index for row in selected_rows]
            attempted_indices.append(indices)
            if len(attempted_indices) == 1:
                return [
                    fake_result(0, "pass"),
                    fake_result(1, None, stop_condition="timeout_reached"),
                    fake_result(2, "fail"),
                ]
            return [fake_result(1, "pass")]

        judgments = await collect_judgments_with_retries(
            rows, run_rows, max_retries=1
        )

        self.assertEqual(attempted_indices, [[0, 1, 2], [1]])
        self.assertEqual(
            [judgment["label"] for judgment in judgments],
            ["pass", "pass", "fail"],
        )

    async def test_exhausted_rubric_retry_reports_last_failure(self):
        rows = [SimpleNamespace(rubric_index=0)]
        attempted_indices = []

        async def run_rows(selected_rows):
            attempted_indices.append([row.rubric_index for row in selected_rows])
            return [
                fake_result(
                    0,
                    None,
                    route="max_iters_exhausted",
                    stop_condition="timeout_reached",
                )
            ]

        with self.assertRaisesRegex(
            RuntimeError,
            r"unfinished judgments after 2 attempt\(s\).*rubric_index=0.*timeout_reached",
        ):
            await collect_judgments_with_retries(rows, run_rows, max_retries=1)

        self.assertEqual(attempted_indices, [[0], [0]])

    async def test_grouped_call_retries_only_missing_rubrics(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement", "Second requirement"],
            model="judge-model",
        )
        record = build_judge_record(request)
        attempted = []

        async def call_model(attempt, rubric_indices):
            attempted.append((attempt, rubric_indices))
            if attempt == 0:
                return (
                    {
                        "judgments": [
                            {
                                "rubric_id": 0,
                                "label": "pass",
                                "feedback": "Present.",
                            }
                        ]
                    },
                    {"attempt": 1},
                )
            return (
                {
                    "judgments": [
                        {
                            "rubric_id": 1,
                            "label": "fail",
                            "feedback": "Missing.",
                        }
                    ]
                },
                {"attempt": 2},
            )

        judgments = await collect_grouped_judgments_with_retries(
            record, call_model, max_retries=1
        )

        self.assertEqual(attempted, [(0, [0, 1]), (1, [1])])
        self.assertEqual([item["label"] for item in judgments], ["pass", "fail"])

    async def test_grouped_evaluation_timeout_caps_all_attempts(self):
        request = JudgeRequest(
            input="Question",
            output="Candidate output",
            rubrics=["First requirement"],
            model="judge-model",
        )
        record = build_judge_record(request)

        async def call_model(_attempt, _rubric_indices):
            await asyncio.sleep(0.05)
            return (
                {
                    "judgments": [
                        {
                            "rubric_id": 0,
                            "label": "pass",
                            "feedback": "Present.",
                        }
                    ]
                },
                {},
            )

        with self.assertRaisesRegex(RuntimeError, "exceeded 0.01 seconds"):
            await collect_grouped_judgments_with_retries(
                record,
                call_model,
                max_retries=1,
                evaluation_timeout=0.01,
            )

    async def test_evaluation_timeout_caps_all_attempts(self):
        rows = [SimpleNamespace(rubric_index=0)]

        async def run_rows(_selected_rows):
            await asyncio.sleep(0.05)
            return [fake_result(0, "pass")]

        with self.assertRaisesRegex(RuntimeError, "exceeded 0.01 seconds"):
            await collect_judgments_with_retries(
                rows, run_rows, max_retries=1, evaluation_timeout=0.01
            )


if __name__ == "__main__":
    unittest.main()


def test_worker_routes_grouped_inference_through_configured_gateway():
    from primebeaker.judge.verifier_runner import run_verifier_tool_judge

    request = JudgeRequest('Question', 'Answer', ['Criterion'], 'judge-model')
    config = SimpleNamespace(model_gateway_url='http://127.0.0.1:57121')
    profile = SimpleNamespace(rubric_mode='grouped_rubrics', source_path='profile.json')
    judgments = [{'rubric_index': 0, 'label': 'pass', 'feedback': 'supported'}]
    with patch('primebeaker.judge.verifier_runner.load_model_profile', return_value=profile), patch(
        'primebeaker.judge.verifier_runner.run_grouped_rubric_judge',
        new_callable=AsyncMock, return_value=judgments,
    ) as run:
        result = asyncio.run(run_verifier_tool_judge(request, config=config))
    assert run.await_args.kwargs['model_server'] == config.model_gateway_url
    assert result['judgments'] == judgments
    assert result['model'] == request.model
