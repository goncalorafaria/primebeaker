"""Two-turn Verifiers environment for JTC label RL with Python execution.

This is separate from `jtc_label_env.py` on purpose:

- `jtc_label_env.py` is the one-step environment.
- This file owns the code-tool variant where the model may first emit code,
  receive a tool message, and then emit the final feedback + label.

This env is a true `vf.MultiTurnEnv`:

1. The first model turn may be final XML or a Python code request.
2. If it requested code, `env_response` executes it asynchronously and returns
   only a tool message.
3. The second model turn must produce final feedback + label XML.
"""

import json
from pathlib import Path

from datasets import Dataset
from datasets import load_dataset
import verifiers as vf

from .jtc_code_tool_env import (
    classify_response,
    final_label_reward,
    final_turn_format_reward,
    first_turn_format_reward,
    normalize_truncation_limit,
)
from literegistry_tool_client import RemoteCodeExecutionClient, code_output_to_tool_content
from .jtc_label_reward import extract_python_code
from .jtc_label_reward import extract_task_output


def _load_jsonl(path, split):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row.setdefault("split", split)
            rows.append(row)
    return Dataset.from_list(rows)


def _load_dataset(dataset, split):
    if Path(dataset).exists():
        return _load_jsonl(dataset, split)
    rows = load_dataset(dataset, split=split)
    if "split" not in rows.column_names:
        rows = rows.map(lambda row: {"split": split})
    return rows


def _message_text(message):
    if isinstance(message, dict):
        return message.get("content") or ""
    return getattr(message, "content", "") or ""


def _trajectory_completion_text(state, index):
    trajectory = state.get("trajectory") or []
    if len(trajectory) <= index:
        return ""
    completion = trajectory[index].get("completion") or []
    if not completion:
        return ""
    return _message_text(completion[-1])


def _task_output(state, messages=None):
    task = state.get("task") or {}
    if isinstance(task, dict):
        output = task.get("output")
        if output is not None:
            return output
    input_data = state.get("input") or {}
    if isinstance(input_data, dict):
        output = input_data.get("output")
        if output is not None:
            return output
    return extract_task_output(messages)


class JTCCodeToolLabelEnv(vf.MultiTurnEnv):
    """Two-turn JTC environment with one optional Python execution."""

    def __init__(
        self,
        *,
        dataset,
        code_server_url="http://127.0.0.1:1212/python",
        timeout=20,
        max_retries=3,
        max_runtime=2,
        code_output_truncation=None,
        tool_role=True,
        max_turns=2,
        **kwargs,
    ):
        self.code_server_url = code_server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.max_runtime = max_runtime
        self.code_output_truncation = normalize_truncation_limit(code_output_truncation)
        self.tool_role = tool_role
        self.code_client = RemoteCodeExecutionClient(
            code_server_url,
            timeout=timeout,
            max_retries=max_retries,
            max_runtime=max_runtime,
        )

        async def valid_first_turn(state):
            return first_turn_format_reward(_trajectory_completion_text(state, 0))

        async def first_turn_uses_code(state):
            first_turn = _trajectory_completion_text(state, 0)
            return 0.5 if classify_response(first_turn) == "code" else 0.0

        async def valid_final_format(completion):
            completion_text = _message_text(completion[-1]) if completion else ""
            return final_turn_format_reward(completion_text)

        async def correct_final_label(completion, answer):
            completion_text = _message_text(completion[-1]) if completion else ""
            return final_label_reward(completion_text, answer)

        rubric = vf.Rubric(
            funcs=[
                valid_first_turn,
                # first_turn_uses_code,
                valid_final_format,
                correct_final_label,
            ]
        )
        super().__init__(
            dataset=dataset,
            rubric=rubric,
            max_turns=max_turns,
            **kwargs,
        )

    @vf.stop(priority=50)
    async def first_turn_is_final_or_invalid(self, state):
        """Stop after turn one unless the model requested valid code."""
        if len(state.get("trajectory") or []) != 1:
            return False
        first_response = _trajectory_completion_text(state, 0)
        route = classify_response(first_response)
        state["jtc_first_route"] = route
        return route in ("final", "invalid")

    async def env_response(self, messages, state, **kwargs):
        """Execute first-turn code and return exactly one tool/user message."""
        first_response = _trajectory_completion_text(state, 0)
        code = extract_python_code(first_response)
        if not code:
            return []

        try:
            code_result = await self.code_client.execute(
                code=code,
                context_payload=_task_output(state, messages),
            )
        except Exception as exc:
            code_result = {
                "stdout": "",
                "stderr": "Code execution failed: {}".format(exc),
                "success": False,
            }

        state["jtc_code"] = code
        state["jtc_code_result"] = code_result
        role = "tool" if self.tool_role else "user"
        return [
            {
                "role": role,
                "content": code_output_to_tool_content(
                    code_result,
                    code_output_truncation=self.code_output_truncation,
                ),
                "tool_call_id": "python",
            }
        ]


def load_environment(
    dataset,
    split="train",
    code_server_url="http://127.0.0.1:1212/python",
    timeout=20,
    max_retries=3,
    max_runtime=2,
    code_output_truncation=None,
    tool_role=True,
    max_turns=2,
):
    rows = _load_dataset(dataset, split)
    return JTCCodeToolLabelEnv(
        dataset=rows,
        code_server_url=code_server_url,
        timeout=timeout,
        max_retries=max_retries,
        max_runtime=max_runtime,
        code_output_truncation=code_output_truncation,
        tool_role=tool_role,
        max_turns=max_turns,
    )
