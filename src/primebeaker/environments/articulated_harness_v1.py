"""verifiers v1 port of ``articulated_harness_env`` (submit-terminated rubric judging).

The v0 environment subclasses the v0 ``JTCToolLabelEnv``; prime-rl >= 0.9 accepts only
v1 sources, so this module rebuilds its contract on the v1 tool-label pieces:

* ``ArticulatedToolset`` is one per-rollout MCP server exposing ``terminal`` (the evaluated
  output as stdin, as in ``jtc_tool_label_v1``) and ``submit`` (v0's ``SubmissionStore``
  semantics and JSON response). Both share the ``max_tool_calls`` budget, as in v0. Each
  row's trained tool schemas are advertised verbatim. Calls are serialized per rollout:
  v1 state sync is last-write-wins, so concurrent calls could drop a submission.
* ``ArticulatedHarnessTask`` keeps v0's reward terms and names for the submit-terminated
  case (no final-answer terms), adds ``rubric_completion_reward`` (+coef per correct
  gold label / N) and ``missing_rubric_penalty`` (-coef per unsubmitted rubric / N), and
  ends the rollout before the next model call once every expected rubric is submitted.

Run it with the ``null`` harness on the ``subprocess`` runtime.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from literegistry_tool_client.output import PlainJsonToolOutputDisplay
from literegistry_tool_client.submission import SUBMIT_TOOL_SPEC, SubmissionStore

from .jtc_terminal_tool_env import final_turn_format_reward
from .jtc_tool_label_v1 import (
    JTCToolLabelData,
    JTCToolLabelState,
    JTCToolLabelTaskConfig,
    TerminalToolset,
    TerminalToolsetConfig,
    _call_arguments,
    _final_assistant,
    _final_text,
)
from .tool_protocol import _tool_arguments_usable

__all__ = [
    "ArticulatedHarnessData",
    "ArticulatedHarnessState",
    "ArticulatedHarnessTask",
    "ArticulatedHarnessTaskConfig",
    "ArticulatedHarnessTaskset",
    "ArticulatedHarnessTasksetConfig",
    "ArticulatedToolset",
    "ArticulatedToolsetConfig",
]

TERMINAL = "terminal"
SUBMIT = "submit"
TOOL_NAMES = frozenset({TERMINAL, SUBMIT})
# The terminal schema most sloth rows were trained on (the safety sources carry the longer
# tool_protocol wording in their own ``tools`` column, which takes precedence).
DEFAULT_TERMINAL_SPEC: dict[str, Any] = {
    "name": TERMINAL,
    "description": "Run a restricted terminal command against the evaluated output supplied on standard input.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
}


def _drop_nulls(value: Any) -> Any:
    """Undo the null padding HF ``datasets`` adds to struct columns."""
    if isinstance(value, Mapping):
        return {k: _drop_nulls(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_drop_nulls(v) for v in value]
    return value


def _tool_specs(tools: Any) -> dict[str, dict[str, Any]]:
    """``{name: {name, description, parameters}}`` from a row's OpenAI-style ``tools``."""
    specs = {TERMINAL: DEFAULT_TERMINAL_SPEC, SUBMIT: SUBMIT_TOOL_SPEC}
    for tool in tools or []:
        function = _drop_nulls(tool).get("function") if isinstance(tool, Mapping) else None
        if isinstance(function, Mapping) and function.get("name") in TOOL_NAMES:
            specs[function["name"]] = dict(function)
    return specs


def _submit_error_output(exc: Exception) -> dict[str, Any]:
    # v0 ``_error_output("submit", exc)``.
    return {
        "success": False,
        "stdout": "",
        "stderr": f"submit execution request failed: {type(exc).__name__}: {exc}",
        "exit_code": 1,
    }


# ----------------------------------------------------------------------------
# Data, state, and configs
# ----------------------------------------------------------------------------


class ArticulatedHarnessData(JTCToolLabelData):
    expected_rubric_ids: list[str] = Field(default_factory=list)
    gold_submissions: list[dict[str, Any]] = Field(default_factory=list)
    """Hidden per-rubric gold ``{rubric_id, label, feedback}``."""
    rubric_bundle_size: int | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list)
    """The row's trained OpenAI-style tool schemas."""


class ArticulatedHarnessState(JTCToolLabelState):
    submissions: dict[str, dict[str, str]] = Field(default_factory=dict)
    submit_call_count: int = 0
    submit_error_count: int = 0


class ArticulatedToolsetConfig(TerminalToolsetConfig):
    max_tool_calls: int = Field(128, ge=1)


class ArticulatedHarnessTaskConfig(JTCToolLabelTaskConfig):
    tools: ArticulatedToolsetConfig = ArticulatedToolsetConfig()
    rubric_completion_reward_coef: float = Field(1.0, ge=0)
    missing_rubric_penalty_coef: float = Field(1.0, ge=0)


class ArticulatedHarnessTasksetConfig(vf.TasksetConfig):
    dataset: str = ""
    """HF ``save_to_disk`` directory (DatasetDict or Dataset) or JSONL file of articulated rows."""
    split: str = "train"
    task: ArticulatedHarnessTaskConfig = ArticulatedHarnessTaskConfig()


# ----------------------------------------------------------------------------
# Tool server
# ----------------------------------------------------------------------------


class ArticulatedToolset(TerminalToolset, vf.Toolset[ArticulatedToolsetConfig, ArticulatedHarnessState]):
    """``terminal`` + ``submit`` sharing one budget and one serialized state channel.

    The generic is re-declared: v1 resolves a toolset's state class from its type arguments,
    and the inherited ``TerminalToolset`` binding would load ``JTCToolLabelState`` (no
    ``submissions``), failing every ``submit``.
    """

    async def setup(self) -> None:
        await super().setup()
        self._lock = asyncio.Lock()
        self._expected: tuple[str, ...] = ()
        self._specs = _tool_specs(None)

    async def setup_task(self, task: ArticulatedHarnessData) -> None:
        await super().setup_task(task)
        self._expected = tuple(str(x) for x in task.expected_rubric_ids)
        self._specs = _tool_specs(task.tools)

    @vf.tool
    async def submit(self, rubric_id: str | int, feedback: str, label: str) -> str:
        state = self.state
        cfg = self.config
        if state.tool_call_count >= cfg.max_tool_calls:
            state.final_warning_sent = True
            state.rejected_over_budget_calls += 1
            return cfg.final_warning

        store = SubmissionStore(self._expected)
        store._submissions = {k: dict(v) for k, v in state.submissions.items()}
        try:
            status = store.submit(rubric_id=rubric_id, feedback=feedback, label=label)
            output: dict[str, Any] = {
                "success": True,
                "stdout": json.dumps(status, ensure_ascii=False),
                "stderr": "",
                "exit_code": 0,
                "data": status,
            }
            state.submissions = store.as_dict()
        except Exception as exc:  # noqa: BLE001 - v0 returned the error to the model
            output = _submit_error_output(exc)
            state.submit_error_count += 1
            # v0 counted a rejected submission as a model-caused command error.
            state.command_error_count += 1
            state.nonempty_stderr_count += 1

        state.tool_call_count += 1
        state.submit_call_count += 1
        content = PlainJsonToolOutputDisplay().render(output)
        if state.tool_call_count == cfg.max_tool_calls:
            state.final_warning_sent = True
            content = f"{content.rstrip()}\n\n{cfg.final_warning}".lstrip()
        return content

    def _serialized(self, fn: Any) -> Any:
        synced = self._with_state(fn)

        async def call(*args: Any, **kwargs: Any) -> Any:
            async with self._lock:
                return await synced(*args, **kwargs)

        call.__signature__ = synced.__signature__  # type: ignore[attr-defined]
        call.__name__ = fn.__name__
        call.__doc__ = fn.__doc__
        return call

    def register(self, mcp: Any) -> None:
        for name, fn in ((TERMINAL, self.terminal), (SUBMIT, self.submit)):
            spec = self._specs[name]
            mcp.add_tool(self._serialized(fn), name=name, description=spec["description"])
            # Advertise the trained JSON schema, not the one derived from the signature.
            mcp._tool_manager._tools[name].parameters = spec["parameters"]


# ----------------------------------------------------------------------------
# Task: stops and rewards
# ----------------------------------------------------------------------------


def _malformed_call(message: vf.AssistantMessage | None) -> bool:
    calls = (message.tool_calls or []) if message is not None else []
    return any(
        call.name in TOOL_NAMES and not _tool_arguments_usable(call.name, _call_arguments(call))
        for call in calls
    )


class ArticulatedHarnessTask(
    vf.Task[ArticulatedHarnessData, ArticulatedHarnessState, ArticulatedHarnessTaskConfig]
):
    @property
    def key(self) -> str:
        ids = ",".join(self.data.expected_rubric_ids)
        return f"{self.data.record_id}:{ids}:{self.data.idx}"

    @classmethod
    def toolsets(cls, config: ArticulatedHarnessTaskConfig) -> list[vf.Toolset]:
        return [ArticulatedToolset(config.tools)]

    def _gold(self) -> dict[str, str]:
        gold: dict[str, str] = {}
        for item in self.data.gold_submissions:
            rubric_id = str(item.get("rubric_id") or "").strip()
            label = str(item.get("label") or "").strip().casefold()
            if rubric_id and label in {"pass", "fail"}:
                gold[rubric_id] = label
        return gold

    def _expected(self) -> tuple[str, ...]:
        return tuple(str(x) for x in self.data.expected_rubric_ids)

    def _waiting(self, trace: vf.Trace) -> list[str]:
        return [key for key in self._expected() if key not in trace.state.submissions]

    # --- stops -------------------------------------------------------------

    @vf.stop
    def all_rubrics_submitted(self, request: vf.Request, trace: vf.Trace) -> bool:
        # A request stop runs on the next model request, after the tool results are
        # appended: the last accepted submit ends the episode without another model turn.
        return bool(self._expected()) and not self._waiting(trace)

    @vf.stop
    def tool_call_after_final_warning(self, response: vf.Response, trace: vf.Trace) -> bool:
        return bool(trace.state.final_warning_sent and response.message.tool_calls)

    @vf.stop
    def malformed_tool_call(self, response: vf.Response) -> bool:
        return _malformed_call(response.message)

    # --- routing (v0) ------------------------------------------------------

    def _route(self, trace: vf.Trace) -> str:
        final = _final_assistant(trace)
        if final is None:
            return "invalid"
        if final.tool_calls:
            if trace.state.final_warning_sent or _malformed_call(final):
                return "invalid"
            return "tool"
        return "final" if final_turn_format_reward(_final_text(final)) else "invalid"

    def _unknown_tool_calls(self, trace: vf.Trace) -> int:
        return sum(
            call.name not in TOOL_NAMES
            for message in trace.messages
            if isinstance(message, vf.AssistantMessage)
            for call in (message.tool_calls or [])
        )

    # --- rewards (v0 names; no final-answer terms for submit-terminated episodes) ---

    @vf.reward
    async def valid_tool_or_final_turn(self, trace: vf.Trace) -> float:
        return 1.0 if self._route(trace) in {"tool", "final"} else 0.0

    @vf.reward
    async def invalid_trace_penalty_reward(self, trace: vf.Trace) -> float:
        return self.config.invalid_trace_penalty if self._route(trace) == "invalid" else 0.0

    @vf.reward
    async def tool_use_reward(self, trace: vf.Trace) -> float:
        budget = self.config.tools.max_tool_calls
        used = min(max(trace.state.tool_call_count, 0), budget)
        return self.config.tool_use_reward_coef * used / budget

    @vf.reward
    async def tool_stderr_penalty_reward(self, trace: vf.Trace) -> float:
        return -self.config.tool_stderr_penalty * max(trace.state.command_error_count, 0)

    @vf.reward
    async def unknown_tool_penalty_reward(self, trace: vf.Trace) -> float:
        return -self.config.unknown_tool_penalty * self._unknown_tool_calls(trace)

    @vf.reward
    async def browse_url_not_in_output_penalty_reward(self, trace: vf.Trace) -> float:
        return -self.config.browse_url_not_in_output_penalty * max(
            trace.state.browse_url_not_in_output_count, 0
        )

    @vf.reward
    async def browse_success_reward(self, trace: vf.Trace) -> float:
        return self.config.browse_success_reward_coef if trace.state.successful_browse_count > 0 else 0.0

    @vf.reward
    async def standalone_command_repeat_penalty_reward(self, trace: vf.Trace) -> float:
        total = max(trace.state.standalone_command_count, 0)
        if total == 0:
            return 0.0
        excess = max(
            trace.state.targeted_standalone_command_count
            - self.config.standalone_command_repeat_free_count,
            0,
        )
        return -self.config.standalone_command_repeat_penalty * excess / total

    @vf.reward
    async def rubric_completion_reward(self, trace: vf.Trace) -> float:
        """+coef/N for each submission matching its hidden gold label."""
        expected = self._expected()
        if not expected:
            return 0.0
        gold = self._gold()
        submissions = trace.state.submissions
        if not gold:
            # v0 kept completion credit for evaluation-only rows without gold labels.
            completed = sum(key in submissions for key in expected)
            return self.config.rubric_completion_reward_coef * completed / len(expected)
        correct = sum(
            str(submissions.get(key, {}).get("label") or "").strip().casefold() == gold.get(key)
            for key in expected
        )
        return self.config.rubric_completion_reward_coef * correct / len(expected)

    @vf.reward
    async def missing_rubric_penalty(self, trace: vf.Trace) -> float:
        """-coef/N for every rubric the model did not submit."""
        expected = self._expected()
        if not expected or not self._gold():
            return 0.0
        return -self.config.missing_rubric_penalty_coef * len(self._waiting(trace)) / len(expected)

    # --- metrics -----------------------------------------------------------

    @vf.metric
    async def tool_call_count(self, trace: vf.Trace) -> float:
        return float(trace.state.tool_call_count)

    @vf.metric
    async def submit_call_count(self, trace: vf.Trace) -> float:
        return float(trace.state.submit_call_count)

    @vf.metric
    async def submit_error_count(self, trace: vf.Trace) -> float:
        return float(trace.state.submit_error_count)

    @vf.metric
    async def submitted_fraction(self, trace: vf.Trace) -> float:
        expected = self._expected()
        return (len(expected) - len(self._waiting(trace))) / len(expected) if expected else 0.0

    @vf.metric
    async def all_rubrics_submitted_rate(self, trace: vf.Trace) -> float:
        return float(bool(self._expected()) and not self._waiting(trace))

    @vf.metric
    async def rubric_bundle_size(self, trace: vf.Trace) -> float:
        return float(len(self._expected()))

    @vf.metric
    async def infrastructure_stderr_count(self, trace: vf.Trace) -> float:
        return float(trace.state.infrastructure_stderr_count)


# ----------------------------------------------------------------------------
# Taskset
# ----------------------------------------------------------------------------

_DATA_FIELDS = (
    "answer", "output", "record_id", "rubric_index", "source",
    "expected_rubric_ids", "gold_submissions", "rubric_bundle_size", "tools",
)


def _rows(path: Path, split: str) -> Iterable[dict[str, Any]]:
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    yield json.loads(line)
        return
    import datasets

    loaded = datasets.load_from_disk(str(path))
    yield from (loaded[split] if isinstance(loaded, datasets.DatasetDict) else loaded)


class ArticulatedHarnessTaskset(vf.Taskset[ArticulatedHarnessTask, ArticulatedHarnessTasksetConfig]):
    def load(self) -> Iterable[ArticulatedHarnessTask]:
        path = Path(self.config.dataset)
        if not path.exists():
            raise FileNotFoundError(f"articulated harness dataset not found: {path}")
        for idx, row in enumerate(_rows(path, self.config.split)):
            if row.get("output") is None:
                raise ValueError(f"{path} row {idx} has no evaluated output")
            fields = {name: row.get(name) for name in _DATA_FIELDS}
            fields["expected_rubric_ids"] = [str(x) for x in fields["expected_rubric_ids"] or []]
            fields["gold_submissions"] = [dict(x) for x in fields["gold_submissions"] or []]
            fields["tools"] = [_drop_nulls(t) for t in fields["tools"] or []]
            data = ArticulatedHarnessData(idx=idx, prompt=row["prompt"], **fields)
            yield ArticulatedHarnessTask(data, self.config.task)


if __name__ == "__main__":
    # verifiers starts each rollout's tool server as `python -m <this module>`.
    ArticulatedToolset.run()
