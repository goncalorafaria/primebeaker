"""Configurable multi-turn Verifiers environment for JTC terminal judging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import verifiers as vf

from literegistry_tool_client import PlainJsonToolOutputDisplay, TerminalExecutionClient
from .tool_protocol import GPTOSS_TERMINAL_TOOL
from primebeaker.runtime.templates import load_prompt_template, render_chat_template
from .jtc_label_reward import extract_task_output
from .jtc_terminal_tool_env import (
    DEFAULT_TERMINAL_SERVER_URL,
    FINAL_WARNING_TEMPLATE_PATH,
    classify_response,
    final_label_reward,
    final_turn_format_reward,
    normalize_truncation_limit,
    strict_final_json_reward,
)


# Keep legacy dataset rows in OpenAI wire shape, while the environment passes
# the canonical provider-agnostic JTC definition to Verifiers.
TERMINAL_TOOL = {"type": "function", "function": GPTOSS_TERMINAL_TOOL}
TERMINAL_TOOL_DEF = GPTOSS_TERMINAL_TOOL


def _with_terminal_tool(row: dict[str, Any]) -> dict[str, Any]:
    """Supply the native Qwen tool schema when a dataset row lacks one."""
    tools = row.get("tools")
    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError:
            tools = None
    if tools:
        return {**row, "tools": tools}
    return {**row, "tools": [TERMINAL_TOOL]}


def _load_jsonl(path: str, split: str) -> Dataset:
    with open(path, "r", encoding="utf-8") as handle:
        rows = [
            _with_terminal_tool(dict(json.loads(line), split=split))
            for line in handle
            if line.strip()
        ]
    return Dataset.from_list(rows)


def _load_dataset(dataset: str, split: str) -> Dataset:
    if Path(dataset).exists():
        return _load_jsonl(dataset, split)
    rows = load_dataset(dataset, split=split)
    if "split" not in rows.column_names:
        rows = rows.map(lambda row: {"split": split})
    return rows.map(_with_terminal_tool)


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _message_reasoning_text(message: Any) -> str:
    """Return the provider's hidden reasoning field, if it was populated."""
    if isinstance(message, dict):
        return str(message.get("reasoning_content") or message.get("reasoning") or "")
    return str(
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )


def _completion_final_text(completion: list[Any]) -> str:
    """Use visible content, falling back only for a reasoning-only final.

    Some Qwen/vLLM traces place a complete final verdict in
    ``reasoning_content`` and leave ``content`` empty. That is a bad output
    format, but it is still an interpretable completed attempt: returning it
    here keeps the rollout in the RL batch so its explicit penalty is learned.
    """
    if not completion:
        return ""
    visible = _message_text(completion[-1])
    return visible if visible.strip() else _message_reasoning_text(completion[-1])


def _reasoning_only_final(completion: list[Any]) -> bool:
    """Whether a parseable final was emitted only into hidden reasoning."""
    if not completion or _message_text(completion[-1]).strip():
        return False
    return final_turn_format_reward(_message_reasoning_text(completion[-1])) == 1.0


def _message_tool_call(message: Any) -> tuple[str, dict[str, Any], str | None] | None:
    """Read one structured Verifiers/OpenAI function call from an assistant turn."""
    tool_calls = (
        message.get("tool_calls")
        if isinstance(message, dict)
        else getattr(message, "tool_calls", None)
    )
    if not isinstance(tool_calls, (list, tuple)) or len(tool_calls) != 1:
        return None
    call = tool_calls[0]
    name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
    arguments = (
        call.get("arguments") if isinstance(call, dict) else getattr(call, "arguments", None)
    )
    call_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
    if not isinstance(name, str) or not name.strip():
        return None
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(arguments, dict):
        return None
    return name, arguments, str(call_id) if call_id else None


def _trajectory_completion_text(state: dict[str, Any], index: int) -> str:
    trajectory = state.get("trajectory") or []
    if len(trajectory) <= index:
        return ""
    completion = trajectory[index].get("completion") or []
    if not completion:
        return ""
    return _completion_final_text(completion)


def _trajectory_tool_call_id(state: dict[str, Any], index: int) -> str:
    trajectory = state.get("trajectory") or []
    if len(trajectory) <= index:
        return "terminal"
    completion = trajectory[index].get("completion") or []
    if not completion:
        return "terminal"
    tool_call = _message_tool_call(completion[-1])
    return tool_call[2] if tool_call and tool_call[2] else "terminal"


def _trajectory_terminal_command(state: dict[str, Any], index: int) -> str | None:
    """Return the terminal command from vLLM's parsed structured tool call."""
    trajectory = state.get("trajectory") or []
    if len(trajectory) <= index:
        return None
    completion = trajectory[index].get("completion") or []
    if not completion:
        return None
    tool_call = _message_tool_call(completion[-1])
    if tool_call is None:
        return None
    name, arguments, _ = tool_call
    command = arguments.get("command") if name == "terminal" else None
    return command.strip() if isinstance(command, str) and command.strip() else None


def _trajectory_route(state: dict[str, Any], index: int) -> str:
    """Classify a structured terminal action or a text-only final response."""
    if _trajectory_terminal_command(state, index):
        return "terminal"
    return classify_response(_trajectory_completion_text(state, index))


def _task_output(state: dict[str, Any], messages: list[dict[str, Any]] | None = None) -> Any:
    for key in ("task", "input"):
        candidate = state.get(key) or {}
        if isinstance(candidate, dict) and candidate.get("output") is not None:
            return candidate["output"]
    return extract_task_output(messages or [])


def _render_final_warning(
    template: dict[str, Any] | None,
    *,
    tool_content: str,
) -> str:
    fallback = (
        "IMPORTANT: That was your last allowed terminal tool call. Do not call "
        'terminal again. Send a final reply that is ONLY a JSON object: '
        '{"feedback": "your concise justification citing the concrete terminal '
        'evidence", "label": "the rubric label"}'
    )
    if template is None:
        return fallback
    try:
        messages = render_chat_template(
            template,
            tool_content=tool_content,
            tool_name="terminal",
            stdout="",
            stderr="",
            code="",
            success=True,
        )
    except (KeyError, ValueError):
        return fallback
    if messages and messages[0].get("role") == "user":
        content = messages[0].get("content")
        if isinstance(content, str) and content.strip():
            return content
    return fallback


def _terminal_route_is_over_limit(state: dict[str, Any], max_terminal_calls: int) -> bool:
    """Whether a terminal call was emitted after finalization was required."""
    return bool(state.get("jtc_terminal_final_warning_sent")) or (
        int(state.get("jtc_terminal_call_count") or 0) >= max_terminal_calls
    )


class JTCTerminalToolLabelEnv(vf.MultiTurnEnv):
    """JTC terminal-tool environment with a forced final-answer warning."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        terminal_server_url: str = DEFAULT_TERMINAL_SERVER_URL,
        timeout: float = 20,
        max_retries: int = 3,
        terminal_truncation: int | None = 2000,
        tool_role: bool = True,
        max_terminal_calls: int = 4,
        max_turns: int | None = None,
        final_warning_template_path: str | None = FINAL_WARNING_TEMPLATE_PATH,
        reasoning_only_final_penalty: float = -4.0,
        invalid_trace_penalty: float = -4.0,
        **kwargs: Any,
    ) -> None:
        if max_terminal_calls < 1:
            raise ValueError("max_terminal_calls must be at least 1")
        if reasoning_only_final_penalty > 0:
            raise ValueError("reasoning_only_final_penalty must be non-positive")
        if invalid_trace_penalty > 0:
            raise ValueError("invalid_trace_penalty must be non-positive")
        self.terminal_server_url = terminal_server_url
        self.timeout = timeout
        self.max_retries = max_retries
        self.terminal_truncation = normalize_truncation_limit(terminal_truncation)
        self.tool_role = tool_role
        self.max_terminal_calls = int(max_terminal_calls)
        self.max_turns = max_turns or self.max_terminal_calls + 1
        self.reasoning_only_final_penalty = float(reasoning_only_final_penalty)
        self.invalid_trace_penalty = float(invalid_trace_penalty)
        self.final_warning_template = (
            load_prompt_template(final_warning_template_path)
            if final_warning_template_path
            else None
        )
        self.terminal_client = TerminalExecutionClient(
            terminal_server_url,
            timeout=timeout,
            max_retries=max_retries,
            truncation=self.terminal_truncation,
        )

        async def valid_terminal_or_final_turn(state: dict[str, Any]) -> float:
            trajectory = state.get("trajectory") or []
            if not trajectory:
                return 0.0
            route = _trajectory_route(state, len(trajectory) - 1)
            return 1.0 if route in {"terminal", "final"} else 0.0

        async def invalid_trace_penalty_reward(state: dict[str, Any]) -> float:
            """Keep model-caused malformed/over-limit turns in the RL signal."""
            trajectory = state.get("trajectory") or []
            if not trajectory:
                return 0.0
            route = _trajectory_route(state, len(trajectory) - 1)
            over_limit = route == "terminal" and _terminal_route_is_over_limit(
                state, self.max_terminal_calls
            )
            if over_limit:
                return -1.0
            return self.invalid_trace_penalty if route == "invalid" else 0.0

        async def valid_final_format(completion: list[Any]) -> float:
            return final_turn_format_reward(_completion_final_text(completion))

        async def strict_final_json(completion: list[Any]) -> float:
            """Non-gating reward for a bare JSON final answer.

            ``valid_final_format`` remains the permissive parseability check
            used by existing runs. This signal makes surrounding prose and
            fenced JSON less attractive without making either invalid.
            """
            return strict_final_json_reward(
                _message_text(completion[-1]) if completion else ""
            )

        async def reasoning_only_final_penalty_reward(completion: list[Any]) -> float:
            """Penalize a verdict hidden in thinking, without dropping it."""
            return (
                self.reasoning_only_final_penalty
                if _reasoning_only_final(completion)
                else 0.0
            )

        async def correct_final_label(completion: list[Any], answer: Any) -> float:
            return final_label_reward(_completion_final_text(completion), answer)

        super().__init__(
            dataset=dataset,
            # Verifiers forwards only environment-level ``tool_defs`` (or
            # ``info.tool_defs``), not a dataset row's legacy ``tools`` key.
            # Supplying this provider-agnostic definition makes the Qwen
            # renderer include its native terminal-tool template every turn.
            tool_defs=[TERMINAL_TOOL_DEF],
            rubric=vf.Rubric(
                funcs=[
                    valid_terminal_or_final_turn,
                    invalid_trace_penalty_reward,
                    valid_final_format,
                    strict_final_json,
                    reasoning_only_final_penalty_reward,
                    correct_final_label,
                ]
            ),
            max_turns=self.max_turns,
            **kwargs,
        )

    @vf.stop(priority=50)
    async def stop_after_final_or_invalid(self, state: dict[str, Any]) -> bool:
        """Continue only for terminal actions before the configured call limit."""
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return False
        route = _trajectory_route(state, len(trajectory) - 1)
        state["jtc_terminal_route"] = route
        if route != "terminal":
            return True
        executed = int(state.get("jtc_terminal_call_count") or 0)
        if state.get("jtc_terminal_final_warning_sent") or executed >= self.max_terminal_calls:
            # Preserve the terminal route; the rubric applies the explicit
            # over-limit negative reward instead of reclassifying it invalid.
            return True
        return False

    async def env_response(
        self,
        messages: list[dict[str, Any]],
        state: dict[str, Any],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Execute the latest terminal call and optionally append the final warning."""
        trajectory = state.get("trajectory") or []
        command = _trajectory_terminal_command(state, len(trajectory) - 1)
        executed = int(state.get("jtc_terminal_call_count") or 0)
        if not command or executed >= self.max_terminal_calls:
            return []
        try:
            result = await self.terminal_client.execute(
                command=command,
                contents=_task_output(state, messages),
            )
        except Exception as exc:
            result = {
                "stdout": "",
                "stderr": f"Terminal execution failed: {exc}",
                "success": False,
                "returncode": None,
            }

        call_number = executed + 1
        state["jtc_terminal_command"] = command
        state["jtc_terminal_result"] = result
        state["jtc_terminal_call_count"] = call_number
        # The terminal client sends ``self.terminal_truncation`` in its request
        # payload, so the server has already applied the configured limit.
        # Preserve that response verbatim when exposing it to the model.
        tool_content = PlainJsonToolOutputDisplay().render(result)
        response_messages = [
            {
                "role": "tool" if self.tool_role else "user",
                "content": tool_content,
                "tool_call_id": _trajectory_tool_call_id(
                    state, len(trajectory) - 1
                ),
            }
        ]
        if call_number == self.max_terminal_calls:
            state["jtc_terminal_final_warning_sent"] = True
            warning = _render_final_warning(
                self.final_warning_template, tool_content=tool_content
            )
            response_messages[-1]["content"] = f"{tool_content}\n\n{warning}"
        return response_messages


def load_environment(
    dataset: str,
    split: str = "train",
    terminal_server_url: str = DEFAULT_TERMINAL_SERVER_URL,
    timeout: float = 20,
    max_retries: int = 3,
    terminal_truncation: int | None = 2000,
    tool_role: bool = True,
    max_terminal_calls: int = 4,
    max_turns: int | None = None,
    final_warning_template_path: str | None = FINAL_WARNING_TEMPLATE_PATH,
    reasoning_only_final_penalty: float = -4.0,
    invalid_trace_penalty: float = -4.0,
) -> JTCTerminalToolLabelEnv:
    """Prime-RL environment factory."""
    return JTCTerminalToolLabelEnv(
        dataset=_load_dataset(dataset, split),
        terminal_server_url=terminal_server_url,
        timeout=timeout,
        max_retries=max_retries,
        terminal_truncation=terminal_truncation,
        tool_role=tool_role,
        max_terminal_calls=max_terminal_calls,
        max_turns=max_turns,
        final_warning_template_path=final_warning_template_path,
        reasoning_only_final_penalty=reasoning_only_final_penalty,
        invalid_trace_penalty=invalid_trace_penalty,
    )
