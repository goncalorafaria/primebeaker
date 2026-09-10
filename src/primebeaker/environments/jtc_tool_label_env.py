"""Generic multi-tool Verifiers environment for JTC Qwen tool-use RL.

The environment receives concrete :class:`ToolClient` instances at startup and
uses the same normalized ``{name, arguments}`` dispatch contract as
``tool_use_workflow.Turn``.  It keeps messages provider-neutral: the Verifiers
renderer/Qwen chat template is responsible for turning a ``role='tool'``
message into ``<tool_response>`` tokens.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import verifiers as vf

from literegistry_tool_client.asset_store import WebAssetStore
from literegistry_tool_client.webterminal import _parse_browse_command
from literegistry_tool_client import (
    FetchClient,
    PlainJsonToolOutputDisplay,
    TerminalExecutionClient,
    ToolClient,
    WebTerminalExecutionClient,
    code_output_to_tool_content,
    terminal_output_to_tool_content,
)
from .tool_protocol import (
    GPTOSS_WEBTERMINAL_TOOL,
    parse_qwen_tool_calls,
    TOOL_SPECS,
    _tool_arguments_usable,
)
from literegistry_tool_client.submission import SubmissionStore, SubmitToolClient
from .jtc_label_reward import extract_task_output
from .jtc_terminal_tool_env import (
    final_label_reward,
    final_turn_format_reward,
    strict_final_json_reward,
)


def _message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _message_reasoning_text(message: Any) -> str:
    """Return provider-specific reasoning text without treating it as content."""
    if isinstance(message, dict):
        return str(message.get("reasoning_content") or message.get("reasoning") or "")
    return str(
        getattr(message, "reasoning_content", None)
        or getattr(message, "reasoning", None)
        or ""
    )


def _completion_final_text(completion: list[Any]) -> str:
    """Use visible content, falling back only for a reasoning-only final."""
    if not completion:
        return ""
    visible = _message_text(completion[-1])
    return visible if visible.strip() else _message_reasoning_text(completion[-1])


def _reasoning_only_final(completion: list[Any]) -> bool:
    """Whether a parseable final was emitted only into hidden reasoning."""
    if not completion or _message_text(completion[-1]).strip():
        return False
    return final_turn_format_reward(_message_reasoning_text(completion[-1])) == 1.0


def _load_dataset(dataset: str, split: str) -> Dataset:
    """Load either a local JSONL task file or a Hugging Face dataset split."""
    path = Path(dataset)
    if path.is_dir():
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                raise KeyError(f"dataset {path} has no {split!r} split")
            rows = loaded[split]
        else:
            rows = loaded
        if "split" not in rows.column_names:
            rows = rows.map(lambda row: {"split": split})
        return rows
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            rows = [
                {**dict(json.loads(line)), "split": split}
                for line in handle
                if line.strip()
            ]
        return Dataset.from_list(rows)
    rows = load_dataset(dataset, split=split)
    if "split" not in rows.column_names:
        rows = rows.map(lambda row: {"split": split})
    return rows


ToolCall = tuple[str, dict[str, Any], str | None]

_COMMAND_CONTROL_SENTINELS = {
    "|": "\ue100",
    ";": "\ue101",
    "<": "\ue102",
    ">": "\ue103",
}
_COMMAND_SEPARATORS = frozenset({"|", "||", ";", ";;", "&&", "&", "(", ")"})
_DEFAULT_SIMPLE_COMMANDS = frozenset({"cat", "sed", "head", "tail"})
_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"\x27\[\](){}]+", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ".,;!?"


def _output_text(value: Any) -> str:
    """Return the evaluated output in the same textual form used for URL checks."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _browse_url_is_in_output(url: str, output: Any) -> bool:
    """Whether an exact HTTP(S) browse target occurs in the evaluated output."""
    if not url.casefold().startswith(("http://", "https://")):
        return False
    output_urls = {
        match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
        for match in _HTTP_URL_RE.finditer(_output_text(output))
    }
    return url in output_urls


def _protect_quoted_command_controls(command: str) -> str:
    """Keep quoted regex/control characters out of pipeline-stage splitting."""

    result: list[str] = []
    quote: str | None = None
    escaped = False
    for character in command:
        if escaped:
            result.append(character)
            escaped = False
            continue
        if quote == '"' and character == "\\":
            result.append(character)
            escaped = True
            continue
        if character in {"'", '"'}:
            if quote is None:
                quote = character
            elif quote == character:
                quote = None
            result.append(character)
        elif quote is not None and character in _COMMAND_CONTROL_SENTINELS:
            result.append(_COMMAND_CONTROL_SENTINELS[character])
        else:
            result.append(character)
    return "".join(result)


def _terminal_command_stage_names(command: str) -> tuple[str, ...]:
    """Return executable names for terminal pipeline/semicolon stages."""

    if not isinstance(command, str) or not command.strip():
        return ()
    try:
        lexer = shlex.shlex(
            _protect_quoted_command_controls(command),
            posix=True,
            punctuation_chars="|;&()",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        # Invalid quoting is rejected by the terminal. Retain only its
        # unambiguous leading stage rather than inventing commands from regexes.
        prefix = command.strip().split(maxsplit=1)
        return (Path(prefix[0]).name,) if prefix else ()

    names: list[str] = []
    at_stage_start = True
    for raw_token in tokens:
        token = raw_token
        for character, sentinel in _COMMAND_CONTROL_SENTINELS.items():
            token = token.replace(sentinel, character)
        is_separator = token in _COMMAND_SEPARATORS or (
            token and all(character in "|;&()" for character in token)
        )
        if is_separator:
            at_stage_start = True
        elif at_stage_start:
            names.append(Path(token).name)
            at_stage_start = False
    return tuple(names)


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _structured_tool_calls(message: Any) -> list[ToolCall] | None:
    """Extract one or more OpenAI/Verifiers structured function calls."""
    calls = _field(message, "tool_calls")
    if not isinstance(calls, (list, tuple)) or not calls:
        return None
    parsed: list[ToolCall] = []
    for call in calls:
        function = _field(call, "function")
        payload = function if function is not None else call
        name = _field(payload, "name")
        arguments = _field(payload, "arguments", {})
        if not isinstance(name, str) or not name.strip():
            return None
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        call_id = _field(call, "id")
        parsed.append((name.strip(), arguments, str(call_id) if call_id else None))
    return parsed


def _completion_tool_calls(message: Any) -> list[ToolCall] | None:
    """Prefer structured calls, with native-Qwen text as a fallback."""
    structured = _structured_tool_calls(message)
    if structured is not None:
        return structured
    calls = parse_qwen_tool_calls(_message_text(message))
    if not calls:
        return None
    return [
        (str(call["name"]), dict(call.get("arguments") or {}), None)
        for call in calls
    ]


def _latest_completion(state: dict[str, Any]) -> Any | None:
    trajectory = state.get("trajectory") or []
    if not trajectory:
        return None
    completion = trajectory[-1].get("completion") or []
    return completion[-1] if completion else None


def _task_output(state: dict[str, Any], messages: vf.Messages) -> Any:
    for key in ("task", "input"):
        candidate = state.get(key) or {}
        if isinstance(candidate, dict) and candidate.get("output") is not None:
            return candidate["output"]
    return extract_task_output(messages)


def _task_web_pages(state: dict[str, Any]) -> Mapping[str, str] | None:
    """Return saved ``{url: markdown}`` evidence attached to this rollout."""
    for key in ("task", "input"):
        candidate = state.get(key) or {}
        if not isinstance(candidate, Mapping):
            continue
        pages = candidate.get("web_pages")
        if pages is None and candidate.get("web_pages_json") is not None:
            raw_pages = candidate["web_pages_json"]
            if not isinstance(raw_pages, str):
                raise TypeError("web_pages_json must be a JSON string")
            try:
                pages = json.loads(raw_pages)
            except json.JSONDecodeError as exc:
                raise ValueError("web_pages_json must contain valid JSON") from exc
        if pages is None or pages == {} or pages == []:
            continue
        if not isinstance(pages, Mapping):
            raise TypeError("web_pages must be a mapping of URL strings to content strings")
        normalized: dict[str, str] = {}
        for url, content in pages.items():
            if not isinstance(url, str) or not url:
                raise ValueError("web_pages keys must be non-empty URL strings")
            if not isinstance(content, str):
                raise TypeError("web_pages values must be content strings")
            normalized[url] = content
        return normalized
    return None


def _expected_ids_from_mapping(candidate: Any) -> tuple[str | int, ...] | None:
    if not isinstance(candidate, Mapping):
        return None
    for field in ("expected_rubric_ids", "rubric_ids"):
        values = candidate.get(field)
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            return tuple(values)
    indexed = candidate.get("indexed_rubrics")
    if isinstance(indexed, Mapping):
        return tuple(indexed)
    if isinstance(indexed, Sequence) and not isinstance(indexed, (str, bytes)):
        identifiers: list[str | int] = []
        for rubric in indexed:
            if not isinstance(rubric, Mapping):
                return None
            identifier = rubric.get("rubric_id", rubric.get("id"))
            if identifier is None:
                return None
            identifiers.append(identifier)
        return tuple(identifiers)
    rubric_index = candidate.get("rubric_index")
    if rubric_index is not None:
        return (rubric_index,)
    return None


def _task_expected_rubric_ids(state: dict[str, Any]) -> tuple[str | int, ...]:
    """Find the rubric IDs declared by this rollout's task payload."""
    for key in ("task", "input"):
        candidate = state.get(key)
        for container in (
            candidate,
            candidate.get("info") if isinstance(candidate, Mapping) else None,
        ):
            identifiers = _expected_ids_from_mapping(container)
            if identifiers is not None:
                return identifiers
    return ()


def _error_output(tool_name: str, exc: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "stdout": "",
        "stderr": f"{tool_name} execution request failed: {type(exc).__name__}: {exc}",
        "exit_code": 1,
    }


def _unknown_tool_output(tool_name: str, available_tools: Collection[str]) -> dict[str, Any]:
    """Return a recoverable, model-facing observation for an unadvertised tool."""
    available = ", ".join(sorted(available_tools))
    return {
        "success": False,
        "stdout": "",
        "stderr": (
            f"unknown tool {tool_name!r}; available tools: {available}. "
            "Use one of the available tools instead."
        ),
        "exit_code": 1,
    }


_MODEL_CAUSED_HTTP_STATUSES = (400, 413, 422)


def _exception_is_model_caused(exc: Exception) -> bool:
    """Separate bad tool requests from transport and service failures."""
    if isinstance(exc, (TypeError, ValueError)):
        return True
    detail = f" {exc} "
    return any(
        f" HTTP {status} " in detail for status in _MODEL_CAUSED_HTTP_STATUSES
    )


def _tool_stderr_is_model_error(
    tool_name: str,
    arguments: Mapping[str, Any],
    output: Mapping[str, Any],
    *,
    request_model_caused: bool,
) -> bool:
    """Whether stderr reflects the model's command rather than infrastructure."""
    stderr = str(output.get("stderr") or "").strip()
    if not stderr or not request_model_caused:
        return False
    if tool_name == "terminal":
        command_stages = _terminal_command_stage_names(
            str(arguments.get("command") or "")
        )
        # URL existence is itself evidence. A well-formed browse probe that
        # returns 404 or another fetch stderr is not a command-syntax mistake.
        if command_stages[:1] == ("browse",):
            return False
        if stderr.casefold().startswith("required command is unavailable:"):
            return False
    return True

def _tool_content(tool_name: str, output: dict[str, Any]) -> str:
    """Match ``MultiToolExecutionMap``'s model-facing result display."""
    if tool_name == "terminal":
        return terminal_output_to_tool_content(output)
    if tool_name == "submit":
        return PlainJsonToolOutputDisplay().render(output)
    if tool_name == "search" and not output.get("stdout") and not output.get("stderr"):
        output = {
            **output,
            "stdout": json.dumps(output.get("data", output), ensure_ascii=False),
            "stderr": "",
            "exit_code": 0 if output.get("success") else 1,
        }
    return code_output_to_tool_content(output)


def _append_warning_to_last_tool_response(messages: vf.Messages, warning: str) -> bool:
    """Append workflow control text to an existing, ID-bearing tool response."""
    for message in reversed(messages):
        if message.get("role") == "tool":
            content = str(message.get("content") or "").rstrip()
            if warning not in content:
                message["content"] = f"{content}\n\n{warning}".lstrip()
            return True
    return False


class JTCToolLabelEnv(vf.MultiTurnEnv):
    """JTC label verifier that dispatches configured native tools per turn.

    ``tool_clients`` is deliberately supplied as live objects, rather than
    server configuration: callers can provide any startup-constructed
    ``ToolClient`` implementation.  The selected names define both dispatch
    and the provider-neutral ``tool_defs`` advertised to Verifiers.  An empty
    selection is a prompt-only, single-turn JSON-label environment.
    """

    def __init__(
        self,
        *,
        dataset: Dataset,
        tool_clients: Mapping[str, ToolClient],
        tools: Sequence[str] | None = None,
        tool_defs: Sequence[Mapping[str, Any]] | None = None,
        max_tool_calls: int = 64,
        tool_role: bool = True,
        max_turns: int | None = None,
        final_warning: str | None = None,
        reasoning_only_final_penalty: float = -4.0,
        invalid_trace_penalty: float = -4.0,
        tool_use_reward_coef: float = 0.0,
        tool_stderr_penalty: float = 0.1,
        unknown_tool_penalty: float = 0.1,
        browse_url_not_in_output_penalty: float = 0.0,
        browse_success_reward_coef: float = 0.0,
        standalone_command_repeat_penalty: float = 0.0,
        standalone_command_repeat_free_count: int = 1,
        standalone_command_repeat_commands: Sequence[str] | None = None,
        final_response_required: bool = True,
        **kwargs: Any,
    ) -> None:
        selected = tuple(dict.fromkeys(tools if tools is not None else tool_clients.keys()))
        unknown = sorted(set(selected).difference(TOOL_SPECS))
        if unknown:
            raise ValueError(f"unknown tools: {unknown}; available tools: {sorted(TOOL_SPECS)}")
        missing = sorted(set(selected).difference(tool_clients))
        if missing:
            raise ValueError(f"missing execution clients for tools: {missing}")
        invalid = [name for name in selected if not isinstance(tool_clients[name], ToolClient)]
        if invalid:
            raise TypeError(f"tool_clients values must be ToolClient instances: {invalid}")
        if tool_defs is None:
            resolved_tool_defs = [TOOL_SPECS[name] for name in selected]
        else:
            resolved_tool_defs = [dict(tool_def) for tool_def in tool_defs]
            invalid_defs = [
                index
                for index, tool_def in enumerate(resolved_tool_defs)
                if not isinstance(tool_def.get("name"), str)
                or not isinstance(tool_def.get("parameters"), Mapping)
            ]
            if invalid_defs:
                raise ValueError(
                    f"tool_defs entries need string names and parameter mappings: {invalid_defs}"
                )
            defined_names = [str(tool_def["name"]) for tool_def in resolved_tool_defs]
            if len(set(defined_names)) != len(defined_names):
                raise ValueError("tool_defs names must be unique")
            if set(defined_names) != set(selected):
                raise ValueError(
                    "tool_defs names must exactly match selected tools: "
                    f"defined={sorted(defined_names)}, selected={sorted(selected)}"
                )
        if max_tool_calls < 0:
            raise ValueError("max_tool_calls must be non-negative")
        if not selected and max_tool_calls != 0:
            raise ValueError("a no-tool verifier must use max_tool_calls=0")
        if reasoning_only_final_penalty > 0:
            raise ValueError("reasoning_only_final_penalty must be non-positive")
        if invalid_trace_penalty > 0:
            raise ValueError("invalid_trace_penalty must be non-positive")
        if tool_stderr_penalty < 0:
            raise ValueError("tool_stderr_penalty must be non-negative")
        if unknown_tool_penalty < 0:
            raise ValueError("unknown_tool_penalty must be non-negative")
        if browse_url_not_in_output_penalty < 0:
            raise ValueError(
                "browse_url_not_in_output_penalty must be non-negative"
            )
        if browse_success_reward_coef < 0:
            raise ValueError("browse_success_reward_coef must be non-negative")
        if standalone_command_repeat_penalty < 0:
            raise ValueError("standalone_command_repeat_penalty must be non-negative")
        if standalone_command_repeat_free_count < 0:
            raise ValueError("standalone_command_repeat_free_count must be non-negative")
        simple_commands = frozenset(
            standalone_command_repeat_commands or _DEFAULT_SIMPLE_COMMANDS
        )
        if not simple_commands or any(
            not isinstance(name, str) or not name or any(char.isspace() for char in name)
            for name in simple_commands
        ):
            raise ValueError(
                "standalone_command_repeat_commands must contain non-empty command names"
            )

        self.tool_names = frozenset(selected)
        self.tool_defs = resolved_tool_defs
        self.tool_clients = {name: tool_clients[name] for name in selected}
        self.tool_role = tool_role
        self.max_tool_calls = int(max_tool_calls)
        self.max_turns = max_turns or self.max_tool_calls + 1
        self.reasoning_only_final_penalty = float(reasoning_only_final_penalty)
        self.invalid_trace_penalty = float(invalid_trace_penalty)
        # Applied once per accepted tool call: cumulative trajectory bonus is
        # tool_use_reward_coef * used_tool_calls / max_tool_calls.
        self.tool_use_reward_coef = float(tool_use_reward_coef)
        # Subtracted once for every non-empty stderr result; exit code alone is free.
        self.tool_stderr_penalty = float(tool_stderr_penalty)
        # Unknown tool names are recoverable so a model can correct itself,
        # but remain a distinct modeling error beyond their stderr result.
        self.unknown_tool_penalty = float(unknown_tool_penalty)
        # These rewards are deliberately independent of stderr. A browse target
        # is judged against the evaluated report, while success is taken from
        # the structured terminal result.
        self.browse_url_not_in_output_penalty = float(
            browse_url_not_in_output_penalty
        )
        self.browse_success_reward_coef = float(browse_success_reward_coef)
        # One targeted standalone call is free by default. Multi-stage commands
        # are excluded from both numerator and denominator.
        self.standalone_command_repeat_penalty = float(standalone_command_repeat_penalty)
        self.standalone_command_repeat_free_count = int(standalone_command_repeat_free_count)
        self.standalone_command_repeat_commands = simple_commands
        self.final_response_required = bool(final_response_required)
        self.final_warning = final_warning or (
            "IMPORTANT: This was your last allowed tool call. Do not call any tool "
            "again. Send a final reply that is ONLY a JSON object with string keys "
            '"feedback" and "label".'
        )

        def scored_route(state: dict[str, Any]) -> str:
            """Use the stop hook final route when it is available.

            A syntactically valid tool call after the final warning parses as
            ``tool`` in isolation, but the stop hook has classified that turn
            as invalid. Re-parsing it here used to reward the over-limit call
            and hid the negative signal.
            """
            route = state.get("jtc_tool_route")
            if route in {"tool", "final", "invalid"}:
                return route
            return self._route(state)

        async def valid_tool_or_final_turn(state: dict[str, Any]) -> float:
            if state.get("jtc_infrastructure_failure") is True:
                return 0.0
            return 1.0 if scored_route(state) in {"tool", "final"} else 0.0

        async def invalid_trace_penalty_reward(state: dict[str, Any]) -> float:
            # Infrastructure-specific environments may recover transport/runtime
            # failures into state so their own rubric can assign one clear penalty.
            if state.get("jtc_infrastructure_failure") is True:
                return 0.0
            return self.invalid_trace_penalty if scored_route(state) == "invalid" else 0.0

        async def tool_use_reward(state: dict[str, Any]) -> float:
            """Return the cumulative, normalized count of accepted tool calls.

            Rubrics score once after a rollout ends, so this intentionally reads
            the count accumulated by env_response instead of inspecting only the
            terminal completion (which is normally the final JSON answer).
            """
            if self.max_tool_calls == 0:
                return 0.0
            used_tool_calls = min(
                max(int(state.get("jtc_tool_call_count") or 0), 0),
                self.max_tool_calls,
            )
            return self.tool_use_reward_coef * used_tool_calls / self.max_tool_calls

        async def tool_stderr_penalty_reward(state: dict[str, Any]) -> float:
            """Penalize model-caused command errors, never service failures."""
            count = max(int(state.get("jtc_tool_command_error_count") or 0), 0)
            return -self.tool_stderr_penalty * count

        async def unknown_tool_penalty_reward(state: dict[str, Any]) -> float:
            count = max(int(state.get("jtc_unknown_tool_call_count") or 0), 0)
            return -self.unknown_tool_penalty * count

        async def browse_url_not_in_output_penalty_reward(
            state: dict[str, Any],
        ) -> float:
            """Penalize every HTTP(S)-citation mistake independently of stderr."""
            count = max(
                int(state.get("jtc_browse_url_not_in_output_count") or 0), 0
            )
            return -self.browse_url_not_in_output_penalty * count

        async def browse_success_reward(state: dict[str, Any]) -> float:
            """Reward a trajectory once after its first successful browse."""
            succeeded = int(state.get("jtc_successful_browse_count") or 0) > 0
            return self.browse_success_reward_coef if succeeded else 0.0

        async def standalone_command_repeat_penalty_reward(
            state: dict[str, Any],
        ) -> float:
            """Penalize excess targeted calls among standalone commands only."""

            total = max(int(state.get("jtc_standalone_command_count") or 0), 0)
            if total == 0:
                return 0.0
            simple = max(int(state.get("jtc_targeted_standalone_command_count") or 0), 0)
            excess = max(simple - self.standalone_command_repeat_free_count, 0)
            return -self.standalone_command_repeat_penalty * excess / total

        async def browse_call_count(state: dict[str, Any]) -> float:
            """Report literal ``browse`` terminal calls without rewarding them."""
            return float(max(int(state.get("jtc_browse_call_count") or 0), 0))

        async def browse_url_not_in_output_count(state: dict[str, Any]) -> float:
            return float(
                max(int(state.get("jtc_browse_url_not_in_output_count") or 0), 0)
            )

        async def successful_browse_count(state: dict[str, Any]) -> float:
            return float(max(int(state.get("jtc_successful_browse_count") or 0), 0))

        async def valid_final_format(completion: list[Any]) -> float:
            return final_turn_format_reward(_completion_final_text(completion))

        async def strict_final_json(completion: list[Any]) -> float:
            return strict_final_json_reward(
                _message_text(completion[-1]) if completion else ""
            )

        async def reasoning_only_final_penalty_reward(completion: list[Any]) -> float:
            return (
                self.reasoning_only_final_penalty
                if _reasoning_only_final(completion)
                else 0.0
            )

        async def correct_final_label(completion: list[Any], answer: Any) -> float:
            return final_label_reward(_completion_final_text(completion), answer)

        reward_funcs = [
            valid_tool_or_final_turn,
            invalid_trace_penalty_reward,
            tool_use_reward,
            tool_stderr_penalty_reward,
            unknown_tool_penalty_reward,
            browse_url_not_in_output_penalty_reward,
            browse_success_reward,
            standalone_command_repeat_penalty_reward,
        ]
        # Submit-terminated episodes have no final JSON answer. Their subclass
        # supplies rubric-level reward instead of these scalar final-answer terms.
        if self.final_response_required:
            reward_funcs.extend([
                valid_final_format,
                strict_final_json,
                reasoning_only_final_penalty_reward,
                correct_final_label,
            ])
        rubric = vf.Rubric(funcs=reward_funcs)
        rubric.add_metric(browse_call_count)
        rubric.add_metric(browse_url_not_in_output_count)
        rubric.add_metric(successful_browse_count)

        super().__init__(
            dataset=dataset,
            tool_defs=self.tool_defs,
            rubric=rubric,
            max_turns=self.max_turns,
            **kwargs,
        )

    def _route(self, state: dict[str, Any]) -> str:
        completion = _latest_completion(state)
        calls = _completion_tool_calls(completion) if completion is not None else None
        if calls is not None:
            for name, arguments, _ in calls:
                # An unadvertised tool receives an error observation below so
                # the model can correct it. Only malformed calls to tools we
                # actually expose make the turn structurally invalid.
                if name in self.tool_names and not _tool_arguments_usable(name, arguments):
                    return "invalid"
            return "tool"
        final_text = _completion_final_text([completion]) if completion is not None else ""
        return "final" if final_turn_format_reward(final_text) else "invalid"

    @vf.stop(priority=50)
    async def stop_after_final_or_invalid(self, state: dict[str, Any]) -> bool:
        if not (state.get("trajectory") or []):
            return False
        route = self._route(state)
        state["jtc_tool_route"] = route
        if route != "tool":
            return True
        executed = int(state.get("jtc_tool_call_count") or 0)
        calls = _completion_tool_calls(_latest_completion(state)) or []
        if state.get("jtc_tool_final_warning_sent"):
            state["jtc_tool_route"] = "invalid"
            return True
        # A single assistant completion may contain multiple calls. If that
        # batch would cross the budget, do not execute any of it, but let
        # env_response issue the final warning and give the model one chance
        # to produce its required JSON answer. Mark it invalid now so no
        # format or tool-use reward is granted before env_response runs.
        if executed + len(calls) > self.max_tool_calls:
            state["jtc_tool_route"] = "invalid"
        return False

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        """Dispatch latest calls concurrently and return ordered tool turns."""
        calls = _completion_tool_calls(_latest_completion(state))
        executed = int(state.get("jtc_tool_call_count") or 0)
        if not calls:
            return []
        if executed + len(calls) > self.max_tool_calls:
            # Parallel calls are one assistant action, but each call consumes
            # budget. Reject the whole over-budget batch and still deliver the
            # same warning used after an exactly-budgeted batch.
            state["jtc_tool_final_warning_sent"] = True
            state["jtc_tool_route"] = "invalid"
            if _append_warning_to_last_tool_response(messages, self.final_warning):
                return []
            # No tool has run yet, so there is no valid tool_call_id to attach.
            return [vf.UserMessage(content=self.final_warning)]

        context = _task_output(state, messages)
        submission_store = state.get("jtc_submit_session")
        if "submit" in self.tool_names and submission_store is None:
            submission_store = SubmitToolClient.new_session(
                _task_expected_rubric_ids(state)
            )
            state["jtc_submit_session"] = submission_store
        asset_store = state.get("jtc_tool_asset_store")
        if asset_store is None:
            asset_store = WebAssetStore(web_pages=_task_web_pages(state))
            state["jtc_tool_asset_store"] = asset_store
        prepared: list[ToolCall] = []
        for tool_name, arguments, call_id in calls:
            tool_arguments = dict(arguments)
            tool_client = self.tool_clients.get(tool_name)
            # The browser-aware backend may be advertised under the canonical
            # training-time name ``terminal``. Enrich by client capability,
            # not by the model-facing alias, so its rollout-local asset cache
            # is always forwarded.
            if isinstance(tool_client, WebTerminalExecutionClient):
                tool_arguments.update(contents=context, asset_store=asset_store)
            elif tool_name == "terminal":
                tool_arguments["contents"] = context
            elif tool_name == "python":
                tool_arguments["context_payload"] = context
            elif tool_name == "submit":
                tool_arguments["submission_store"] = submission_store
            prepared.append((tool_name, tool_arguments, call_id))

        async def execute_one(tool_name: str, tool_arguments: dict[str, Any]) -> tuple[dict[str, Any], bool]:
            if tool_name not in self.tool_names:
                return _unknown_tool_output(tool_name, self.tool_names), False
            try:
                output = await self.tool_clients[tool_name].execute(**tool_arguments)
                if not isinstance(output, dict):
                    raise RuntimeError("tool returned a non-object response")
                return output, True
            except Exception as exc:  # noqa: BLE001 - retain the rollout turn
                return _error_output(tool_name, exc), _exception_is_model_caused(exc)

        if any(name == "submit" for name, _, _ in prepared):
            # A batch of submissions is cumulative: preserve assistant call order
            # so every response reports the keys completed by earlier calls.
            executions = [
                await execute_one(name, arguments)
                for name, arguments, _ in prepared
            ]
        else:
            executions = await asyncio.gather(
                *(execute_one(name, arguments) for name, arguments, _ in prepared)
            )
        outputs = [output for output, _ in executions]
        nonempty_stderr_flags = [
            bool(str(output.get("stderr") or "").strip())
            for output in outputs
        ]
        model_error_flags = [
            _tool_stderr_is_model_error(
                name,
                arguments,
                output,
                request_model_caused=request_model_caused,
            )
            for (name, arguments, _), (output, request_model_caused) in zip(
                prepared, executions
            )
        ]
        call_number = executed + len(prepared)
        nonempty_stderr_count = int(
            state.get("jtc_tool_nonempty_stderr_count") or 0
        ) + sum(nonempty_stderr_flags)
        command_error_count = int(
            state.get("jtc_tool_command_error_count") or 0
        ) + sum(model_error_flags)
        unknown_tool_call_count = int(
            state.get("jtc_unknown_tool_call_count") or 0
        ) + sum(name not in self.tool_names for name, _, _ in prepared)
        infrastructure_stderr_count = int(
            state.get("jtc_tool_infrastructure_stderr_count") or 0
        ) + sum(
            has_stderr and not is_model_error
            for has_stderr, is_model_error in zip(
                nonempty_stderr_flags, model_error_flags
            )
        )
        terminal_command_stages = tuple(
            _terminal_command_stage_names(str(arguments.get("command") or ""))
            for tool_name, arguments, _ in prepared
            if tool_name == "terminal"
        )
        browse_call_count = int(state.get("jtc_browse_call_count") or 0) + sum(
            stages[:1] == ("browse",) for stages in terminal_command_stages
        )
        browse_urls = tuple(
            url
            for tool_name, arguments, _ in prepared
            if tool_name == "terminal"
            for url, _ in (_parse_browse_command(str(arguments.get("command") or "")),)
            if url is not None
        )
        browse_url_not_in_output_count = int(
            state.get("jtc_browse_url_not_in_output_count") or 0
        ) + sum(not _browse_url_is_in_output(url, context) for url in browse_urls)
        successful_browse_count = int(
            state.get("jtc_successful_browse_count") or 0
        ) + sum(
            output.get("success") is True
            for (tool_name, arguments, _), output in zip(prepared, outputs)
            if tool_name == "terminal"
            and _parse_browse_command(str(arguments.get("command") or ""))[0]
            is not None
        )
        standalone_names = tuple(
            stages[0] for stages in terminal_command_stages if len(stages) == 1
        )
        standalone_command_count = int(
            state.get("jtc_standalone_command_count") or 0
        ) + len(standalone_names)
        targeted_standalone_command_count = int(
            state.get("jtc_targeted_standalone_command_count") or 0
        ) + sum(
            name in self.standalone_command_repeat_commands
            for name in standalone_names
        )
        targeted_standalone_excess_fraction = (
            max(
                targeted_standalone_command_count
                - self.standalone_command_repeat_free_count,
                0,
            )
            / standalone_command_count
            if standalone_command_count
            else 0.0
        )
        if isinstance(submission_store, SubmissionStore):
            state["jtc_submit_submissions"] = submission_store.as_dict()
            state["jtc_submit_complete"] = submission_store.all_completed
        state.update(
            # ``jtc_tool_calls`` is the model's wire request, suitable for an
            # assistant OpenAI ``tool_calls`` transcript.  ``prepared`` adds
            # private harness context (for example terminal ``contents``), so
            # retain it separately for execution diagnostics.
            jtc_tool_calls=[
                {"name": name, "arguments": dict(arguments), "id": call_id}
                for name, arguments, call_id in calls
            ],
            jtc_tool_execution_calls=[
                {"name": name, "arguments": arguments, "id": call_id}
                for name, arguments, call_id in prepared
            ],
            jtc_tool_results=outputs,
            jtc_tool_call_count=call_number,
            jtc_tool_nonempty_stderr_count=nonempty_stderr_count,
            jtc_tool_command_error_count=command_error_count,
            jtc_unknown_tool_call_count=unknown_tool_call_count,
            jtc_tool_infrastructure_stderr_count=infrastructure_stderr_count,
            jtc_browse_call_count=browse_call_count,
            jtc_browse_url_not_in_output_count=(
                browse_url_not_in_output_count
            ),
            jtc_successful_browse_count=successful_browse_count,
            jtc_standalone_command_count=standalone_command_count,
            jtc_targeted_standalone_command_count=targeted_standalone_command_count,
            jtc_targeted_standalone_excess_fraction=(
                targeted_standalone_excess_fraction
            ),
        )
        history = state.setdefault("jtc_tool_history", [])
        history.extend(
            {
                "name": name,
                "arguments": dict(arguments),
                "id": call_id,
                "output": output,
            }
            for (name, arguments, call_id), output in zip(calls, outputs)
        )
        response: vf.Messages = []
        for index, ((tool_name, _, call_id), output) in enumerate(
            zip(prepared, outputs), start=1
        ):
            content = _tool_content(tool_name, output)
            tool_call_id = call_id or f"{tool_name}_{executed + index}"
            if self.tool_role:
                response.append(
                    vf.ToolMessage(
                        content=content,
                        tool_call_id=tool_call_id,
                        name=tool_name,
                    )
                )
            else:
                response.append(
                    vf.UserMessage(
                        content=content,
                        name=tool_name,
                        tool_call_id=tool_call_id,
                    )
                )
        if call_number == self.max_tool_calls:
            state["jtc_tool_final_warning_sent"] = True
            _append_warning_to_last_tool_response(response, self.final_warning)
        return response


def load_environment(
    dataset: str,
    split: str = "train",
    tools: Sequence[str] | None = None,
    terminal_server_url: str = "http://127.0.0.1:1212/terminal",
    timeout: float = 20,
    max_retries: int = 3,
    terminal_truncation: int | None = 2000,
    browser_aware_terminal: bool = False,
    fetch_server_url: str | None = None,
    fetch_timeout: float = 65,
    fetch_max_retries: int = 3,
    local_search_server_url: str | None = None,
    local_search_model_path: str | None = None,
    capture_web_assets: bool = False,
    tool_role: bool = True,
    max_tool_calls: int = 64,
    max_turns: int | None = None,
    final_warning: str | None = None,
    reasoning_only_final_penalty: float = -4.0,
    invalid_trace_penalty: float = -4.0,
    tool_use_reward_coef: float = 0.0,
    tool_stderr_penalty: float = 0.1,
    unknown_tool_penalty: float = 0.1,
    browse_url_not_in_output_penalty: float = 0.0,
    browse_success_reward_coef: float = 0.0,
    standalone_command_repeat_penalty: float = 0.0,
    standalone_command_repeat_free_count: int = 1,
    standalone_command_repeat_commands: Sequence[str] | None = None,
) -> JTCToolLabelEnv:
    """PrimeRL factory that constructs the configured terminal client.

    ``browser_aware_terminal`` retains the model-facing ``terminal`` name but
    binds it to :class:`WebTerminalExecutionClient`. Saved per-task pages are
    served from the rollout-local asset store before any fetch fallback.
    """
    selected = tuple(dict.fromkeys(tools if tools is not None else ("terminal",)))
    terminal_client = TerminalExecutionClient(
        terminal_server_url,
        timeout=timeout,
        max_retries=max_retries,
        truncation=terminal_truncation,
    )
    clients: dict[str, ToolClient] = {"terminal": terminal_client}
    tool_defs: list[dict[str, Any]] | None = None
    if browser_aware_terminal:
        fetch_kwargs: dict[str, Any] = {
            "timeout": fetch_timeout,
            "max_retries": fetch_max_retries,
            "local_search_server_url": local_search_server_url,
            "local_search_model_path": local_search_model_path,
        }
        if fetch_server_url is not None:
            fetch_kwargs["server_url"] = fetch_server_url
        clients["terminal"] = WebTerminalExecutionClient(
            terminal_client,
            FetchClient(**fetch_kwargs),
            capture_web_assets=capture_web_assets,
        )
        tool_defs = []
        for name in selected:
            definition = dict(TOOL_SPECS[name])
            if name == "terminal":
                definition = dict(GPTOSS_WEBTERMINAL_TOOL)
                definition["name"] = "terminal"
            tool_defs.append(definition)
    if "submit" in selected:
        clients["submit"] = SubmitToolClient()
    return JTCToolLabelEnv(
        dataset=_load_dataset(dataset, split),
        tool_clients=clients,
        tools=selected,
        tool_defs=tool_defs,
        max_tool_calls=max_tool_calls,
        tool_role=tool_role,
        max_turns=max_turns,
        final_warning=final_warning,
        reasoning_only_final_penalty=reasoning_only_final_penalty,
        invalid_trace_penalty=invalid_trace_penalty,
        tool_use_reward_coef=tool_use_reward_coef,
        tool_stderr_penalty=tool_stderr_penalty,
        unknown_tool_penalty=unknown_tool_penalty,
        browse_url_not_in_output_penalty=browse_url_not_in_output_penalty,
        browse_success_reward_coef=browse_success_reward_coef,
        standalone_command_repeat_penalty=standalone_command_repeat_penalty,
        standalone_command_repeat_free_count=standalone_command_repeat_free_count,
        standalone_command_repeat_commands=standalone_command_repeat_commands,
    )
