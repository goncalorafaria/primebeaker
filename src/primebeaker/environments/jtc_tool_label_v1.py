"""verifiers v1 port of ``jtc_tool_label_env`` for prime-rl >= 0.9.

prime-rl 0.9 accepts only v1 sources (``env = { taskset = { id = ... } }``), so
the v0 ``MultiTurnEnv`` cannot run there. This module keeps the v0 reward
contract for the terminal-only, final-JSON-label case:

* ``TerminalToolset`` is a per-rollout MCP server exposing the bare ``terminal``
  tool. It injects the hidden evaluated output as stdin, enforces the tool
  budget, appends the final warning, and accumulates the penalty counters in
  ``JTCToolLabelState``.
* ``JTCToolLabelTask`` scores the finished trace with the same reward terms and
  names as v0, and stops a rollout on a tool call after the final warning or a
  malformed ``terminal`` call (both of which v0 routed as ``invalid``).

Run it with the ``null`` harness on the ``subprocess`` runtime.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar

import verifiers.v1 as vf
from pydantic import Field

from literegistry_tool_client import (
    FetchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
    terminal_output_to_tool_content,
)
from literegistry_tool_client.asset_store import WebAssetStore
from literegistry_tool_client.webterminal import _parse_browse_command

from .jtc_terminal_tool_env import (
    final_label_reward,
    final_turn_format_reward,
    strict_final_json_reward,
)
from .tool_protocol import GPTOSS_WEBTERMINAL_TOOL, TOOL_SPECS, _tool_arguments_usable

__all__ = [
    "JTCToolLabelData",
    "JTCToolLabelState",
    "JTCToolLabelTask",
    "JTCToolLabelTaskConfig",
    "JTCToolLabelTaskset",
    "JTCToolLabelTasksetConfig",
    "TerminalToolset",
    "TerminalToolsetConfig",
]

TOOL_NAME = "terminal"
DEFAULT_FINAL_WARNING = (
    "IMPORTANT: This was your last allowed tool call. Do not call any tool "
    "again. Send a final reply that is ONLY a JSON object with string keys "
    '"feedback" and "label".'
)
# v0 advertised the browser-aware schema renamed to ``terminal``; keep the text
# the checkpoints were trained on.
_BROWSER_DESCRIPTION = str(GPTOSS_WEBTERMINAL_TOOL["description"])
_PLAIN_DESCRIPTION = str(TOOL_SPECS[TOOL_NAME]["description"])

# ----------------------------------------------------------------------------
# Pure helpers copied from the v0 module so the per-rollout tool subprocess does
# not import ``datasets`` or the v0 environment.
# ----------------------------------------------------------------------------

_COMMAND_CONTROL_SENTINELS = {"|": "", ";": "", "<": "", ">": ""}
_COMMAND_SEPARATORS = frozenset({"|", "||", ";", ";;", "&&", "&", "(", ")"})
_DEFAULT_SIMPLE_COMMANDS = ("cat", "sed", "head", "tail")
_HTTP_URL_RE = re.compile(r"https?://[^\s<>\"\x27\[\](){}]+", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ".,;!?"
_MODEL_CAUSED_HTTP_STATUSES = (400, 413, 422)


def _output_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _browse_url_is_in_output(url: str, output: Any) -> bool:
    if not url.casefold().startswith(("http://", "https://")):
        return False
    output_urls = {
        match.group(0).rstrip(_TRAILING_URL_PUNCTUATION)
        for match in _HTTP_URL_RE.finditer(_output_text(output))
    }
    return url in output_urls


def _protect_quoted_command_controls(command: str) -> str:
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


def _error_output(exc: Exception) -> dict[str, Any]:
    return {
        "success": False,
        "stdout": "",
        "stderr": f"{TOOL_NAME} execution request failed: {type(exc).__name__}: {exc}",
        "exit_code": 1,
    }


def _exception_is_model_caused(exc: Exception) -> bool:
    if isinstance(exc, (TypeError, ValueError)):
        return True
    detail = f" {exc} "
    return any(f" HTTP {status} " in detail for status in _MODEL_CAUSED_HTTP_STATUSES)


def _stderr_is_model_error(command: str, output: Mapping[str, Any], *, model_caused: bool) -> bool:
    stderr = str(output.get("stderr") or "").strip()
    if not stderr or not model_caused:
        return False
    # A well-formed browse probe that 404s is evidence, not a command mistake.
    if _terminal_command_stage_names(command)[:1] == ("browse",):
        return False
    return not stderr.casefold().startswith("required command is unavailable:")


def _web_pages(raw: str | None) -> dict[str, str] | None:
    if not raw:
        return None
    pages = json.loads(raw)
    if not pages:
        return None
    if not isinstance(pages, Mapping):
        raise TypeError("web_pages_json must decode to a mapping of URL to content")
    return {str(url): str(content) for url, content in pages.items()}


# ----------------------------------------------------------------------------
# Data, state, and configs
# ----------------------------------------------------------------------------


class JTCToolLabelData(vf.TaskData):
    answer: Any = None
    """Gold rubric label."""
    output: Any = None
    """Evaluated response; fed to ``terminal`` as stdin and never shown in the prompt."""
    web_pages_json: str | None = None
    """Saved ``{url: markdown}`` evidence served to ``browse`` before any fetch."""
    record_id: str | None = None
    rubric_index: int | None = None
    source: str | None = None


class JTCToolLabelState(vf.State):
    tool_call_count: int = 0
    nonempty_stderr_count: int = 0
    command_error_count: int = 0
    infrastructure_stderr_count: int = 0
    browse_call_count: int = 0
    browse_url_not_in_output_count: int = 0
    successful_browse_count: int = 0
    standalone_command_count: int = 0
    targeted_standalone_command_count: int = 0
    final_warning_sent: bool = False
    rejected_over_budget_calls: int = 0


class TerminalToolsetConfig(vf.ToolsetConfig):
    terminal_server_url: str = "http://127.0.0.1:1212/terminal"
    timeout: float = 20
    max_retries: int = 3
    terminal_truncation: int | None = 2000
    browser_aware_terminal: bool = False
    fetch_server_url: str | None = None
    fetch_timeout: float = 65
    fetch_max_retries: int = 3
    local_search_server_url: str | None = None
    local_search_model_path: str | None = None
    capture_web_assets: bool = False
    max_tool_calls: int = Field(64, ge=1)
    final_warning: str = DEFAULT_FINAL_WARNING
    standalone_command_repeat_commands: list[str] = Field(
        default_factory=lambda: list(_DEFAULT_SIMPLE_COMMANDS)
    )


class JTCToolLabelTaskConfig(vf.TaskConfig):
    tools: TerminalToolsetConfig = TerminalToolsetConfig()
    reasoning_only_final_penalty: float = Field(-4.0, le=0)
    invalid_trace_penalty: float = Field(-4.0, le=0)
    tool_use_reward_coef: float = 0.0
    tool_stderr_penalty: float = Field(0.1, ge=0)
    unknown_tool_penalty: float = Field(0.1, ge=0)
    browse_url_not_in_output_penalty: float = Field(0.0, ge=0)
    browse_success_reward_coef: float = Field(0.0, ge=0)
    standalone_command_repeat_penalty: float = Field(0.0, ge=0)
    standalone_command_repeat_free_count: int = Field(1, ge=0)


class JTCToolLabelTasksetConfig(vf.TasksetConfig):
    dataset: str = ""
    """Local JSONL file of JTC tool-label rows."""
    split: str = "train"
    task: JTCToolLabelTaskConfig = JTCToolLabelTaskConfig()


# ----------------------------------------------------------------------------
# Tool server
# ----------------------------------------------------------------------------


class TerminalToolset(vf.Toolset[TerminalToolsetConfig, JTCToolLabelState]):
    TOOL_PREFIX: ClassVar[str | None] = None

    async def setup(self) -> None:
        cfg = self.config
        terminal = TerminalExecutionClient(
            cfg.terminal_server_url,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            truncation=cfg.terminal_truncation,
        )
        self._web = cfg.browser_aware_terminal
        if self._web:
            fetch_kwargs: dict[str, Any] = {
                "timeout": cfg.fetch_timeout,
                "max_retries": cfg.fetch_max_retries,
                "local_search_server_url": cfg.local_search_server_url,
                "local_search_model_path": cfg.local_search_model_path,
            }
            if cfg.fetch_server_url is not None:
                fetch_kwargs["server_url"] = cfg.fetch_server_url
            self._client: Any = WebTerminalExecutionClient(
                terminal,
                FetchClient(**fetch_kwargs),
                capture_web_assets=cfg.capture_web_assets,
            )
        else:
            self._client = terminal
        self._simple_commands = frozenset(cfg.standalone_command_repeat_commands)
        self._contents: Any = None
        self._asset_store = WebAssetStore(web_pages=None)

    async def setup_task(self, task: JTCToolLabelData) -> None:
        self._contents = task.output
        self._asset_store = WebAssetStore(web_pages=_web_pages(task.web_pages_json))

    @vf.tool
    async def terminal(self, command: str) -> str:
        state = self.state
        cfg = self.config
        if state.tool_call_count >= cfg.max_tool_calls:
            state.final_warning_sent = True
            state.rejected_over_budget_calls += 1
            return cfg.final_warning

        arguments: dict[str, Any] = {"command": command, "contents": self._contents}
        if self._web:
            arguments["asset_store"] = self._asset_store
        try:
            output = await self._client.execute(**arguments)
            if not isinstance(output, dict):
                raise RuntimeError("tool returned a non-object response")
            model_caused = True
        except Exception as exc:  # noqa: BLE001 - the rollout keeps its turn
            output = _error_output(exc)
            model_caused = _exception_is_model_caused(exc)

        has_stderr = bool(str(output.get("stderr") or "").strip())
        is_model_error = _stderr_is_model_error(command, output, model_caused=model_caused)
        stages = _terminal_command_stage_names(command)
        url, _ = _parse_browse_command(command)

        state.tool_call_count += 1
        state.nonempty_stderr_count += int(has_stderr)
        state.command_error_count += int(is_model_error)
        state.infrastructure_stderr_count += int(has_stderr and not is_model_error)
        state.browse_call_count += int(stages[:1] == ("browse",))
        if url is not None:
            state.browse_url_not_in_output_count += int(
                not _browse_url_is_in_output(url, self._contents)
            )
            state.successful_browse_count += int(output.get("success") is True)
        if len(stages) == 1:
            state.standalone_command_count += 1
            state.targeted_standalone_command_count += int(stages[0] in self._simple_commands)

        content = terminal_output_to_tool_content(output)
        if state.tool_call_count == cfg.max_tool_calls:
            state.final_warning_sent = True
            content = f"{content.rstrip()}\n\n{cfg.final_warning}".lstrip()
        return content

    def register(self, mcp: Any) -> None:
        # The advertised description must match what the checkpoint was trained
        # on, and it depends on the configured mode, so it can't be a docstring.
        mcp.add_tool(
            self._with_state(self.terminal),
            name=TOOL_NAME,
            description=(
                _BROWSER_DESCRIPTION if self.config.browser_aware_terminal else _PLAIN_DESCRIPTION
            ),
        )


# ----------------------------------------------------------------------------
# Task: stops and rewards
# ----------------------------------------------------------------------------


def _final_assistant(trace: vf.Trace) -> vf.AssistantMessage | None:
    for message in reversed(trace.messages):
        if isinstance(message, vf.AssistantMessage):
            return message
    return None


def _visible_text(message: vf.AssistantMessage | None) -> str:
    return str(message.content or "") if message is not None else ""


def _final_text(message: vf.AssistantMessage | None) -> str:
    """Visible content, falling back to reasoning only for a reasoning-only final."""
    if message is None:
        return ""
    visible = _visible_text(message)
    return visible if visible.strip() else str(message.reasoning_content or "")


def _call_arguments(call: Any) -> Any:
    try:
        return json.loads(call.arguments or "{}")
    except (TypeError, json.JSONDecodeError):
        return None


def _malformed_terminal_call(message: vf.AssistantMessage | None) -> bool:
    calls = (message.tool_calls or []) if message is not None else []
    return any(
        call.name == TOOL_NAME and not _tool_arguments_usable(TOOL_NAME, _call_arguments(call))
        for call in calls
    )


class JTCToolLabelTask(vf.Task[JTCToolLabelData, JTCToolLabelState, JTCToolLabelTaskConfig]):
    @property
    def key(self) -> str:
        # (record_id, rubric_index) repeats across distinct rows (and a few rows
        # are exact duplicates); v0 trained on every row, so the file position
        # keeps each one a distinct task.
        if self.data.record_id is not None:
            return f"{self.data.record_id}:{self.data.rubric_index}:{self.data.idx}"
        return self.hash

    @classmethod
    def toolsets(cls, config: JTCToolLabelTaskConfig) -> list[vf.Toolset]:
        return [TerminalToolset(config.tools)]

    # --- stops -------------------------------------------------------------

    @vf.stop
    def tool_call_after_final_warning(self, response: vf.Response, trace: vf.Trace) -> bool:
        return bool(trace.state.final_warning_sent and response.message.tool_calls)

    @vf.stop
    def malformed_terminal_call(self, response: vf.Response) -> bool:
        return _malformed_terminal_call(response.message)

    # --- routing -----------------------------------------------------------

    def _route(self, trace: vf.Trace) -> str:
        final = _final_assistant(trace)
        if final is None:
            return "invalid"
        if final.tool_calls:
            # Ending on a tool call after the warning or with unusable
            # arguments is invalid; running out of turns mid-budget stays a
            # valid tool turn, as in v0.
            if trace.state.final_warning_sent or _malformed_terminal_call(final):
                return "invalid"
            return "tool"
        return "final" if final_turn_format_reward(_final_text(final)) else "invalid"

    def _unknown_tool_calls(self, trace: vf.Trace) -> int:
        return sum(
            call.name != TOOL_NAME
            for message in trace.messages
            if isinstance(message, vf.AssistantMessage)
            for call in (message.tool_calls or [])
        )

    # --- rewards (names match v0 for metric continuity) --------------------

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
    async def valid_final_format(self, trace: vf.Trace) -> float:
        return final_turn_format_reward(_final_text(_final_assistant(trace)))

    @vf.reward
    async def strict_final_json(self, trace: vf.Trace) -> float:
        return strict_final_json_reward(_visible_text(_final_assistant(trace)))

    @vf.reward
    async def reasoning_only_final_penalty_reward(self, trace: vf.Trace) -> float:
        final = _final_assistant(trace)
        if final is None or final.tool_calls or _visible_text(final).strip():
            return 0.0
        reasoning_final = final_turn_format_reward(str(final.reasoning_content or "")) == 1.0
        return self.config.reasoning_only_final_penalty if reasoning_final else 0.0

    @vf.reward
    async def correct_final_label(self, trace: vf.Trace) -> float:
        return final_label_reward(_final_text(_final_assistant(trace)), self.data.answer)

    # --- metrics -----------------------------------------------------------

    @vf.metric
    async def tool_call_count(self, trace: vf.Trace) -> float:
        return float(trace.state.tool_call_count)

    @vf.metric
    async def browse_call_count(self, trace: vf.Trace) -> float:
        return float(trace.state.browse_call_count)

    @vf.metric
    async def browse_url_not_in_output_count(self, trace: vf.Trace) -> float:
        return float(trace.state.browse_url_not_in_output_count)

    @vf.metric
    async def successful_browse_count(self, trace: vf.Trace) -> float:
        return float(trace.state.successful_browse_count)

    @vf.metric
    async def infrastructure_stderr_count(self, trace: vf.Trace) -> float:
        return float(trace.state.infrastructure_stderr_count)


# ----------------------------------------------------------------------------
# Taskset
# ----------------------------------------------------------------------------

_DATA_FIELDS = ("answer", "output", "web_pages_json", "record_id", "rubric_index", "source")


class JTCToolLabelTaskset(vf.Taskset[JTCToolLabelTask, JTCToolLabelTasksetConfig]):
    def load(self) -> Iterable[JTCToolLabelTask]:
        path = Path(self.config.dataset)
        if not path.is_file():
            raise FileNotFoundError(f"JTC tool-label dataset must be a JSONL file: {path}")
        with path.open("r", encoding="utf-8") as handle:
            for idx, line in enumerate(line for line in handle if line.strip()):
                row = json.loads(line)
                output = row.get("output")
                if output is None:
                    raise ValueError(f"{path} row {idx} has no evaluated output")
                data = JTCToolLabelData(
                    idx=idx,
                    prompt=row["prompt"],
                    **{name: row.get(name) for name in _DATA_FIELDS},
                )
                yield JTCToolLabelTask(data, self.config.task)


if __name__ == "__main__":
    # verifiers starts each rollout's tool server as `python -m <this module>`.
    TerminalToolset.run()
