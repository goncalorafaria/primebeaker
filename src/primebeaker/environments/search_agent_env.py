"""Verifiers search-agent environment with search and web-page inspection."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, cast

import verifiers as vf
from verifiers.types import ToolMessage

from literegistry_tool_client.asset_store import WebAssetStore
from literegistry_tool_client import (
    PlainJsonToolOutputDisplay,
    ToolClient,
    terminal_output_to_tool_content,
)


FINAL_TOOL_CALL_WARNING = """IMPORTANT: This was your last allowed tool call.
Do not call another tool. Give your final report now, with
supporting source URLs, then finish with a fenced JSON block exactly like:
```json
{"answer": "your short canonical answer"}
```"""


def _search_results(output: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    data = output.get("data") or {}
    if not isinstance(data, Mapping):
        return []
    results = data.get("organic") or data.get("results") or []
    if not isinstance(results, list):
        return []
    return [item for item in results if isinstance(item, Mapping)]


def _format_search_output(output: Mapping[str, Any]) -> str:
    results = _search_results(output)
    if not results:
        return json.dumps(dict(output), ensure_ascii=False)
    rendered: list[str] = []
    for result in results:
        title = str(result.get("title") or "Untitled result").strip()
        snippet = str(result.get("snippet") or result.get("description") or "").strip()
        url = str(result.get("link") or result.get("url") or "").strip()
        lines = [f"**{title}**"]
        if snippet:
            lines.append(snippet)
        if url:
            lines.append(f"Source: {url}")
        rendered.append("\n".join(lines))
    return "\n\n".join(rendered)


class SearchAgentEnv(vf.StatefulToolEnv):
    """Stateful research environment backed by concrete async tool clients."""

    _ASSET_STORE_STATE_KEY = "search_agent_asset_store"
    _TOOL_CALL_COUNT_STATE_KEY = "search_agent_tool_call_count"
    _FINAL_WARNING_SENT_STATE_KEY = "search_agent_final_warning_sent"
    _TOKEN_BUDGET_WARNING_SENT_STATE_KEY = "search_agent_token_budget_warning_sent"
    _TURN_BUDGET_WARNING_SENT_STATE_KEY = "search_agent_turn_budget_warning_sent"

    def __init__(
        self,
        *,
        search_client: ToolClient,
        webterminal_client: ToolClient,
        max_turns: int = 10,
        max_tool_calls: int = 20,
        final_tool_call_warning: str = FINAL_TOOL_CALL_WARNING,
        token_budget_warning_fraction: float | None = None,
        turn_budget_warning_fraction: float | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(search_client, ToolClient):
            raise TypeError("search_client must be a ToolClient")
        if not isinstance(webterminal_client, ToolClient):
            raise TypeError("webterminal_client must be a ToolClient")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.search_client = search_client
        self.webterminal_client = webterminal_client
        self.max_tool_calls = int(max_tool_calls)
        self.final_tool_call_warning = final_tool_call_warning
        self.token_budget_warning_fraction = self._validate_warning_fraction(
            token_budget_warning_fraction,
            "token_budget_warning_fraction",
        )
        self.turn_budget_warning_fraction = self._validate_warning_fraction(
            turn_budget_warning_fraction,
            "turn_budget_warning_fraction",
        )
        super().__init__(
            tools=[],
            max_turns=max_turns,
            **kwargs,
        )
        self.add_tool(self.search)
        self.add_tool(self.webterminal, args_to_skip=["asset_store"])

    @staticmethod
    def _validate_warning_fraction(value: float | None, name: str) -> float | None:
        if value is None:
            return None
        fraction = float(value)
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{name} must be in (0, 1]")
        return fraction

    @vf.stop(priority=50)
    async def stop_after_tool_budget(self, state: vf.State) -> bool:
        """Reject an additional tool turn once the tool-call budget is spent.

        The final allowed call itself is executed and receives the warning below;
        this hook then still permits one ordinary assistant final-answer turn.
        """
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return False
        completion = trajectory[-1].get("completion") or []
        if not completion:
            return False
        last_message = completion[-1]
        tool_calls = getattr(last_message, "tool_calls", None)
        return bool(tool_calls) and int(
            state.get(self._TOOL_CALL_COUNT_STATE_KEY) or 0
        ) >= self.max_tool_calls

    def record_webterminal_result(
        self,
        state: vf.State,
        tool_args: Mapping[str, Any],
        output: Mapping[str, Any] | None,
        error: Exception | None,
    ) -> None:
        """Observe a raw WebTerminal result before it is rendered for the model.

        The base search harness deliberately leaves this as a no-op. Specialised
        environments can use the structured terminal result for rewards without
        exposing extra diagnostics in the tool response.
        """
        del state, tool_args, output, error

    def _budget_warning(self, state: vf.State) -> str:
        """Return one-time near-exhaustion notices for the next tool response."""
        warnings: list[str] = []
        token_fraction = getattr(self, "token_budget_warning_fraction", None)
        token_limit = int(getattr(self, "max_total_completion_tokens", -1) or -1)
        if (
            token_fraction is not None
            and token_limit > 0
            and not state.get(self._TOKEN_BUDGET_WARNING_SENT_STATE_KEY)
        ):
            usage = self.get_state_usage(state)
            used_tokens = 0.0 if usage is None else float(usage["output_tokens"])
            remaining_tokens = max(0, token_limit - math.ceil(used_tokens))
            if remaining_tokens <= math.floor(token_limit * token_fraction):
                state[self._TOKEN_BUDGET_WARNING_SENT_STATE_KEY] = True
                warnings.append(
                    "[TOKEN BUDGET EXCEEDED] You are in the final 10% of the total "
                    f"completion-token budget ({remaining_tokens:,} of {token_limit:,} "
                    "tokens remain). You should now do your best to compose the final "
                    "answer."
                )

        turn_fraction = getattr(self, "turn_budget_warning_fraction", None)
        max_turns = int(getattr(self, "max_turns", -1) or -1)
        if (
            turn_fraction is not None
            and max_turns > 0
            and not state.get(self._TURN_BUDGET_WARNING_SENT_STATE_KEY)
        ):
            turns_used = len(state.get("trajectory") or [])
            remaining_turns = max(0, max_turns - turns_used)
            threshold = max(1, math.floor(max_turns * turn_fraction))
            if remaining_turns <= threshold:
                state[self._TURN_BUDGET_WARNING_SENT_STATE_KEY] = True
                warnings.append(
                    "IMPORTANT: You are in the final 10% of the turn budget "
                    f"({remaining_turns} of {max_turns} assistant turns remain). "
                    "Finish gathering essential evidence and prepare your final answer."
                )
        return "\n\n".join(warnings)

    async def env_response(
        self, messages: vf.Messages, state: vf.State, **kwargs: Any
    ) -> vf.Messages:
        """Execute calls, cap the total at ``max_tool_calls``, and warn on the last."""
        last_message = cast(vf.AssistantMessage, messages[-1])
        assert last_message.tool_calls is not None
        tool_messages: list[ToolMessage] = []
        for tool_call in last_message.tool_calls:
            tool_call_id = tool_call.id
            executed = int(state.get(self._TOOL_CALL_COUNT_STATE_KEY) or 0)
            if executed >= self.max_tool_calls:
                # The final admissible call already received the warning as an
                # append to its tool result. Do not create a fresh response
                # after the tool budget is exhausted.
                return tool_messages
            try:
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)
                if not isinstance(tool_args, dict):
                    raise ValueError("Tool arguments must be a JSON object")
                tool_args = self.update_tool_args(
                    tool_name, tool_args, messages, state, **kwargs
                )
            except Exception as exc:
                if self._should_stop_for_error(exc):
                    raise vf.ToolParseError from exc
                tool_messages.append(
                    ToolMessage(
                        role="tool",
                        content=self.error_formatter(exc),
                        tool_call_id=tool_call_id,
                    )
                )
                continue

            call_number = executed + 1
            state[self._TOOL_CALL_COUNT_STATE_KEY] = call_number
            raw_webterminal_output: Mapping[str, Any] | None = None
            try:
                if tool_name == "webterminal":
                    output = await self.webterminal_client.execute(
                        asset_store=tool_args["asset_store"],
                        command=tool_args["command"],
                    )
                    if not isinstance(output, Mapping):
                        raise TypeError("webterminal client returned a non-object response")
                    raw_webterminal_output = dict(output)
                    tool_message = ToolMessage(
                        role="tool",
                        content=terminal_output_to_tool_content(raw_webterminal_output),
                        tool_call_id=tool_call_id,
                    )
                else:
                    tool_message = await self.call_tool(
                        tool_name,
                        tool_args,
                        tool_call_id,
                    )
            except Exception as exc:
                if tool_name == "webterminal":
                    self.record_webterminal_result(
                        state,
                        tool_args,
                        output=None,
                        error=exc,
                    )
                if self._should_stop_for_error(exc):
                    raise vf.ToolCallError from exc
                tool_message = ToolMessage(
                    role="tool",
                    content=self.error_formatter(exc),
                    tool_call_id=tool_call_id,
                )
            else:
                if tool_name == "webterminal":
                    self.record_webterminal_result(
                        state,
                        tool_args,
                        output=raw_webterminal_output,
                        error=None,
                    )
            budget_warning = self._budget_warning(state)
            if budget_warning:
                tool_message.content = f"{tool_message.content}\n\n{budget_warning}"
            if call_number == self.max_tool_calls:
                state[self._FINAL_WARNING_SENT_STATE_KEY] = True
                tool_message.content = (
                    f"{tool_message.content}\n\n{self.final_tool_call_warning}"
                )
            tool_messages.append(tool_message)
        return tool_messages

    async def search(self, query: str, num_results: int = 10) -> str:
        """Find candidate web sources for a query.

        Use this for discovery. The returned titles and snippets help select
        candidate URLs, but must not be treated as verified source evidence.

        Args:
            query: Focused web search query.
            num_results: Maximum number of candidate results to return.
        """
        output = await self.search_client.execute(
            query=query,
            num_results=num_results,
        )
        if not isinstance(output, Mapping):
            raise TypeError("search client returned a non-object response")
        return _format_search_output(output)

    async def webterminal(self, command: str, asset_store: WebAssetStore) -> str:
        """Open a source URL and inspect its page contents with a terminal pipeline.

        Use this after search: a search snippet is only a lead. Inspect the
        source itself with commands such as `browse <url> | rg -n -i pattern`.

        Args:
            command: A `browse <url>` command, optionally followed by a pipeline.
            asset_store: Hidden rollout-local cache injected by the environment.
        """
        output = await self.webterminal_client.execute(
            asset_store=asset_store,
            command=command,
        )
        if not isinstance(output, dict):
            raise TypeError("webterminal client returned a non-object response")
        return terminal_output_to_tool_content(output)

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict:
        """Inject hidden state needed by WebTerminal for this rollout."""
        del messages, kwargs
        updated = dict(tool_args)
        if tool_name == "webterminal":
            updated["asset_store"] = state.setdefault(
                self._ASSET_STORE_STATE_KEY,
                WebAssetStore(),
            )
        return updated
