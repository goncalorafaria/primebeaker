"""Search-agent environment with direct local-corpus document visits.

This variant deliberately exposes ``search`` and ``visit`` only.  Unlike the
webterminal harness, it does not start or depend on a terminal service.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import verifiers as vf
from verifiers.types import ToolCall

from literegistry_tool_client import FetchClient, ToolClient
from .search_agent_env import (
    FINAL_TOOL_CALL_WARNING,
    SearchAgentEnv,
    _format_search_output,
)


def _format_visit_output(output: Mapping[str, Any]) -> str:
    """Render a local FetchClient response as readable document evidence."""
    data = output.get("data")
    if isinstance(data, Mapping):
        title = str(data.get("title") or "").strip()
        content = data.get("content") or data.get("text") or data.get("markdown")
        if isinstance(content, str) and content.strip():
            heading = f"# {title}\n\n" if title else ""
            return heading + content.strip()
    if isinstance(data, str) and data.strip():
        return data.strip()
    return json.dumps(dict(output), ensure_ascii=False)


class SearchVisitAgentEnv(vf.StatefulToolEnv):
    """Research environment using discovery search plus direct page visits."""

    _TOOL_CALL_COUNT_STATE_KEY = "search_agent_tool_call_count"
    _FINAL_WARNING_SENT_STATE_KEY = "search_agent_final_warning_sent"
    _TOKEN_BUDGET_WARNING_SENT_STATE_KEY = (
        SearchAgentEnv._TOKEN_BUDGET_WARNING_SENT_STATE_KEY
    )
    _TURN_BUDGET_WARNING_SENT_STATE_KEY = (
        SearchAgentEnv._TURN_BUDGET_WARNING_SENT_STATE_KEY
    )

    def __init__(
        self,
        *,
        search_client: ToolClient,
        fetch_client: FetchClient,
        max_turns: int = 10,
        max_tool_calls: int = 20,
        final_tool_call_warning: str = FINAL_TOOL_CALL_WARNING,
        **kwargs: Any,
    ) -> None:
        if not isinstance(search_client, ToolClient):
            raise TypeError("search_client must be a ToolClient")
        if not isinstance(fetch_client, FetchClient):
            raise TypeError("fetch_client must be a FetchClient")
        if max_tool_calls < 1:
            raise ValueError("max_tool_calls must be at least 1")
        self.search_client = search_client
        self.fetch_client = fetch_client
        self.max_tool_calls = int(max_tool_calls)
        self.final_tool_call_warning = final_tool_call_warning
        super().__init__(tools=[], max_turns=max_turns, **kwargs)
        self.add_tool(self.search)
        self.add_tool(self.visit)

    # The search and visit environments share the same call-budget semantics.
    stop_after_tool_budget = SearchAgentEnv.stop_after_tool_budget
    _budget_warning = SearchAgentEnv._budget_warning
    env_response = SearchAgentEnv.env_response

    async def search(self, query: str, num_results: int = 10) -> str:
        """Discover candidate URLs; snippets are leads, not evidence.

        Args:
            query: Focused web search query.
            num_results: Maximum number of candidate results to return.
        """
        output = await self.search_client.execute(query=query, num_results=num_results)
        if not isinstance(output, Mapping):
            raise TypeError("search client returned a non-object response")
        return _format_search_output(output)

    async def visit(self, url: str) -> str:
        """Read a document directly from the configured local search corpus.

        Call this on URLs returned by search before using a source as evidence.

        Args:
            url: A URL returned by the search tool.
        """
        output = await self.fetch_client.execute(url=url)
        if not isinstance(output, Mapping):
            raise TypeError("fetch client returned a non-object response")
        return _format_visit_output(output)

    def update_tool_args(
        self, tool_name: str, tool_args: dict[str, Any],
        messages: vf.Messages, state: vf.State, **kwargs: Any,
    ) -> dict[str, Any]:
        """Direct visits need no hidden terminal or asset-store state."""
        del tool_name, messages, state, kwargs
        return dict(tool_args)
