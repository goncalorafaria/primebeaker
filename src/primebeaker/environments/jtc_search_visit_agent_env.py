"""Legacy Verifiers factory for the direct-visit search-agent environment."""

from __future__ import annotations

from typing import Any

import verifiers as vf

from primebeaker.client import FetchClient, SearchClient
from .jtc_search_agent_env import _load_dataset, answer_exact_match, final_json_format
from .search_visit_agent_env import SearchVisitAgentEnv


def load_environment(
    dataset: str,
    split: str = "train",
    search_server_url: str = "http://127.0.0.1:1212/search",
    local_search_model_path: str = "localsearch:bc-v2-72k",
    timeout: float = 65,
    max_retries: int = 3,
    visit_max_content_length: int | None = 64000,
    max_turns: int | None = None,
    max_tool_calls: int = 150,
    template_path: str = "templates/search-agent-visit.json",
    **_: Any,
) -> SearchVisitAgentEnv:
    """Build search + direct ``visit(url)`` over a named local corpus."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")
    resolved_max_turns = max_turns or max_tool_calls + 1
    if resolved_max_turns < max_tool_calls + 1:
        raise ValueError("max_turns must reserve a final-answer turn")
    return SearchVisitAgentEnv(
        dataset=_load_dataset(dataset, split, template_path),
        search_client=SearchClient(
            search_server_url, model_path=local_search_model_path,
            timeout=timeout, max_retries=max_retries,
        ),
        fetch_client=FetchClient(
            local_search_server_url=search_server_url,
            local_search_model_path=local_search_model_path,
            timeout=timeout,
            max_retries=max_retries,
            max_content_length=visit_max_content_length,
        ),
        max_turns=resolved_max_turns,
        max_tool_calls=max_tool_calls,
        rubric=vf.Rubric(funcs=[final_json_format, answer_exact_match]),
    )
