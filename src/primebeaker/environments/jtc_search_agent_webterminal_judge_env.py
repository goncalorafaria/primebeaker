"""Judge-scored search-agent environment with a browser-aware web terminal."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import verifiers as vf

from primebeaker.client import (
    FetchClient,
    SearchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
)

from .jtc_search_agent_env import _has_tool_calls, _load_dataset, _message_content
from .jtc_tool_label_env import (
    _error_output,
    _exception_is_model_caused,
    _tool_stderr_is_model_error,
)
from .jtc_search_visit_agent_judge_env import (
    DEFAULT_JUDGE_RUBRIC_TEMPLATE,
    GatewayJudgeClient,
    JudgeAnswerRubric,
)
from .search_agent_env import SearchAgentEnv


class WebTerminalJudgeSearchAgentEnv(SearchAgentEnv):
    """WebTerminal search harness with a judge plus terminal-only penalties."""

    _MODEL_CAUSED_TERMINAL_STDERR_COUNT_STATE_KEY = (
        "search_agent_model_caused_terminal_stderr_count"
    )

    def __init__(
        self,
        *,
        judge_rubric: JudgeAnswerRubric,
        tool_stderr_penalty: float = 0.05,
        missing_final_answer_penalty: float = -1.0,
        **kwargs: Any,
    ) -> None:
        if tool_stderr_penalty < 0:
            raise ValueError("tool_stderr_penalty must be non-negative")
        if missing_final_answer_penalty > 0:
            raise ValueError("missing_final_answer_penalty must be non-positive")
        self.tool_stderr_penalty = float(tool_stderr_penalty)
        self.missing_final_answer_penalty = float(missing_final_answer_penalty)

        async def tool_stderr_penalty_reward(state: vf.State) -> float:
            """Penalize only model-caused WebTerminal stderr results."""
            count = max(
                int(state.get(self._MODEL_CAUSED_TERMINAL_STDERR_COUNT_STATE_KEY) or 0),
                0,
            )
            return -self.tool_stderr_penalty * count

        async def missing_final_answer_penalty_reward(completion: list[Any]) -> float:
            """Penalize an exhausted or failed trajectory without a final answer."""
            if not completion or _has_tool_calls(completion[-1]):
                return self.missing_final_answer_penalty
            return (
                0.0
                if _message_content(completion[-1]).strip()
                else self.missing_final_answer_penalty
            )

        # Keep the judge rubric object itself so its teardown remains registered.
        judge_rubric.add_reward_func(tool_stderr_penalty_reward)
        judge_rubric.add_reward_func(missing_final_answer_penalty_reward)
        super().__init__(rubric=judge_rubric, **kwargs)

    def record_webterminal_result(
        self,
        state: vf.State,
        tool_args: Mapping[str, Any],
        output: Mapping[str, Any] | None,
        error: Exception | None,
    ) -> None:
        """Count model-caused stderr, excluding fetch and service failures."""
        if output is None:
            if error is None:
                return
            raw_output = _error_output("terminal", error)
            request_model_caused = _exception_is_model_caused(error)
        else:
            raw_output = dict(output)
            request_model_caused = True
        if _tool_stderr_is_model_error(
            "terminal",
            tool_args,
            raw_output,
            request_model_caused=request_model_caused,
        ):
            state[self._MODEL_CAUSED_TERMINAL_STDERR_COUNT_STATE_KEY] = (
                int(
                    state.get(
                        self._MODEL_CAUSED_TERMINAL_STDERR_COUNT_STATE_KEY,
                    )
                    or 0
                )
                + 1
            )


def load_environment(
    dataset: str,
    judge_model_path: str,
    split: str = "train",
    search_server_url: str = "http://127.0.0.1:1212/search",
    terminal_server_url: str = "http://127.0.0.1:1212/terminal",
    jina_reader_url: str = "https://r.jina.ai",
    local_search_model_path: str | None = None,
    judge_server_url: str | None = None,
    judge_service_model_path: str = "judge",
    judge_timeout: float = 240,
    judge_max_retries: int = 3,
    judge_rubric_template: str = DEFAULT_JUDGE_RUBRIC_TEMPLATE,
    timeout: float = 65,
    max_retries: int = 3,
    terminal_truncation: int | None = 2000,
    webterminal_max_content_length: int | None = 64000,
    max_turns: int | None = None,
    max_tool_calls: int = 150,
    tool_stderr_penalty: float = 0.05,
    missing_final_answer_penalty: float = -1.0,
    token_budget_warning_fraction: float | None = 0.1,
    turn_budget_warning_fraction: float | None = 0.1,
    template_path: str = "templates/search-agent.json",
    **_: Any,
) -> SearchAgentEnv:
    """Build the WebTerminal search agent with a gateway-routed judge reward."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")
    resolved_max_turns = max_turns or max_tool_calls + 1
    if resolved_max_turns < max_tool_calls + 1:
        raise ValueError("max_turns must reserve a final-answer turn")
    if judge_server_url is None:
        base_search_url = search_server_url.rstrip("/")
        if not base_search_url.endswith("/search"):
            raise ValueError("judge_server_url is required when search_server_url is not a /search URL")
        judge_server_url = base_search_url[:-len("/search")] + "/judge"
    local_search_model_path = local_search_model_path or None
    terminal = TerminalExecutionClient(
        terminal_server_url,
        timeout=timeout,
        max_retries=max_retries,
        truncation=terminal_truncation,
    )
    return WebTerminalJudgeSearchAgentEnv(
        dataset=_load_dataset(dataset, split, template_path),
        search_client=SearchClient(
            search_server_url,
            model_path=local_search_model_path,
            timeout=timeout,
            max_retries=max_retries,
        ),
        webterminal_client=WebTerminalExecutionClient(
            terminal,
            FetchClient(
                jina_reader_url,
                timeout=timeout,
                max_retries=max_retries,
                max_content_length=webterminal_max_content_length,
                local_search_server_url=search_server_url,
                local_search_model_path=local_search_model_path,
            ),
        ),
        max_turns=resolved_max_turns,
        max_tool_calls=max_tool_calls,
        token_budget_warning_fraction=token_budget_warning_fraction,
        turn_budget_warning_fraction=turn_budget_warning_fraction,
        tool_stderr_penalty=tool_stderr_penalty,
        missing_final_answer_penalty=missing_final_answer_penalty,
        judge_rubric=JudgeAnswerRubric(
            judge_client=GatewayJudgeClient(
                judge_server_url,
                service_model_path=judge_service_model_path,
                timeout=judge_timeout,
                max_retries=judge_max_retries,
            ),
            judge_model_path=judge_model_path,
            rubric_template=judge_rubric_template,
        ),
    )


__all__ = ["WebTerminalJudgeSearchAgentEnv", "load_environment"]
