"""Registry for all bundled Prime-RL environment factories."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType


ENVIRONMENTS: dict[str, str] = {
    "articulated-harness": "articulated_harness_env",
    "code-tool-label": "jtc_code_tool_label_env",
    "label": "jtc_label_env",
    "multiple-choice-accuracy": "jtc_multiple_choice_accuracy_env",
    "podman-terminal": "podman_terminal_env",
    "reward-model": "jtc_reward_model_env",
    "rubrichub-judge": "jtc_rubrichub_judge_env",
    "search-agent-dataset": "jtc_search_agent_env",
    "search-agent-visit": "jtc_search_agent_visit_env",
    "search-agent-visit-judge": "jtc_search_agent_visit_judge_env",
    "search-agent-webterminal": "jtc_search_agent_webterminal_env",
    "search-agent-webterminal-judge": "jtc_search_agent_webterminal_judge_env",
    "search-visit-agent-dataset": "jtc_search_visit_agent_env",
    "search-visit-agent-judge": "jtc_search_visit_agent_judge_env",
    "terminal-tool-label": "jtc_terminal_tool_label_env",
    "tool-label": "jtc_tool_label_env",
}

# Existing identifiers remain aliases so current Prime-RL TOMLs do not break.
ENVIRONMENTS.update(
    {
        "podman-terminal-verifier": "podman_terminal_env",
        "jtc-code-tool-label": "jtc_code_tool_label_env",
        "jtc-label": "jtc_label_env",
        "jtc-multiple-choice-accuracy": "jtc_multiple_choice_accuracy_env",
        "jtc-reward-model": "jtc_reward_model_env",
        "jtc-rubrichub-judge": "jtc_rubrichub_judge_env",
        "jtc-search-agent": "jtc_search_agent_env",
        "jtc-search-agent-visit": "jtc_search_agent_visit_env",
        "jtc-search-agent-visit-judge": "jtc_search_agent_visit_judge_env",
        "jtc-search-agent-webterminal": "jtc_search_agent_webterminal_env",
        "jtc-search-agent-webterminal-judge": "jtc_search_agent_webterminal_judge_env",
        "jtc-search-visit-agent": "jtc_search_visit_agent_env",
        "jtc-search-visit-agent-judge": "jtc_search_visit_agent_judge_env",
        "jtc-terminal-tool-label": "jtc_terminal_tool_label_env",
        "jtc-tool-label": "jtc_tool_label_env",
    }
)


def environment_names() -> tuple[str, ...]:
    return tuple(sorted(ENVIRONMENTS))


def load_environment_module(name: str) -> ModuleType:
    normalized = name.removesuffix(".py").replace("_", "-")
    module_name = ENVIRONMENTS.get(normalized)
    if module_name is None and name in ENVIRONMENTS.values():
        module_name = name
    if module_name is None:
        available = ", ".join(environment_names())
        raise KeyError(f"unknown environment {name!r}; choose one of: {available}")
    return import_module(f"primebeaker.environments.{module_name}")
