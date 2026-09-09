"""Backward-compatible legacy factory for the search + webterminal harness.

Prefer the explicit ``jtc-search-agent-webterminal-env`` ID for new TOMLs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
import verifiers as vf

from primebeaker.runtime.templates import resolve_resource_path
from literegistry_tool_client import (
    FetchClient,
    SearchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
)
from .search_agent_env import SearchAgentEnv


def _render_prompt(question: str, template_path: str) -> list[dict[str, str]]:
    template = json.loads(resolve_resource_path(template_path).read_text(encoding="utf-8"))
    messages = template.get("chat_template") if isinstance(template, dict) else None
    if not isinstance(messages, list):
        raise ValueError(f"{template_path}: expected chat_template list")
    return [
        {"role": str(message["role"]), "content": str(message["content"]).format(question=question)}
        for message in messages
        if isinstance(message, dict) and isinstance(message.get("role"), str)
        and isinstance(message.get("content"), str)
    ]


def _with_runtime_prompts(rows: list[dict[str, Any]], template_path: str) -> list[dict[str, Any]]:
    for row in rows:
        if isinstance(row.get("prompt"), list) and row["prompt"]:
            continue
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("search-agent task needs prompt or non-empty question")
        row["prompt"] = _render_prompt(question.strip(), template_path)
    return rows


def _load_dataset(dataset: str, split: str, template_path: str = "templates/search-agent.json") -> Dataset:
    path = Path(dataset)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return Dataset.from_list(_with_runtime_prompts(rows, template_path))
    return Dataset.from_list(_with_runtime_prompts(list(load_dataset(dataset, split=split)), template_path))


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _has_tool_calls(message: Any) -> bool:
    """Return whether an assistant turn invokes a tool rather than answers."""
    if isinstance(message, dict):
        return bool(message.get("tool_calls"))
    return bool(getattr(message, "tool_calls", None))


def extract_final_answer(text: str) -> str:
    """Extract the ``answer`` only from the response's final fenced JSON block."""
    text = text.strip()
    if not text:
        return ""
    match = re.search(
        r"```json\s*(\{.*?\})\s*```\s*\Z", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return ""
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ""
    if not isinstance(payload, dict) or set(payload) != {"answer"}:
        return ""
    answer = payload.get("answer")
    return answer.strip() if isinstance(answer, str) else ""


def _casefolded(text: str) -> str:
    return text.strip().casefold()


async def final_json_format(completion: list[Any]) -> float:
    """Reward only a non-tool final turn with a valid JSON ``answer`` object."""
    if not completion or _has_tool_calls(completion[-1]):
        return 0.0
    return 1.0 if extract_final_answer(_message_content(completion[-1])) else 0.0


async def answer_exact_match(completion: list[Any], answer: Any) -> float:
    """Match the final JSON answer to ground truth, ignoring capitalization only."""
    if (
        not completion
        or _has_tool_calls(completion[-1])
        or not isinstance(answer, str)
        or not answer.strip()
    ):
        return 0.0
    final = extract_final_answer(_message_content(completion[-1]))
    return 1.0 if final and _casefolded(final) == _casefolded(answer) else 0.0


def load_environment(
    dataset: str,
    split: str = "train",
    search_server_url: str = "http://127.0.0.1:1212/search",
    terminal_server_url: str = "http://127.0.0.1:1212/terminal",
    jina_reader_url: str = "https://r.jina.ai",
    local_search_model_path: str | None = None,
    timeout: float = 65,
    max_retries: int = 3,
    terminal_truncation: int | None = 2000,
    max_turns: int | None = None,
    max_tool_calls: int = 20,
    template_path: str = "templates/search-agent.json",
) -> SearchAgentEnv:
    """Construct the TOML-configured SEC search environment.

    Reserve one assistant turn after the final allowed tool call so the agent
    can act on its final-turn warning and provide its answer.
    """
    local_search_model_path = local_search_model_path or None
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")
    resolved_max_turns = max_turns or max_tool_calls + 1
    if resolved_max_turns < max_tool_calls + 1:
        raise ValueError(
            "max_turns must allow every tool call plus one final-answer turn "
            f"(at least {max_tool_calls + 1})"
        )
    terminal = TerminalExecutionClient(
        terminal_server_url,
        timeout=timeout,
        max_retries=max_retries,
        truncation=terminal_truncation,
    )
    return SearchAgentEnv(
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
                local_search_server_url=search_server_url,
                local_search_model_path=local_search_model_path,
            ),
        ),
        max_turns=resolved_max_turns,
        max_tool_calls=max_tool_calls,
        rubric=vf.Rubric(funcs=[final_json_format, answer_exact_match]),
    )
