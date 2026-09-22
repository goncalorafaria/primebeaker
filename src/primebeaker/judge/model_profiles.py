"""Load and validate model-specific verifier workflow profiles."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ModelProfile:
    model: str
    source_path: Path
    rubric_mode: str
    prompt_template_path: str
    max_turns: int
    max_tool_calls: int
    max_tokens: int
    timeout: float
    max_retries: int
    rubric_max_retries: int
    rollout_timeout: float
    evaluation_timeout: float
    tools: tuple[str, ...]
    tool_server_url: str
    local_search_model_path: str | None
    temperature: float


def _require_number(payload: dict[str, Any], key: str, default: float) -> float:
    value = payload.get(key, default)
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"profile field {key!r} must be a positive number")
    return float(value)


def _require_int(payload: dict[str, Any], key: str, default: int, minimum: int) -> int:
    value = payload.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"profile field {key!r} must be an integer >= {minimum}")
    return value


def load_model_profile(model: str, config: Any) -> ModelProfile:
    """Load the sole profile whose ``model`` field exactly matches ``model``."""
    directory = Path(config.model_profiles_dir)
    if not directory.is_dir():
        raise ValueError(f"model profiles directory does not exist: {directory}")

    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON profile {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"model profile {path} must contain a JSON object")
        if payload.get("model") == model:
            matches.append((path, payload))

    if not matches:
        raise ValueError(f"no model profile found for {model!r} in {directory}")
    if len(matches) > 1:
        paths = ", ".join(str(path) for path, _ in matches)
        raise ValueError(f"multiple model profiles found for {model!r}: {paths}")

    path, payload = matches[0]
    template = payload.get("prompt_template_path", config.prompt_template_path)
    if not isinstance(template, str) or not template:
        raise ValueError("profile field 'prompt_template_path' must be a non-empty path")
    template_path = Path(template)
    if not template_path.is_absolute():
        template_path = path.parent / template_path
    if not template_path.is_file():
        raise ValueError(f"profile template does not exist: {template_path}")

    raw_tools = payload.get("tools", list(config.tools))
    if isinstance(raw_tools, str):
        tools = tuple(item.strip() for item in raw_tools.split(",") if item.strip())
    elif isinstance(raw_tools, list) and all(isinstance(item, str) for item in raw_tools):
        tools = tuple(item.strip() for item in raw_tools if item.strip())
    else:
        raise ValueError("profile field 'tools' must be a list of names or CSV string")

    max_turns = _require_int(
        payload, "max_turns", config.max_tool_calls + 1, minimum=1
    )
    default_calls = max_turns - 1
    max_tool_calls = _require_int(
        payload, "max_tool_calls", default_calls, minimum=0
    )
    if max_tool_calls >= max_turns:
        raise ValueError("profile max_tool_calls must be smaller than max_turns")
    tool_server_url = payload.get("tool_server_url", config.tool_server_url)
    if not isinstance(tool_server_url, str) or not tool_server_url:
        raise ValueError("profile field 'tool_server_url' must be a non-empty URL")
    local_search_model_path = payload.get("local_search_model_path")
    if local_search_model_path is not None and (
        not isinstance(local_search_model_path, str)
        or not local_search_model_path.strip()
    ):
        raise ValueError(
            "profile field 'local_search_model_path' must be a non-empty string"
        )
    temperature = payload.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise ValueError("profile field 'temperature' must be a number")
    rubric_mode = payload.get("rubric_mode", "per_rubric")
    if rubric_mode not in {"per_rubric", "grouped_rubrics"}:
        raise ValueError(
            "profile field 'rubric_mode' must be 'per_rubric' or "
            "'grouped_rubrics'"
        )

    return ModelProfile(
        model=model,
        source_path=path,
        rubric_mode=rubric_mode,
        prompt_template_path=str(template_path),
        max_turns=max_turns,
        max_tool_calls=max_tool_calls,
        max_tokens=_require_int(payload, "max_tokens", config.max_tokens, minimum=1),
        timeout=_require_number(payload, "timeout", config.timeout),
        max_retries=_require_int(payload, "max_retries", config.max_retries, minimum=0),
        rubric_max_retries=_require_int(
            payload,
            "rubric_max_retries",
            getattr(config, "rubric_max_retries", 1),
            minimum=0,
        ),
        rollout_timeout=_require_number(
            payload, "rollout_timeout", config.rollout_timeout
        ),
        evaluation_timeout=_require_number(
            payload,
            "evaluation_timeout",
            getattr(config, "evaluation_timeout", config.rollout_timeout),
        ),
        tools=tools,
        tool_server_url=tool_server_url,
        local_search_model_path=local_search_model_path,
        temperature=float(temperature),
    )
