"""Prompt template helpers for bundled training environments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def resolve_resource_path(path: str | Path) -> Path:
    """Resolve a repo-relative runtime asset from an installed wheel.

    Existing absolute paths and paths relative to the current working directory
    continue to win. A missing ``templates/...`` path falls back to the copy
    bundled under ``primebeaker/resources``.
    """
    requested = Path(path)
    if requested.exists() or requested.is_absolute():
        return requested
    packaged = Path(__file__).resolve().parents[1] / "resources" / requested
    return packaged if packaged.exists() else requested


def load_prompt_template(path: str | Path) -> dict[str, Any]:
    with resolve_resource_path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def render_chat_template(
    template: dict[str, Any],
    **values: Any,
) -> list[dict[str, str]]:
    missing = set(template["input_variables"]) - set(values)
    if missing:
        raise ValueError(f"Missing template variables: {sorted(missing)}")
    return [
        {
            "role": message["role"],
            "content": message["content"].format(**values),
        }
        for message in template["chat_template"]
    ]
