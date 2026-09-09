"""Core logic for Qwen/Tapir multi-turn terminal-tool RL.

The model can either return a final JSON verdict immediately, or issue one
native Qwen ``terminal`` tool call.  The Verifiers wrapper executes commands
against the hidden judged output and supplies the result on subsequent turns.
"""

from __future__ import annotations

import json
from typing import Any

from .tool_protocol import (
    parse_qwen_tool_calls,
    strip_qwen_special_blocks,
)
from .jtc_label_reward import normalize_label
from primebeaker.runtime.json_utils import extract_json_from_response


DEFAULT_TERMINAL_SERVER_URL = "http://127.0.0.1:1212/terminal"
FINAL_WARNING_TEMPLATE_PATH = "templates/multistep-final-warning-harmony.json"


def normalize_truncation_limit(limit: Any) -> int | None:
    """Coerce optional TOML output-truncation values."""
    if limit is None or limit == "None":
        return None
    return int(limit)


def extract_terminal_command(text: str) -> str | None:
    """Return the command from exactly one valid native terminal tool call."""
    calls = parse_qwen_tool_calls(text or "")
    if len(calls) != 1:
        return None
    call = calls[0]
    if call.get("name") != "terminal":
        return None
    command = (call.get("arguments") or {}).get("command")
    if not isinstance(command, str):
        return None
    command = command.strip()
    return command or None


def _final_object(text: str) -> dict[str, str] | None:
    """Parse a final verdict using the workflow's JSON extraction rules.

    extract_json_from_response accepts raw JSON, fenced JSON, and JSON
    embedded in surrounding prose. The verifier still enforces the exact
    feedback/label object schema after extraction.
    """
    cleaned = strip_qwen_special_blocks(text or "")
    if not cleaned:
        return None
    payload = extract_json_from_response(cleaned)
    if not isinstance(payload, dict) or set(payload) != {"feedback", "label"}:
        return None
    feedback, label = payload["feedback"], payload["label"]
    if not isinstance(feedback, str) or not isinstance(label, str):
        return None
    feedback, label = feedback.strip(), label.strip()
    if not feedback or not label:
        return None
    return {"feedback": feedback, "label": label}


def strict_final_object(text: str) -> dict[str, str] | None:
    """Return a final verdict only when the visible final is one JSON object.

    Qwen thinking and native tool markup are allowed before the visible final
    because ``strip_qwen_special_blocks`` removes them. Unlike
    :func:`_final_object`, this deliberately does *not* recover JSON from
    fences or surrounding prose: it is a format-quality signal, not the
    environment's validity gate.
    """
    cleaned = strip_qwen_special_blocks(text or "").strip()
    if not cleaned:
        return None
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or set(payload) != {"feedback", "label"}:
        return None
    feedback, label = payload["feedback"], payload["label"]
    if not isinstance(feedback, str) or not isinstance(label, str):
        return None
    feedback, label = feedback.strip(), label.strip()
    if not feedback or not label:
        return None
    return {"feedback": feedback, "label": label}


def extract_final_label(text: str) -> str | None:
    """Return a normalized label from a valid final JSON answer."""
    final = _final_object(text)
    return normalize_label(final["label"]) if final else None


def classify_response(text: str) -> str:
    """Classify a completion as ``terminal``, ``final``, or ``invalid``."""
    if extract_terminal_command(text):
        return "terminal"
    if _final_object(text):
        return "final"
    return "invalid"


def first_turn_format_reward(text: str) -> float:
    """Reward either a valid terminal action or a valid direct JSON verdict."""
    return 1.0 if classify_response(text) in {"terminal", "final"} else 0.0


def final_turn_format_reward(text: str) -> float:
    """Reward exactly one complete JSON feedback/label verdict."""
    return 1.0 if _final_object(text) else 0.0


def strict_final_json_reward(text: str) -> float:
    """Reward a bare final JSON object, without changing validity semantics."""
    return 1.0 if strict_final_object(text) else 0.0


def final_label_reward(text: str, answer: Any) -> float:
    """Reward an exactly matching normalized final label."""
    predicted = extract_final_label(text)
    expected = normalize_label(answer)
    return 1.0 if predicted is not None and predicted == expected else 0.0
