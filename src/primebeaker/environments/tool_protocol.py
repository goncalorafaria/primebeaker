"""Wire-format parsing and tool schemas used by verifier environments.

This is intentionally not a model renderer. Prime-RL owns prompt/completion
rendering; environments only need to recover raw tool calls and advertise
their callable tools.
"""

from __future__ import annotations

import json
import re
from typing import Any

from primebeaker.client.submission import SUBMIT_TOOL_SPEC


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
_REDACTED_THINK_RE = re.compile(
    r"<redacted_thinking>(.*?)</redacted_thinking>", re.DOTALL | re.IGNORECASE
)
_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)
_TOOL_CALL_JSON_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_TOOL_CALL_XML_RE = re.compile(
    r"<tool_call>\s*<function=(?P<name>[^\s>]+)>\s*(?P<body>.*?)\s*</function>\s*</tool_call>",
    re.DOTALL,
)
_TOOL_PARAM_RE = re.compile(
    r"<parameter=(?P<key>[^>]+)>\s*(?P<value>.*?)\s*</parameter>", re.DOTALL
)
_TOOL_CALL_STRIP_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_QWEN_END_TOKEN_RE = re.compile(r"<\|(?:im_end|endoftext|return)\|>", re.IGNORECASE)


def parse_qwen_thinking(text: str) -> str:
    for pattern in (_THINK_RE, _REDACTED_THINK_RE):
        match = pattern.search(text or "")
        if match:
            return match.group(1).strip()
    close_tag = _THINK_CLOSE_RE.search(text or "")
    return (text or "")[: close_tag.start()].strip() if close_tag else ""


def _coerce_arguments(arguments: Any) -> dict[str, Any]:
    if arguments is None:
        return {}
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments.strip())
        except json.JSONDecodeError:
            return {"raw": arguments}
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    return {"raw": arguments}


def parse_qwen_tool_calls(text: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for block in _TOOL_CALL_JSON_RE.findall(text or ""):
        try:
            payload = json.loads(block)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict) and payload.get("name"):
            calls.append({
                "name": str(payload["name"]),
                "arguments": _coerce_arguments(payload.get("arguments")),
            })
    if calls:
        return calls
    for match in _TOOL_CALL_XML_RE.finditer(text or ""):
        arguments: dict[str, Any] = {}
        for parameter in _TOOL_PARAM_RE.finditer(match.group("body")):
            value = parameter.group("value").strip()
            try:
                arguments[parameter.group("key").strip()] = json.loads(value)
            except json.JSONDecodeError:
                arguments[parameter.group("key").strip()] = value
        calls.append({"name": match.group("name").strip(), "arguments": arguments})
    return calls


def strip_qwen_special_blocks(text: str) -> str:
    cleaned = _THINK_RE.sub("", text or "")
    cleaned = _REDACTED_THINK_RE.sub("", cleaned)
    cleaned = _TOOL_CALL_STRIP_RE.sub("", cleaned)
    cleaned = _QWEN_END_TOKEN_RE.sub("", cleaned)
    if "</think>" in cleaned.lower():
        cleaned = re.split(r"</think>", cleaned, flags=re.IGNORECASE)[-1]
    return cleaned.strip()


GPTOSS_PYTHON_TOOL: dict[str, Any] = {
    "name": "python",
    "description": "Execute Python code and return stdout/stderr.",
    "parameters": {
        "type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]
    },
}
GPTOSS_TERMINAL_TOOL: dict[str, Any] = {
    "name": "terminal",
    "description": (
        "Run a restricted terminal command/pipeline against the evaluated output "
        "supplied on standard input, then return stdout, stderr, and the exit code."
    ),
    "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    },
}
GPTOSS_WEBTERMINAL_TOOL: dict[str, Any] = {
    "name": "webterminal",
    "description": (
        "Run a restricted terminal command/pipeline against the evaluated output "
        "or web content. Use `browse <url>` to fetch a web page into standard "
        "input, or `cat <asset-id>` to reuse a previously stored web asset."
    ),
    "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]
    },
}
GPTOSS_SEARCH_TOOL: dict[str, Any] = {
    "name": "search",
    "description": "Search the web and return matching links and snippets.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "num_results": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["query"],
    },
}
GPTOSS_SUBMIT_TOOL = SUBMIT_TOOL_SPEC


TOOL_SPECS: dict[str, dict[str, Any]] = {
    "python": GPTOSS_PYTHON_TOOL,
    "terminal": GPTOSS_TERMINAL_TOOL,
    "webterminal": GPTOSS_WEBTERMINAL_TOOL,
    "search": GPTOSS_SEARCH_TOOL,
    "submit": GPTOSS_SUBMIT_TOOL,
}

def _tool_arguments_usable(tool_name: str, arguments: Any) -> bool:
    """Return whether every advertised required argument is populated."""
    if tool_name not in TOOL_SPECS or not isinstance(arguments, dict):
        return False
    required = TOOL_SPECS[tool_name]["parameters"].get("required") or []
    return all(
        arguments.get(name) is not None
        and (not isinstance(arguments.get(name), str) or arguments[name].strip())
        for name in required
    )
