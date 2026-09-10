from __future__ import annotations

from pathlib import Path
import tomllib

from primebeaker.environments.registry import ENVIRONMENTS, load_environment_module
from primebeaker.environments.tool_protocol import (
    GPTOSS_TERMINAL_TOOL,
    GPTOSS_WEBTERMINAL_TOOL,
    parse_qwen_tool_calls,
)
from primebeaker.runtime.json_utils import extract_json_from_response


ROOT = Path(__file__).resolve().parents[2]
VERIFIERS_REQUIREMENT = "verifiers[harbor]==0.2.1"


def test_every_registered_environment_has_a_factory() -> None:
    modules = {load_environment_module(name) for name in ENVIRONMENTS}

    assert len(modules) == 16
    assert all(module.__name__.startswith("primebeaker.environments.") for module in modules)
    assert all(callable(getattr(module, "load_environment", None)) for module in modules)


def test_local_protocol_adapter_preserves_environment_wire_contract() -> None:
    text = (
        '<think>inspect</think><tool_call>{"name":"terminal",'
        '"arguments":{"command":"rg -n \\"error\\""}}</tool_call>'
    )
    assert parse_qwen_tool_calls(text) == [
        {"name": "terminal", "arguments": {"command": 'rg -n "error"'}}
    ]
    assert "standard input" in GPTOSS_TERMINAL_TOOL["description"]
    assert "browse <url>" in GPTOSS_WEBTERMINAL_TOOL["description"]
    assert "cat <asset-id>" in GPTOSS_WEBTERMINAL_TOOL["description"]


def test_json_extraction_is_local_and_preserves_permissive_routing() -> None:
    expected = {"feedback": "ok", "label": "pass"}
    assert extract_json_from_response(expected) is expected
    assert extract_json_from_response(f"prefix {expected!r}") is None
    assert extract_json_from_response('prefix {"feedback":"ok","label":"pass"}') == expected


def test_runtime_uses_only_released_pypi_packages() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    runtime = metadata["project"]["optional-dependencies"]["runtime"]
    verifier = next(item for item in runtime if item.startswith("verifiers[harbor]"))
    assert verifier == VERIFIERS_REQUIREMENT
    assert all("git+" not in requirement and " @ " not in requirement for requirement in runtime)

    dockerfile = (
        ROOT / "src/primebeaker/images/Dockerfile.runtime"
    ).read_text(encoding="utf-8")
    assert "primebeaker[runtime]==" in dockerfile
    assert "jtc[harness]==" in dockerfile
    assert "COPY " not in dockerfile
    assert "git+" not in dockerfile
    assert " -e " not in dockerfile
    assert not (ROOT / "src/primebeaker/images/Dockerfile.full").exists()
