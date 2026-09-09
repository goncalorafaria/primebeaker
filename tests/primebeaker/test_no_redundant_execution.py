from primebeaker.client import (
    RemoteCodeExecutionClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
)
from primebeaker.environments import jtc_code_tool_env, jtc_terminal_tool_env
from primebeaker.environments.registry import ENVIRONMENTS


def test_dead_procedural_execution_harnesses_are_absent() -> None:
    for module, names in (
        (
            jtc_code_tool_env,
            (
                "append_tool_turn",
                "build_second_turn_prompt",
                "execute_first_turn_if_needed_async",
                "execute_python_code_async",
                "code_result_to_tool_content",
            ),
        ),
        (
            jtc_terminal_tool_env,
            (
                "append_tool_turn",
                "build_next_turn_prompt",
                "execute_turn_if_needed_async",
                "execute_terminal_command_async",
                "terminal_result_to_tool_content",
            ),
        ),
    ):
        assert all(not hasattr(module, name) for name in names)


def test_clients_expose_one_canonical_execution_method() -> None:
    assert not hasattr(RemoteCodeExecutionClient, "execute_code")
    assert not hasattr(TerminalExecutionClient, "execute_terminal")
    assert not hasattr(WebTerminalExecutionClient, "execute_terminal")


def test_helper_modules_are_not_registered_as_environments() -> None:
    assert "jtc_code_tool_env" not in ENVIRONMENTS.values()
    assert "jtc_terminal_tool_env" not in ENVIRONMENTS.values()
    assert "search_agent_env" not in ENVIRONMENTS.values()
    assert "search_visit_agent_env" not in ENVIRONMENTS.values()
