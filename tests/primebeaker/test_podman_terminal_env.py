from __future__ import annotations

import asyncio
from typing import Any

from datasets import Dataset

from primebeaker.environments.podman_terminal_env import (
    DEFAULT_REWARD_FILE_PATH,
    DEFAULT_TEST_COMMAND,
    PodmanTerminalEnv,
    episode_termination_credit,
    has_completion_marker,
    is_completion_command,
    parse_reward_file_output,
    podman_error_is_oom,
    podman_failure_penalty,
    reward_file_clear_command,
    reward_file_probe_command,
    reward_file_score,
    start_task_container,
    task_image,
)


class FakePodmanClient:
    def __init__(self, test_reward_file: str | None = None) -> None:
        self.started = False
        self.container_id = "a" * 64
        self.affinity_id = self.container_id
        self.image: str | None = None
        self.commands: list[tuple[str, float]] = []
        self.closed = False
        self.test_reward_file = test_reward_file
        self.reward_file: str | None = "stale-or-model-written"

    async def start(self, *, image: str | None = None) -> dict[str, Any]:
        self.started = True
        self.image = image
        return {
            "container_id": self.container_id,
            "affinity_id": self.affinity_id,
        }

    async def execute(
        self,
        *,
        command: str,
        timeout: float = 10,
        **_: Any,
    ) -> dict[str, Any]:
        self.commands.append((command, timeout))
        if command == reward_file_clear_command(DEFAULT_REWARD_FILE_PATH):
            self.reward_file = None
            return {
                "stdout": "",
                "stderr": "",
                "success": True,
                "exit_code": 0,
            }
        if command == DEFAULT_TEST_COMMAND:
            self.reward_file = self.test_reward_file
            return {
                "stdout": "unit tests executed\n",
                "stderr": "",
                "success": True,
                "exit_code": 0,
            }
        if command == reward_file_probe_command(DEFAULT_REWARD_FILE_PATH):
            stdout = (
                f"__PRIMEBEAKER_REWARD_FILE__\n{self.reward_file}\n"
                if self.reward_file is not None
                else ""
            )
            return {
                "stdout": stdout,
                "stderr": "",
                "success": True,
                "exit_code": 0,
            }
        return {
            "stdout": (
                "TERMINAL_COMPLETE\n"
                if command == "echo TERMINAL_COMPLETE"
                else "ok\n"
            ),
            "stderr": "",
            "success": True,
            "exit_code": 0,
        }

    async def close(self) -> dict[str, Any]:
        self.started = False
        self.closed = True
        return {"removed": True}


def _completion_state() -> dict[str, Any]:
    return {
        "task": {"original_image": "owner/task:tag"},
        "prompt": [],
        "completion": [],
        "trajectory": [
            {
                "completion": [
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "done-1",
                                "type": "function",
                                "function": {
                                    "name": "bash",
                                    "arguments": (
                                        '{"command":"echo TERMINAL_COMPLETE"}'
                                    ),
                                },
                            }
                        ],
                    }
                ]
            }
        ],
        "final_env_response": None,
    }


def test_completion_marker_requires_successful_standalone_echo() -> None:
    assert is_completion_command("echo TERMINAL_COMPLETE", "TERMINAL_COMPLETE")
    assert is_completion_command("echo 'TERMINAL_COMPLETE'", "TERMINAL_COMPLETE")
    assert not is_completion_command("printf TERMINAL_COMPLETE", "TERMINAL_COMPLETE")
    assert not is_completion_command(
        "echo TERMINAL_COMPLETE && touch /tmp/x",
        "TERMINAL_COMPLETE",
    )
    assert has_completion_marker(
        "echo TERMINAL_COMPLETE",
        {"success": True, "stdout": "TERMINAL_COMPLETE\n"},
        "TERMINAL_COMPLETE",
    )
    assert not has_completion_marker(
        "echo TERMINAL_COMPLETE",
        {"success": False, "stdout": "TERMINAL_COMPLETE\n"},
        "TERMINAL_COMPLETE",
    )


def test_protocol_rubrics_are_independent_raw_scores() -> None:
    assert reward_file_score({"podman_reward_file_score": 0.75}) == 0.75
    assert reward_file_score({}) == 0.0
    assert episode_termination_credit({"podman_terminal_complete": True}) == 1.0
    assert episode_termination_credit({}) == 0.0
    assert podman_failure_penalty({"podman_failure": True}) == -1.0
    assert podman_failure_penalty({}) == 0.0


def test_reward_file_parser_scores_missing_and_invalid_as_zero() -> None:
    assert parse_reward_file_output({"stdout": ""}) == (0.0, False, None)
    value, present, error = parse_reward_file_output(
        {"stdout": "__PRIMEBEAKER_REWARD_FILE__\nnot-a-number\n"}
    )
    assert value == 0.0
    assert present is True
    assert error is not None
    assert parse_reward_file_output(
        {"stdout": "__PRIMEBEAKER_REWARD_FILE__\n1.5\n"}
    ) == (1.0, True, None)


def test_task_image_is_resolved_without_any_trace() -> None:
    assert task_image({"task": {"original_image": "owner/task:tag"}}) == (
        "owner/task:tag"
    )
    try:
        task_image(
            {
                "task": {
                    "tmax_trace": {
                        "commands": ["touch /replayed"],
                        "original_image": "trace/image:old",
                    }
                }
            }
        )
    except ValueError as error:
        assert "needs original_image" in str(error)
    else:
        raise AssertionError("saved trace contents must not provide the task image")


def test_start_task_container_does_not_replay_commands() -> None:
    client = FakePodmanClient()
    container_id = asyncio.run(start_task_container(client, "owner/task:tag"))

    assert container_id == client.container_id
    assert client.image == "docker.io/owner/task:tag"
    assert client.commands == []
    assert client.started is True


def test_tests_run_only_after_live_agent_completes() -> None:
    client = FakePodmanClient(test_reward_file="0.75")
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "solve"}]}]
    )
    env = PodmanTerminalEnv(
        dataset=dataset,
        podman_client_factory=lambda: client,
        reward_file_weight=2.0,
        termination_reward_weight=0.25,
        podman_failure_penalty_weight=3.0,
    )
    state = _completion_state()

    async def exercise() -> None:
        await env.setup_state(state)  # type: ignore[arg-type]
        assert client.commands == []
        assert state["podman_tests_ran"] is False
        assert state["podman_reward_file_score"] == 0.0

        await env.podman_terminal_client.execute(command="touch /solution")
        assert state["podman_tests_ran"] is False
        assert state["podman_test_output"] is None

        response = await env.env_response([], state)  # type: ignore[arg-type]
        assert response
        assert state["podman_terminal_complete"] is True
        assert state["podman_tests_ran"] is True
        assert state["podman_reward_file_present"] is True
        assert state["podman_reward_file_score"] == 0.75
        assert state["podman_test_output"]["stdout"] == "unit tests executed\n"
        assert state["final_env_response"] == response
        await env.close_podman_session(state)  # type: ignore[arg-type]

    asyncio.run(exercise())

    weights = {
        func.__name__: weight
        for func, weight in zip(
            env.rubric._get_reward_funcs(),
            env.rubric._get_reward_weights(),
        )
    }
    assert weights["task_reward_file"] == 2.0
    assert weights["episode_terminated"] == 0.25
    assert weights["podman_failed_reward"] == 3.0
    assert weights["fake_tool_penalty"] == 0.1
    assert client.commands == [
        ("touch /solution", 60),
        ("echo TERMINAL_COMPLETE", 60),
        (reward_file_clear_command(DEFAULT_REWARD_FILE_PATH), 600),
        (DEFAULT_TEST_COMMAND, 600),
        (reward_file_probe_command(DEFAULT_REWARD_FILE_PATH), 600),
    ]
    assert client.closed is True


def test_missing_image_is_scored_as_podman_failure() -> None:
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "solve"}]}]
    )
    env = PodmanTerminalEnv(dataset=dataset)
    state: dict[str, Any] = {
        "task": {
            "tmax_trace": {
                "commands": ["touch /must-not-run"],
                "original_image": "trace/image:old",
            }
        },
        "prompt": [],
        "completion": [],
        "trajectory": [],
    }

    async def exercise() -> None:
        returned = await env.setup_state(state)  # type: ignore[arg-type]
        assert returned is state
        assert state["podman_failure"] is True
        assert state["podman_failure_phase"] == "setup"
        assert state["podman_failure_type"] == "ValueError"
        assert state["podman_failure_oom"] is False
        assert state["jtc_infrastructure_failure"] is True
        assert await env.podman_failed(state) is True  # type: ignore[arg-type]
        await env.rubric.score_rollout(state)  # type: ignore[arg-type]

    asyncio.run(exercise())

    assert state["reward"] == -1.0


def test_runtime_oom_stops_without_running_tests() -> None:
    class RuntimeOOMPodmanClient(FakePodmanClient):
        async def execute(
            self,
            *,
            command: str,
            timeout: float = 10,
            **kwargs: Any,
        ) -> dict[str, Any]:
            if command == "echo TERMINAL_COMPLETE":
                raise RuntimeError("container was OOM-killed")
            return await super().execute(
                command=command, timeout=timeout, **kwargs
            )

    client = RuntimeOOMPodmanClient()
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "solve"}]}]
    )
    env = PodmanTerminalEnv(
        dataset=dataset,
        podman_client_factory=lambda: client,
    )
    state = _completion_state()

    async def exercise() -> None:
        await env.setup_state(state)  # type: ignore[arg-type]
        responses = await env.env_response([], state)  # type: ignore[arg-type]
        assert state["podman_failure"] is True
        assert state["podman_failure_phase"] == "execute"
        assert state["podman_failure_type"] == "RuntimeError"
        assert state["podman_failure_oom"] is True
        assert state["podman_tests_ran"] is False
        assert state["final_env_response"] == responses
        assert await env.podman_failed(state) is True  # type: ignore[arg-type]
        await env.rubric.score_rollout(state)  # type: ignore[arg-type]
        await env.close_podman_session(state)  # type: ignore[arg-type]

    asyncio.run(exercise())

    assert podman_error_is_oom("container was OOM-killed")
    assert state["reward"] == -1.0
    assert client.closed is True


def test_missing_reward_file_gets_only_termination_credit() -> None:
    client = FakePodmanClient(test_reward_file=None)
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "solve"}]}]
    )
    env = PodmanTerminalEnv(
        dataset=dataset,
        podman_client_factory=lambda: client,
        termination_reward_weight=0.1,
    )
    state = _completion_state()

    async def exercise() -> None:
        await env.setup_state(state)  # type: ignore[arg-type]
        await env.env_response([], state)  # type: ignore[arg-type]
        await env.rubric.score_rollout(state)  # type: ignore[arg-type]
        await env.close_podman_session(state)  # type: ignore[arg-type]

    asyncio.run(exercise())

    assert state["podman_reward_file_present"] is False
    assert state["podman_reward_file_score"] == 0.0
    assert state["reward"] == 0.1


def test_fake_tool_attempts_receive_the_configured_penalty() -> None:
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "solve"}]}]
    )
    env = PodmanTerminalEnv(dataset=dataset, fake_tool_penalty_weight=0.25)
    state: dict[str, Any] = {
        "task": {"original_image": "owner/task:tag"},
        "prompt": [],
        "completion": [],
        "trajectory": [],
        "jtc_unknown_tool_call_count": 2,
    }

    asyncio.run(env.rubric.score_rollout(state))  # type: ignore[arg-type]

    assert state["reward"] == -0.5


def test_unfinished_episode_never_runs_tests_and_scores_zero() -> None:
    client = FakePodmanClient(test_reward_file="1")
    dataset = Dataset.from_list(
        [{"prompt": [{"role": "user", "content": "solve"}]}]
    )
    env = PodmanTerminalEnv(dataset=dataset, podman_client_factory=lambda: client)
    state: dict[str, Any] = {
        "task": {"original_image": "owner/task:tag"},
        "prompt": [],
        "completion": [],
        "trajectory": [],
    }

    async def exercise() -> None:
        await env.setup_state(state)  # type: ignore[arg-type]
        await env.rubric.score_rollout(state)  # type: ignore[arg-type]
        await env.close_podman_session(state)  # type: ignore[arg-type]

    asyncio.run(exercise())

    assert state["podman_tests_ran"] is False
    assert state["podman_reward_file_score"] == 0.0
    assert state["reward"] == 0.0
    assert client.commands == []
