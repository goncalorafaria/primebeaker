"""Rule-scored Podman environment in which an agent solves the task live.

Each rollout starts a fresh container from the task image. The model works in
that same container through bash. Only after the successful standalone
"echo TERMINAL_COMPLETE" protocol does PrimeBeaker run the private unit-test
command and read its reward file.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
import re
import shlex
from typing import Any, Callable, Protocol
from uuid import uuid4

from datasets import Dataset
import verifiers as vf

from literegistry_tool_client import PodmanExecutionClient, ToolClient
from literegistry_tool_client.submission import SubmitToolClient

from .articulated_harness_env import JTCArticulatedHarnessEnv
from .jtc_tool_label_env import _load_dataset
from .tool_protocol import GPTOSS_SUBMIT_TOOL, GPTOSS_TERMINAL_TOOL


DEFAULT_COMPLETION_MARKER = "TERMINAL_COMPLETE"
DEFAULT_PODMAN_GATEWAY_URL = "http://127.0.0.1:1212"
DEFAULT_REWARD_FILE_PATH = "/logs/verifier/reward.txt"
DEFAULT_TEST_COMMAND = "bash /tests/test.sh"
DEFAULT_TEST_FILE_PATH = "/tmp/primebeaker-test-final-state.py"
_REWARD_FILE_SENTINEL = "__PRIMEBEAKER_REWARD_FILE__"
_PODMAN_EXECUTION_FAILURE_PREFIX = "bash execution request failed:"


class PodmanCLI(Protocol):
    """Podman client operations needed by one live task rollout."""

    started: bool
    container_id: str | None
    affinity_id: str | None

    async def start(self, *, image: str | None = None) -> Mapping[str, Any]: ...

    async def execute(
        self, *, command: str, stdin: str = "", timeout: float = 10
    ) -> Mapping[str, Any]: ...

    async def close(self) -> Mapping[str, Any] | None: ...


def podman_error_is_oom(error: BaseException | str) -> bool:
    """Return whether a Podman failure reports an out-of-memory condition."""

    detail = (
        f"{type(error).__name__}: {error}"
        if isinstance(error, BaseException)
        else error
    )
    normalized = detail.casefold()
    return (
        "out of memory" in normalized
        or "oomerror" in normalized
        or re.search(r"(?<![a-z0-9])oom(?:[_-]?killed)?(?![a-z0-9])", normalized)
        is not None
    )


def podman_execution_failure(output: Mapping[str, Any]) -> str | None:
    """Extract an exception-backed bash failure produced by the shared harness."""

    stderr = str(output.get("stderr") or "").strip()
    if stderr.casefold().startswith(_PODMAN_EXECUTION_FAILURE_PREFIX):
        return stderr[len(_PODMAN_EXECUTION_FAILURE_PREFIX) :].strip()
    return None


class ContextBoundPodmanTerminalClient(ToolClient):
    """Dispatch terminal calls to the Podman session bound to this rollout."""

    def __init__(self, *, command_timeout: float = 60) -> None:
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        super().__init__(
            "context://podman-terminal-solver",
            timeout=command_timeout,
            max_retries=1,
        )
        self.command_timeout = float(command_timeout)
        self._current: ContextVar[PodmanCLI | None] = ContextVar(
            f"podman_terminal_verifier_{id(self)}",
            default=None,
        )

    @property
    def request_name(self) -> str:
        return "Podman task terminal execution"

    @property
    def current_client(self) -> PodmanCLI | None:
        return self._current.get()

    @contextmanager
    def bind(self, client: PodmanCLI) -> Iterator[None]:
        if self.current_client is not None:
            raise RuntimeError("a Podman task session is already bound")
        token = self._current.set(client)
        try:
            yield
        finally:
            self._current.reset(token)

    async def execute(
        self,
        *,
        command: str,
        contents: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        del contents
        client = self.current_client
        if client is None or not client.started:
            raise RuntimeError("no live Podman task session is bound")
        result = await client.execute(command=command, timeout=self.command_timeout)
        if not isinstance(result, Mapping):
            raise RuntimeError("Podman task terminal returned a non-object")
        return dict(result)


def is_completion_command(command: str, marker: str) -> bool:
    """Return whether command is exactly echo plus the configured marker."""

    if not isinstance(command, str) or not marker:
        return False
    try:
        return shlex.split(command) == ["echo", marker]
    except ValueError:
        return False


def has_completion_marker(
    command: str,
    output: Mapping[str, Any],
    marker: str,
) -> bool:
    """Accept the marker only from the successful standalone echo protocol."""

    if not is_completion_command(command, marker):
        return False
    if output.get("success") is not True:
        return False
    return marker in str(output.get("stdout") or "").splitlines()


def reward_file_score(state: Mapping[str, Any]) -> float:
    """Return the privately captured verifier reward, defaulting to zero."""

    value = state.get("podman_reward_file_score", state.get("podman_verifier_reward", 0.0))
    if not isinstance(value, (int, float)):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def episode_termination_credit(state: Mapping[str, Any]) -> float:
    """Give raw unit credit only for the explicit standalone echo protocol."""

    return float(state.get("podman_terminal_complete") is True)


def podman_failure_penalty(state: Mapping[str, Any]) -> float:
    """Return one raw negative unit when Podman setup or execution failed."""

    return -float(state.get("podman_failure") is True)


def parse_reward_file_output(output: Mapping[str, Any]) -> tuple[float, bool, str | None]:
    """Parse a private reward-file probe; missing or invalid data scores zero."""

    lines = str(output.get("stdout") or "").splitlines()
    if not lines or lines[0] != _REWARD_FILE_SENTINEL:
        return 0.0, False, None
    raw = "\n".join(lines[1:]).strip()
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0, True, f"invalid reward file value: {raw!r}"
    return max(0.0, min(1.0, value)), True, None


def reward_file_probe_command(path: str) -> str:
    """Build a non-failing private probe that distinguishes a missing file."""

    quoted_path = shlex.quote(path)
    quoted_sentinel = shlex.quote(_REWARD_FILE_SENTINEL)
    return (
        f"if [ -f {quoted_path} ]; then "
        f"printf '%s\\n' {quoted_sentinel}; cat -- {quoted_path}; fi"
    )


def reward_file_clear_command(path: str) -> str:
    """Build the private command used to prevent stale or model-written rewards."""

    return f"rm -f -- {shlex.quote(path)}"


def _task_candidates(state: Mapping[str, Any]) -> Iterator[Mapping[str, Any]]:
    for key in ("task", "input"):
        candidate = state.get(key)
        if not isinstance(candidate, Mapping):
            continue
        yield candidate
        for nested_key in ("info", "source", "misc", "raw"):
            nested = candidate.get(nested_key)
            if not isinstance(nested, Mapping):
                continue
            yield nested
            for child_key in ("misc", "raw"):
                child = nested.get(child_key)
                if isinstance(child, Mapping):
                    yield child


def qualify_image(image: str) -> str:
    """Turn a short task image name into an explicit registry reference."""

    normalized = image.strip()
    if not normalized:
        raise ValueError("task image must be non-empty")
    if "/" not in normalized:
        return f"docker.io/library/{normalized}"
    first_component = normalized.split("/", 1)[0]
    if first_component == "localhost" or any(
        marker in first_component for marker in (".", ":")
    ):
        return normalized
    return f"docker.io/{normalized}"


def task_image(state: Mapping[str, Any]) -> str:
    """Resolve the initial container image without consulting any saved trace."""

    for candidate in _task_candidates(state):
        for key in (
            "original_image",
            "image",
            "container_image",
            "published_container_image",
            "resolved_container_image",
        ):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise ValueError(
        "dataset row needs original_image, image, or container_image"
    )


def task_final_state_tests(state: Mapping[str, Any]) -> str | None:
    """Return private per-task tests without consulting a saved trajectory."""

    for candidate in _task_candidates(state):
        for key in ("test_final_state", "unit_tests"):
            value = candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return None


def private_test_install_command(path: str) -> str:
    """Build a command that receives private test source only through stdin."""

    return (
        "python3 -c 'import pathlib,sys; "
        "pathlib.Path(sys.argv[1]).write_text(sys.stdin.read())' "
        f"{shlex.quote(path)}"
    )


async def start_task_container(podman_cli: PodmanCLI, image: str) -> str:
    """Start one clean task container and validate its affinity identity."""

    if podman_cli.started:
        raise RuntimeError("Podman client already owns a live session")
    await podman_cli.start(image=qualify_image(image))
    container_id = podman_cli.container_id
    affinity_id = podman_cli.affinity_id
    if not isinstance(container_id, str) or not container_id:
        raise RuntimeError("Podman client returned no container ID")
    if not isinstance(affinity_id, str) or affinity_id != container_id:
        raise RuntimeError("Podman container and affinity IDs do not match")
    return container_id


class PodmanTerminalEnv(JTCArticulatedHarnessEnv):
    """Let an agent solve a task in a fresh Podman container."""

    def __init__(
        self,
        *,
        dataset: Dataset,
        podman_gateway_url: str = DEFAULT_PODMAN_GATEWAY_URL,
        podman_workdir: str = "/home/user",
        command_timeout: float = 60,
        request_timeout: float = 150,
        handshake_timeout: float = 600,
        max_retries: int = 3,
        service: str = "podman",
        client_id_prefix: str = "primebeaker-solver",
        completion_marker: str = DEFAULT_COMPLETION_MARKER,
        reward_file_path: str = DEFAULT_REWARD_FILE_PATH,
        test_command: str = DEFAULT_TEST_COMMAND,
        test_file_path: str = DEFAULT_TEST_FILE_PATH,
        test_timeout: float = 600,
        reward_file_weight: float = 1.0,
        verifier_reward_weight: float | None = None,
        termination_reward_weight: float = 0.1,
        podman_failure_penalty_weight: float = 1.0,
        fake_tool_penalty_weight: float = 0.1,
        max_tool_calls: int = 64,
        max_turns: int | None = None,
        podman_client_factory: Callable[[], PodmanCLI] | None = None,
        **kwargs: Any,
    ) -> None:
        if not podman_gateway_url.strip():
            raise ValueError("podman_gateway_url must be non-empty")
        if not podman_workdir.strip():
            raise ValueError("podman_workdir must be non-empty")
        if command_timeout <= 0:
            raise ValueError("command_timeout must be positive")
        if request_timeout <= 0 or handshake_timeout <= 0:
            raise ValueError("request timeouts must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1")
        if not service.strip() or not client_id_prefix.strip():
            raise ValueError("service and client_id_prefix must be non-empty")
        if not completion_marker.strip():
            raise ValueError("completion_marker must be non-empty")
        if not reward_file_path.strip():
            raise ValueError("reward_file_path must be non-empty")
        if not test_command.strip():
            raise ValueError("test_command must be non-empty")
        if not test_file_path.strip():
            raise ValueError("test_file_path must be non-empty")
        if test_timeout <= 0:
            raise ValueError("test_timeout must be positive")
        if verifier_reward_weight is not None:
            if reward_file_weight != 1.0:
                raise ValueError(
                    "supply only reward_file_weight or legacy verifier_reward_weight"
                )
            reward_file_weight = verifier_reward_weight
        if reward_file_weight < 0 or termination_reward_weight < 0:
            raise ValueError("reward weights must be non-negative")
        if podman_failure_penalty_weight < 0 or fake_tool_penalty_weight < 0:
            raise ValueError("penalty weights must be non-negative")

        self.podman_gateway_url = podman_gateway_url.rstrip("/")
        self.podman_workdir = podman_workdir
        self.command_timeout = float(command_timeout)
        self.request_timeout = float(request_timeout)
        self.handshake_timeout = float(handshake_timeout)
        self.max_retries = int(max_retries)
        self.service = service
        self.client_id_prefix = client_id_prefix
        self.completion_marker = completion_marker.strip()
        self.reward_file_path = reward_file_path.strip()
        self.test_command = test_command.strip()
        self.test_file_path = test_file_path.strip()
        self.test_timeout = float(test_timeout)
        self.reward_file_weight = float(reward_file_weight)
        self.termination_reward_weight = float(termination_reward_weight)
        self.podman_failure_penalty_weight = float(podman_failure_penalty_weight)
        self.fake_tool_penalty_weight = float(fake_tool_penalty_weight)
        self.podman_terminal_client = ContextBoundPodmanTerminalClient(
            command_timeout=self.command_timeout
        )
        self._podman_client_factory = (
            podman_client_factory or self._new_podman_client
        )

        bash_tool = {
            **GPTOSS_TERMINAL_TOOL,
            "name": "bash",
            "description": (
                "Work directly in the task's fresh Podman container. Inspect files, "
                "edit the solution, and run useful checks. When the task is solved, "
                f"call exactly echo {self.completion_marker}."
            ),
        }
        super().__init__(
            dataset=dataset,
            tool_clients={
                "bash": self.podman_terminal_client,
                "submit": SubmitToolClient(),
            },
            tools=("bash", "submit"),
            tool_defs=(bash_tool, GPTOSS_SUBMIT_TOOL),
            max_tool_calls=max_tool_calls,
            max_turns=max_turns,
            **kwargs,
        )

        # The inherited harness supplies dispatch and trajectory bookkeeping.
        # Solving rewards are intentionally limited to this environment's rules.
        self.rubric = vf.Rubric()

        async def task_reward_file(state: vf.State) -> float:
            return reward_file_score(state)

        async def episode_terminated(state: vf.State) -> float:
            return episode_termination_credit(state)

        async def podman_failed_reward(state: vf.State) -> float:
            return podman_failure_penalty(state)

        async def fake_tool_penalty(state: vf.State) -> float:
            return -float(max(int(state.get("jtc_unknown_tool_call_count") or 0), 0))

        async def terminal_completion_metric(state: vf.State) -> float:
            return episode_termination_credit(state)

        async def reward_file_present_metric(state: vf.State) -> float:
            return float(state.get("podman_reward_file_present") is True)

        async def podman_test_success_metric(state: vf.State) -> float:
            output = state.get("podman_test_output")
            if not isinstance(output, Mapping):
                return 0.0
            succeeded = (
                output.get("success") is True or output.get("exit_code") == 0
            )
            return float(succeeded)

        async def podman_failure_metric(state: vf.State) -> float:
            return float(state.get("podman_failure") is True)

        async def podman_oom_metric(state: vf.State) -> float:
            return float(state.get("podman_failure_oom") is True)

        async def fake_tool_count_metric(state: vf.State) -> float:
            return float(max(int(state.get("jtc_unknown_tool_call_count") or 0), 0))

        self.rubric.add_reward_func(
            task_reward_file, weight=self.reward_file_weight
        )
        self.rubric.add_reward_func(
            episode_terminated, weight=self.termination_reward_weight
        )
        self.rubric.add_reward_func(
            podman_failed_reward, weight=self.podman_failure_penalty_weight
        )
        self.rubric.add_reward_func(
            fake_tool_penalty, weight=self.fake_tool_penalty_weight
        )
        self.rubric.add_metric(terminal_completion_metric)
        self.rubric.add_metric(reward_file_present_metric)
        self.rubric.add_metric(podman_test_success_metric)
        self.rubric.add_metric(podman_failure_metric)
        self.rubric.add_metric(podman_oom_metric)
        self.rubric.add_metric(fake_tool_count_metric)

    def _new_podman_client(self) -> PodmanExecutionClient:
        return PodmanExecutionClient(
            self.podman_gateway_url,
            client_id=f"{self.client_id_prefix}-{uuid4().hex}",
            service=self.service,
            timeout=self.request_timeout,
            handshake_timeout=self.handshake_timeout,
            max_retries=self.max_retries,
            workdir=self.podman_workdir,
        )

    async def setup_state(self, state: vf.State) -> vf.State:
        client: PodmanCLI | None = None
        try:
            image = task_image(state)
            client = self._podman_client_factory()
            container_id = await start_task_container(client, image)
            binding = self.podman_terminal_client.bind(client)
            binding.__enter__()
            state.update(
                podman_client=client,
                podman_container_id=container_id,
                podman_original_image=image,
                podman_terminal_binding=binding,
                podman_terminal_complete=False,
                podman_tests_ran=False,
                podman_test_command=self.test_command,
                podman_test_file_path=self.test_file_path,
                podman_test_source=task_final_state_tests(state),
                podman_test_install_output=None,
                podman_test_output=None,
                podman_reward_file_clear_output=None,
                podman_reward_file_path=self.reward_file_path,
                podman_reward_file_present=False,
                podman_reward_file_error=None,
                podman_reward_file_score=0.0,
            )
        except Exception as error:  # noqa: BLE001 - recover rollout for scoring
            if client is not None and client.started:
                try:
                    result = await client.close()
                    state["podman_cleanup"] = dict(result or {})
                except Exception as cleanup_error:  # noqa: BLE001
                    state["podman_cleanup_error"] = (
                        f"{type(cleanup_error).__name__}: {cleanup_error}"
                    )
            self._record_podman_failure(state, phase="setup", error=error)
        return state

    async def _grade_completed_task(self, state: vf.State) -> None:
        """Run private tests once, after the model explicitly completes."""

        if state.get("podman_tests_ran") is True:
            return
        state["podman_tests_ran"] = True
        client = state.get("podman_client")
        if client is None or not client.started:
            self._record_podman_failure(
                state,
                phase="test",
                error="no live Podman task session is available for tests",
            )
            return
        try:
            reward_clear_output = await client.execute(
                command=reward_file_clear_command(self.reward_file_path),
                timeout=self.test_timeout,
            )
            if not isinstance(reward_clear_output, Mapping):
                raise RuntimeError("Podman reward-file clear returned a non-object")
            test_source = state.get("podman_test_source")
            if isinstance(test_source, str):
                install_output = await client.execute(
                    command=private_test_install_command(self.test_file_path),
                    stdin=test_source,
                    timeout=self.test_timeout,
                )
                if not isinstance(install_output, Mapping):
                    raise RuntimeError(
                        "Podman private-test install returned a non-object"
                    )
                state["podman_test_install_output"] = dict(install_output)
                install_succeeded = (
                    install_output.get("success") is True
                    or install_output.get("exit_code") == 0
                )
                if not install_succeeded:
                    raise RuntimeError("Podman private-test install failed")
            test_output = await client.execute(
                command=self.test_command,
                timeout=self.test_timeout,
            )
            if not isinstance(test_output, Mapping):
                raise RuntimeError("Podman test command returned a non-object")
            reward_output = await client.execute(
                command=reward_file_probe_command(self.reward_file_path),
                timeout=self.test_timeout,
            )
            if not isinstance(reward_output, Mapping):
                raise RuntimeError("Podman reward-file probe returned a non-object")
            reward_value, reward_present, reward_error = parse_reward_file_output(
                reward_output
            )
            clear_succeeded = (
                reward_clear_output.get("success") is True
                or reward_clear_output.get("exit_code") == 0
            )
            if not clear_succeeded:
                reward_value = 0.0
                reward_present = False
                reward_error = "could not clear reward file before tests"
            state.update(
                podman_test_output=dict(test_output),
                podman_reward_file_clear_output=dict(reward_clear_output),
                podman_reward_file_present=reward_present,
                podman_reward_file_error=reward_error,
                podman_reward_file_score=reward_value,
            )
        except Exception as error:  # noqa: BLE001 - recover rollout for scoring
            self._record_podman_failure(state, phase="test", error=error)

    @staticmethod
    def _record_podman_failure(
        state: vf.State,
        *,
        phase: str,
        error: BaseException | str,
    ) -> None:
        detail = (
            f"{type(error).__name__}: {error}"
            if isinstance(error, BaseException)
            else error
        )
        error_type, separator, message = detail.partition(":")
        state.update(
            podman_failure=True,
            podman_failure_phase=phase,
            podman_failure_type=(
                error_type.strip() if separator else "PodmanError"
            ),
            podman_failure_message=message.strip() if separator else detail,
            podman_failure_oom=podman_error_is_oom(detail),
            jtc_infrastructure_failure=True,
        )

    @vf.stop(priority=90)
    async def podman_failed(self, state: vf.State) -> bool:
        return state.get("podman_failure") is True

    # Completed submissions do not end this environment. The explicit terminal
    # marker is the only successful stop; inherited error and budget stops remain.
    async def all_rubrics_submitted(self, state: vf.State) -> bool:
        del state
        return False

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        history_start = len(state.get("jtc_tool_history") or [])
        responses = await super().env_response(messages, state, **kwargs)
        # The articulated base normally ends after the final submit.
        state["final_env_response"] = None
        history = state.get("jtc_tool_history") or []
        for item in history[history_start:]:
            if not isinstance(item, Mapping) or item.get("name") != "bash":
                continue
            arguments = item.get("arguments")
            output = item.get("output")
            if isinstance(output, Mapping):
                failure = podman_execution_failure(output)
                if failure is not None:
                    self._record_podman_failure(
                        state, phase="execute", error=failure
                    )
                    state["final_env_response"] = responses
                    break
            command = (
                arguments.get("command")
                if isinstance(arguments, Mapping)
                else None
            )
            if (
                isinstance(command, str)
                and isinstance(output, Mapping)
                and has_completion_marker(command, output, self.completion_marker)
            ):
                state["podman_terminal_complete"] = True
                state["podman_terminal_completion_command"] = command
                await self._grade_completed_task(state)
                state["final_env_response"] = responses
                break
        return responses

    @vf.stop(priority=75)
    async def terminal_complete(self, state: vf.State) -> bool:
        return state.get("podman_terminal_complete") is True

    @vf.cleanup(priority=50)
    async def close_podman_session(self, state: vf.State) -> None:
        binding = state.pop("podman_terminal_binding", None)
        client = state.pop("podman_client", None)
        if binding is not None:
            binding.__exit__(None, None, None)
        if client is not None and client.started:
            result = await client.close()
            state["podman_cleanup"] = dict(result or {})

PodmanTerminalVerifierEnv = PodmanTerminalEnv


def load_environment(
    dataset: str,
    split: str = "train",
    **kwargs: Any,
) -> PodmanTerminalEnv:
    """Prime-RL factory for live Podman task solving."""

    return PodmanTerminalEnv(
        dataset=_load_dataset(dataset, split),
        **kwargs,
    )


__all__ = [
    "ContextBoundPodmanTerminalClient",
    "DEFAULT_COMPLETION_MARKER",
    "DEFAULT_REWARD_FILE_PATH",
    "DEFAULT_TEST_COMMAND",
    "DEFAULT_TEST_FILE_PATH",
    "PodmanTerminalEnv",
    "PodmanTerminalVerifierEnv",
    "PodmanCLI",
    "episode_termination_credit",
    "has_completion_marker",
    "is_completion_command",
    "load_environment",
    "podman_error_is_oom",
    "podman_execution_failure",
    "podman_failure_penalty",
    "parse_reward_file_output",
    "reward_file_clear_command",
    "reward_file_probe_command",
    "reward_file_score",
    "qualify_image",
    "start_task_container",
    "task_final_state_tests",
    "task_image",
    "private_test_install_command",
]
