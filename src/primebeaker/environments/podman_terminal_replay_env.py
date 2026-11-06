"""Continue a successful terminal trace after replaying its command prefix.

Replay data is private row metadata. Only executed commands and their fresh
outputs enter the prompt; the reference suffix never enters model context.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
import shlex
from typing import Any

from datasets import Dataset
import verifiers as vf
from verifiers.utils.message_utils import normalize_messages

from .jtc_tool_label_env import _load_dataset, _latest_completion, _completion_tool_calls
from .podman_terminal_env import (
    PodmanTerminalEnv,
    _task_candidates,
    task_image,
    podman_execution_failure,
)

from .replay_shell import ReplayShellClient
from .replay_podman import ReplayPodmanClient


def replay_exit_matches(step: Mapping[str, Any], output: Mapping[str, Any]) -> bool:
    """TMAX's generator serialized success as 0 and every failure as 1."""
    actual = output.get("exit_code")
    if type(actual) is not int or output.get("timed_out") or podman_execution_failure(output):
        return False
    if step.get("exit_code_semantics", "exact") == "success_boolean":
        return int(actual != 0) == step["exit_code"]
    return actual == step["exit_code"]


def replay_plan(row: Mapping[str, Any], fraction: float) -> list[dict[str, Any]]:
    """Validate the explicit replay schema and choose complete command turns."""
    if not 0 < fraction < 1:
        raise ValueError("replay_fraction must be strictly between zero and one")
    replay = row.get("replay")
    if isinstance(replay, str):
        replay = json.loads(replay)
    if not isinstance(replay, Mapping) or replay.get("source_success") is not True:
        raise ValueError("replay needs an explicitly successful reference trace")
    if not replay.get("source_dataset") or not replay.get("source_id"):
        raise ValueError("replay needs source_dataset and source_id provenance")
    steps = replay.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        raise ValueError("replay needs at least two bash command turns")
    for step in steps:
        if not isinstance(step, Mapping) or not isinstance(step.get("command"), str):
            raise ValueError("each replay step needs a command string")
        if not step["command"].strip():
            raise ValueError("empty replay command")
        # Submission is a protocol action, never part of reconstructing a state.
        try:
            command_tokens = shlex.split(step["command"])
        except ValueError:
            command_tokens = None
        if command_tokens in (
            ["echo", "TERMINAL_COMPLETE"],
            ["echo", "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"],
        ):
            raise ValueError("remove completion/submission commands from replay steps")
        if type(step.get("exit_code")) is not int:
            raise ValueError("each replay step needs the recorded integer exit_code")
        if not isinstance(step.get("assistant_content", ""), str):
            raise ValueError("assistant_content must be a string")
    count = min(len(steps) - 1, math.floor(len(steps) * fraction))
    if count < 1:
        raise ValueError("selected prefix contains no complete command turn")
    selected = deepcopy(steps[:count])
    if replay.get("source_dataset") in {"allenai/tmax-sft-glm-52", "allenai/tmax-sft-16.5k-glm-5.3-max-thinking", "allenai/tmax-sft"}:
        for step in selected:
            step.setdefault("exit_code_semantics", "success_boolean")
    return selected


# Keep the historical wrapper name for callers of the local replay module.
VanilluxReplayShellClient = ReplayShellClient


class PodmanTerminalReplayEnv(PodmanTerminalEnv):
    """Reconstruct a reference prefix in each fresh rollout's own container."""

    def __init__(
        self, *, dataset: Dataset, replay_fraction: float = 0.8,
        include_replay_history: bool = True,
        replay_tokenizer: str = "Qwen/Qwen3-4B",
        max_replay_prompt_tokens: int = 16384,
        task_setup_timeout: float = 1200, **kwargs: Any,
    ) -> None:
        self.replay_fraction = replay_fraction
        self.include_replay_history = include_replay_history
        from transformers import AutoTokenizer
        self.replay_tokenizer = AutoTokenizer.from_pretrained(replay_tokenizer)
        self.max_replay_prompt_tokens = max_replay_prompt_tokens
        self.task_setup_timeout = task_setup_timeout
        # Fail before allocating containers for malformed/private metadata.
        for row in dataset:
            row = dict(row)
            if isinstance(row.get("info"), str):
                row["info"] = json.loads(row["info"])
            prompt = row.get("prompt")
            if isinstance(prompt, list) and any(
                message.get("role") not in {"system", "user"} for message in prompt
            ):
                raise ValueError("row prompt must contain only the original system/user task")
            candidates = list(_task_candidates({"task": row}))
            if any(r.get("task_setup_archive") or r.get("task_start_services") for r in candidates):
                raise ValueError("replay requires prebuilt source images without setup archives or service injection")
            image = task_image({"task": row})
            if not image.startswith("docker.io/"):
                raise ValueError("replay requires an original Docker Hub image reference")
            candidate = next((r for r in candidates if "replay" in r), None)
            if candidate is None:
                raise ValueError("dataset row has no replay metadata")
            replay_plan(candidate, replay_fraction)
        # Reward passing private tests, rather than merely stopping the episode.
        kwargs.setdefault("termination_reward_weight", 0.0)
        super().__init__(dataset=dataset, **kwargs)

    def _new_podman_client(self):
        return ReplayPodmanClient(self)

    async def _initialize_task(self, state: vf.State) -> None:
        # Inputs are checked before allocation in __init__. Never reconstruct
        # fixtures or inject task services into a published source image.
        return None

    @vf.stop(priority=50)
    async def stop_after_final_or_invalid(self, state: vf.State) -> bool:
        stop = await super().stop_after_final_or_invalid(state)
        if (stop and state.get("podman_replay_complete")
                and not state.get("podman_failure")
                and not state.get("podman_terminal_complete")
                and not state.get("replay_format_retry_used")
                and not state.get("jtc_tool_final_warning_sent")
                and not _completion_tool_calls(_latest_completion(state))):
            state["replay_format_retry_used"] = True
            state["replay_format_retry_pending"] = True
            return False
        return stop

    async def env_response(self, messages: vf.Messages, state: vf.State, **kwargs: Any) -> vf.Messages:
        if state.pop("replay_format_retry_pending", False):
            return normalize_messages([{
                "role": "user",
                "content": (
                    "Your last response did not include a bash tool call. "
                    "Use the bash tool to continue working. If you have finished, "
                    "submit by calling the bash tool with this exact JSON argument: "
                    + json.dumps({"command": f"echo {self.completion_marker}"})
                ),
            }], field_name="replay_format_correction")
        return await super().env_response(messages, state, **kwargs)

    async def _grade_completed_task(self, state: vf.State) -> None:
        # Grading uses the original workdir and a clean shell, as in the base harness.
        client = state.get("podman_client")
        if isinstance(client, ReplayShellClient):
            state["podman_client"] = client.client
        try:
            await super()._grade_completed_task(state)
        finally:
            state["podman_client"] = client

    async def setup_state(self, state: vf.State) -> vf.State:
        state = await super().setup_state(state)
        if state.get("podman_failure"):
            return state
        state["podman_replay_history"] = []
        state["podman_replay_complete"] = False
        try:
            await self._initialize_task(state)
            candidate = next(r for r in _task_candidates(state) if "replay" in r)
            steps = replay_plan(candidate, self.replay_fraction)
            replay_metadata = candidate["replay"]
            if isinstance(replay_metadata, str):
                replay_metadata = json.loads(replay_metadata)
            shell = ReplayShellClient(state["podman_client"])
            await shell.initialize()
            state["podman_terminal_binding"].__exit__(None, None, None)
            binding = self.podman_terminal_client.bind(shell)
            binding.__enter__()
            state["podman_terminal_binding"] = binding
            state["podman_client"] = shell
            state["podman_replay_prefix_sha256"] = hashlib.sha256(
                json.dumps(steps, sort_keys=True).encode()
            ).hexdigest()
            state["podman_replay_planned_commands"] = len(steps)
            history = []
            for index, step in enumerate(steps):
                result = dict(await state["podman_client"].execute(
                    command=step["command"], timeout=self.command_timeout,
                ))
                state["podman_replay_history"].append({
                    "command": step["command"], "output": result,
                })
                failure = podman_execution_failure(result)
                if failure or not replay_exit_matches(step, result):
                    raise RuntimeError(
                        f"Replay command {index} diverged: expected exit "
                        f"{step['exit_code']}, got {result.get('exit_code')}; "
                        f"{failure or result.get('stderr', '')}"
                    )
                call_id = f"replay_{index}"
                history.extend([
                    {"role": "assistant", "content": step.get("assistant_content", ""),
                     "reasoning_content": step.get("reasoning_content", ""),
                     "tool_calls": [{"id": call_id, "name": "bash",
                                     "arguments": json.dumps({"command": step["command"]})}]},
                    {"role": "tool", "tool_call_id": call_id,
                     "content": json.dumps(result)},
                ])
            # Prompt tokens have no policy loss. Replay is deliberately absent
            # from trajectory and jtc_tool_history, keeping continuation budgets
            # and tool counters independent of reference-prefix length.
            if self.include_replay_history:
                state["prompt"] = [*state["prompt"], *normalize_messages(
                    history, field_name="replay_history",
                )]
            messages = [m.model_dump(exclude_none=True) if hasattr(m, "model_dump") else dict(m)
                        for m in state["prompt"]]
            for message in messages:
                if message.get("tool_calls"):
                    message["tool_calls"] = [{"type": "function", "id": c["id"],
                        "function": {"name": c["name"], "arguments": json.loads(c["arguments"])}}
                        for c in message["tool_calls"]]
                # Preserve native reasoning_content. With no added user handoff,
                # Qwen keeps reasoning throughout the ongoing tool interaction.
            encoded = self.replay_tokenizer.apply_chat_template(
                messages, tools=[{"type": "function", "function": (
                    t.model_dump(exclude_none=True) if hasattr(t, "model_dump") else dict(t)
                )} for t in self.tool_defs],
                tokenize=True, add_generation_prompt=True, enable_thinking=True,
            )
            token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            if hasattr(token_ids, "tolist"):
                token_ids = token_ids.tolist()
            if token_ids and isinstance(token_ids[0], list):
                if len(token_ids) != 1:
                    raise ValueError("Expected exactly one replay prompt")
                token_ids = token_ids[0]
            token_count = len(token_ids)
            state["podman_replay_prompt_tokens"] = token_count
            if token_count > self.max_replay_prompt_tokens:
                raise RuntimeError(f"Replay prompt has {token_count} tokens; limit {self.max_replay_prompt_tokens}")
            state["prompt"] = normalize_messages(messages, field_name="replay_prompt")
            state["podman_replay_complete"] = True
        except Exception as error:  # Match the parent's infrastructure-failure path.
            self._record_podman_failure(state, phase="replay", error=error)
        return state


def load_environment(dataset: str, split: str = "train", **kwargs: Any) -> PodmanTerminalReplayEnv:
    return PodmanTerminalReplayEnv(dataset=_load_dataset(dataset, split), **kwargs)
