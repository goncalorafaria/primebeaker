"""Native taskset for fresh-container terminal solving with private pytest grading."""
from __future__ import annotations

import json
from pathlib import Path

import verifiers.v1 as vf
from pydantic import Field
from pydantic_config import BaseConfig

__all__ = ["PodmanSolverTaskset"]

SYSTEM_PROMPT = """Solve the task in the fresh container using the bash tool.
Only bash is available. Commands share a container; cwd and exported environment
variables persist between calls, but each call starts a new Bash process.
Tool output may be truncated; inspect files in small sections when needed.
When finished, call bash with exactly: echo TERMINAL_COMPLETE
This ends the episode and runs private tests. You cannot continue after submission.
"""


class ExecutionConfig(BaseConfig):
    gateway_url: str = ""
    service: str = "podman"
    workdir: str = "/home/user"
    request_timeout: float = Field(900, gt=0)
    command_timeout: float = Field(60, gt=0, le=300)
    test_timeout: float = Field(60, gt=0, le=300)
    max_tool_calls: int = Field(128, ge=1)
    output_characters: int = Field(4000, ge=1)


class SolverData(vf.TaskData):
    record_id: str
    original_image: str
    test_final_state: str
    execution: ExecutionConfig


class SolverState(vf.State):
    container_id: str | None = None
    tool_calls: int = 0
    invalid_tool_calls: int = 0
    submitted: bool = False
    tests_ran: bool = False
    tests_passed: bool = False
    infrastructure_error: str | None = None
    cleanup_error: str | None = None


class SolverTaskConfig(vf.TaskConfig):
    execution: ExecutionConfig = ExecutionConfig()
    test_reward: float = Field(1.0, ge=0)
    completion_reward: float = Field(0.0, ge=0)
    invalid_tool_penalty: float = Field(0.1, ge=0)


class SolverTasksetConfig(vf.TasksetConfig):
    dataset: str = ""
    task: SolverTaskConfig = SolverTaskConfig()


class SolverTask(vf.Task[SolverData, SolverState, SolverTaskConfig]):
    @property
    def key(self):
        return self.data.record_id

    @vf.stop
    def terminal_complete(self, trace: vf.Trace) -> bool:
        return trace.state.submitted

    @vf.reward
    async def private_tests(self, trace: vf.Trace) -> float:
        if trace.state.infrastructure_error:
            raise RuntimeError(trace.state.infrastructure_error)
        return self.config.test_reward * float(trace.state.tests_passed)

    @vf.reward
    async def completion(self, trace: vf.Trace) -> float:
        return self.config.completion_reward * float(trace.state.submitted)

    @vf.reward
    async def invalid_tools(self, trace: vf.Trace) -> float:
        return -self.config.invalid_tool_penalty * trace.state.invalid_tool_calls

    @vf.metric
    async def task_success(self, trace: vf.Trace) -> float:
        return float(trace.state.tests_passed)

    @vf.metric
    async def submitted(self, trace: vf.Trace) -> float:
        return float(trace.state.submitted)

    @vf.metric
    async def bash_calls(self, trace: vf.Trace) -> float:
        return float(trace.state.tool_calls)

    @vf.metric
    async def tool_budget_exhausted(self, trace: vf.Trace) -> float:
        return float(not trace.state.submitted and
                     trace.state.tool_calls >= self.config.execution.max_tool_calls)

    @vf.metric
    async def cleanup_failed(self, trace: vf.Trace) -> float:
        return float(trace.state.cleanup_error is not None)


class PodmanSolverTaskset(vf.Taskset[SolverTask, SolverTasksetConfig]):
    def load(self):
        seen = set()
        with Path(self.config.dataset).open() as handle:
            for idx, line in enumerate(handle):
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row.get("task_id") or row["record_id"])
                if record_id in seen:
                    raise ValueError(f"Duplicate task ID: {record_id}")
                seen.add(record_id)
                # Reconstruct only the original user task, never reference turns or hints.
                prompt = row["prompt"]
                if isinstance(prompt, list):
                    users = [m["content"] for m in prompt if m["role"] == "user"]
                    if len(users) != 1 or not isinstance(users[0], str):
                        raise ValueError("Expected exactly one original user task")
                    prompt = users[0]
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Missing task prompt")
                image = row["original_image"]
                tests = row["test_final_state"]
                if not isinstance(image, str) or not image.strip():
                    raise ValueError("Missing initial image")
                if not isinstance(tests, str) or not tests.strip():
                    raise ValueError("Missing private final-state tests")
                data = SolverData(
                    idx=idx, record_id=record_id, original_image=image,
                    test_final_state=tests, execution=self.config.task.execution,
                    prompt=[{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}],
                )
                yield SolverTask(data, self.config.task)
