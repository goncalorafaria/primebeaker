"""Direct terminal harness; no replay prefix, judge labels, or reference trajectory."""
from __future__ import annotations

import json
import logging
import os
import shlex
from dataclasses import asdict
from uuid import uuid4

import httpx
from openai import AsyncOpenAI
from literegistry_podman_client import PodmanClient
from verifiers.v1.configs.harness import HarnessConfig
from verifiers.v1.dialects.chat import message_to_wire
from verifiers.v1.harness import Harness
from verifiers.v1.harnesses.utils.core import chat, gate_tool_call
from verifiers.v1.runtimes import ProgramResult

from primebeaker_podman_solver import SolverData, SolverState

__all__ = ["PodmanSolverHarness"]
log = logging.getLogger(__name__)
TOOLS = [{"type": "function", "function": {
    "name": "bash", "description": "Execute a Bash command in the task container.",
    "parameters": {"type": "object", "properties": {"command": {"type": "string"}},
                   "required": ["command"], "additionalProperties": False},
}}]


class SolverHarnessConfig(HarnessConfig):
    pass


class PodmanSolverHarness(Harness[SolverHarnessConfig]):
    APPENDS_SYSTEM_PROMPT = True
    EXECUTES_CODE = False
    NEEDS_CONTAINER = False
    SUPPORTS_TOOL_INTERCEPTION = True

    async def launch(self, ctx, trace, runtime, endpoint, secret, mcp_urls, data,
                     tool_interception_url=None):
        if mcp_urls:
            raise ValueError("Podman solver uses its own bash tool, not MCP")
        if not isinstance(data, SolverData) or not isinstance(trace.state, SolverState):
            raise TypeError("Use the primebeaker-podman-solver taskset")
        cfg, state = data.execution, trace.state
        gateway = cfg.gateway_url or os.environ.get("PODMAN_GATEWAY_URL", "")
        if not gateway:
            raise ValueError("Set execution.gateway_url or PODMAN_GATEWAY_URL")
        client = PodmanClient(gateway, service=cfg.service, workdir=cfg.workdir,
                              request_timeout=cfg.request_timeout)
        session = None
        system, prompt = self.resolve_prompt(data)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if isinstance(prompt, str):
            messages.append({"role": "user", "content": prompt})
        elif prompt is not None:
            messages.extend(message_to_wire(m) for m in prompt)
        while len(messages) > 1 and messages[0]["role"] == messages[1]["role"] == "system":
            messages[0]["content"] += "\n\n" + messages.pop(1)["content"]
        try:
            await client.open()
            session = await client.handshake(image=data.original_image,
                                             client_id="terminal-solver-" + uuid4().hex)
            state.container_id = session.container_id
            # Persist cwd/exports without relying on a persistent interactive process.
            shell_root = "/tmp/primebeaker-shell-" + uuid4().hex
            wrapper = f"""#!/bin/bash
set -a
source {shell_root}.env
set +a
cd "$(cat {shell_root}.cwd)" || exit 1
trap 'code=$?; export -p > {shell_root}.env; pwd > {shell_root}.cwd; exit "$code"' EXIT
payload=$(cat "$1"; printf .)
payload=${{payload%.}}
rm -f -- "$1"
eval "$payload"
"""
            initialized = await session.execute(
                f"printf '%s\\n' {shlex.quote(cfg.workdir)} > {shell_root}.cwd; "
                f": > {shell_root}.env; cat > {shell_root}.sh", stdin=wrapper,
                timeout=cfg.command_timeout,
            )
            if initialized.exit_code != 0:
                raise RuntimeError("Failed to initialize task shell")
            timeout = httpx.Timeout(None, connect=30, pool=30, write=300)
            async with (
                AsyncOpenAI(base_url=endpoint, api_key=secret, max_retries=0,
                            http_client=httpx.AsyncClient(timeout=timeout, trust_env=False)) as model,
                httpx.AsyncClient(timeout=timeout, trust_env=False) as gate,
            ):
                while trace.stop_condition is None and not state.submitted:
                    if state.tool_calls >= cfg.max_tool_calls:
                        break
                    completion = await chat(model, ctx.model, messages, TOOLS)
                    message = completion.choices[0].message
                    messages.append(message.model_dump(exclude_none=True))
                    if (completion.choices[0].finish_reason == "length" or
                            trace.stop_condition is not None or not message.tool_calls):
                        break
                    for call in message.tool_calls:
                        if state.submitted or trace.stop_condition is not None:
                            break
                        if state.tool_calls >= cfg.max_tool_calls:
                            break
                        if tool_interception_url:
                            decision = await gate_tool_call(gate, tool_interception_url, secret, call)
                            if decision["action"] == "deny":
                                messages.append(decision["message"])
                                continue
                        state.tool_calls += 1
                        try:
                            args = json.loads(call.function.arguments or "{}")
                            if call.function.name != "bash":
                                raise ValueError("Only bash is available")
                            if not isinstance(args, dict) or set(args) != {"command"}:
                                raise ValueError("bash requires only a command argument")
                            command = args["command"]
                            if not isinstance(command, str) or not command.strip():
                                raise ValueError("command must be a non-empty string")
                        except (ValueError, TypeError) as error:
                            state.invalid_tool_calls += 1
                            content = json.dumps({"error": str(error)})
                        else:
                            # Upload as stdin; do not retry commands with uncertain outcomes.
                            payload_path = shell_root + ".command-" + uuid4().hex
                            upload = await session.execute("cat > " + payload_path,
                                                           stdin=command, timeout=cfg.command_timeout)
                            if upload.exit_code != 0:
                                raise RuntimeError("Failed to transfer terminal command")
                            result = await session.execute(f"bash {shell_root}.sh {payload_path}",
                                                           timeout=cfg.command_timeout)
                            output = asdict(result)
                            for key in ("stdout", "stderr"):
                                text = output[key]
                                if len(text) > cfg.output_characters:
                                    output[key] = text[:cfg.output_characters] + "\n[output truncated]"
                            content = json.dumps(output)
                            try:
                                is_submit = shlex.split(command) == ["echo", "TERMINAL_COMPLETE"]
                            except ValueError:
                                is_submit = False
                            if (is_submit and result.exit_code == 0 and not result.timed_out
                                    and result.stdout.strip() == "TERMINAL_COMPLETE"):
                                # End model access before placing private tests in the container.
                                state.submitted = True
                                await self._grade(session, data, state)
                        messages.append({"role": "tool", "tool_call_id": call.id,
                                         "name": call.function.name, "content": content})
            return ProgramResult(exit_code=0, stdout="", stderr="")
        except Exception as error:
            state.infrastructure_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            try:
                if session is not None:
                    await session.close()
            except Exception as error:
                state.cleanup_error = f"{type(error).__name__}: {error}"
                log.warning("Podman cleanup failed: task=%s container=%s error=%s",
                            data.record_id, state.container_id, state.cleanup_error)
            finally:
                await client.aclose()

    async def _grade(self, session, data, state):
        cfg = data.execution
        # Host-owned test source is staged only after submission. No reward file is trusted.
        directory = "/tmp/primebeaker-private-" + uuid4().hex
        installed = await session.execute(
            f"mkdir -m 700 {directory} && cat > {directory}/test_final_state.py",
            stdin=data.test_final_state, timeout=cfg.test_timeout,
        )
        if installed.exit_code != 0:
            raise RuntimeError("Private-test installation failed")
        state.tests_ran = True
        result = await session.execute(
            f"python3 -m pytest -q --rootdir={directory} --confcutdir={directory} "
            f"{directory}/test_final_state.py", timeout=cfg.test_timeout,
        )
        if result.timed_out or result.exit_code not in (0, 1):
            raise RuntimeError(f"Private grader failed: exit={result.exit_code}, timeout={result.timed_out}")
        state.tests_passed = result.exit_code == 0
