"""SWERL fresh-Bash replay with cwd and exported-variable persistence."""

from __future__ import annotations

import asyncio


class ReplayShellClient:
    """SWERL Bash semantics: persist cwd/exports, not a running Bash process."""

    def __init__(self, client):
        self.client = client
        self.lock = asyncio.Lock()

    def __getattr__(self, name):
        return getattr(self.client, name)

    async def close(self):
        return await self.client.close()

    async def initialize(self):
        result = await self.client.execute(command=(
            'mkdir -p /workspace /output /logs/verifier /root && cd /workspace && '
            '[ -d /app ] || { _P="$(pwd)"; [ "$_P" != "/" ] && ln -sf "$_P" /app; } && '
            'printf "%s\n" /app > /tmp/.swerl_vanillux_cwd && '
            ': > /tmp/.swerl_vanillux_env'
        ), timeout=300, workdir="/")
        if result.get("exit_code") != 0:
            raise RuntimeError(f"Could not initialize Vanillux replay: {result}")
        wrapper = """#!/bin/bash
set -a
source /tmp/.swerl_vanillux_env 2>/dev/null || true
set +a
_cwd="$(cat /tmp/.swerl_vanillux_cwd 2>/dev/null || echo /app)"
cd "$_cwd" 2>/dev/null || cd /workspace || exit 1
eval "$1"
_exit_code=$?
export -p > /tmp/.swerl_vanillux_env
pwd > /tmp/.swerl_vanillux_cwd
exit $_exit_code
"""
        result = await self.client.execute(command="cat > /tmp/.swerl_vanillux_bash_wrapper.sh",
                                           stdin=wrapper, timeout=300, workdir="/")
        if result.get("exit_code") != 0:
            raise RuntimeError(f"Could not install Vanillux wrapper: {result}")

    async def execute(self, *, command, timeout=300, stdin=None, workdir=None, **kwargs):
        if stdin is not None or workdir is not None:
            return await self.client.execute(command=command, timeout=timeout, stdin=stdin, workdir=workdir, **kwargs)
        import shlex
        async with self.lock:
            wrapped = "bash /tmp/.swerl_vanillux_bash_wrapper.sh " + shlex.quote(command)
            if len(wrapped) > 16384:
                # Preserve the command argument verbatim while keeping the API
                # command field within its limit. Upload only in this container.
                import uuid
                path = "/tmp/.primebeaker-replay-command-" + uuid.uuid4().hex
                uploaded = await self.client.execute(
                    command="cat > " + shlex.quote(path), stdin=command,
                    timeout=timeout, workdir="/")
                if uploaded.get("exit_code") != 0:
                    raise RuntimeError(f"Could not transfer replay command: {uploaded}")
                # Sentinel preserves trailing newlines removed by substitution.
                wrapped = ("_primebeaker_payload=$(cat " + shlex.quote(path)
                    + "; printf .); _primebeaker_payload=${_primebeaker_payload%.}; "
                    + "rm -f " + shlex.quote(path)
                    + "; bash /tmp/.swerl_vanillux_bash_wrapper.sh \"$_primebeaker_payload\"")
            return await self.client.execute(
                command=wrapped, timeout=timeout, workdir="/", **kwargs)
