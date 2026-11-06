"""Bind a standalone LiteRegistry Podman session to a replay rollout."""
from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

from literegistry_podman_client import PodmanClient


class ReplayPodmanClient:
    """Execute once in a known session; never retry unknown command outcomes."""

    def __init__(self, env):
        self.client = PodmanClient(
            gateway_url=env.podman_gateway_url,
            service=env.service,
            workdir=env.podman_workdir,
            request_timeout=max(env.request_timeout, env.handshake_timeout),
        )
        self.client_id = f"{env.client_id_prefix}-{uuid4().hex}"
        self.session = None

    @property
    def started(self):
        return self.session is not None

    @property
    def container_id(self):
        return self.session.container_id if self.session else None

    @property
    def affinity_id(self):
        return self.session.affinity_id if self.session else None

    async def start(self, *, image):
        if self.session is not None:
            raise RuntimeError("replay client already owns a container")
        await self.client.open()
        try:
            self.session = await self.client.handshake(image=image, client_id=self.client_id)
        except BaseException:
            await self.client.aclose()
            raise
        return {"container_id": self.container_id, "affinity_id": self.affinity_id,
                "worker_instance_id": self.session.instance_id}

    async def execute(self, *, command, timeout=300, stdin="", workdir=None):
        if self.session is None:
            raise RuntimeError("replay session has not started")
        return asdict(await self.session.execute(
            command, timeout=timeout, stdin=stdin, workdir=workdir,
        ))

    async def close(self):
        try:
            if self.session is not None:
                result = await self.session.close()
                self.session = None
                return result
        finally:
            await self.client.aclose()
