#!/usr/bin/env python3
"""LiteRegistry-backed rubric judge implemented with Starlette and uvicorn.

The service registers itself as a ``model_path`` in LiteRegistry (``judge`` by
default).  A request supplies the model path of the LLM that should perform the
judgment; the service sends inference through its local LiteRegistry gateway,
which selects live replicas to evaluate the output against the supplied rubrics.
"""

from __future__ import annotations

import asyncio
import logging
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fire
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from literegistry import ServerRegistry, get_kvstore
from literegistry.shared_session import get_session_manager
import uvicorn
from primebeaker.judge.verifier_runner import run_verifier_tool_judge


logger = logging.getLogger(__name__)


_ENV_PREFIX = "JUDGE_SERVER_"


@dataclass(frozen=True)
class JudgeServerConfig:
    host: str = "0.0.0.0"
    port: int = 8090
    registry: str = "redis://localhost:6379"
    model_path: str = "judge"
    model_gateway_url: str = "http://127.0.0.1:1212"
    heartbeat_interval: float = 10.0
    timeout: float = 90.0
    max_retries: int = 3
    rubric_max_retries: int = 1
    max_tokens: int = 8192
    max_tool_calls: int = 8
    rollout_timeout: float = 3600.0
    tool_server_url: str = "http://127.0.0.1:1212"
    tools: tuple[str, ...] = ("terminal",)
    prompt_template_path: str = str(
        Path(__file__).resolve().parents[1]
        / "resources/judge_templates/jtc-io-repl-prompt-gptoss-harmony.json"
    )
    model_profiles_dir: str = str(
        Path(__file__).resolve().parents[1] / "resources/judge_profiles"
    )
    allowed_origin: str = "*"


@dataclass(frozen=True)
class JudgeRequest:
    input: str
    output: str
    rubrics: list[str]
    model: str

    @classmethod
    def from_json(cls, payload: Any) -> "JudgeRequest":
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")

        required = ("input", "output", "rubrics", "model")
        missing = [field for field in required if field not in payload]
        if missing:
            raise ValueError(f"missing required field(s): {', '.join(missing)}")

        input_text = payload["input"]
        output_text = payload["output"]
        model = payload["model"]
        rubrics = payload["rubrics"]
        if not isinstance(input_text, str):
            raise ValueError("input must be text")
        if not isinstance(output_text, str):
            raise ValueError("output must be text")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a non-empty model path")
        if not isinstance(rubrics, list) or not rubrics:
            raise ValueError("rubrics must be a non-empty list of text")
        if not all(isinstance(rubric, str) and rubric.strip() for rubric in rubrics):
            raise ValueError("each rubric must be non-empty text")
        return cls(
            input=input_text,
            output=output_text,
            rubrics=rubrics,
            model=model.strip(),
        )

class JudgeServer:
    """Starlette judge worker with LiteRegistry registration and heartbeats."""

    def __init__(self, config: JudgeServerConfig | None = None):
        self.config = config or JudgeServerConfig()
        self.store = get_kvstore(self.config.registry)
        self.server_registry = ServerRegistry(store=self.store)
        self.url = f"http://{socket.getfqdn()}"
        self._running = False
        self._heartbeat_task: asyncio.Task[None] | None = None
        self.app = self._create_app()

    def _metadata(self) -> dict[str, Any]:
        return {
            "model_path": self.config.model_path,
            "host": self.config.host,
            "port": self.config.port,
            "backend": "rubric-judge",
            "worker_pid": os.getpid(),
            "extra_kwargs": {
                "endpoint": "/judge",
                "request_fields": ["input", "output", "rubrics", "model"],
                "response_format": "rubric pass/fail JSON",
            },
        }

    async def start(self) -> None:
        await self.server_registry.register_server(
            url=self.url, port=self.config.port, metadata=self._metadata()
        )
        self._running = True
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
        logger.info(
            "Registered judge server at %s:%s as model_path=%s",
            self.url,
            self.config.port,
            self.config.model_path,
        )

    async def _heartbeat_loop(self) -> None:
        while self._running:
            try:
                await self.server_registry.heartbeat(self.url, self.config.port)
            except Exception:
                logger.exception("Judge server heartbeat failed")
            await asyncio.sleep(self.config.heartbeat_interval)

    async def stop(self) -> None:
        self._running = False
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None
        try:
            await self.server_registry.deregister()
        finally:
            close = getattr(self.store, "close", None)
            if close is not None:
                await close()

    async def judge(self, request: Request) -> JSONResponse:
        try:
            judge_request = JudgeRequest.from_json(await request.json())
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)

        try:
            # Each synchronous workflow gets its own worker, with no shared
            # request queue. Rubric batching remains internal to the workflow.
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="judge-workflow")
            try:
                result = await run_verifier_tool_judge(
                    judge_request,
                    config=self.config,
                    executor=executor,
                )
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
            return JSONResponse(result)
        except Exception as exc:
            logger.exception("Judge request failed for model=%s", judge_request.model)
            return JSONResponse({"error": str(exc), "status": "failed"}, status_code=502)

    async def health(self, request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "status": "healthy" if self._running else "starting",
                "service": "judge-server",
                "model_path": self.config.model_path,
            }
        )

    async def metadata(self, request: Request) -> JSONResponse:
        return JSONResponse(self._metadata())

    def _create_app(self) -> Starlette:
        @asynccontextmanager
        async def lifespan(app: Starlette):
            session_manager = get_session_manager()
            await session_manager.initialize()
            await self.start()
            try:
                yield
            finally:
                await self.stop()
                await session_manager.shutdown()

        app = Starlette(
            routes=[
                Route("/health", self.health, methods=["GET"]),
                Route("/metadata", self.metadata, methods=["GET"]),
                Route("/judge", self.judge, methods=["POST"]),
            ],
            lifespan=lifespan,
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=[self.config.allowed_origin],
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type"],
        )
        return app


def _parse_tools(tools: str) -> tuple[str, ...]:
    return tuple(tool.strip() for tool in tools.split(",") if tool.strip())


def _config_from_environment() -> JudgeServerConfig:
    """Build a worker configuration passed by the Fire parent process."""
    defaults = JudgeServerConfig()
    environment = os.environ
    return JudgeServerConfig(
        host=environment.get(f"{_ENV_PREFIX}HOST", defaults.host),
        port=int(environment.get(f"{_ENV_PREFIX}PORT", defaults.port)),
        registry=environment.get(f"{_ENV_PREFIX}REGISTRY", defaults.registry),
        model_path=environment.get(f"{_ENV_PREFIX}MODEL_PATH", defaults.model_path),
        model_gateway_url=environment.get(f"{_ENV_PREFIX}MODEL_GATEWAY_URL", defaults.model_gateway_url),
        heartbeat_interval=float(
            environment.get(f"{_ENV_PREFIX}HEARTBEAT_INTERVAL", defaults.heartbeat_interval)
        ),
        timeout=float(environment.get(f"{_ENV_PREFIX}TIMEOUT", defaults.timeout)),
        max_retries=int(environment.get(f"{_ENV_PREFIX}MAX_RETRIES", defaults.max_retries)),
        rubric_max_retries=int(
            environment.get(
                f"{_ENV_PREFIX}RUBRIC_MAX_RETRIES", defaults.rubric_max_retries
            )
        ),
        max_tokens=int(environment.get(f"{_ENV_PREFIX}MAX_TOKENS", defaults.max_tokens)),
        max_tool_calls=int(
            environment.get(f"{_ENV_PREFIX}MAX_TOOL_CALLS", defaults.max_tool_calls)
        ),
        rollout_timeout=float(
            environment.get(f"{_ENV_PREFIX}ROLLOUT_TIMEOUT", defaults.rollout_timeout)
        ),
        tool_server_url=environment.get(
            f"{_ENV_PREFIX}TOOL_SERVER_URL", defaults.tool_server_url
        ),
        tools=_parse_tools(environment.get(f"{_ENV_PREFIX}TOOLS", ",".join(defaults.tools))),
        prompt_template_path=environment.get(
            f"{_ENV_PREFIX}PROMPT_TEMPLATE_PATH", defaults.prompt_template_path
        ),
        model_profiles_dir=environment.get(
            f"{_ENV_PREFIX}MODEL_PROFILES_DIR", defaults.model_profiles_dir
        ),
        allowed_origin=environment.get(f"{_ENV_PREFIX}ALLOWED_ORIGIN", defaults.allowed_origin),
    )


def create_app() -> Starlette:
    """Uvicorn factory used when ``--workers`` starts child processes."""
    return JudgeServer(_config_from_environment()).app


def _publish_worker_configuration(config: JudgeServerConfig) -> None:
    values = {
        "HOST": config.host,
        "PORT": config.port,
        "REGISTRY": config.registry,
        "MODEL_PATH": config.model_path,
        "MODEL_GATEWAY_URL": config.model_gateway_url,
        "HEARTBEAT_INTERVAL": config.heartbeat_interval,
        "TIMEOUT": config.timeout,
        "MAX_RETRIES": config.max_retries,
        "RUBRIC_MAX_RETRIES": config.rubric_max_retries,
        "MAX_TOKENS": config.max_tokens,
        "MAX_TOOL_CALLS": config.max_tool_calls,
        "ROLLOUT_TIMEOUT": config.rollout_timeout,
        "TOOL_SERVER_URL": config.tool_server_url,
        "TOOLS": ",".join(config.tools),
        "PROMPT_TEMPLATE_PATH": config.prompt_template_path,
        "MODEL_PROFILES_DIR": config.model_profiles_dir,
        "ALLOWED_ORIGIN": config.allowed_origin,
    }
    for name, value in values.items():
        os.environ[f"{_ENV_PREFIX}{name}"] = str(value)


def main(
    host: str = "0.0.0.0",
    port: int = 8090,
    registry: str = "redis://localhost:6379",
    model_path: str = "judge",
    model_gateway_url: str = JudgeServerConfig.model_gateway_url,
    heartbeat_interval: float = 10.0,
    timeout: float = 90.0,
    max_retries: int = 3,
    rubric_max_retries: int = 1,
    max_tokens: int = 8192,
    max_tool_calls: int = 8,
    rollout_timeout: float = 3600.0,
    tool_server_url: str = "http://127.0.0.1:1212",
    tools: str = "terminal",
    prompt_template_path: str = JudgeServerConfig.prompt_template_path,
    model_profiles_dir: str = JudgeServerConfig.model_profiles_dir,
    allowed_origin: str = "*",
    workers: int = 1,
) -> None:
    """Start the LiteRegistry rubric judge server."""
    workers = int(workers)
    if workers < 1:
        raise ValueError("workers must be at least 1")

    logging.basicConfig(level=logging.INFO)
    config = JudgeServerConfig(
        host=host,
        port=port,
        registry=registry,
        model_path=model_path,
        model_gateway_url=model_gateway_url,
        heartbeat_interval=heartbeat_interval,
        timeout=timeout,
        max_retries=max_retries,
        rubric_max_retries=rubric_max_retries,
        max_tokens=max_tokens,
        max_tool_calls=max_tool_calls,
        rollout_timeout=rollout_timeout,
        tool_server_url=tool_server_url,
        tools=_parse_tools(tools),
        prompt_template_path=prompt_template_path,
        model_profiles_dir=model_profiles_dir,
        allowed_origin=allowed_origin,
    )
    if workers == 1:
        uvicorn.run(
            JudgeServer(config).app,
            host=config.host,
            port=config.port,
            log_level="info",
            access_log=False,
        )
        return

    _publish_worker_configuration(config)
    uvicorn.run(
        "primebeaker.judge.judge_server:create_app",
        host=config.host,
        port=config.port,
        workers=workers,
        factory=True,
        log_level="info",
        access_log=False,
    )


def cli() -> None:
    fire.Fire(main)


if __name__ == "__main__":
    cli()
