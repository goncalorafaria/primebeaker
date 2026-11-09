"""Capture judge HTTP attempts without changing transport retry/error semantics."""
from __future__ import annotations

from typing import Any

import aiohttp
from literegistry_tool_client import JudgeClient


class JudgeTraceError(RuntimeError):
    def __init__(self, message: str, attempts: list[dict[str, Any]]):
        super().__init__(message)
        self.judge_attempts = attempts


class TracedJudgeClient(JudgeClient):
    def _prepare_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**super()._prepare_payload(payload), "include_judge_trace": True}

    async def verify_output(self, **kwargs: Any) -> dict[str, Any]:
        # Keep buffers request-local: this client is shared by concurrent rollouts.
        attempts: list[dict[str, Any]] = []
        observer = aiohttp.TraceConfig()

        async def capture(session, context, params):
            response = params.response
            entry: dict[str, Any] = {"http_status": response.status}
            attempts.append(entry)
            try:
                entry["response"] = await response.json(content_type=None)
            except Exception as exc:
                entry["capture_error"] = f"{type(exc).__name__}: {exc}"

        observer.on_request_end.append(capture)
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout),
                trust_env=False,
                trace_configs=[observer],
            ) as session:
                client = TracedJudgeClient(
                    self.server_url,
                    service_model_path=self.service_model_path,
                    timeout=self.timeout,
                    max_retries=self.max_retries,
                    http_session=session,
                )
                # execute uses the original transport, including its HTTP retries.
                result = await client.execute(**kwargs)
            return {**result, "judge_http_attempts": attempts}
        except Exception as exc:
            raise JudgeTraceError(str(exc), attempts) from exc
