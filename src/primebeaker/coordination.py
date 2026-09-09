"""PrimeBeaker readiness adapters and their Fire CLI."""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import fire
from literegistry import RegistryClient, get_kvstore
from literegistry.head_registry import is_head_registry_uri
from literegistry.coop.redis import wait_for_redis as _literegistry_wait_for_redis


def _shutdown_requested(stop_file: str | Path | None, *, operation: str) -> None:
    if stop_file is None:
        return
    stop_path = Path(stop_file)
    if stop_path.is_file():
        raise RuntimeError(f"shutdown requested while {operation}: {stop_path}")


async def _head_registry_ping(registry: str, *, timeout: float) -> None:
    """Resolve and ping the live Redis selected by a LiteRegistry head URI."""

    store = get_kvstore(registry, raise_on_error=True)
    try:
        ready = await asyncio.wait_for(store.ping(), timeout=timeout)
        if ready is False:
            raise ConnectionError("resolved Redis did not answer PING")
    finally:
        await store.close()


def wait_for_redis(
    registry: str,
    *,
    timeout: float = 600.0,
    poll_interval: float = 2.0,
    stop_file: str | Path | None = None,
) -> str:
    """Use LiteRegistry's Redis barrier with shutdown cancellation."""
    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("timeouts must be positive")
    deadline = time.monotonic() + timeout
    last_error: BaseException | None = None
    while time.monotonic() < deadline:
        _shutdown_requested(stop_file, operation="waiting for Redis")
        remaining = max(0.0, deadline - time.monotonic())
        attempt_timeout = min(poll_interval, remaining)
        try:
            if is_head_registry_uri(registry):
                asyncio.run(
                    _head_registry_ping(registry, timeout=attempt_timeout)
                )
            else:
                _literegistry_wait_for_redis(
                    registry,
                    timeout=attempt_timeout,
                    poll_interval=min(poll_interval, attempt_timeout),
                )
        except (OSError, ValueError, ConnectionError, TimeoutError) as error:
            last_error = error
        else:
            return registry
    raise TimeoutError(f"timed out waiting for Redis at {registry}: {last_error}")


async def _service_counts_async(registry: str) -> dict[str, int]:
    client = RegistryClient(
        get_kvstore(registry, raise_on_error=True),
        service_type="model_path",
        cache_ttl=1,
        max_heartbeat_interval=240,
    )
    try:
        services = await client.models(force=True)
        return {
            name: len(servers)
            for name, servers in services.items()
            if isinstance(name, str) and name and isinstance(servers, list)
        }
    finally:
        await client.close()


def _service_counts(registry: str) -> dict[str, int]:
    return asyncio.run(_service_counts_async(registry))


def wait_for_services(
    registry: str,
    requirements: Mapping[str, int],
    *,
    timeout: float = 1800.0,
    poll_interval: float = 2.0,
    stop_file: str | Path | None = None,
) -> dict[str, int]:
    """Wait for minimum live service counts through LiteRegistry."""
    expected = dict(requirements)
    if any(
        not isinstance(name, str)
        or not name
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        for name, count in expected.items()
    ):
        raise ValueError("service requirements need names and positive integer counts")
    if not expected:
        return {}
    if timeout <= 0 or poll_interval <= 0:
        raise ValueError("timeouts must be positive")

    deadline = time.monotonic() + timeout
    observed: dict[str, int] = {}
    last_error = ""
    while time.monotonic() < deadline:
        _shutdown_requested(stop_file, operation="waiting for services")
        try:
            observed = _service_counts(registry)
            last_error = ""
        except Exception as error:  # noqa: BLE001 - retry until the deadline
            last_error = str(error)
        if not last_error and all(
            observed.get(name, 0) >= count for name, count in expected.items()
        ):
            return observed
        time.sleep(min(poll_interval, max(0.0, deadline - time.monotonic())))
    missing = {
        name: {"expected": count, "observed": observed.get(name, 0)}
        for name, count in expected.items()
        if observed.get(name, 0) < count
    }
    raise TimeoutError(
        f"timed out waiting for LiteRegistry services: {last_error or missing}"
    )


def probe_gateway(
    url: str,
    *,
    model_paths: Sequence[str] = (),
    required_paths: Sequence[str] = (),
    timeout: float = 70.0,
) -> None:
    """Probe PrimeBeaker routes after generic gateway readiness."""
    base = url.rstrip("/")
    try:
        with urlopen(base + "/health", timeout=min(timeout, 10.0)) as response:
            if response.status != 200:
                raise RuntimeError(f"gateway health returned HTTP {response.status}")
        for path in required_paths:
            try:
                with urlopen(
                    Request(base + path, method="OPTIONS"),
                    timeout=10.0,
                ) as response:
                    if response.status not in {200, 204}:
                        raise RuntimeError(
                            f"gateway route {path} returned HTTP {response.status}"
                        )
            except HTTPError as error:
                if error.code != 405:
                    raise
        for model_path in model_paths:
            payload = json.dumps(
                {
                    "mode": "query",
                    "query": "PrimeBeaker readiness probe",
                    "model_path": model_path,
                }
            ).encode()
            request = Request(
                base + "/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urlopen(request, timeout=timeout) as response:
                if response.status != 200:
                    raise RuntimeError(
                        f"gateway search probe returned HTTP {response.status}"
                    )
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(
            f"gateway probe returned HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"gateway probe could not reach {url}: {error.reason}"
        ) from error


class CoordinationCLI:
    """Wait for Redis/services and probe the PrimeBeaker gateway."""

    def wait_redis(
        self,
        registry: str,
        timeout: float = 600.0,
        stop_file: str | None = None,
    ) -> str:
        return wait_for_redis(registry, timeout=timeout, stop_file=stop_file)

    def wait_services(
        self,
        registry: str,
        requirements_json: str,
        timeout: float = 1800.0,
        stop_file: str | None = None,
    ) -> dict[str, int]:
        requirements = (
            json.loads(requirements_json)
            if isinstance(requirements_json, str)
            else requirements_json
        )
        if not isinstance(requirements, dict):
            raise ValueError("requirements_json must decode to an object")
        return wait_for_services(
            registry,
            requirements,
            timeout=timeout,
            stop_file=stop_file,
        )

    def probe_gateway(
        self,
        url: str,
        model_paths_json: str = "[]",
        required_path: str | Sequence[str] | None = None,
        timeout: float = 70.0,
    ) -> None:
        model_paths = (
            json.loads(model_paths_json)
            if isinstance(model_paths_json, str)
            else model_paths_json
        )
        if not isinstance(model_paths, list):
            raise ValueError("model_paths_json must decode to a list")
        paths = (
            []
            if required_path is None
            else [required_path]
            if isinstance(required_path, str)
            else list(required_path)
        )
        probe_gateway(
            url,
            model_paths=model_paths,
            required_paths=paths,
            timeout=timeout,
        )


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return "" if value is None else str(value)


def main(argv: Sequence[str] | None = None) -> int:
    command = None if argv is None else list(argv)
    try:
        fire.Fire(CoordinationCLI(), command=command, serialize=_serialize)
        return 0
    except (OSError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"primebeaker.coordination: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
