"""LiteRegistry gateway used by tool-enabled Prime-RL environments."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import fire
from literegistry import get_kvstore
from literegistry.client import RegistryClient

try:
    from literegistry.gateway.affinity import StrictAffinityGateway
except ImportError:  # LiteRegistry < 1.0.39
    from literegistry.gateway_affinity import StrictAffinityGateway
from literegistry.gateway import (
    Gateway,
    GatewayConfig,
    GatewayRequestError,
    ProxyRoute,
    RetryConfig,
    RoutingPolicy,
    advertised_gateway_url,
    default_proxy_routes,
)


def prepare_judge(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    forwarded = dict(payload)
    service = forwarded.pop("model_path", "judge")
    if not isinstance(service, str) or not service:
        raise GatewayRequestError("model_path must be a non-empty string")
    return service, forwarded


def prepare_search(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    forwarded = dict(payload)
    mode = forwarded.get("mode")
    if mode not in {"query", "url"}:
        raise GatewayRequestError("mode must be either 'query' or 'url'")
    required = "query" if mode == "query" else "url"
    if not forwarded.get(required):
        raise GatewayRequestError(f"{required} parameter required for {mode} mode")
    service = forwarded.pop("model_path", "search")
    if not isinstance(service, str) or not service:
        raise GatewayRequestError("model_path must be a non-empty string")
    return service, forwarded


def gateway_config() -> GatewayConfig:
    config = GatewayConfig.from_env()
    retry = dict(config.retry)
    judge_timeout = float(os.getenv("JUDGE_TIMEOUT", "240"))
    retry["judge"] = RetryConfig(
        timeout=judge_timeout,
        connect_timeout=3,
        max_retries=int(os.getenv("JUDGE_MAX_RETRIES", "1")),
        retry_budget_seconds=judge_timeout,
        retry_backoff_seconds=0.1,
    )
    return replace(config, retry=retry)


class PrimeBeakerGateway(Gateway):
    """Native LiteRegistry routes plus search selection and judge routing."""

    def __init__(
        self,
        registry: RegistryClient,
        config: GatewayConfig | None = None,
        *,
        routing: RoutingPolicy | None = None,
        strict_affinity: StrictAffinityGateway | None = None,
    ) -> None:
        routes = [
            *(route for route in default_proxy_routes() if route.path != "/search"),
            ProxyRoute(
                path="/search",
                upstream_endpoint="search",
                prepare=prepare_search,
                retry="search",
                name="search",
            ),
            ProxyRoute(
                path="/judge",
                upstream_endpoint="judge",
                prepare=prepare_judge,
                retry="judge",
                name="judge",
            ),
        ]
        super().__init__(
            registry,
            config=config or gateway_config(),
            routes=routes,
            routing=routing,
            strict_affinity=strict_affinity,
            enable_strict_affinity=True,
            enable_docker_mirror=False,
        )


def create_app():
    registry = RegistryClient(
        get_kvstore(os.environ["REGISTRY_PATH"]),
        service_type="model_path",
        cache_ttl=int(os.getenv("REGISTRY_CACHE_TTL_SECONDS", "5")),
    )
    return PrimeBeakerGateway(registry).app


def serve(
    registry: str,
    port: int,
    advertise_host: str | None = None,
    workers: int = 1,
    affinity_ttl_seconds: float = 900,
    registry_cache_ttl_seconds: int = 5,
    timeout: float = 65,
    judge_timeout: float = 240,
) -> None:
    """Run the PrimeBeaker LiteRegistry gateway."""
    os.environ.update(
        REGISTRY_PATH=registry,
        AFFINITY_TTL_SECONDS=str(affinity_ttl_seconds),
        REGISTRY_CACHE_TTL_SECONDS=str(registry_cache_ttl_seconds),
        TIMEOUT=str(timeout),
        JUDGE_TIMEOUT=str(judge_timeout),
    )
    print(
        f"GATEWAY_URL={advertised_gateway_url(int(port), advertise_host)}",
        flush=True,
    )
    import uvicorn

    uvicorn.run(
        "primebeaker.gateway:create_app",
        factory=True,
        host="0.0.0.0",
        port=int(port),
        workers=int(workers),
    )


def main(argv: Sequence[str] | None = None) -> Any:
    command = None if argv is None else list(argv)
    return fire.Fire(serve, command=command)


if __name__ == "__main__":
    main()
