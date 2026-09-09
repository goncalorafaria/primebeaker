from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from primebeaker import coordination


def test_wait_for_redis_delegates_to_literegistry(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, float, float]] = []

    def ready(registry: str, *, timeout: float, poll_interval: float) -> None:
        calls.append((registry, timeout, poll_interval))

    monkeypatch.setattr(coordination, "_literegistry_wait_for_redis", ready)

    assert coordination.wait_for_redis(
        "redis://registry:6379", timeout=10, poll_interval=2
    ) == "redis://registry:6379"
    assert calls == [("redis://registry:6379", pytest.approx(2), pytest.approx(2))]


def test_wait_for_redis_preserves_shutdown_file_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stop_file = tmp_path / "shutdown.requested"
    stop_file.touch()
    monkeypatch.setattr(
        coordination,
        "_literegistry_wait_for_redis",
        lambda *args, **kwargs: pytest.fail("native wait should not run after shutdown"),
    )

    with pytest.raises(RuntimeError, match="shutdown requested"):
        coordination.wait_for_redis(
            "redis://registry:6379", timeout=10, stop_file=stop_file
        )


def test_wait_for_redis_resolves_literegistry_head_uri(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}

    class Store:
        async def ping(self) -> bool:
            state["pinged"] = True
            return True

        async def close(self) -> None:
            state["closed"] = True

    def get_store(registry: str, **kwargs: object) -> Store:
        state["registry"] = registry
        state["kwargs"] = kwargs
        return Store()

    monkeypatch.setattr(coordination, "get_kvstore", get_store)

    registry = "head+file:///weka/shared/services"
    assert coordination.wait_for_redis(
        registry, timeout=10, poll_interval=2
    ) == registry
    assert state == {
        "registry": registry,
        "kwargs": {"raise_on_error": True},
        "pinged": True,
        "closed": True,
    }


def test_service_counts_use_literegistry_live_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state: dict[str, object] = {}

    class FakeRegistryClient:
        def __init__(self, store: object, **kwargs: object) -> None:
            state["store"] = store
            state["kwargs"] = kwargs

        async def models(self, force: bool = False) -> dict[str, list[object]]:
            state["force"] = force
            return {"terminal": [{}, {}], "judge": [{}]}

        async def close(self) -> None:
            state["closed"] = True

    store = object()
    monkeypatch.setattr(coordination, "get_kvstore", lambda *args, **kwargs: store)
    monkeypatch.setattr(coordination, "RegistryClient", FakeRegistryClient)

    assert asyncio.run(coordination._service_counts_async("redis://registry:6379")) == {
        "terminal": 2,
        "judge": 1,
    }
    assert state == {
        "store": store,
        "kwargs": {
            "service_type": "model_path",
            "cache_ttl": 1,
            "max_heartbeat_interval": 240,
        },
        "force": True,
        "closed": True,
    }


def test_coordination_contains_no_private_redis_protocol() -> None:
    source = Path(coordination.__file__).read_text(encoding="utf-8")
    assert "_resp_command" not in source
    assert 'command("SCAN"' not in source
