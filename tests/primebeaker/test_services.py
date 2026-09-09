from __future__ import annotations

from primebeaker.services import LiteRegistryServices
from primebeaker.cli import run


def test_services_delegate_config_and_actions_to_literegistry(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class Config:
        def __init__(self, **kwargs):
            seen["config"] = kwargs

    class Launcher:
        def __init__(self, config):
            seen["launcher_config"] = config

        def preview(self):
            return {"action": "preview"}

        def submit(self):
            return {"action": "launch"}

        @staticmethod
        def stop(experiment_id, dry_run=False):
            return {"experiment_id": experiment_id, "dry_run": dry_run}

    monkeypatch.setattr(
        "primebeaker.services._native", lambda: (Config, Launcher)
    )
    services = LiteRegistryServices()

    assert services.preview(
        head_registry="/weka/registry",
        service_cluster="ai2/jupiter,ai2/saturn",
        terminal_replicas=3,
    ) == {"action": "preview"}
    assert seen["config"] == {
        "head_registry": "/weka/registry",
        "service_clusters": ("ai2/jupiter", "ai2/saturn"),
        "terminal_replicas": 3,
    }
    assert services.launch(registry="redis://host:6379") == {"action": "launch"}
    assert services.stop("01STACK", dry_run=True) == {
        "experiment_id": "01STACK",
        "dry_run": True,
    }


def test_fire_services_command_forwards_native_config(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class Config:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class Launcher:
        def __init__(self, config):
            pass

        def preview(self):
            return {"native": True}

    monkeypatch.setattr(
        "primebeaker.services._native", lambda: (Config, Launcher)
    )

    result = run([
        "services",
        "preview",
        "--head-registry=/weka/registry",
        "--terminal-replicas=3",
    ])

    assert result == {"native": True}
    assert seen == {
        "head_registry": "/weka/registry",
        "terminal_replicas": 3,
    }


def test_fire_podman_services_command_uses_native_launcher(monkeypatch) -> None:
    seen: dict[str, object] = {}

    class Config:
        def __init__(self, **kwargs):
            seen.update(kwargs)

    class Launcher:
        def __init__(self, config):
            pass

        def preview(self):
            return {"podman": True}

    monkeypatch.setattr(
        "primebeaker.services._native_podman", lambda: (Config, Launcher)
    )

    result = run([
        "services",
        "podman",
        "preview",
        "--service-cluster=ai2/jupiter",
        "--podman-replicas=4",
    ])

    assert result == {"podman": True}
    assert seen == {
        "service_clusters": ("ai2/jupiter",),
        "podman_replicas": 4,
    }
