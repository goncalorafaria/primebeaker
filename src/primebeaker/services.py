"""Thin Fire-compatible adapter for LiteRegistry's native Beaker deployment."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
import re
from typing import Any

from primebeaker.service_images import LiteRegistryImageInstaller


_YAML_SCHEMA = "primebeaker.services/v1"
_UNRESOLVED_ENVIRONMENT = re.compile(
    r"\$[A-Za-z_][A-Za-z0-9_]*|\$" r"\{[^}]+\}"
)


def _expand_environment(value: Any) -> Any:
    """Expand environment variables in a declarative services configuration."""
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if _UNRESOLVED_ENVIRONMENT.search(expanded):
            raise ValueError(f"unresolved environment variable in YAML value {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def load_services_yaml(path: str | Path) -> dict[str, Any]:
    """Load one strict, environment-expandable LiteRegistry service YAML file."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML service configuration requires PyYAML; install primebeaker[runtime]"
        ) from error
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("service YAML must contain an object")
    if document.get("schema") != _YAML_SCHEMA:
        raise ValueError(f"service YAML schema must be {_YAML_SCHEMA!r}")
    if set(document) != {"schema", "services"}:
        raise ValueError("service YAML may contain only schema and services")
    services = document["services"]
    if not isinstance(services, Mapping):
        raise ValueError("service YAML services must contain an object")
    return _expand_environment(dict(services))


def _native() -> tuple[type[Any], type[Any]]:
    try:
        from literegistry_base_deployment import (
            BaseDeploymentConfig,
            BaseDeploymentLauncher,
        )
    except ImportError as error:
        raise RuntimeError(
            "LiteRegistry deployment support is optional; install primebeaker[runtime] "
            "or literegistry-base-deployment"
        ) from error
    return BaseDeploymentConfig, BaseDeploymentLauncher


def _native_podman() -> tuple[type[Any], type[Any]]:
    try:
        from literegistry_podman_beaker import PodmanStackConfig, PodmanStackLauncher
    except ImportError as error:
        raise RuntimeError(
            "LiteRegistry Podman deployment support is optional; install "
            "primebeaker[runtime] or literegistry-podman-beaker"
        ) from error
    return PodmanStackConfig, PodmanStackLauncher


class LiteRegistryPodmanServices:
    """Delegate Podman and Docker-mirror deployment to LiteRegistry."""

    def __init__(self) -> None:
        self.yaml = LiteRegistryPodmanYamlServices()

    @staticmethod
    def _run(action: str, **config: Any) -> dict[str, Any]:
        config_type, launcher_type = _native_podman()
        service_cluster = config.pop("service_cluster", None)
        if service_cluster is not None:
            if "service_clusters" in config:
                raise ValueError("supply only service_cluster or service_clusters")
            config["service_clusters"] = tuple(
                item.strip() for item in str(service_cluster).split(",") if item.strip()
            )
        launcher = launcher_type(config_type(**config))
        return launcher.preview() if action == "preview" else launcher.submit()

    def preview(self, **config: Any) -> dict[str, Any]:
        """Render LiteRegistry's native Podman/mirror Beaker stack."""

        return self._run("preview", **config)

    def launch(self, **config: Any) -> dict[str, Any]:
        """Launch LiteRegistry's native Podman/mirror Beaker stack."""

        return self._run("launch", **config)

    def stop(self, experiment_id: str, dry_run: bool = False) -> dict[str, Any]:
        """Stop a LiteRegistry Podman-stack experiment by ID."""

        _, launcher_type = _native_podman()
        return launcher_type.stop(experiment_id, dry_run=dry_run)


class LiteRegistryPodmanYamlServices:
    """Load a Podman service-stack YAML and delegate it to LiteRegistry."""

    def preview(self, config: str | Path) -> dict[str, Any]:
        """Render the native Podman stack described by config."""
        return LiteRegistryPodmanServices._run(
            "preview", **load_services_yaml(config)
        )

    def launch(self, config: str | Path) -> dict[str, Any]:
        """Launch the native Podman stack described by config."""
        return LiteRegistryPodmanServices._run(
            "launch", **load_services_yaml(config)
        )


class LiteRegistryServices:
    """Delegate stack preview, launch, and stop to LiteRegistry's own package."""

    def __init__(self) -> None:
        self.podman = LiteRegistryPodmanServices()
        self.images = LiteRegistryImageInstaller()
        self.yaml = LiteRegistryYamlServices()

    @staticmethod
    def _run(action: str, **config: Any) -> dict[str, Any]:
        config_type, launcher_type = _native()
        service_cluster = config.pop("service_cluster", None)
        if service_cluster is not None:
            if "service_clusters" in config:
                raise ValueError("supply only service_cluster or service_clusters")
            config["service_clusters"] = tuple(
                item.strip() for item in str(service_cluster).split(",") if item.strip()
            )
        launcher = launcher_type(config_type(**config))
        return launcher.preview() if action == "preview" else launcher.submit()

    def preview(self, **config: Any) -> dict[str, Any]:
        """Render the native LiteRegistry service-stack Beaker experiments."""

        return self._run("preview", **config)

    def launch(self, **config: Any) -> dict[str, Any]:
        """Launch the native LiteRegistry service stack."""

        return self._run("launch", **config)

    def stop(self, experiment_id: str, dry_run: bool = False) -> dict[str, Any]:
        """Stop a LiteRegistry service-stack experiment by ID."""

        _, launcher_type = _native()
        return launcher_type.stop(experiment_id, dry_run=dry_run)


class LiteRegistryYamlServices:
    """Load a service-stack YAML and delegate it to LiteRegistry unchanged."""

    def preview(self, config: str | Path) -> dict[str, Any]:
        """Render the native service stack described by config."""
        return LiteRegistryServices._run("preview", **load_services_yaml(config))

    def launch(self, config: str | Path) -> dict[str, Any]:
        """Launch the native service stack described by config."""
        return LiteRegistryServices._run("launch", **load_services_yaml(config))
