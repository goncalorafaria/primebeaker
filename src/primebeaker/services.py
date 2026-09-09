"""Thin Fire-compatible adapter for LiteRegistry's native Beaker deployment."""

from __future__ import annotations

from typing import Any


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


class LiteRegistryServices:
    """Delegate stack preview, launch, and stop to LiteRegistry's own package."""

    def __init__(self) -> None:
        self.podman = LiteRegistryPodmanServices()

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
