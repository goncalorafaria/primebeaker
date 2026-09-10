"""Declarative joint LiteRegistry-service and RL launches."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
from typing import Any

from primebeaker.services import LiteRegistryServices


_RUN_SCHEMA = "primebeaker.run/v1"
_CLEANUP_EXPERIMENT_ENV = "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID"
_DEFAULT_BEAKER_TOKEN_SECRET = "BEAKER_TOKEN"


def _expand_environment(value: Any) -> Any:
    """Expand environment variables and reject unresolved substitutions."""
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        if "$" in expanded:
            raise ValueError(f"unresolved environment variable in run YAML value {value!r}")
        return expanded
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _expand_environment(item) for key, item in value.items()}
    return value


def load_run_yaml(path: str | Path) -> dict[str, Any]:
    """Load one strict service + RL run description."""
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError(
            "YAML run configuration requires PyYAML; install primebeaker[runtime]"
        ) from error

    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("run YAML must contain an object")
    allowed = {"schema", "service", "rl", "lifecycle"}
    unknown = sorted(set(document) - allowed)
    if unknown:
        raise ValueError(f"run YAML contains unsupported keys: {unknown}")
    if document.get("schema") != _RUN_SCHEMA:
        raise ValueError(f"run YAML schema must be {_RUN_SCHEMA!r}")
    for section in ("service", "rl"):
        if not isinstance(document.get(section), Mapping):
            raise ValueError(f"run YAML {section} must contain an object")
    lifecycle = document.get("lifecycle", {})
    if not isinstance(lifecycle, Mapping):
        raise ValueError("run YAML lifecycle must contain an object")
    lifecycle_allowed = {"cleanup_service_on_training_exit", "beaker_token_secret"}
    lifecycle_unknown = sorted(set(lifecycle) - lifecycle_allowed)
    if lifecycle_unknown:
        raise ValueError(
            f"run YAML lifecycle contains unsupported keys: {lifecycle_unknown}"
        )

    expanded = _expand_environment(dict(document))
    rl = dict(expanded["rl"])
    toml = rl.get("toml")
    if not isinstance(toml, str) or not toml.strip():
        raise ValueError("run YAML rl.toml must be a non-empty path")
    toml_path = Path(toml).expanduser()
    if not toml_path.is_absolute():
        toml_path = config_path.parent / toml_path
    rl["toml"] = str(toml_path.resolve())
    expanded["rl"] = rl
    expanded["lifecycle"] = dict(expanded.get("lifecycle", {}))
    expanded["config_path"] = str(config_path)
    return expanded


def _pairs(value: Any, *, name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise ValueError(f"run YAML rl.{name} must contain an object")
    return [f"{key}={item}" for key, item in value.items()]


def _derive_registry(service: Mapping[str, Any]) -> str | None:
    registry = service.get("registry")
    if isinstance(registry, str) and registry:
        return registry
    head = service.get("head_registry")
    if not isinstance(head, str) or not head:
        return None
    return head if head.startswith("head+") else f"head+{head}"


def _rl_kwargs(
    document: Mapping[str, Any],
    *,
    service_experiment_id: str | None,
) -> dict[str, Any]:
    service = document["service"]
    rl = dict(document["rl"])
    assert isinstance(service, Mapping)

    if "clusters" in rl:
        if "cluster" in rl:
            raise ValueError("run YAML rl may supply only cluster or clusters")
        rl["cluster"] = rl.pop("clusters")
    if "environment" in rl:
        if "env" in rl:
            raise ValueError("run YAML rl may supply only env or environment")
        rl["env"] = _pairs(rl.pop("environment"), name="environment")
    if "secrets" in rl:
        if "secret" in rl:
            raise ValueError("run YAML rl may supply only secret or secrets")
        rl["secret"] = _pairs(rl.pop("secrets"), name="secrets")
    if "required_services" in rl:
        if "required_service" in rl:
            raise ValueError(
                "run YAML rl may supply only required_service or required_services"
            )
        rl["required_service"] = _pairs(
            rl.pop("required_services"), name="required_services"
        )
    if not rl.get("registry"):
        registry = _derive_registry(service)
        if registry:
            rl["registry"] = registry

    lifecycle = document.get("lifecycle", {})
    assert isinstance(lifecycle, Mapping)
    cleanup = lifecycle.get("cleanup_service_on_training_exit", True)
    if not isinstance(cleanup, bool):
        raise ValueError(
            "run YAML lifecycle.cleanup_service_on_training_exit must be boolean"
        )
    if cleanup:
        from primebeaker.multinode import deployment_type

        if deployment_type(rl["toml"]) != "multi_node":
            raise ValueError(
                "trainer-side service cleanup currently requires multi-node RL; "
                "set lifecycle.cleanup_service_on_training_exit=false for single-node RL"
            )
        token_secret = lifecycle.get(
            "beaker_token_secret", _DEFAULT_BEAKER_TOKEN_SECRET
        )
        if not isinstance(token_secret, str) or not token_secret.strip():
            raise ValueError(
                "run YAML lifecycle.beaker_token_secret must be a non-empty secret name"
            )
        environment = list(rl.get("env") or [])
        if any(
            str(item).partition("=")[0] == _CLEANUP_EXPERIMENT_ENV
            for item in environment
        ):
            raise ValueError(
                f"rl.environment may not override {_CLEANUP_EXPERIMENT_ENV}"
            )
        environment.append(
            f"{_CLEANUP_EXPERIMENT_ENV}="
            f"{service_experiment_id or '<created-at-launch>'}"
        )
        rl["env"] = environment

        secrets = list(rl.get("secret") or [])
        if any(
            str(item).partition("=")[0] == "BEAKER_TOKEN" for item in secrets
        ):
            raise ValueError("rl.secrets may not override managed BEAKER_TOKEN")
        secrets.append(f"BEAKER_TOKEN={token_secret.strip()}")
        rl["secret"] = secrets
    return rl


def _experiment_id(receipt: Mapping[str, Any]) -> str:
    """Extract the service experiment ID from a native launcher receipt."""

    def direct(value: Any) -> list[str]:
        values: list[str] = []
        records = (
            value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else (value,)
        )
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for key in ("experiment_id", "experimentId", "id"):
                candidate = record.get(key)
                if isinstance(candidate, str) and candidate:
                    values.append(candidate)
                    break
        return values

    beaker = receipt.get("beaker", receipt)
    candidates = direct(beaker)
    if not candidates and isinstance(beaker, Mapping):
        for key in ("experiment", "experiments"):
            candidates.extend(direct(beaker.get(key)))
    unique = list(dict.fromkeys(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            "LiteRegistry launch receipt did not contain exactly one service "
            f"experiment ID: {unique}"
        )
    return unique[0]


def _training(
    action: str,
    document: Mapping[str, Any],
    service_id: str | None,
) -> dict[str, Any]:
    # Imported lazily to avoid a module cycle with the root Fire CLI.
    from primebeaker.cli import _launch

    return _launch(
        "rl",
        action,
        **_rl_kwargs(document, service_experiment_id=service_id),
    )


class JointRunCLI:
    """Preview or launch a service stack and its dependent RL experiment."""

    def preview(self, config: str | Path) -> dict[str, Any]:
        """Preview both experiments without submitting either one."""
        document = load_run_yaml(config)
        service = LiteRegistryServices._run("preview", **document["service"])
        training = _training("preview", document, None)
        return {"service": service, "rl": training}

    def launch(self, config: str | Path) -> dict[str, Any]:
        """Launch services then RL, rolling services back if RL submission fails."""
        document = load_run_yaml(config)
        service = LiteRegistryServices._run("launch", **document["service"])
        service_id = _experiment_id(service)
        try:
            training = _training("submit", document, service_id)
        except BaseException as error:
            try:
                rollback = LiteRegistryServices().stop(service_id)
            except Exception as cleanup_error:
                error.add_note(
                    f"also failed to stop service experiment {service_id}: "
                    f"{cleanup_error}"
                )
            else:
                error.add_note(
                    f"stopped service experiment {service_id}: {rollback}"
                )
            raise
        return {
            "service_experiment_id": service_id,
            "service": service,
            "rl": training,
        }


__all__ = ["JointRunCLI", "load_run_yaml"]
