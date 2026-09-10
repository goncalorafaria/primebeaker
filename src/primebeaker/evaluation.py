"""Managed Beaker scheduling for Python evaluation workloads.

PrimeBeaker owns placement and service lifetime.  The application image owns
the evaluation module itself; this package deliberately treats that module and
its arguments as an opaque Python entry point.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
import subprocess
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from primebeaker.common import environment_entries, image_source, require_under_mount


_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")


def _native_launcher(config: Mapping[str, Any]) -> Any:
    """Construct LiteRegistry's native base-deployment launcher lazily."""

    try:
        from literegistry_base_deployment import (
            BaseDeploymentConfig,
            BaseDeploymentLauncher,
        )
    except ImportError as error:
        raise RuntimeError(
            "managed evaluations require primebeaker[runtime] or "
            "literegistry-base-deployment"
        ) from error
    return BaseDeploymentLauncher(BaseDeploymentConfig(**dict(config)))


def experiment_id(receipt: Mapping[str, Any]) -> str:
    """Extract exactly one experiment ID from a Beaker/native receipt."""

    def candidates(value: Any) -> list[str]:
        records = (
            value
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes))
            else (value,)
        )
        result: list[str] = []
        for record in records:
            if not isinstance(record, Mapping):
                continue
            for key in ("experiment_id", "experimentId", "id"):
                item = record.get(key)
                if isinstance(item, str) and item:
                    result.append(item)
                    break
        return result

    beaker = receipt.get("beaker", receipt)
    values = candidates(beaker)
    if not values and isinstance(beaker, Mapping):
        for key in ("experiment", "experiments"):
            values.extend(candidates(beaker.get(key)))
    unique = list(dict.fromkeys(values))
    if len(unique) != 1:
        raise RuntimeError(
            "Beaker receipt did not contain exactly one experiment ID: "
            f"{unique}"
        )
    return unique[0]


class PythonEvaluationRequest(BaseModel):
    """One Python evaluator and the LiteRegistry capacity it depends on."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    module: str = Field(min_length=1)
    arguments: tuple[str, ...] = ()
    image: str = Field(min_length=1)
    run_dir: Path
    services: dict[str, Any]
    required_services: dict[str, int] = Field(default_factory=dict)
    workspace: str = Field(default="ai2/oe-agents", min_length=1)
    evaluation_cluster: str = Field(default="ai2/jupiter", min_length=1)
    priority: Literal["normal", "high", "urgent"] = "high"
    min_runtime_hours: int = Field(default=0, ge=0)
    mount_path: Path = Path("/weka")
    dataset: str = Field(default="oe-adapt-default", min_length=1)
    environment: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    beaker_token_secret: str | None = "BEAKER_TOKEN"
    keep_services: bool = False
    readiness_timeout_seconds: float = Field(default=7200, gt=0)
    experiment_name: str = Field(default="jtceval", min_length=1)
    budget: str | None = "ai2/oe-omai"

    @model_validator(mode="after")
    def _valid_python_workload(self) -> "PythonEvaluationRequest":
        if not _MODULE.fullmatch(self.module):
            raise ValueError(f"invalid Python module path: {self.module!r}")
        if any(not isinstance(value, str) for value in self.arguments):
            raise ValueError("Python module arguments must all be strings")
        if any(not name or count < 1 for name, count in self.required_services.items()):
            raise ValueError("required services need names and positive counts")
        if not self.keep_services and not self.beaker_token_secret:
            raise ValueError(
                "managed service cleanup requires a Beaker token secret"
            )
        require_under_mount((self.run_dir,), self.mount_path)
        return self

    def service_config(self) -> dict[str, Any]:
        """Return native LiteRegistry config with scheduler-owned placement."""

        values = dict(self.services)
        for key, value in (
            ("workspace", self.workspace),
            ("budget", self.budget),
        ):
            if key in values and values[key] != value:
                raise ValueError(
                    f"services.{key} conflicts with evaluation scheduler value"
                )
            if value is not None:
                values[key] = value
        return values


class PythonEvaluationPreview(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    service: dict[str, Any]
    evaluation_experiment_name: str
    coordination_root: str
    evaluation_spec: dict[str, Any]


class PythonEvaluationScheduler:
    """Schedule native services followed by one supervised Python evaluator."""

    beaker_binary = "beaker"

    @staticmethod
    def _coordination_root(request: PythonEvaluationRequest, service_name: str) -> str:
        shared_dir = str(request.service_config().get("shared_dir", "/weka/gfaria"))
        return f"{shared_dir.rstrip('/')}/.literegistry-coop/{service_name}"

    def preview(self, request: PythonEvaluationRequest) -> PythonEvaluationPreview:
        service = _native_launcher(request.service_config()).preview()
        service_name = str(service["experiment_name"])
        coordination_root = self._coordination_root(request, service_name)
        return PythonEvaluationPreview(
            service=service,
            evaluation_experiment_name=f"{request.experiment_name}-{service_name[-10:]}",
            coordination_root=coordination_root,
            evaluation_spec=self._spec(
                request,
                service_experiment_id="<created-at-launch>",
                coordination_root=coordination_root,
            ),
        )

    def submit(self, request: PythonEvaluationRequest) -> dict[str, Any]:
        native = _native_launcher(request.service_config())
        service = native.submit()
        service_id = experiment_id(service)
        service_name = str(service["experiment_name"])
        coordination_root = self._coordination_root(request, service_name)
        evaluation_name = f"{request.experiment_name}-{service_name[-10:]}"
        spec = self._spec(
            request,
            service_experiment_id=service_id,
            coordination_root=coordination_root,
        )
        try:
            evaluation = self._submit_spec(
                name=evaluation_name,
                workspace=request.workspace,
                spec=spec,
            )
        except BaseException as error:
            try:
                native.stop(service_id)
            except Exception as cleanup_error:
                error.add_note(
                    f"also failed to stop service experiment {service_id}: "
                    f"{cleanup_error}"
                )
            else:
                error.add_note(
                    f"stopped service experiment {service_id} after evaluator "
                    "submission failed"
                )
            raise
        return {
            "service_experiment_id": service_id,
            "evaluation_experiment_id": experiment_id(evaluation),
            "coordination_root": coordination_root,
            "service": service,
            "evaluation": evaluation,
        }

    def _submit_spec(
        self,
        *,
        name: str,
        workspace: str,
        spec: Mapping[str, Any],
    ) -> dict[str, Any]:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix=f"{name}-", encoding="utf-8"
        ) as handle:
            json.dump(spec, handle, indent=2)
            handle.write("\n")
            handle.flush()
            try:
                completed = subprocess.run(
                    (
                        self.beaker_binary,
                        "--format",
                        "json",
                        "experiment",
                        "create",
                        "--name",
                        name,
                        "--workspace",
                        workspace,
                        handle.name,
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as error:
                detail = error.stderr.strip() or error.stdout.strip() or "no detail"
                raise RuntimeError(
                    f"Beaker could not create evaluator {name}: {detail}"
                ) from error
        return {"experiment_name": name, "beaker": json.loads(completed.stdout)}

    @staticmethod
    def _spec(
        request: PythonEvaluationRequest,
        *,
        service_experiment_id: str,
        coordination_root: str,
    ) -> dict[str, Any]:
        reserved = {
            "PRIMEBEAKER_EVALUATION_MODULE": request.module,
            "PRIMEBEAKER_EVALUATION_ARGUMENTS_JSON": json.dumps(request.arguments),
            "PRIMEBEAKER_COORDINATION_ROOT": coordination_root,
            "PRIMEBEAKER_REQUIRED_SERVICES_JSON": json.dumps(
                request.required_services, sort_keys=True
            ),
            "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID": service_experiment_id,
            "PRIMEBEAKER_KEEP_SERVICES": str(request.keep_services),
            "PRIMEBEAKER_READINESS_TIMEOUT_SECONDS": str(
                request.readiness_timeout_seconds
            ),
            "PRIMEBEAKER_RUN_DIR": str(request.run_dir),
        }
        registry = request.service_config().get("registry")
        if registry:
            reserved["PRIMEBEAKER_REGISTRY"] = str(registry)
        overlap = sorted(set(reserved) & set(request.environment))
        if overlap:
            raise ValueError(
                f"reserved evaluation environment names cannot be overridden: {overlap}"
            )
        secrets = dict(request.secrets)
        if not request.keep_services:
            if "BEAKER_TOKEN" in secrets:
                raise ValueError("evaluation secrets may not override managed BEAKER_TOKEN")
            assert request.beaker_token_secret is not None
            secrets["BEAKER_TOKEN"] = request.beaker_token_secret
        task = {
            "name": "evaluation",
            "image": image_source(request.image),
            "command": ["python3", "-m", "primebeaker.evaluation_worker"],
            "envVars": environment_entries(
                {**reserved, **request.environment}, secrets
            ),
            "datasets": [
                {
                    "mountPath": str(request.mount_path),
                    "source": {"weka": request.dataset},
                }
            ],
            "result": {"path": "/tmp/beaker-result"},
            "context": {
                "priority": request.priority,
                "minRuntime": f"{request.min_runtime_hours}h",
                "autoResume": True,
            },
            "constraints": {"cluster": [request.evaluation_cluster]},
        }
        spec: dict[str, Any] = {
            "version": "v2",
            "description": (
                f"PrimeBeaker managed Python evaluation: {request.experiment_name}"
            ),
            "tasks": [task],
        }
        if request.budget:
            spec["budget"] = request.budget
        return spec


__all__ = [
    "PythonEvaluationPreview",
    "PythonEvaluationRequest",
    "PythonEvaluationScheduler",
    "experiment_id",
]
