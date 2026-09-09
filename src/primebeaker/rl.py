"""Typed Beaker launcher for composed single-node Prime-RL TOMLs."""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import shlex
import subprocess
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from primebeaker.common import environment_entries, image_source, require_under_mount
from primebeaker.config import load_toml
from primebeaker.wandb import WandbRunReference


class RLTomlMetadata(BaseModel):
    """Launch-critical values derived exclusively from a Prime-RL TOML."""

    model_config = ConfigDict(frozen=True)

    toml_path: Path
    output_dir: Path
    run_name: str = Field(min_length=1)
    wandb_project: str = Field(min_length=1)
    wandb_entity: str | None = None
    wandb_offline: bool = False
    deployment_type: str = Field(min_length=1)
    train_gpus: int = Field(gt=0)
    inference_gpus: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)

    @property
    def num_gpus(self) -> int:
        return self.train_gpus + self.inference_gpus

    @classmethod
    def from_path(cls, path: str | Path) -> "RLTomlMetadata":
        toml_path = Path(path).resolve()
        if not toml_path.is_file():
            raise FileNotFoundError(toml_path)
        config = load_toml(toml_path)
        wandb = config.get("wandb")
        deployment = config.get("deployment")
        if not isinstance(wandb, dict) or not isinstance(deployment, dict):
            raise ValueError("RL TOML needs [wandb] and [deployment] tables")
        try:
            return cls(
                toml_path=toml_path,
                output_dir=Path(str(config["output_dir"])),
                run_name=str(wandb["name"]),
                wandb_project=str(wandb["project"]),
                wandb_entity=str(wandb["entity"]) if wandb.get("entity") else None,
                wandb_offline=bool(wandb.get("offline", False)),
                deployment_type=str(deployment["type"]),
                train_gpus=int(deployment["num_train_gpus"]),
                inference_gpus=int(deployment["num_infer_gpus"]),
                gpus_per_node=int(deployment["gpus_per_node"]),
            )
        except KeyError as error:
            raise ValueError(f"RL TOML is missing launch field {error.args[0]!r}") from error

    @model_validator(mode="after")
    def _single_node_topology(self) -> "RLTomlMetadata":
        if self.deployment_type != "single_node":
            raise ValueError(
                "the direct `rl @` Beaker launcher supports deployment.type = 'single_node'"
            )
        if self.num_gpus != self.gpus_per_node:
            raise ValueError(
                "single-node RL requires num_train_gpus + num_infer_gpus == gpus_per_node"
            )
        return self


class RLLaunchRequest(BaseModel):
    """Immutable placement and environment policy for one Prime-RL launch."""

    model_config = ConfigDict(frozen=True)

    metadata: RLTomlMetadata
    image: str = Field(min_length=1)
    workspace: str = Field(default="ai2/oe-agents", min_length=1)
    clusters: tuple[str, ...] = ("ai2/holmes",)
    priority: Literal["normal", "high", "urgent"] = "high"
    min_runtime_hours: int = Field(default=0, ge=0)
    mount_path: Path = Path("/weka")
    dataset: str = Field(default="oe-adapt-default", min_length=1)
    working_dir: Path = Path("/weka")
    scratch_dir: Path = Path("/weka")
    home_dir: Path = Path("/weka")
    wandb_secret: str | None = "WANDB_API_KEY"
    hf_secret: str | None = "HF_TOKEN"
    wandb_run_id: str = Field(default_factory=lambda: uuid4().hex, min_length=1)
    wandb_entity: str | None = None
    registry: str | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    setup_command: str | None = None
    budget: str | None = None
    description: str | None = None

    @classmethod
    def from_toml(cls, toml_path: str | Path, *, image: str, **kwargs: Any) -> "RLLaunchRequest":
        return cls(metadata=RLTomlMetadata.from_path(toml_path), image=image, **kwargs)

    @model_validator(mode="after")
    def _paths_and_clusters(self) -> "RLLaunchRequest":
        if not self.clusters or any(not cluster for cluster in self.clusters):
            raise ValueError("at least one non-empty Beaker cluster is required")
        require_under_mount(
            (
                self.metadata.toml_path,
                self.metadata.output_dir,
                self.working_dir,
                self.scratch_dir,
                self.home_dir,
            ),
            self.mount_path,
        )
        return self

    @property
    def experiment_name(self) -> str:
        return self.metadata.run_name

    @property
    def wandb_run(self) -> WandbRunReference:
        return WandbRunReference(
            run_id=self.wandb_run_id,
            project=self.metadata.wandb_project,
            entity=self.metadata.wandb_entity or self.wandb_entity,
            name=self.metadata.run_name,
            offline=self.metadata.wandb_offline,
        )


class RLLaunchPreview(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    spec_path: Path
    spec: dict[str, Any]
    submit_command: tuple[str, ...]
    wandb_run: WandbRunReference


class RLLaunchReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    spec_path: Path
    stdout: str
    wandb_run: WandbRunReference


class RLLaunchBackend(ABC):
    @abstractmethod
    def preview(self, request: RLLaunchRequest) -> RLLaunchPreview: ...

    @abstractmethod
    def submit(self, request: RLLaunchRequest) -> RLLaunchReceipt: ...


class BeakerRLBackend(RLLaunchBackend):
    beaker_binary: str = "beaker"

    def preview(self, request: RLLaunchRequest) -> RLLaunchPreview:
        spec_path = (
            request.scratch_dir
            / "primebeaker"
            / "beaker_experiments"
            / f"{request.experiment_name}.json"
        )
        return RLLaunchPreview(
            experiment_name=request.experiment_name,
            spec_path=spec_path,
            spec=self._spec(request),
            submit_command=(
                self.beaker_binary,
                "experiment",
                "create",
                "--name",
                request.experiment_name,
                "--workspace",
                request.workspace,
                str(spec_path),
            ),
            wandb_run=request.wandb_run,
        )

    def write_preview(self, preview: RLLaunchPreview) -> Path:
        preview.spec_path.parent.mkdir(parents=True, exist_ok=True)
        preview.spec_path.write_text(json.dumps(preview.spec, indent=2) + "\n", encoding="utf-8")
        return preview.spec_path

    def submit(self, request: RLLaunchRequest) -> RLLaunchReceipt:
        preview = self.preview(request)
        self.write_preview(preview)
        completed = subprocess.run(
            preview.submit_command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return RLLaunchReceipt(
            experiment_name=preview.experiment_name,
            spec_path=preview.spec_path,
            stdout=completed.stdout,
            wandb_run=preview.wandb_run,
        )

    def _spec(self, request: RLLaunchRequest) -> dict[str, Any]:
        metadata = request.metadata
        values = {
            "HOME": str(request.home_dir),
            "CONFIG_PATH": str(metadata.toml_path),
            "OUTPUT_DIR": str(metadata.output_dir),
            "EXPERIMENT_NAME": metadata.run_name,
            "NUM_GPUS": str(metadata.num_gpus),
            "WANDB_RUN_ID": request.wandb_run_id,
            "WANDB_SHARED_RUN_ID": request.wandb_run_id,
            "WANDB_RESUME": "allow",
        }
        if request.wandb_run.entity:
            values["WANDB_ENTITY"] = request.wandb_run.entity
        if request.registry:
            values["REGISTRY"] = request.registry
        overlap = sorted(set(values) & set(request.environment))
        if overlap:
            raise ValueError(f"reserved environment names cannot be overridden: {overlap}")
        secret_values = dict(request.secrets)
        if request.wandb_secret:
            secret_values["WANDB_API_KEY"] = request.wandb_secret
        if request.hf_secret:
            secret_values["HF_TOKEN"] = request.hf_secret
        task = {
            "name": "rl",
            "image": image_source(request.image),
            "command": ["bash", "-c", _container_command(request)],
            "envVars": environment_entries({**values, **request.environment}, secret_values),
            "datasets": [
                {"mountPath": str(request.mount_path), "source": {"weka": request.dataset}}
            ],
            "result": {"path": "/tmp/beaker-result"},
            "resources": {"gpuCount": metadata.num_gpus},
            "context": {
                "priority": request.priority,
                "minRuntime": f"{request.min_runtime_hours}h",
                "autoResume": True,
            },
            "constraints": {"cluster": list(request.clusters)},
        }
        spec: dict[str, Any] = {
            "version": "v2",
            "description": request.description or f"PrimeBeaker RL {request.experiment_name}",
            "tasks": [task],
        }
        if request.budget:
            spec["budget"] = request.budget
        return spec


class RLLauncher:
    def __init__(self, backend: RLLaunchBackend | None = None) -> None:
        self.backend = backend or BeakerRLBackend()

    def preview_from_toml(self, toml_path: str | Path, *, image: str, **kwargs: Any) -> RLLaunchPreview:
        return self.backend.preview(RLLaunchRequest.from_toml(toml_path, image=image, **kwargs))

    def submit_from_toml(self, toml_path: str | Path, *, image: str, **kwargs: Any) -> RLLaunchReceipt:
        return self.backend.submit(RLLaunchRequest.from_toml(toml_path, image=image, **kwargs))


def _container_command(request: RLLaunchRequest) -> str:
    setup = f"\n{request.setup_command}" if request.setup_command else ""
    config_path = shlex.quote(str(request.metadata.toml_path))
    runtime_config = shlex.quote("/tmp/primebeaker-runtime.toml")
    rewrite_command = shlex.join((
        "python3",
        "-c",
        "from primebeaker.config import RLTrainingToml; import sys; "
        "RLTrainingToml.from_path(sys.argv[1]).with_environment_namespace().write(sys.argv[2])",
        str(request.metadata.toml_path),
        "/tmp/primebeaker-runtime.toml",
    ))
    output_dir = shlex.quote(str(request.metadata.output_dir))
    working_dir = shlex.quote(str(request.working_dir))
    scratch = shlex.quote(str(request.scratch_dir))
    return f"""set -euo pipefail
mkdir -p /tmp/beaker-result
cd {working_dir}{setup}
command -v rl >/dev/null 2>&1
export XDG_CACHE_HOME=${{XDG_CACHE_HOME:-{scratch}/.cache}}
export HF_HOME=${{HF_HOME:-$XDG_CACHE_HOME/huggingface}}
export TORCH_HOME=${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}
export TORCHINDUCTOR_CACHE_DIR=${{TORCHINDUCTOR_CACHE_DIR:-$TORCH_HOME/inductor}}
export TRITON_CACHE_DIR=${{TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}}
export VLLM_CACHE_ROOT=${{VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}}
export TMPDIR=${{TMPDIR:-{scratch}/tmp}}
mkdir -p {output_dir} "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
test -f {config_path}
{rewrite_command}
exec rl @ {runtime_config}"""
