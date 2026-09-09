"""Typed Beaker launcher for Prime-RL SFT TOMLs."""

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


class SFTTomlMetadata(BaseModel):
    """Launch-critical values derived exclusively from an SFT TOML."""

    model_config = ConfigDict(frozen=True)

    toml_path: Path
    output_dir: Path
    train_data_path: Path
    validation_data_path: Path
    run_name: str = Field(min_length=1)
    wandb_project: str = Field(min_length=1)
    wandb_entity: str | None = None
    wandb_offline: bool = False
    max_steps: int = Field(gt=0)
    validation_interval: int = Field(gt=0)
    validation_eval_on_start: bool
    checkpoint_interval: int = Field(gt=0)
    deployment_type: str = Field(min_length=1)
    num_gpus: int = Field(gt=0)
    gpus_per_node: int = Field(gt=0)

    @classmethod
    def from_path(cls, path: str | Path) -> "SFTTomlMetadata":
        toml_path = Path(path).resolve()
        if not toml_path.is_file():
            raise FileNotFoundError(toml_path)
        config = load_toml(toml_path)
        deployment = config.get("deployment")
        wandb = config.get("wandb")
        data = config.get("data")
        validation = config.get("val")
        validation_data = validation.get("data") if isinstance(validation, dict) else None
        checkpoint = config.get("ckpt")
        if not all(
            isinstance(value, dict)
            for value in (deployment, wandb, data, validation, validation_data, checkpoint)
        ):
            raise ValueError(
                "SFT TOML needs [deployment], [wandb], [data], [val], [val.data], and [ckpt] tables"
            )
        try:
            return cls(
                toml_path=toml_path,
                output_dir=Path(str(config["output_dir"])),
                train_data_path=Path(str(data["name"])),
                validation_data_path=Path(str(validation_data["name"])),
                run_name=str(wandb["name"]),
                wandb_project=str(wandb["project"]),
                wandb_entity=str(wandb["entity"]) if wandb.get("entity") else None,
                wandb_offline=bool(wandb.get("offline", False)),
                max_steps=int(config["max_steps"]),
                validation_interval=int(validation["interval"]),
                validation_eval_on_start=bool(validation["eval_on_start"]),
                checkpoint_interval=int(checkpoint["interval"]),
                deployment_type=str(deployment["type"]),
                num_gpus=int(deployment["num_gpus"]),
                gpus_per_node=int(deployment["gpus_per_node"]),
            )
        except KeyError as error:
            raise ValueError(f"SFT TOML is missing launch field {error.args[0]!r}") from error

    @model_validator(mode="after")
    def _topology_and_intervals(self) -> "SFTTomlMetadata":
        if self.deployment_type != "single_node":
            raise ValueError("the direct SFT launcher supports deployment.type = 'single_node'")
        if self.num_gpus != self.gpus_per_node:
            raise ValueError("single-node SFT requires num_gpus == gpus_per_node")
        if not self.validation_eval_on_start:
            raise ValueError("SFT TOML must set val.eval_on_start = true")
        if self.validation_interval > self.max_steps:
            raise ValueError("SFT val.interval must not exceed max_steps")
        if self.validation_interval > self.checkpoint_interval:
            raise ValueError("SFT val.interval must be no larger than ckpt.interval")
        return self


class SFTLaunchRequest(BaseModel):
    """Immutable placement and environment policy for one SFT launch."""

    model_config = ConfigDict(frozen=True)

    metadata: SFTTomlMetadata
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
    environment: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    setup_command: str | None = None
    budget: str | None = None
    description: str | None = None

    @classmethod
    def from_toml(cls, toml_path: str | Path, *, image: str, **kwargs: Any) -> "SFTLaunchRequest":
        return cls(metadata=SFTTomlMetadata.from_path(toml_path), image=image, **kwargs)

    @model_validator(mode="after")
    def _paths_and_clusters(self) -> "SFTLaunchRequest":
        if not self.clusters or any(not cluster for cluster in self.clusters):
            raise ValueError("at least one non-empty Beaker cluster is required")
        require_under_mount(
            (
                self.metadata.toml_path,
                self.metadata.output_dir,
                self.metadata.train_data_path,
                self.metadata.validation_data_path,
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
            entity=self.metadata.wandb_entity,
            name=self.metadata.run_name,
            offline=self.metadata.wandb_offline,
        )


class SFTLaunchPreview(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    spec_path: Path
    spec: dict[str, Any]
    submit_command: tuple[str, ...]
    wandb_run: WandbRunReference


class SFTLaunchReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    spec_path: Path
    stdout: str
    wandb_run: WandbRunReference


class SFTLaunchBackend(ABC):
    @abstractmethod
    def preview(self, request: SFTLaunchRequest) -> SFTLaunchPreview: ...

    @abstractmethod
    def submit(self, request: SFTLaunchRequest) -> SFTLaunchReceipt: ...


class BeakerSFTBackend(SFTLaunchBackend):
    beaker_binary: str = "beaker"

    def preview(self, request: SFTLaunchRequest) -> SFTLaunchPreview:
        spec_path = (
            request.scratch_dir
            / "primebeaker"
            / "beaker_experiments"
            / f"{request.experiment_name}.json"
        )
        return SFTLaunchPreview(
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

    def write_preview(self, preview: SFTLaunchPreview) -> Path:
        preview.spec_path.parent.mkdir(parents=True, exist_ok=True)
        preview.spec_path.write_text(json.dumps(preview.spec, indent=2) + "\n", encoding="utf-8")
        return preview.spec_path

    def submit(self, request: SFTLaunchRequest) -> SFTLaunchReceipt:
        preview = self.preview(request)
        self.write_preview(preview)
        completed = subprocess.run(
            preview.submit_command,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return SFTLaunchReceipt(
            experiment_name=preview.experiment_name,
            spec_path=preview.spec_path,
            stdout=completed.stdout,
            wandb_run=preview.wandb_run,
        )

    def _spec(self, request: SFTLaunchRequest) -> dict[str, Any]:
        metadata = request.metadata
        values = {
            "HOME": str(request.home_dir),
            "CONFIG_PATH": str(metadata.toml_path),
            "OUTPUT_DIR": str(metadata.output_dir),
            "EXPERIMENT_NAME": metadata.run_name,
            "NUM_GPUS": str(metadata.num_gpus),
            "WANDB_RUN_ID": request.wandb_run_id,
            "WANDB_RESUME": "allow",
        }
        overlap = sorted(set(values) & set(request.environment))
        if overlap:
            raise ValueError(f"reserved environment names cannot be overridden: {overlap}")
        secret_values = dict(request.secrets)
        if request.wandb_secret:
            secret_values["WANDB_API_KEY"] = request.wandb_secret
        if request.hf_secret:
            secret_values["HF_TOKEN"] = request.hf_secret
        task = {
            "name": "sft",
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
            "description": request.description or f"PrimeBeaker SFT {request.experiment_name}",
            "tasks": [task],
        }
        if request.budget:
            spec["budget"] = request.budget
        return spec


class SFTLauncher:
    def __init__(self, backend: SFTLaunchBackend | None = None) -> None:
        self.backend = backend or BeakerSFTBackend()

    def preview_from_toml(self, toml_path: str | Path, *, image: str, **kwargs: Any) -> SFTLaunchPreview:
        return self.backend.preview(SFTLaunchRequest.from_toml(toml_path, image=image, **kwargs))

    def submit_from_toml(self, toml_path: str | Path, *, image: str, **kwargs: Any) -> SFTLaunchReceipt:
        return self.backend.submit(SFTLaunchRequest.from_toml(toml_path, image=image, **kwargs))


def _container_command(request: SFTLaunchRequest) -> str:
    metadata = request.metadata
    setup = f"\n{request.setup_command}" if request.setup_command else ""
    output_dir = shlex.quote(str(metadata.output_dir))
    checkpoint_dir = shlex.quote(str(metadata.output_dir / "checkpoints"))
    config_path = shlex.quote(str(metadata.toml_path))
    train_path = shlex.quote(str(metadata.train_data_path))
    validation_path = shlex.quote(str(metadata.validation_data_path))
    working_dir = shlex.quote(str(request.working_dir))
    scratch = shlex.quote(str(request.scratch_dir))
    return f"""set -euo pipefail
mkdir -p /tmp/beaker-result
cd {working_dir}{setup}
command -v sft >/dev/null 2>&1
export XDG_CACHE_HOME=${{XDG_CACHE_HOME:-{scratch}/.cache}}
export HF_HOME=${{HF_HOME:-$XDG_CACHE_HOME/huggingface}}
export TORCH_HOME=${{TORCH_HOME:-$XDG_CACHE_HOME/torch}}
export TORCHINDUCTOR_CACHE_DIR=${{TORCHINDUCTOR_CACHE_DIR:-$TORCH_HOME/inductor}}
export TRITON_CACHE_DIR=${{TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}}
export VLLM_CACHE_ROOT=${{VLLM_CACHE_ROOT:-$XDG_CACHE_HOME/vllm}}
export TMPDIR=${{TMPDIR:-{scratch}/tmp}}
mkdir -p {output_dir} "$XDG_CACHE_HOME" "$HF_HOME" "$TORCH_HOME" "$TMPDIR"
test -f {config_path}
test -r {train_path}
test -r {validation_path}
if [[ -d {checkpoint_dir} ]]; then
  for step_dir in {checkpoint_dir}/step_*; do
    [[ -d "$step_dir/trainer" ]] || continue
    if [[ ! -f "$step_dir/trainer/.metadata" ]]; then
      quarantine="${{step_dir}}.incomplete-${{BEAKER_JOB_ID:-manual}}"
      echo "Quarantining incomplete checkpoint: $step_dir -> $quarantine"
      mv "$step_dir" "$quarantine"
    fi
  done
fi
exec sft @ {config_path}"""
