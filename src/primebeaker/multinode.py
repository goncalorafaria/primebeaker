"""Self-contained Beaker launchers and role preparation for multi-node Prime-RL."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
from typing import Any, Literal, Mapping, Sequence
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from primebeaker.common import environment_entries, image_source, require_under_mount
from primebeaker.config import RLTrainingToml, load_toml
from primebeaker.wandb import WandbRunReference


def deployment_type(path: str | Path) -> str:
    deployment = load_toml(path).get("deployment")
    if not isinstance(deployment, dict) or not isinstance(deployment.get("type"), str):
        raise ValueError("training TOML needs deployment.type")
    return deployment["type"]


def _table(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"training TOML needs a [{name}] table")
    return value


def _positive(table: Mapping[str, object], key: str, default: int | None = None) -> int:
    value = table.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{key} must be a positive integer")
    return value


def _local_search_model_paths(config: Mapping[str, object]) -> tuple[str, ...]:
    orchestrator = config.get("orchestrator")
    if not isinstance(orchestrator, dict):
        return ()
    paths: set[str] = set()
    for phase_name in ("train", "eval"):
        phase = orchestrator.get(phase_name)
        sources = phase.get("source", ()) if isinstance(phase, dict) else ()
        for source in sources if isinstance(sources, list) else ():
            legacy = source.get("legacy") if isinstance(source, dict) else None
            args = legacy.get("args") if isinstance(legacy, dict) else None
            path = args.get("local_search_model_path") if isinstance(args, dict) else None
            if path is not None:
                if not isinstance(path, str) or not path:
                    raise ValueError("local_search_model_path must be a non-empty string")
                paths.add(path)
    return tuple(sorted(paths))


class MultiNodeRLMetadata(BaseModel):
    """Validated heterogeneous topology read without importing Prime-RL."""

    model_config = ConfigDict(frozen=True)

    toml_path: Path
    output_dir: Path
    run_name: str = Field(min_length=1)
    wandb_project: str = Field(min_length=1)
    wandb_entity: str | None = None
    wandb_offline: bool = False
    gpus_per_node: int = Field(gt=0)
    num_train_nodes: int = Field(gt=0)
    num_infer_nodes: int = Field(gt=0)
    num_infer_replicas: int = Field(gt=0)
    inference_tp: int = Field(gt=0)
    inference_dp: int = Field(gt=0)
    api_server_count: int = Field(gt=0)
    trainer_gpus: int = Field(gt=0)
    local_search_model_paths: tuple[str, ...] = ()

    @property
    def has_mixed_node(self) -> bool:
        return self.trainer_gpus < self.gpus_per_node

    @property
    def mixed_node_infer_gpus(self) -> int:
        return self.gpus_per_node - self.trainer_gpus if self.has_mixed_node else 0

    @property
    def total_nodes(self) -> int:
        return self.num_infer_nodes if self.has_mixed_node else self.num_train_nodes + self.num_infer_nodes

    @property
    def total_infer_gpus(self) -> int:
        dedicated = self.num_infer_nodes - int(self.has_mixed_node)
        return self.mixed_node_infer_gpus + dedicated * self.gpus_per_node

    @classmethod
    def from_path(cls, path: str | Path) -> "MultiNodeRLMetadata":
        toml_path = Path(path).resolve()
        config = load_toml(toml_path)
        deployment = _table(config.get("deployment"), "deployment")
        if deployment.get("type") != "multi_node":
            raise ValueError("multi-node RL requires deployment.type = 'multi_node'")
        inference = _table(config.get("inference"), "inference")
        parallel = _table(inference.get("parallel"), "inference.parallel")
        trainer = _table(config.get("trainer"), "trainer")
        trainer_model = _table(trainer.get("model"), "trainer.model")
        wandb = _table(config.get("wandb"), "wandb")
        gpus_per_node = _positive(deployment, "gpus_per_node", 8)
        num_train_nodes = _positive(deployment, "num_train_nodes")
        num_infer_nodes = _positive(deployment, "num_infer_nodes")
        num_infer_replicas = _positive(deployment, "num_infer_replicas", 1)
        trainer_gpus = _positive(trainer_model, "cp", 1) * _positive(
            trainer_model, "dp_replicate", 1
        )
        inference_tp = _positive(parallel, "tp", 1)
        inference_dp = _positive(parallel, "dp", 1)
        api_server_count = _positive(inference, "api_server_count", 1)
        if num_train_nodes != 1:
            raise ValueError("heterogeneous Beaker RL currently supports exactly one trainer node")
        if num_infer_replicas != 1:
            raise ValueError("heterogeneous Beaker RL requires num_infer_replicas = 1")
        if trainer_gpus > gpus_per_node:
            raise ValueError("trainer GPU count exceeds gpus_per_node")
        mixed = gpus_per_node - trainer_gpus if trainer_gpus < gpus_per_node else 0
        if gpus_per_node % inference_tp or (mixed and mixed % inference_tp):
            raise ValueError("each inference-bearing GPU slice must be divisible by inference.parallel.tp")
        total_infer_gpus = mixed + (num_infer_nodes - int(bool(mixed))) * gpus_per_node
        expected_dp = total_infer_gpus // inference_tp
        if inference_dp != expected_dp or api_server_count != expected_dp:
            raise ValueError(
                "inference.parallel.dp and inference.api_server_count must both equal "
                f"the external-LB rank count ({expected_dp})"
            )
        return cls(
            toml_path=toml_path,
            output_dir=Path(str(config["output_dir"])),
            run_name=str(wandb["name"]),
            wandb_project=str(wandb["project"]),
            wandb_entity=str(wandb["entity"]) if wandb.get("entity") else None,
            wandb_offline=bool(wandb.get("offline", False)),
            gpus_per_node=gpus_per_node,
            num_train_nodes=num_train_nodes,
            num_infer_nodes=num_infer_nodes,
            num_infer_replicas=num_infer_replicas,
            inference_tp=inference_tp,
            inference_dp=inference_dp,
            api_server_count=api_server_count,
            trainer_gpus=trainer_gpus,
            local_search_model_paths=_local_search_model_paths(config),
        )


class MultiNodeSFTMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    toml_path: Path
    output_dir: Path
    train_data_path: Path
    validation_data_path: Path
    run_name: str = Field(min_length=1)
    wandb_project: str = Field(min_length=1)
    wandb_entity: str | None = None
    wandb_offline: bool = False
    num_nodes: int = Field(gt=1)
    gpus_per_node: int = Field(gt=0)

    @classmethod
    def from_path(cls, path: str | Path) -> "MultiNodeSFTMetadata":
        toml_path = Path(path).resolve()
        config = load_toml(toml_path)
        deployment = _table(config.get("deployment"), "deployment")
        if deployment.get("type") != "multi_node":
            raise ValueError("multi-node SFT requires deployment.type = 'multi_node'")
        wandb = _table(config.get("wandb"), "wandb")
        data = _table(config.get("data"), "data")
        validation = _table(config.get("val"), "val")
        validation_data = _table(validation.get("data"), "val.data")
        return cls(
            toml_path=toml_path,
            output_dir=Path(str(config["output_dir"])),
            train_data_path=Path(str(data["name"])),
            validation_data_path=Path(str(validation_data["name"])),
            run_name=str(wandb["name"]),
            wandb_project=str(wandb["project"]),
            wandb_entity=str(wandb["entity"]) if wandb.get("entity") else None,
            wandb_offline=bool(wandb.get("offline", False)),
            num_nodes=_positive(deployment, "num_nodes", 2),
            gpus_per_node=_positive(deployment, "gpus_per_node", 8),
        )


class _MultiNodeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

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
    environment: dict[str, str] = Field(default_factory=dict)
    secrets: dict[str, str] = Field(default_factory=dict)
    setup_command: str | None = None
    budget: str | None = None
    description: str | None = None
    rendezvous_timeout_seconds: int = Field(default=1800, gt=0)

    def _validate_common_paths(self, paths: Sequence[Path]) -> None:
        if not self.clusters or any(not cluster for cluster in self.clusters):
            raise ValueError("at least one non-empty Beaker cluster is required")
        require_under_mount((*paths, self.working_dir, self.scratch_dir, self.home_dir), self.mount_path)


class MultiNodeRLLaunchRequest(_MultiNodeRequest):
    metadata: MultiNodeRLMetadata
    registry: str = Field(min_length=1)
    gateway_port: int = Field(default=1212, gt=0, lt=65536)
    gateway_workers: int = Field(default=8, gt=0)
    required_services: dict[str, int] = Field(default_factory=dict)

    @classmethod
    def from_toml(cls, path: str | Path, *, image: str, **kwargs: Any) -> "MultiNodeRLLaunchRequest":
        return cls(metadata=MultiNodeRLMetadata.from_path(path), image=image, **kwargs)

    @model_validator(mode="after")
    def _validate_request(self) -> "MultiNodeRLLaunchRequest":
        self._validate_common_paths((self.metadata.toml_path, self.metadata.output_dir))
        if any(not name or count < 1 for name, count in self.required_services.items()):
            raise ValueError("required services need non-empty names and positive counts")
        return self


class MultiNodeSFTLaunchRequest(_MultiNodeRequest):
    metadata: MultiNodeSFTMetadata

    @classmethod
    def from_toml(cls, path: str | Path, *, image: str, **kwargs: Any) -> "MultiNodeSFTLaunchRequest":
        return cls(metadata=MultiNodeSFTMetadata.from_path(path), image=image, **kwargs)

    @model_validator(mode="after")
    def _validate_request(self) -> "MultiNodeSFTLaunchRequest":
        self._validate_common_paths((
            self.metadata.toml_path, self.metadata.output_dir,
            self.metadata.train_data_path, self.metadata.validation_data_path,
        ))
        return self


class MultiNodeLaunchPreview(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    launch_id: str
    spec_path: Path
    spec: dict[str, Any]
    submit_command: tuple[str, ...]
    wandb_run: WandbRunReference


class MultiNodeLaunchReceipt(BaseModel):
    model_config = ConfigDict(frozen=True)

    experiment_name: str
    launch_id: str
    spec_path: Path
    stdout: str
    wandb_run: WandbRunReference


class _BeakerMultiNodeBackend:
    beaker_binary = "beaker"

    def write_preview(self, preview: MultiNodeLaunchPreview) -> Path:
        preview.spec_path.parent.mkdir(parents=True, exist_ok=True)
        preview.spec_path.write_text(json.dumps(preview.spec, indent=2) + "\n", encoding="utf-8")
        return preview.spec_path

    def submit(self, request: Any) -> MultiNodeLaunchReceipt:
        preview = self.preview(request)
        self.write_preview(preview)
        try:
            completed = subprocess.run(
                preview.submit_command, check=True, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as error:
            detail = (error.stderr or "").strip() or (error.stdout or "").strip()
            raise RuntimeError(f"Beaker could not create {preview.experiment_name}: {detail}") from error
        return MultiNodeLaunchReceipt(
            experiment_name=preview.experiment_name,
            launch_id=preview.launch_id,
            spec_path=preview.spec_path,
            stdout=completed.stdout,
            wandb_run=preview.wandb_run,
        )


def _base_values(request: _MultiNodeRequest, *, launch_id: str, output_dir: Path) -> dict[str, str]:
    return {
        "HOME": str(request.home_dir),
        "WORKING_DIR": str(request.working_dir),
        "SCRATCH": str(request.scratch_dir),
        "CONFIG_PATH": str(request.metadata.toml_path),
        "OUTPUT_DIR": str(output_dir),
        "LAUNCH_ID": launch_id,
        "RENDEZVOUS_DIR": str(
            request.scratch_dir / "primebeaker/rendezvous" / f"{request.metadata.run_name}-{launch_id}"
        ),
        "RENDEZVOUS_TIMEOUT_SECONDS": str(request.rendezvous_timeout_seconds),
        "WANDB_RUN_ID": request.wandb_run_id,
        "WANDB_SHARED_RUN_ID": request.wandb_run_id,
        "WANDB_RESUME": "allow",
    }


def _task_environment(request: _MultiNodeRequest, values: dict[str, str]) -> list[dict[str, str]]:
    overlap = sorted(set(values) & set(request.environment))
    if overlap:
        raise ValueError(f"reserved environment names cannot be overridden: {overlap}")
    secrets = dict(request.secrets)
    if request.wandb_secret:
        secrets["WANDB_API_KEY"] = request.wandb_secret
    if request.hf_secret:
        secrets["HF_TOKEN"] = request.hf_secret
    return environment_entries({**values, **request.environment}, secrets)


def _preview(
    request: _MultiNodeRequest, *, launch_id: str, spec: dict[str, Any]
) -> MultiNodeLaunchPreview:
    name = request.metadata.run_name
    path = request.scratch_dir / "primebeaker/beaker_experiments" / f"{name}-{launch_id}.json"
    run = WandbRunReference(
        run_id=request.wandb_run_id,
        project=request.metadata.wandb_project,
        entity=request.metadata.wandb_entity or request.wandb_entity,
        name=name,
        offline=request.metadata.wandb_offline,
    )
    return MultiNodeLaunchPreview(
        experiment_name=name,
        launch_id=launch_id,
        spec_path=path,
        spec=spec,
        submit_command=(
            "beaker", "experiment", "create", "--name", name,
            "--workspace", request.workspace, str(path)
        ),
        wandb_run=run,
    )


class BeakerMultiNodeRLBackend(_BeakerMultiNodeBackend):
    def preview(
        self, request: MultiNodeRLLaunchRequest, *, launch_id: str | None = None
    ) -> MultiNodeLaunchPreview:
        launch_id = launch_id or uuid4().hex
        metadata = request.metadata
        values = {
            **_base_values(request, launch_id=launch_id, output_dir=metadata.output_dir),
            "REGISTRY": request.registry,
            "TOTAL_NODES": str(metadata.total_nodes),
            "TOTAL_INFER_RANKS": str(metadata.inference_dp),
            "GPUS_PER_NODE": str(metadata.gpus_per_node),
            "TRAINER_GPU_COUNT": str(metadata.trainer_gpus),
            "MIXED_INFER_GPU_COUNT": str(metadata.mixed_node_infer_gpus),
            "GATEWAY_PORT": str(request.gateway_port),
            "GATEWAY_WORKERS": str(request.gateway_workers),
            "REQUIRED_LITEREGISTRY_SERVICES": json.dumps(
                request.required_services, separators=(",", ":"), sort_keys=True
            ),
            "LOCAL_SEARCH_MODEL_PATHS_JSON": json.dumps(metadata.local_search_model_paths),
        }
        if request.setup_command:
            values["PRIMEBEAKER_SETUP_COMMAND"] = request.setup_command
        task = {
            "name": "prime-rl-multinode",
            "image": image_source(request.image),
            "command": ["python3", "-m", "primebeaker.multinode", "role", "rl"],
            "replicas": metadata.total_nodes,
            "leaderSelection": True,
            "synchronizedStartTimeout": "4h",
            "hostNetworking": True,
            "propagateFailure": True,
            "propagatePreemption": True,
            "envVars": _task_environment(request, values),
            "datasets": [{"mountPath": str(request.mount_path), "source": {"weka": request.dataset}}],
            "result": {"path": "/tmp/beaker-result"},
            "resources": {"gpuCount": metadata.gpus_per_node},
            "context": {
                "priority": request.priority,
                "minRuntime": f"{request.min_runtime_hours}h",
                "autoResume": False,
            },
            "constraints": {"cluster": list(request.clusters)},
        }
        spec: dict[str, Any] = {
            "version": "v2",
            "description": request.description or f"PrimeBeaker multi-node RL {metadata.run_name}",
            "tasks": [task],
        }
        if request.budget:
            spec["budget"] = request.budget
        return _preview(request, launch_id=launch_id, spec=spec)


class BeakerMultiNodeSFTBackend(_BeakerMultiNodeBackend):
    def preview(
        self, request: MultiNodeSFTLaunchRequest, *, launch_id: str | None = None
    ) -> MultiNodeLaunchPreview:
        launch_id = launch_id or uuid4().hex
        metadata = request.metadata
        values = {
            **_base_values(request, launch_id=launch_id, output_dir=metadata.output_dir),
            "TOTAL_NODES": str(metadata.num_nodes),
            "GPUS_PER_NODE": str(metadata.gpus_per_node),
        }
        if request.setup_command:
            values["PRIMEBEAKER_SETUP_COMMAND"] = request.setup_command
        task = {
            "name": "prime-sft-multinode",
            "image": image_source(request.image),
            "command": ["python3", "-m", "primebeaker.multinode", "role", "sft"],
            "replicas": metadata.num_nodes,
            "leaderSelection": True,
            "synchronizedStartTimeout": "4h",
            "hostNetworking": True,
            "propagateFailure": True,
            "propagatePreemption": True,
            "envVars": _task_environment(request, values),
            "datasets": [{"mountPath": str(request.mount_path), "source": {"weka": request.dataset}}],
            "result": {"path": "/tmp/beaker-result"},
            "resources": {"gpuCount": metadata.gpus_per_node},
            "context": {
                "priority": request.priority,
                "minRuntime": f"{request.min_runtime_hours}h",
                "autoResume": False,
            },
            "constraints": {"cluster": list(request.clusters)},
        }
        spec: dict[str, Any] = {
            "version": "v2",
            "description": request.description or f"PrimeBeaker multi-node SFT {metadata.run_name}",
            "tasks": [task],
        }
        if request.budget:
            spec["budget"] = request.budget
        return _preview(request, launch_id=launch_id, spec=spec)


def _atomic_shell_env(path: Path, values: Mapping[str, object]) -> None:
    lines: list[str] = []
    for key, value in values.items():
        if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", key):
            raise ValueError(f"unsafe environment variable name: {key!r}")
        lines.append(f"export {key}={shlex.quote(str(value))}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def _step_directories(directory: Path) -> list[tuple[int, Path]]:
    result: list[tuple[int, Path]] = []
    for path in directory.glob("step_*"):
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            continue
        if path.is_dir():
            result.append((step, path))
    return sorted(result)


def clean_stale_runtime_state(config: Any) -> int | None:
    checkpoint = config.ckpt
    resume_step = checkpoint.resume_step if checkpoint is not None else None
    if resume_step == -1:
        base = checkpoint.output_dir or config.output_dir
        steps = _step_directories(base / "checkpoints")
        resume_step = steps[-1][0] if steps else None
    cutoff = resume_step if resume_step is not None else -1
    for directory in (
        config.output_dir / "rollouts",
        config.output_dir / "run_default/rollouts",
        config.output_dir / "run_default/broadcasts",
    ):
        for step, path in _step_directories(directory):
            if step > cutoff:
                shutil.rmtree(path)
    return resume_step


def prepare_rl_runtime(config_path: Path, config_dir: Path, runtime_dir: Path) -> None:
    from prime_rl.configs.rl import RLConfig
    from prime_rl.entrypoints.rl import write_subconfigs
    from prime_rl.utils.process import (
        DEFAULT_COMMON_ENV_VARS, DEFAULT_INFERENCE_ENV_VARS, DEFAULT_TRAINER_ENV_VARS,
    )

    raw = RLTrainingToml.from_path(config_path).with_environment_namespace().values
    raw.setdefault("slurm", {})
    inference_raw = raw.get("inference")
    if isinstance(inference_raw, dict):
        inference_deployment = inference_raw.get("deployment")
        if isinstance(inference_deployment, dict) and inference_deployment.get("type") in {
            "multi_node", "disaggregated"
        }:
            inference_raw.setdefault("slurm", {})
    config = RLConfig.model_validate(raw)
    if config.inference is None or config.inference.router is None:
        raise ValueError("multi-node RL requires inference with a router")
    if config.inference.enable_expert_parallel:
        raise ValueError("heterogeneous Beaker RL currently supports dense inference")
    clean_stale_runtime_state(config)
    if config.weight_broadcast.type in {"nccl", "nixl"}:
        world_size = config.inference.parallel.dp * config.inference.parallel.tp
        config.trainer.weight_broadcast.inference_world_size = world_size
        config.orchestrator.weight_broadcast.inference_world_size = world_size
        if config.weight_broadcast.type == "nccl":
            config.trainer.weight_broadcast.host = "0.0.0.0"
    router_policy = config.inference.router.policy
    config.inference.router = None
    config_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_subconfigs(config, config_dir)
    common = {**DEFAULT_COMMON_ENV_VARS, **config.env_vars}
    _atomic_shell_env(runtime_dir / "trainer.env", {**common, **DEFAULT_TRAINER_ENV_VARS, **config.trainer.env_vars})
    _atomic_shell_env(runtime_dir / "orchestrator.env", {**common, **config.orchestrator.env_vars})
    _atomic_shell_env(runtime_dir / "inference.env", {**common, **DEFAULT_INFERENCE_ENV_VARS, **config.inference.env_vars})
    _atomic_shell_env(runtime_dir / "runtime.env", {
        "ROUTER_PORT": config.inference.server.port,
        "ROUTER_POLICY": router_policy,
        "BACKEND_PORT": config.inference.backend_port,
        "INFERENCE_TP": config.inference.parallel.tp,
        "TRAINER_RANKS_FILTER": ",".join(str(rank) for rank in config.trainer.log.ranks_filter),
        "USE_NCCL_BROADCAST": int(config.weight_broadcast.type == "nccl"),
        "USE_ZMQ_TRANSPORT": int(config.rollout_transport.type == "zmq"),
    })
    (runtime_dir / "runtime.ready").write_text("ready\n", encoding="utf-8")


def prepare_sft_runtime(config_path: Path, output_path: Path, runtime_dir: Path) -> None:
    from prime_rl.configs.sft import SFTConfig
    from prime_rl.entrypoints.sft import write_config
    from prime_rl.utils.process import DEFAULT_COMMON_ENV_VARS, DEFAULT_TRAINER_ENV_VARS

    raw = load_toml(config_path)
    raw.setdefault("slurm", {})
    config = SFTConfig.model_validate(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    runtime_dir.mkdir(parents=True, exist_ok=True)
    write_config(config, output_path, exclude={"deployment", "slurm", "dry_run", "clean_output_dir"})
    _atomic_shell_env(runtime_dir / "trainer.env", {
        **DEFAULT_COMMON_ENV_VARS, **DEFAULT_TRAINER_ENV_VARS, **config.env_vars,
    })
    _atomic_shell_env(runtime_dir / "runtime.env", {
        "TRAINER_RANKS_FILTER": ",".join(str(rank) for rank in config.log.ranks_filter),
    })
    (runtime_dir / "runtime.ready").write_text("ready\n", encoding="utf-8")


def _role(kind: str) -> None:
    script = Path(__file__).with_name(f"multinode_{kind}_role.sh")
    if not script.is_file():
        raise FileNotFoundError(script)
    os.execv("/bin/bash", ("bash", str(script)))


def main(argv: Sequence[str] | None = None) -> int:
    from primebeaker.multinode_cli import run

    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
