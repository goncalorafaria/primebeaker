"""Safe whole-topology checkpoint resume for multi-node Prime-RL launches."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any, Mapping, Sequence
from uuid import uuid4

from primebeaker.beaker_resources import validate_beaker_resources
from primebeaker.multinode import (
    BeakerMultiNodeRLBackend,
    MultiNodeLaunchPreview,
    MultiNodeRLLaunchRequest,
    MultiNodeRLMetadata,
)


_STEP_PATTERN = re.compile(r"step_(\d+)$")
_EXPERIMENT_ID_PATTERN = re.compile(r"Experiment\s+([A-Z0-9]+)\s+submitted")
_RESERVED_ENVIRONMENT = {
    "HOME",
    "WORKING_DIR",
    "SCRATCH",
    "CONFIG_PATH",
    "OUTPUT_DIR",
    "LAUNCH_ID",
    "RENDEZVOUS_DIR",
    "RENDEZVOUS_TIMEOUT_SECONDS",
    "WANDB_RUN_ID",
    "WANDB_SHARED_RUN_ID",
    "WANDB_RESUME",
    "REGISTRY",
    "TOTAL_NODES",
    "TOTAL_INFER_RANKS",
    "GPUS_PER_NODE",
    "TRAINER_GPU_COUNT",
    "MIXED_INFER_GPU_COUNT",
    "GATEWAY_PORT",
    "GATEWAY_WORKERS",
    "REQUIRED_LITEREGISTRY_SERVICES",
    "LOCAL_SEARCH_MODEL_PATHS_JSON",
    "PRIMEBEAKER_SETUP_COMMAND",
}


def _safe_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", value).strip("-.")
    return value or "primebeaker"


def _resume_experiment_name(base_name: str, *, resume_step: int, attempt_id: str) -> str:
    suffix = f"-resume-step{resume_step}-{attempt_id[:8]}"
    prefix = _safe_name(base_name)[: 128 - len(suffix)].rstrip("-.")
    return f"{prefix}{suffix}"


@dataclass(frozen=True)
class CommonCheckpoint:
    step: int
    trainer_dir: Path
    trainer_metadata: Path
    trainer_shards: tuple[Path, ...]
    orchestrator_progress: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "trainer_dir": str(self.trainer_dir),
            "trainer_metadata": str(self.trainer_metadata),
            "trainer_shards": [
                {"path": str(path), "size": path.stat().st_size}
                for path in self.trainer_shards
            ],
            "orchestrator_progress": str(self.orchestrator_progress),
            "orchestrator_progress_size": self.orchestrator_progress.stat().st_size,
        }


@dataclass(frozen=True)
class SourceExperiment:
    experiment_id: str
    experiment_name: str
    workspace: str
    job_ids: tuple[str, ...]
    config_path: Path
    output_dir: Path
    registry: str | None
    image: str
    clusters: tuple[str, ...]
    priority: str
    min_runtime_hours: int
    wandb_shared_run_id: str | None
    mount_path: Path
    dataset: str
    working_dir: Path
    scratch_dir: Path
    home_dir: Path
    gateway_port: int
    gateway_workers: int
    required_services: dict[str, int]
    rendezvous_timeout_seconds: int
    wandb_secret: str | None
    hf_secret: str | None
    environment: dict[str, str]
    secrets: dict[str, str]
    setup_command: str | None


@dataclass(frozen=True)
class ResumeAttempt:
    source: SourceExperiment
    checkpoint: CommonCheckpoint
    source_config_path: Path
    generated_config_path: Path
    generated_config_text: str
    manifest_path: Path
    experiment_name: str
    attempt_id: str
    workspace: str
    preview: MultiNodeLaunchPreview

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_experiment_id": self.source.experiment_id,
            "source_experiment_name": self.source.experiment_name,
            "checkpoint": self.checkpoint.as_dict(),
            "source_config_path": str(self.source_config_path),
            "generated_config_path": str(self.generated_config_path),
            "manifest_path": str(self.manifest_path),
            "experiment_name": self.experiment_name,
            "attempt_id": self.attempt_id,
            "launch": self.preview.model_dump(mode="json"),
        }


def _nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def discover_common_checkpoints(
    output_dir: str | Path, *, expected_trainer_shards: int
) -> tuple[CommonCheckpoint, ...]:
    """Find steps complete for both trainer and orchestrator checkpoint owners."""

    if expected_trainer_shards < 1:
        raise ValueError("expected_trainer_shards must be positive")
    output = Path(output_dir)
    trainer_root = output / "checkpoints"
    orchestrator_root = output / "run_default/checkpoints"
    if not trainer_root.is_dir():
        raise FileNotFoundError(f"trainer checkpoint root not found: {trainer_root}")
    if not orchestrator_root.is_dir():
        raise FileNotFoundError(f"orchestrator checkpoint root not found: {orchestrator_root}")

    result: list[CommonCheckpoint] = []
    for step_dir in trainer_root.iterdir():
        match = _STEP_PATTERN.fullmatch(step_dir.name)
        if match is None or not step_dir.is_dir():
            continue
        trainer_dir = step_dir / "trainer"
        metadata = trainer_dir / ".metadata"
        expected = tuple(
            trainer_dir / f"__{rank}_0.distcp"
            for rank in range(expected_trainer_shards)
        )
        progress = orchestrator_root / step_dir.name / "orchestrator/progress.pt"
        actual = tuple(sorted(trainer_dir.glob("*.distcp")))
        if (
            _nonempty_file(metadata)
            and all(_nonempty_file(path) for path in expected)
            and len(actual) == expected_trainer_shards
            and _nonempty_file(progress)
        ):
            result.append(
                CommonCheckpoint(
                    step=int(match.group(1)),
                    trainer_dir=trainer_dir,
                    trainer_metadata=metadata,
                    trainer_shards=actual,
                    orchestrator_progress=progress,
                )
            )
    return tuple(sorted(result, key=lambda checkpoint: checkpoint.step))


def select_common_checkpoint(
    output_dir: str | Path,
    *,
    expected_trainer_shards: int,
    resume_step: int | None = None,
) -> CommonCheckpoint:
    checkpoints = discover_common_checkpoints(
        output_dir, expected_trainer_shards=expected_trainer_shards
    )
    if not checkpoints:
        raise RuntimeError(
            "no complete common checkpoint: a resumable step needs every trainer shard, "
            "trainer .metadata, and orchestrator progress.pt"
        )
    if resume_step is None:
        return checkpoints[-1]
    for checkpoint in checkpoints:
        if checkpoint.step == resume_step:
            return checkpoint
    available = ", ".join(str(checkpoint.step) for checkpoint in checkpoints)
    raise RuntimeError(
        f"step {resume_step} is not a complete common checkpoint; available: {available}"
    )


def _set_toml_key(
    text: str,
    *,
    table: str | None,
    key: str,
    rendered_value: str,
    create: bool = True,
) -> str:
    lines = text.splitlines(keepends=True)
    header = f"[{table}]" if table else None
    start = 0
    end = next(
        (index for index, line in enumerate(lines) if line.lstrip().startswith("[")),
        len(lines),
    )
    if header is not None:
        start = next(
            (index + 1 for index, line in enumerate(lines) if line.strip() == header),
            -1,
        )
        if start < 0:
            if not create:
                return text
            suffix = "" if text.endswith("\n") else "\n"
            return f"{text}{suffix}\n{header}\n{key} = {rendered_value}\n"
        end = next(
            (
                index
                for index in range(start, len(lines))
                if lines[index].lstrip().startswith("[")
            ),
            len(lines),
        )
    pattern = re.compile(rf"^\s*{re.escape(key)}\s*=")
    for index in range(start, end):
        if pattern.match(lines[index]):
            newline = "\n" if lines[index].endswith("\n") else ""
            lines[index] = f"{key} = {rendered_value}{newline}"
            return "".join(lines)
    lines.insert(end, f"{key} = {rendered_value}\n")
    return "".join(lines)


def render_resume_config(source_text: str, *, resume_step: int, wandb_name: str) -> str:
    rendered = _set_toml_key(
        source_text,
        table=None,
        key="clean_output_dir",
        rendered_value="false",
    )
    rendered = _set_toml_key(
        rendered,
        table="ckpt",
        key="resume_step",
        rendered_value=str(resume_step),
    )
    rendered = _set_toml_key(
        rendered,
        table="wandb",
        key="name",
        rendered_value=json.dumps(wandb_name),
        create=False,
    )
    parsed = tomllib.loads(rendered)
    if parsed.get("clean_output_dir") is not False:
        raise AssertionError("resume config must preserve its output directory")
    if parsed.get("ckpt", {}).get("resume_step") != resume_step:
        raise AssertionError("resume config did not record the selected checkpoint")
    return rendered


def _environment(spec: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in spec.get("envVars", []):
        if (
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("value"), str)
        ):
            result[item["name"]] = item["value"]
    return result


def _secret_environment(spec: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in spec.get("envVars", []):
        if (
            isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and isinstance(item.get("secret"), str)
        ):
            result[item["name"]] = item["secret"]
    return result


def _run_json(command: Sequence[str]) -> Any:
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip() or (error.stdout or "").strip()
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}") from error
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"command returned invalid JSON: {' '.join(command)}") from error


def _positive_env(env: Mapping[str, str], name: str, default: int) -> int:
    value = int(env.get(name, default))
    if value < 1:
        raise RuntimeError(f"source experiment has invalid {name}={value}")
    return value


def _runtime_hours(context: object) -> int:
    value = context.get("minRuntime", "0h") if isinstance(context, dict) else "0h"
    match = re.fullmatch(r"(\d+)h", str(value))
    return int(match.group(1)) if match else 0


def inspect_source_experiment(experiment_id: str) -> SourceExperiment:
    """Recover all reproducibility inputs from a PrimeBeaker Beaker launch."""

    payload = _run_json(
        ("beaker", "experiment", "inspect", experiment_id, "--format", "json")
    )
    experiment = payload[0] if isinstance(payload, list) and payload else payload
    if not isinstance(experiment, dict):
        raise RuntimeError(f"Beaker did not return experiment {experiment_id}")
    jobs = experiment.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise RuntimeError(f"experiment {experiment_id} has no jobs")

    prime_spec: Mapping[str, Any] | None = None
    for job in jobs:
        execution = job.get("execution") if isinstance(job, dict) else None
        spec = execution.get("spec") if isinstance(execution, dict) else None
        env = _environment(spec) if isinstance(spec, dict) else {}
        if "CONFIG_PATH" in env and "OUTPUT_DIR" in env:
            prime_spec = spec
            break
    if prime_spec is None:
        raise RuntimeError(
            f"experiment {experiment_id} is not a PrimeBeaker/Prime-RL launch"
        )
    env = _environment(prime_spec)
    secret_env = _secret_environment(prime_spec)
    image = prime_spec.get("image", {})
    image_ref = image.get("beaker") if isinstance(image, dict) else None
    if not isinstance(image_ref, str) or not image_ref:
        raise RuntimeError(f"experiment {experiment_id} has no Beaker image")
    workspace_ref = experiment.get("workspaceRef")
    workspace = (
        workspace_ref.get("fullName") if isinstance(workspace_ref, dict) else None
    )
    constraints = prime_spec.get("constraints")
    clusters = constraints.get("cluster") if isinstance(constraints, dict) else None
    context = prime_spec.get("context")
    datasets = prime_spec.get("datasets")
    dataset_item = datasets[0] if isinstance(datasets, list) and datasets else {}
    source = dataset_item.get("source") if isinstance(dataset_item, dict) else {}
    required_raw = env.get("REQUIRED_LITEREGISTRY_SERVICES", "{}")
    try:
        required = json.loads(required_raw)
    except json.JSONDecodeError as error:
        raise RuntimeError("source has invalid REQUIRED_LITEREGISTRY_SERVICES") from error
    if not isinstance(required, dict):
        raise RuntimeError("source REQUIRED_LITEREGISTRY_SERVICES must be an object")
    return SourceExperiment(
        experiment_id=experiment_id,
        experiment_name=str(experiment.get("name", experiment_id)).split("/", 1)[-1],
        workspace=str(workspace or "ai2/oe-agents"),
        job_ids=tuple(
            str(job["id"]) for job in jobs if isinstance(job, dict) and "id" in job
        ),
        config_path=Path(env["CONFIG_PATH"]),
        output_dir=Path(env["OUTPUT_DIR"]),
        registry=env.get("REGISTRY"),
        image=image_ref if image_ref.startswith("beaker://") else f"beaker://{image_ref}",
        clusters=tuple(str(cluster) for cluster in (clusters or ("ai2/holmes",))),
        priority=str(context.get("priority", "high")) if isinstance(context, dict) else "high",
        min_runtime_hours=_runtime_hours(context),
        wandb_shared_run_id=env.get("WANDB_SHARED_RUN_ID"),
        mount_path=Path(str(dataset_item.get("mountPath", "/weka"))),
        dataset=str(source.get("weka", "oe-adapt-default")) if isinstance(source, dict) else "oe-adapt-default",
        working_dir=Path(env.get("WORKING_DIR", "/weka")),
        scratch_dir=Path(env.get("SCRATCH", "/weka")),
        home_dir=Path(env.get("HOME", "/weka")),
        gateway_port=_positive_env(env, "GATEWAY_PORT", 1212),
        gateway_workers=_positive_env(env, "GATEWAY_WORKERS", 8),
        required_services={str(name): int(count) for name, count in required.items()},
        rendezvous_timeout_seconds=_positive_env(
            env, "RENDEZVOUS_TIMEOUT_SECONDS", 1800
        ),
        wandb_secret=secret_env.get("WANDB_API_KEY"),
        hf_secret=secret_env.get("HF_TOKEN"),
        environment={
            name: value
            for name, value in env.items()
            if name not in _RESERVED_ENVIRONMENT
            and not name.startswith("DATADEV_")
            and not name.startswith("PRIMEBEAKER_RESUME_")
        },
        secrets={
            name: value
            for name, value in secret_env.items()
            if name not in {"WANDB_API_KEY", "HF_TOKEN"}
        },
        setup_command=env.get("PRIMEBEAKER_SETUP_COMMAND"),
    )


def ensure_source_experiment_terminal(source: SourceExperiment) -> None:
    payload = _run_json(
        ("beaker", "job", "inspect", *source.job_ids, "--format", "json")
    )
    jobs = payload if isinstance(payload, list) else [payload]
    active: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = job.get("status")
        if not isinstance(status, dict) or not ({"exited", "finalized"} & status.keys()):
            active.append(f"{job.get('name', 'unknown')} ({job.get('id', 'unknown')})")
    if active:
        raise RuntimeError(
            "refusing to resume while the source topology is active; two launches must not "
            f"mutate one output directory. Active jobs: {', '.join(active)}"
        )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite resume artifact: {path}")
    _atomic_write(path, content)


def archive_stale_broadcast_state(
    output_dir: str | Path, *, attempt_id: str
) -> Path | None:
    """Move transient broadcasts aside without touching durable checkpoints."""

    output = Path(output_dir)
    source = output / "run_default/broadcasts"
    if not source.exists():
        return None
    if not source.is_dir():
        raise RuntimeError(f"Prime-RL broadcast state is not a directory: {source}")
    destination = output / "primebeaker_resume/broadcast_archives" / attempt_id
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite broadcast archive: {destination}")
    source.replace(destination)
    return destination


def plan_resume(
    from_experiment: str,
    *,
    config: str | Path | None = None,
    registry: str | None = None,
    resume_step: int | None = None,
    experiment_name: str | None = None,
    attempt_id: str | None = None,
    image: str | None = None,
    workspace: str | None = None,
    cluster: str | Sequence[str] | None = None,
    priority: str | None = None,
    min_runtime_hours: int | None = None,
    budget: str | None = None,
    source: SourceExperiment | None = None,
) -> ResumeAttempt:
    """Build a non-mutating resume plan; omitted ``resume_step`` selects latest."""

    source = source or inspect_source_experiment(from_experiment)
    config_path = Path(config or source.config_path).resolve()
    metadata = MultiNodeRLMetadata.from_path(config_path)
    if metadata.output_dir.resolve() != source.output_dir.resolve():
        raise RuntimeError(
            f"config output_dir {metadata.output_dir} does not match source {source.output_dir}"
        )
    checkpoint = select_common_checkpoint(
        metadata.output_dir,
        expected_trainer_shards=metadata.trainer_gpus,
        resume_step=resume_step,
    )
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    max_steps = raw.get("max_steps")
    if isinstance(max_steps, int) and checkpoint.step >= max_steps:
        raise RuntimeError(
            f"checkpoint step {checkpoint.step} already reached max_steps={max_steps}"
        )
    resolved_registry = registry or source.registry
    if not resolved_registry:
        raise RuntimeError(
            "the source did not record an external LiteRegistry endpoint; deploy one with "
            "`primebeaker services launch` and pass --registry"
        )

    attempt_id = attempt_id or uuid4().hex
    name = _resume_experiment_name(
        experiment_name or source.experiment_name,
        resume_step=checkpoint.step,
        attempt_id=attempt_id,
    )
    artifact_root = metadata.output_dir / "primebeaker_resume"
    generated_path = artifact_root / "configs" / f"{name}-{attempt_id}.toml"
    manifest_path = artifact_root / "attempts" / f"{attempt_id}.json"
    wandb = raw.get("wandb")
    wandb_name = (
        wandb.get("name")
        if isinstance(wandb, dict) and isinstance(wandb.get("name"), str)
        else name
    )
    generated_text = render_resume_config(
        config_path.read_text(encoding="utf-8"),
        resume_step=checkpoint.step,
        wandb_name=wandb_name,
    )
    resumed_metadata = metadata.model_copy(
        update={"toml_path": generated_path, "run_name": name}
    )
    clusters = (
        (cluster,) if isinstance(cluster, str) else tuple(cluster)
        if cluster is not None
        else source.clusters
    )
    destination_workspace = workspace or source.workspace
    request = MultiNodeRLLaunchRequest(
        metadata=resumed_metadata,
        registry=resolved_registry,
        image=image or source.image,
        workspace=destination_workspace,
        clusters=clusters,
        priority=priority or source.priority,
        min_runtime_hours=(
            source.min_runtime_hours if min_runtime_hours is None else min_runtime_hours
        ),
        mount_path=source.mount_path,
        dataset=source.dataset,
        working_dir=source.working_dir,
        scratch_dir=source.scratch_dir,
        home_dir=source.home_dir,
        wandb_run_id=source.wandb_shared_run_id or uuid4().hex,
        budget=budget,
        description=(
            f"PrimeBeaker whole-topology resume of {source.experiment_id} from "
            f"common checkpoint step {checkpoint.step}"
        ),
        gateway_port=source.gateway_port,
        gateway_workers=source.gateway_workers,
        required_services=source.required_services,
        rendezvous_timeout_seconds=source.rendezvous_timeout_seconds,
        wandb_secret=source.wandb_secret,
        hf_secret=source.hf_secret,
        secrets=source.secrets,
        setup_command=source.setup_command,
        environment={
            **source.environment,
            "PRIMEBEAKER_RESUME_PARENT_EXPERIMENT_ID": source.experiment_id,
            "PRIMEBEAKER_RESUME_STEP": str(checkpoint.step),
        },
    )
    preview = BeakerMultiNodeRLBackend().preview(request, launch_id=attempt_id)
    return ResumeAttempt(
        source=source,
        checkpoint=checkpoint,
        source_config_path=config_path,
        generated_config_path=generated_path,
        generated_config_text=generated_text,
        manifest_path=manifest_path,
        experiment_name=name,
        attempt_id=attempt_id,
        workspace=destination_workspace,
        preview=preview,
    )


def _manifest(
    attempt: ResumeAttempt,
    *,
    status: str,
    child_experiment_id: str | None = None,
    broadcast_archive: Path | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        **attempt.as_dict(),
    }
    if child_experiment_id:
        result["child_experiment_id"] = child_experiment_id
    if broadcast_archive:
        result["broadcast_archive"] = str(broadcast_archive)
    if error:
        result["error"] = error
    return result


def resume(
    from_experiment: str,
    config: str | Path | None = None,
    registry: str | None = None,
    resume_step: int | None = None,
    experiment_name: str | None = None,
    attempt_id: str | None = None,
    image: str | None = None,
    workspace: str | None = None,
    cluster: str | Sequence[str] | None = None,
    priority: str | None = None,
    min_runtime_hours: int | None = None,
    budget: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Resume all RL replicas from a complete common checkpoint using a fresh launch."""

    attempt = plan_resume(
        from_experiment,
        config=config,
        registry=registry,
        resume_step=resume_step,
        experiment_name=experiment_name,
        attempt_id=attempt_id,
        image=image,
        workspace=workspace,
        cluster=cluster,
        priority=priority,
        min_runtime_hours=min_runtime_hours,
        budget=budget,
    )
    if dry_run:
        return {
            "resume": attempt.as_dict(),
            "generated_config": attempt.generated_config_text,
        }

    ensure_source_experiment_terminal(attempt.source)
    validate_beaker_resources(workspace=attempt.workspace, spec=attempt.preview.spec)
    selected = select_common_checkpoint(
        attempt.source.output_dir,
        expected_trainer_shards=len(attempt.checkpoint.trainer_shards),
        resume_step=attempt.checkpoint.step,
    )
    if selected.as_dict() != attempt.checkpoint.as_dict():
        raise RuntimeError("selected checkpoint changed while preparing the resume attempt")
    archive = archive_stale_broadcast_state(
        attempt.source.output_dir, attempt_id=attempt.attempt_id
    )
    _write_new(attempt.generated_config_path, attempt.generated_config_text)
    try:
        BeakerMultiNodeRLBackend().write_preview(attempt.preview)
        completed = subprocess.run(
            attempt.preview.submit_command,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception as error:
        _write_new(
            attempt.manifest_path,
            json.dumps(
                _manifest(
                    attempt,
                    status="submission_failed",
                    broadcast_archive=archive,
                    error=str(error),
                ),
                indent=2,
            )
            + "\n",
        )
        raise
    match = _EXPERIMENT_ID_PATTERN.search(completed.stdout)
    child_id = match.group(1) if match else None
    _write_new(
        attempt.manifest_path,
        json.dumps(
            _manifest(
                attempt,
                status="submitted",
                child_experiment_id=child_id,
                broadcast_archive=archive,
            ),
            indent=2,
        )
        + "\n",
    )
    return {
        "resume": attempt.as_dict(),
        "child_experiment_id": child_id,
        "stdout": completed.stdout,
    }
