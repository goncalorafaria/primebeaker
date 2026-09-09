from __future__ import annotations

import json
import tomllib
from pathlib import Path
import subprocess

import pytest

from primebeaker.resume import (
    SourceExperiment,
    archive_stale_broadcast_state,
    discover_common_checkpoints,
    inspect_source_experiment,
    plan_resume,
    render_resume_config,
    resume,
    select_common_checkpoint,
)


def _rl_toml(root: Path) -> Path:
    output = root / "outputs/rl"
    path = root / "rl.toml"
    path.write_text(
        f'''max_steps = 100
output_dir = "{output}"
clean_output_dir = true

[wandb]
name = "multinode-rl"
project = "training"

[deployment]
type = "multi_node"
gpus_per_node = 8
num_train_nodes = 1
num_infer_nodes = 2
num_infer_replicas = 1

[trainer.model]
name = "model"
cp = 2
dp_replicate = 2

[inference]
api_server_count = 6

[inference.parallel]
tp = 2
dp = 6

[ckpt]
interval = 10
resume_step = -1
''',
        encoding="utf-8",
    )
    return path


def _checkpoint(output: Path, step: int, *, shards: int, progress: bool = True) -> None:
    trainer = output / f"checkpoints/step_{step}/trainer"
    trainer.mkdir(parents=True)
    (trainer / ".metadata").write_bytes(b"metadata")
    for rank in range(shards):
        (trainer / f"__{rank}_0.distcp").write_bytes(b"shard")
    orchestrator = output / f"run_default/checkpoints/step_{step}/orchestrator"
    orchestrator.mkdir(parents=True)
    if progress:
        (orchestrator / "progress.pt").write_bytes(b"progress")


def _source(root: Path, config: Path) -> SourceExperiment:
    return SourceExperiment(
        experiment_id="01PARENT",
        experiment_name="parent",
        workspace="ai2/oe-agents",
        job_ids=("01JOB",),
        config_path=config,
        output_dir=root / "outputs/rl",
        registry="redis://registry:6379",
        image="beaker://01IMAGE",
        clusters=("ai2/holmes",),
        priority="urgent",
        min_runtime_hours=4,
        wandb_shared_run_id="wandb-run",
        mount_path=root,
        dataset="oe-adapt-default",
        working_dir=root,
        scratch_dir=root,
        home_dir=root,
        gateway_port=1212,
        gateway_workers=8,
        required_services={"terminal": 2},
        rendezvous_timeout_seconds=1800,
        wandb_secret="WANDB_API_KEY",
        hf_secret="HF_TOKEN",
        environment={"CUSTOM_SETTING": "kept"},
        secrets={"CUSTOM_SECRET": "SOURCE_SECRET"},
        setup_command="setup-prime",
    )


def test_common_checkpoint_requires_both_checkpoint_owners(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _checkpoint(output, 10, shards=2)
    _checkpoint(output, 20, shards=2, progress=False)
    _checkpoint(output, 30, shards=1)

    assert [item.step for item in discover_common_checkpoints(
        output, expected_trainer_shards=2
    )] == [10]
    assert select_common_checkpoint(output, expected_trainer_shards=2).step == 10
    with pytest.raises(RuntimeError, match="step 20 is not a complete"):
        select_common_checkpoint(
            output, expected_trainer_shards=2, resume_step=20
        )


def test_resume_config_is_lossless_except_resume_controls() -> None:
    source = '''max_steps = 100
clean_output_dir = true
custom_key = "preserved"

[wandb]
name = "same-run"

[ckpt]
interval = 10
resume_step = -1
'''
    result = render_resume_config(source, resume_step=20, wandb_name="same-run")
    parsed = tomllib.loads(result)

    assert parsed["custom_key"] == "preserved"
    assert parsed["clean_output_dir"] is False
    assert parsed["ckpt"] == {"interval": 10, "resume_step": 20}
    assert parsed["wandb"]["name"] == "same-run"


def test_inspect_source_accepts_primebeaker_replicated_task_without_node_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = [{
        "id": "01PARENT",
        "name": "parent",
        "workspaceRef": {"fullName": "ai2/oe-agents"},
        "jobs": [{
            "id": "01JOB",
            "execution": {"spec": {
                "image": {"beaker": "01IMAGE"},
                "envVars": [
                    {"name": "CONFIG_PATH", "value": "/weka/config.toml"},
                    {"name": "OUTPUT_DIR", "value": "/weka/output"},
                    {"name": "REGISTRY", "value": "redis://registry:6379"},
                    {"name": "REQUIRED_LITEREGISTRY_SERVICES", "value": "{\"terminal\":2}"},
                    {"name": "CUSTOM_SETTING", "value": "kept"},
                    {"name": "WANDB_API_KEY", "secret": "WANDB_SECRET"},
                ],
                "datasets": [{"mountPath": "/weka", "source": {"weka": "dataset"}}],
                "context": {"priority": "urgent", "minRuntime": "4h"},
                "constraints": {"cluster": ["ai2/holmes"]},
            }},
        }],
    }]
    monkeypatch.setattr("primebeaker.resume._run_json", lambda command: payload)

    source = inspect_source_experiment("01PARENT")

    assert source.config_path == Path("/weka/config.toml")
    assert source.registry == "redis://registry:6379"
    assert source.required_services == {"terminal": 2}
    assert source.min_runtime_hours == 4
    assert source.environment == {"CUSTOM_SETTING": "kept"}
    assert source.wandb_secret == "WANDB_SECRET"


def test_plan_resume_preserves_topology_and_lineage(tmp_path: Path) -> None:
    config = _rl_toml(tmp_path)
    source = _source(tmp_path, config)
    _checkpoint(source.output_dir, 20, shards=4)

    attempt = plan_resume(
        source.experiment_id,
        source=source,
        attempt_id="unitattempt",
    )
    [task] = attempt.preview.spec["tasks"]
    env = {
        item["name"]: item.get("value")
        for item in task["envVars"]
        if "value" in item
    }

    assert attempt.checkpoint.step == 20
    assert not attempt.generated_config_path.exists()
    assert task["replicas"] == 2
    assert env["CONFIG_PATH"] == str(attempt.generated_config_path)
    assert env["PRIMEBEAKER_RESUME_PARENT_EXPERIMENT_ID"] == "01PARENT"
    assert env["PRIMEBEAKER_RESUME_STEP"] == "20"
    assert env["WANDB_SHARED_RUN_ID"] == "wandb-run"
    assert env["REQUIRED_LITEREGISTRY_SERVICES"] == '{"terminal":2}'
    assert env["CUSTOM_SETTING"] == "kept"
    secret_env = {
        item["name"]: item.get("secret")
        for item in task["envVars"]
        if "secret" in item
    }
    assert secret_env["CUSTOM_SECRET"] == "SOURCE_SECRET"


def test_resume_archives_only_transient_broadcasts(tmp_path: Path) -> None:
    output = tmp_path / "output"
    _checkpoint(output, 20, shards=2)
    stale = output / "run_default/broadcasts/step_29/STABLE"
    stale.parent.mkdir(parents=True)
    stale.touch()

    archive = archive_stale_broadcast_state(output, attempt_id="attempt")

    assert archive == output / "primebeaker_resume/broadcast_archives/attempt"
    assert (archive / "step_29/STABLE").is_file()
    assert select_common_checkpoint(output, expected_trainer_shards=2).step == 20


def test_submit_materializes_config_spec_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _rl_toml(tmp_path)
    source = _source(tmp_path, config)
    _checkpoint(source.output_dir, 20, shards=4)
    attempt = plan_resume(
        source.experiment_id,
        source=source,
        attempt_id="submitattempt",
    )
    monkeypatch.setattr(
        "primebeaker.resume.plan_resume", lambda *args, **kwargs: attempt
    )
    monkeypatch.setattr(
        "primebeaker.resume.ensure_source_experiment_terminal", lambda source: None
    )
    monkeypatch.setattr(
        "primebeaker.resume.validate_beaker_resources", lambda **kwargs: None
    )
    monkeypatch.setattr(
        "primebeaker.resume.subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 0, stdout="Experiment 01CHILD submitted.\n", stderr=""
        ),
    )

    result = resume(source.experiment_id)

    assert result["child_experiment_id"] == "01CHILD"
    assert attempt.generated_config_path.is_file()
    assert attempt.preview.spec_path.is_file()
    manifest = tomllib.loads(
        attempt.generated_config_path.read_text(encoding="utf-8")
    )
    assert manifest["ckpt"]["resume_step"] == 20
    attempt_manifest = json.loads(
        attempt.manifest_path.read_text(encoding="utf-8")
    )
    assert attempt_manifest["status"] == "submitted"
    assert attempt_manifest["source_experiment_id"] == "01PARENT"
