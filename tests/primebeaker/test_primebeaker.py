from __future__ import annotations

from pathlib import Path
import tomllib

import pytest

from primebeaker.cli import run
from primebeaker.config import RLTrainingToml, SFTTrainingToml, TrainingData
from primebeaker.rl import BeakerRLBackend, RLLaunchRequest, RLTomlMetadata
from primebeaker.sft import BeakerSFTBackend, SFTLaunchRequest


def _sft_toml(root: Path) -> Path:
    path = root / "sft.toml"
    path.write_text(
        f'''output_dir = "{root / 'outputs/sft'}"
max_steps = 20

[deployment]
type = "single_node"
num_gpus = 4
gpus_per_node = 4

[wandb]
name = "sft-example"
project = "training"
entity = "team"
tags = ["sft"]

[data]
type = "sft"
name = "{root / 'data'}"
splits = ["train"]

[val]
interval = 10
eval_on_start = true

[val.data]
type = "sft"
name = "{root / 'data'}"
splits = ["validation"]

[ckpt]
interval = 20
''',
        encoding="utf-8",
    )
    return path


def _rl_toml(root: Path, *, deployment_type: str = "single_node") -> Path:
    path = root / "rl.toml"
    path.write_text(
        f'''output_dir = "{root / 'outputs/rl'}"

[deployment]
type = "{deployment_type}"
num_train_gpus = 2
num_infer_gpus = 6
gpus_per_node = 8

[wandb]
name = "rl-example"
project = "training"
entity = "team"
tags = ["rl"]

[trainer.optim]
lr = 0.000004

[trainer.loss]
kl_tau = 0.001

[trainer.model]
name = "model-a"

[orchestrator]
oversampling_factor = 1

[orchestrator.model]
name = "model-a"

[orchestrator.train]
[[orchestrator.train.source]]
name = "train-source"
[orchestrator.train.source.legacy.args]
dataset = "old-train.jsonl"

[orchestrator.eval]
[[orchestrator.eval.source]]
name = "eval-source"
num_examples = 10
[orchestrator.eval.source.legacy.args]
dataset = "old-validation.jsonl"

[inference.model]
name = "model-a"
''',
        encoding="utf-8",
    )
    return path


def _request_paths(root: Path) -> dict[str, Path]:
    return {
        "mount_path": root,
        "working_dir": root,
        "scratch_dir": root,
        "home_dir": root,
    }


def _env(task: dict[str, object]) -> dict[str, str]:
    entries = task["envVars"]
    assert isinstance(entries, list)
    return {
        entry["name"]: entry.get("value", entry.get("secret", ""))
        for entry in entries
    }


def test_sft_preview_derives_topology_and_runtime_command(tmp_path: Path) -> None:
    toml = _sft_toml(tmp_path)
    request = SFTLaunchRequest.from_toml(
        toml,
        image="beaker://account/runtime",
        wandb_run_id="sft-run-id",
        **_request_paths(tmp_path),
    )
    preview = BeakerSFTBackend().preview(request)
    [task] = preview.spec["tasks"]

    assert task["image"] == {"beaker": "account/runtime"}
    assert task["resources"] == {"gpuCount": 4}
    assert task["datasets"] == [
        {"mountPath": str(tmp_path), "source": {"weka": "oe-adapt-default"}}
    ]
    assert _env(task)["WANDB_RUN_ID"] == "sft-run-id"
    assert "exec sft @" in task["command"][2]
    assert "Quarantining incomplete checkpoint" in task["command"][2]
    assert preview.wandb_run.url == "https://wandb.ai/team/training/runs/sft-run-id"
    assert "primebeaker/beaker_experiments" in str(preview.spec_path)


def test_rl_preview_runs_prime_rl_and_accepts_environment_policy(tmp_path: Path) -> None:
    toml = _rl_toml(tmp_path)
    request = RLLaunchRequest.from_toml(
        toml,
        image="registry.example/runtime:latest",
        registry="redis://service:6379",
        environment={"CUSTOM_ENDPOINT": "http://service"},
        secrets={"SERVICE_TOKEN": "beaker-service-token"},
        wandb_run_id="rl-run-id",
        **_request_paths(tmp_path),
    )
    preview = BeakerRLBackend().preview(request)
    [task] = preview.spec["tasks"]
    environment = _env(task)

    assert task["image"] == {"docker": "registry.example/runtime:latest"}
    assert task["resources"] == {"gpuCount": 8}
    assert environment["REGISTRY"] == "redis://service:6379"
    assert environment["CUSTOM_ENDPOINT"] == "http://service"
    assert environment["SERVICE_TOKEN"] == "beaker-service-token"
    assert environment["WANDB_SHARED_RUN_ID"] == "rl-run-id"
    assert "exec rl @" in task["command"][2]


def test_launch_paths_must_be_visible_under_the_configured_mount(tmp_path: Path) -> None:
    toml = _sft_toml(tmp_path)
    with pytest.raises(ValueError, match="launch path must be under mounted path"):
        SFTLaunchRequest.from_toml(
            toml,
            image="beaker://account/runtime",
            mount_path=tmp_path / "different-mount",
            working_dir=tmp_path,
            scratch_dir=tmp_path,
            home_dir=tmp_path,
        )


def test_direct_rl_launcher_rejects_multi_node_toml(tmp_path: Path) -> None:
    toml = _rl_toml(tmp_path, deployment_type="multi_node")
    with pytest.raises(ValueError, match="single_node"):
        RLTomlMetadata.from_path(toml)


def test_rl_toml_edits_and_round_trips_without_mutating_source(tmp_path: Path) -> None:
    toml = _rl_toml(tmp_path)
    source = RLTrainingToml.from_path(toml)
    edited = (
        source.with_learning_rate(2e-6)
        .with_model_checkpoint("model-b")
        .with_wandb_tags("candidate")
        .bind_data(
            TrainingData(
                kind="rl",
                train_path="new-train.jsonl",
                validation_path="new-validation.jsonl",
                environment_args={"timeout": 30},
            )
        )
    )
    output = edited.write(tmp_path / "rendered.toml")
    with output.open("rb") as stream:
        rendered = tomllib.load(stream)

    assert source.values["trainer"]["optim"]["lr"] == 4e-6
    assert rendered["trainer"]["optim"]["lr"] == 2e-6
    for section, dataset in (("train", "new-train.jsonl"), ("eval", "new-validation.jsonl")):
        args = rendered["orchestrator"][section]["source"][0]["legacy"]["args"]
        assert args == {"dataset": dataset, "timeout": 30}
    for table in ("trainer", "orchestrator", "inference"):
        assert rendered[table]["model"]["name"] == "model-b"
    assert rendered["wandb"]["tags"] == ["rl", "candidate"]


def test_sft_data_binding_uses_one_dataset_with_distinct_splits(tmp_path: Path) -> None:
    source = SFTTrainingToml.from_path(_sft_toml(tmp_path))
    edited = source.bind_data(
        TrainingData(
            kind="sft",
            train_path="ignored-train.jsonl",
            validation_path="ignored-validation.jsonl",
            hf_dataset_path="/mounted/dataset",
        )
    )

    assert source.values["data"]["name"] == str(tmp_path / "data")
    assert edited.values["data"]["name"] == "/mounted/dataset"
    assert edited.values["data"]["splits"] == ["train"]
    assert edited.values["val"]["data"]["splits"] == ["validation"]


def test_cli_preview_is_non_mutating(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    toml = _rl_toml(tmp_path)
    result = run(
        [
            "rl",
            "preview",
            "--toml",
            str(toml),
            "--image",
            "beaker://account/runtime",
            "--mount-path",
            str(tmp_path),
            "--working-dir",
            str(tmp_path),
            "--scratch-dir",
            str(tmp_path),
            "--home-dir",
            str(tmp_path),
            "--env",
            "FEATURE=true",
        ]
    )

    assert result["experiment_name"] == "rl-example"
    assert result["spec"]["tasks"][0]["resources"]["gpuCount"] == 8
    assert not Path(result["spec_path"]).exists()
    assert '"experiment_name": "rl-example"' in capsys.readouterr().out
