from __future__ import annotations

from pathlib import Path

import pytest

from primebeaker.cli import run
from primebeaker.multinode import (
    BeakerMultiNodeRLBackend,
    BeakerMultiNodeSFTBackend,
    MultiNodeRLLaunchRequest,
    MultiNodeSFTLaunchRequest,
)


def _env(task: dict[str, object]) -> dict[str, str]:
    entries = task["envVars"]
    assert isinstance(entries, list)
    return {entry["name"]: entry.get("value", entry.get("secret", "")) for entry in entries}


def _paths(root: Path) -> dict[str, Path]:
    return {
        "mount_path": root,
        "working_dir": root,
        "scratch_dir": root,
        "home_dir": root,
    }


def _rl_toml(root: Path, *, dp: int = 6) -> Path:
    path = root / "rl.toml"
    path.write_text(
        f'''output_dir = "{root / "outputs/rl"}"

[wandb]
name = "multinode-rl"
project = "training"
entity = "team"

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
api_server_count = {dp}

[inference.parallel]
tp = 2
dp = {dp}
''',
        encoding="utf-8",
    )
    return path


def _sft_toml(root: Path) -> Path:
    path = root / "sft.toml"
    path.write_text(
        f'''output_dir = "{root / "outputs/sft"}"

[wandb]
name = "multinode-sft"
project = "training"
entity = "team"

[deployment]
type = "multi_node"
num_nodes = 3
gpus_per_node = 8

[data]
type = "sft"
name = "{root / "data"}"

[val]
interval = 10
eval_on_start = true

[val.data]
type = "sft"
name = "{root / "data"}"
''',
        encoding="utf-8",
    )
    return path


def test_multinode_rl_preview_preserves_mixed_train_inference_topology(tmp_path: Path) -> None:
    request = MultiNodeRLLaunchRequest.from_toml(
        _rl_toml(tmp_path),
        image="beaker://runtime",
        registry="redis://registry.example:6379",
        wandb_run_id="shared-run",
        required_services={"terminal": 4},
        **_paths(tmp_path),
    )
    preview = BeakerMultiNodeRLBackend().preview(request, launch_id="launch")
    [task] = preview.spec["tasks"]
    environment = _env(task)

    assert task["replicas"] == 2
    assert task["leaderSelection"] is True
    assert task["hostNetworking"] is True
    assert task["resources"] == {"gpuCount": 8}
    assert task["command"] == ["python3", "-m", "primebeaker.multinode", "role", "rl"]
    assert environment["TRAINER_GPU_COUNT"] == "4"
    assert environment["MIXED_INFER_GPU_COUNT"] == "4"
    assert environment["TOTAL_INFER_RANKS"] == "6"
    assert environment["WANDB_SHARED_RUN_ID"] == "shared-run"
    assert environment["REQUIRED_LITEREGISTRY_SERVICES"] == '{"terminal":4}'


def test_multinode_sft_preview_uses_one_gang_started_replica_per_node(tmp_path: Path) -> None:
    request = MultiNodeSFTLaunchRequest.from_toml(
        _sft_toml(tmp_path), image="runtime:latest", **_paths(tmp_path)
    )
    preview = BeakerMultiNodeSFTBackend().preview(request, launch_id="launch")
    [task] = preview.spec["tasks"]

    assert task["replicas"] == 3
    assert task["leaderSelection"] is True
    assert task["synchronizedStartTimeout"] == "4h"
    assert task["resources"] == {"gpuCount": 8}
    assert task["command"] == ["python3", "-m", "primebeaker.multinode", "role", "sft"]
    assert _env(task)["TOTAL_NODES"] == "3"


def test_cli_auto_selects_multinode_backend(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = run([
        "rl", "preview", "--toml", str(_rl_toml(tmp_path)),
        "--image", "beaker://runtime",
        "--registry", "redis://registry.example:6379",
        "--mount-path", str(tmp_path),
        "--working-dir", str(tmp_path),
        "--scratch-dir", str(tmp_path),
        "--home-dir", str(tmp_path),
    ])

    assert payload["spec"]["tasks"][0]["replicas"] == 2
    assert "primebeaker.multinode" in payload["spec"]["tasks"][0]["command"]


def test_multinode_rl_rejects_inconsistent_external_lb_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="external-LB rank count"):
        MultiNodeRLLaunchRequest.from_toml(
            _rl_toml(tmp_path, dp=5),
            image="beaker://runtime",
            registry="redis://registry.example:6379",
            **_paths(tmp_path),
        )
