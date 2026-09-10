from __future__ import annotations

import json
from pathlib import Path
import tomllib

import pytest

from primebeaker.config import RLTrainingToml, SFTTrainingToml
from primebeaker.judge_catalog import load_judge_model_profile
from primebeaker.multinode import MultiNodeRLMetadata
from primebeaker.rl import RLTomlMetadata
from primebeaker.run import _rl_kwargs, load_run_yaml
from primebeaker.services import load_services_yaml
from primebeaker.sft import SFTTomlMetadata


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_tiny_sft_example_preserves_message_and_tool_shapes(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    config_path = ROOT / "examples/sft/config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    train = _rows(ROOT / "examples/sft/data/train.jsonl")
    validation = _rows(ROOT / "examples/sft/data/validation.jsonl")

    assert len(train) == 4
    assert len(validation) == 2
    assert all(set(row) == {"messages", "tools"} for row in train + validation)
    assert isinstance(train[0]["tools"], str)
    assert json.loads(train[0]["tools"])[0]["function"]["name"] == "terminal"
    assert train[0]["messages"][2]["tool_calls"][0]["function"]["name"] == "terminal"
    assert config["data"]["name"] == "examples/sft/data"
    assert config["val"]["data"]["name"] == "examples/sft/data"
    SFTTrainingToml.from_path(config_path)
    SFTTomlMetadata.from_path(config_path)


def test_tiny_rl_example_preserves_verifier_record_shape(monkeypatch) -> None:
    monkeypatch.chdir(ROOT)
    config_path = ROOT / "examples/rl/config.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    train = _rows(ROOT / "examples/rl/data/train.jsonl")
    validation = _rows(ROOT / "examples/rl/data/validation.jsonl")
    required = {
        "prompt",
        "answer",
        "output",
        "feedback",
        "record_id",
        "rubric_index",
        "source",
        "tools",
    }

    assert len(train) == 4
    assert len(validation) == 2
    assert all(set(row) == required for row in train + validation)
    assert {row["answer"] for row in train} == {"pass", "fail"}
    train_source = config["orchestrator"]["train"]["source"][0]
    assert train_source["legacy"]["id"] == "primebeaker.environments.jtc_label_env"
    assert train_source["legacy"]["args"]["dataset"] == "examples/rl/data/train.jsonl"
    RLTrainingToml.from_path(config_path)
    RLTomlMetadata.from_path(config_path)


def test_search_agent_webterminal_example_has_matching_multinode_topology() -> None:
    config_path = ROOT / "examples/search-agent-webterminal/multinode-rl.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["deployment"] == {
        "type": "multi_node",
        "gpus_per_node": 8,
        "num_train_nodes": 1,
        "num_infer_nodes": 3,
        "num_infer_replicas": 1,
    }
    assert config["trainer"]["model"]["dp_replicate"] == 8
    assert config["inference"]["parallel"] == {"tp": 1, "dp": 24}
    assert config["inference"]["api_server_count"] == 24
    assert config["orchestrator"]["batch_size"] == 256
    assert config["orchestrator"]["group_size"] == 16
    train_args = config["orchestrator"]["train"]["source"][0]["legacy"]["args"]
    assert (
        config["orchestrator"]["train"]["source"][0]["legacy"]["id"]
        == "primebeaker.environments.jtc_search_agent_webterminal_judge_env"
    )
    assert train_args["local_search_model_path"] == "localsearch:bc-rl-v1"
    assert train_args["max_tool_calls"] == 150
    assert train_args["judge_service_model_path"] == "judge"
    assert train_args["judge_server_url"] == "http://127.0.0.1:1212/judge"
    assert train_args["validate_judge_profile"] is True
    assert (
        load_judge_model_profile(train_args["judge_model_path"]).source_path.name
        == "quokka_sft_qwen35_9b_rltracer_xmlv1_search25_ba8d07_step400.json"
    )
    RLTrainingToml.from_path(config_path)
    metadata = MultiNodeRLMetadata.from_path(config_path)
    assert metadata.total_nodes == 4
    assert metadata.total_infer_gpus == 24


def test_search_agent_webterminal_services_create_redis_through_sqlite_head() -> None:
    yaml = pytest.importorskip("yaml")
    config_path = ROOT / "examples/search-agent-webterminal/services.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    services = load_services_yaml(config_path)

    assert document["schema"] == "primebeaker.services/v1"
    assert services["head_registry"].startswith("sqlite:///")
    assert "registry" not in services
    assert services["terminal_replicas"] == 32
    assert services["local_search_replicas"] == 32


def test_search_agent_joint_run_matches_independent_launches() -> None:
    config_path = ROOT / "examples/search-agent-webterminal/run.yaml"
    document = load_run_yaml(config_path)
    kwargs = _rl_kwargs(document, service_experiment_id="01SERVICE")

    assert document["service"]["terminal_replicas"] == 32
    assert document["service"]["local_search_replicas"] == 32
    assert kwargs["registry"].startswith("head+sqlite:///")
    assert set(kwargs["required_service"]) == {
        "terminal=32",
        "localsearch:bc-rl-v1=32",
        "judge=1",
    }
    assert "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID=01SERVICE" in kwargs["env"]


def test_podman_terminal_single_node_example_is_live_and_rule_scored() -> None:
    config_path = ROOT / "examples/podman-terminal/rl.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    train = _rows(ROOT / "examples/podman-terminal/data/train.jsonl")
    validation = _rows(ROOT / "examples/podman-terminal/data/validation.jsonl")

    assert len(train) == 2
    assert len(validation) == 1
    assert all(set(row) == {"prompt", "original_image", "record_id"} for row in train + validation)
    assert all(row["original_image"].endswith("python:3.12-slim") for row in train + validation)
    source = config["orchestrator"]["train"]["source"][0]
    assert source["legacy"]["id"] == "primebeaker.environments.podman_terminal_env"
    args = source["legacy"]["args"]
    assert args["completion_marker"] == "TERMINAL_COMPLETE"
    assert args["reward_file_weight"] == 1.0
    assert args["termination_reward_weight"] == 0.1
    assert args["podman_failure_penalty_weight"] == 1.0
    assert args["fake_tool_penalty_weight"] == 0.1
    assert "primebeaker-answer.txt" in args["test_command"]
    assert config["deployment"] == {
        "type": "single_node",
        "gpus_per_node": 2,
        "num_train_gpus": 1,
        "num_infer_gpus": 1,
    }
    RLTrainingToml.from_path(config_path)
    metadata = RLTomlMetadata.from_path(config_path)
    assert metadata.num_gpus == 2


def test_podman_terminal_multinode_example_has_matching_topology() -> None:
    config_path = ROOT / "examples/podman-terminal/multinode-rl.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))

    assert config["deployment"] == {
        "type": "multi_node",
        "gpus_per_node": 8,
        "num_train_nodes": 1,
        "num_infer_nodes": 1,
        "num_infer_replicas": 1,
    }
    assert config["trainer"]["model"]["dp_replicate"] == 8
    assert config["inference"]["parallel"] == {"tp": 1, "dp": 8}
    assert config["inference"]["api_server_count"] == 8
    source = config["orchestrator"]["train"]["source"][0]
    assert source["legacy"]["id"] == "primebeaker.environments.podman_terminal_env"
    RLTrainingToml.from_path(config_path)
    metadata = MultiNodeRLMetadata.from_path(config_path)
    assert metadata.total_nodes == 2
    assert metadata.total_infer_gpus == 8


def test_podman_terminal_services_match_training_registry(monkeypatch) -> None:
    yaml = pytest.importorskip("yaml")
    monkeypatch.setenv("REGISTRY", "redis://registry.internal:6379")
    config_path = ROOT / "examples/podman-terminal/services.yaml"
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    services = load_services_yaml(config_path)

    assert document["schema"] == "primebeaker.services/v1"
    assert services["registry"] == "redis://registry.internal:6379"
    assert services["podman_replicas"] == 16
    assert services["docker_mirror_replicas"] == 2
    assert services["podman_session_image"] == "docker.io/library/python:3.12-slim"
