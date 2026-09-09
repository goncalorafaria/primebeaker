from __future__ import annotations

import json
from pathlib import Path
import tomllib

from primebeaker.config import RLTrainingToml, SFTTrainingToml
from primebeaker.rl import RLTomlMetadata
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
