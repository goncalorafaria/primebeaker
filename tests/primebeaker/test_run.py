from __future__ import annotations

from pathlib import Path

import pytest

from primebeaker.run import JointRunCLI, _rl_kwargs, load_run_yaml
from primebeaker.services import LiteRegistryServices


def _config(tmp_path: Path) -> Path:
    toml = tmp_path / "train.toml"
    toml.write_text(
        """output_dir = "/weka/output"

[deployment]
type = "multi_node"
""",
        encoding="utf-8",
    )
    config = tmp_path / "run.yaml"
    config.write_text(
        """schema: primebeaker.run/v1
service:
  head_registry: sqlite:///weka/run.sqlite3
  terminal_replicas: 4
rl:
  toml: train.toml
  image: beaker://runtime
  workspace: ai2/training
  clusters: [ai2/holmes]
  environment:
    FEATURE: enabled
  secrets:
    HF_TOKEN: custom-hf-secret
  required_services:
    terminal: 4
lifecycle:
  cleanup_service_on_training_exit: true
  beaker_token_secret: training-beaker-token
""",
        encoding="utf-8",
    )
    return config


def test_run_yaml_resolves_toml_and_natural_yaml_mappings(tmp_path: Path) -> None:
    document = load_run_yaml(_config(tmp_path))
    kwargs = _rl_kwargs(document, service_experiment_id="01SERVICE")

    assert kwargs["toml"] == str((tmp_path / "train.toml").resolve())
    assert kwargs["registry"] == "head+sqlite:///weka/run.sqlite3"
    assert kwargs["cluster"] == ["ai2/holmes"]
    assert kwargs["required_service"] == ["terminal=4"]
    assert set(kwargs["env"]) == {
        "FEATURE=enabled",
        "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID=01SERVICE",
    }
    assert set(kwargs["secret"]) == {
        "HF_TOKEN=custom-hf-secret",
        "BEAKER_TOKEN=training-beaker-token",
    }


def test_joint_preview_previews_services_then_rl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def service_run(action: str, **config: object) -> dict[str, object]:
        calls.append(("service", action))
        assert config["terminal_replicas"] == 4
        return {"service_preview": True}

    def training_launch(
        program: str, action: str, **kwargs: object
    ) -> dict[str, object]:
        calls.append((program, action))
        assert (
            "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID=<created-at-launch>"
            in kwargs["env"]
        )
        return {"rl_preview": True}

    monkeypatch.setattr(LiteRegistryServices, "_run", staticmethod(service_run))
    monkeypatch.setattr("primebeaker.cli._launch", training_launch)

    result = JointRunCLI().preview(_config(tmp_path))

    assert calls == [("service", "preview"), ("rl", "preview")]
    assert result == {
        "service": {"service_preview": True},
        "rl": {"rl_preview": True},
    }


def test_joint_launch_injects_created_service_experiment_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []

    def service_run(action: str, **config: object) -> dict[str, object]:
        calls.append(("service", action))
        return {
            "experiment_name": "services",
            "beaker": {"id": "01SERVICE", "jobs": [{"id": "01JOB"}]},
        }

    def training_launch(
        program: str, action: str, **kwargs: object
    ) -> dict[str, object]:
        calls.append((program, action))
        assert (
            "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID=01SERVICE"
            in kwargs["env"]
        )
        return {"training": "submitted"}

    monkeypatch.setattr(LiteRegistryServices, "_run", staticmethod(service_run))
    monkeypatch.setattr("primebeaker.cli._launch", training_launch)

    result = JointRunCLI().launch(_config(tmp_path))

    assert calls == [("service", "launch"), ("rl", "submit")]
    assert result["service_experiment_id"] == "01SERVICE"
    assert result["rl"] == {"training": "submitted"}


def test_joint_launch_rolls_back_services_when_rl_submission_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped: list[str] = []

    monkeypatch.setattr(
        LiteRegistryServices,
        "_run",
        staticmethod(
            lambda action, **config: {"beaker": {"experimentId": "01SERVICE"}}
        ),
    )

    def fail_training(*args: object, **kwargs: object) -> dict[str, object]:
        raise RuntimeError("GPU submission failed")

    monkeypatch.setattr("primebeaker.cli._launch", fail_training)
    monkeypatch.setattr(
        LiteRegistryServices,
        "stop",
        lambda self, experiment_id, dry_run=False: stopped.append(experiment_id)
        or {"stopped": experiment_id},
    )

    with pytest.raises(RuntimeError, match="GPU submission failed"):
        JointRunCLI().launch(_config(tmp_path))

    assert stopped == ["01SERVICE"]


def test_managed_cleanup_requires_multinode_rl(tmp_path: Path) -> None:
    config = _config(tmp_path)
    (tmp_path / "train.toml").write_text(
        """output_dir = "/weka/output"

[deployment]
type = "single_node"
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="requires multi-node RL"):
        _rl_kwargs(load_run_yaml(config), service_experiment_id="01SERVICE")


def test_multinode_trainer_owns_managed_service_cleanup() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "src/primebeaker/multinode_rl_role.sh"
    ).read_text(encoding="utf-8")

    assert 'if (( RANK == 0 )) && [[ -n "${PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID:-}" ]]' in script
    assert 'beaker experiment stop "$PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID"' in script
    assert "missing BEAKER_TOKEN for managed service cleanup" in script
