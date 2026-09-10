from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from primebeaker.evaluation import (
    PythonEvaluationRequest,
    PythonEvaluationScheduler,
)
from primebeaker.evaluation_worker import ManagedEvaluationWorker


def _request(tmp_path: Path) -> PythonEvaluationRequest:
    return PythonEvaluationRequest(
        module="jtc.eval.worker",
        arguments=("--config-json={}",),
        image="beaker://01EVAL",
        run_dir=tmp_path,
        mount_path=tmp_path.parent,
        workspace="ai2/evals",
        evaluation_cluster="ai2/jupiter",
        services={
            "generation_model": "Qwen/Qwen3.5-4B",
            "generation_replicas": 2,
            "terminal_replicas": 4,
            "python_replicas": 0,
            "web_search_replicas": 0,
            "service_clusters": ("ai2/jupiter",),
            "model_cluster": "ai2/holmes",
            "shared_dir": str(tmp_path.parent),
        },
        required_services={"Qwen/Qwen3.5-4B": 1, "terminal": 1},
        beaker_token_secret="eval-token",
    )


def test_preview_uses_native_services_and_python_only_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Native:
        def preview(self) -> dict[str, object]:
            return {"experiment_name": "native-services", "spec": {"tasks": []}}

    monkeypatch.setattr("primebeaker.evaluation._native_launcher", lambda config: Native())
    preview = PythonEvaluationScheduler().preview(_request(tmp_path))
    task = preview.evaluation_spec["tasks"][0]

    assert task["command"] == ["python3", "-m", "primebeaker.evaluation_worker"]
    assert task["constraints"] == {"cluster": ["ai2/jupiter"]}
    assert task["image"] == {"beaker": "01EVAL"}
    env = {item["name"]: item for item in task["envVars"]}
    assert env["BEAKER_TOKEN"] == {"name": "BEAKER_TOKEN", "secret": "eval-token"}
    assert env["PRIMEBEAKER_EVALUATION_MODULE"]["value"] == "jtc.eval.worker"
    assert preview.coordination_root.endswith("/.literegistry-coop/native-services")


def test_submit_rolls_back_native_services_when_evaluator_submit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stopped: list[str] = []

    class Native:
        def submit(self) -> dict[str, object]:
            return {
                "experiment_name": "native-services",
                "beaker": {"id": "01SERVICES"},
            }

        def stop(self, experiment_id: str) -> None:
            stopped.append(experiment_id)

    scheduler = PythonEvaluationScheduler()
    monkeypatch.setattr("primebeaker.evaluation._native_launcher", lambda config: Native())
    monkeypatch.setattr(
        scheduler,
        "_submit_spec",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("submit failed")),
    )

    with pytest.raises(RuntimeError, match="submit failed"):
        scheduler.submit(_request(tmp_path))
    assert stopped == ["01SERVICES"]


def test_worker_resolves_native_endpoints_runs_module_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = {
        "PRIMEBEAKER_EVALUATION_MODULE": "jtc.eval.worker",
        "PRIMEBEAKER_EVALUATION_ARGUMENTS_JSON": json.dumps(["--config-json={}"]),
        "PRIMEBEAKER_COORDINATION_ROOT": "/weka/.coop/run",
        "PRIMEBEAKER_REQUIRED_SERVICES_JSON": json.dumps({"terminal": 1}),
        "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID": "01SERVICES",
        "PRIMEBEAKER_KEEP_SERVICES": "False",
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    endpoints: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "primebeaker.evaluation_worker.wait_endpoint",
        lambda root, name, **kwargs: endpoints.append((root, name))
        or ("redis://redis:6379" if name == "redis" else "http://gateway:8080"),
    )
    requirements: list[tuple[str, dict[str, int]]] = []
    monkeypatch.setattr(
        "primebeaker.evaluation_worker.wait_for_services",
        lambda registry, expected, **kwargs: requirements.append(
            (registry, expected)
        ),
    )
    spawned: list[tuple[tuple[str, ...], dict[str, str]]] = []

    class Process:
        pid = 123

        def __init__(self, command, *, env, start_new_session):
            spawned.append((tuple(command), env))

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0

    stopped: list[tuple[str, ...]] = []
    monkeypatch.setattr("primebeaker.evaluation_worker.subprocess.Popen", Process)
    monkeypatch.setattr(
        "primebeaker.evaluation_worker.subprocess.run",
        lambda command, **kwargs: stopped.append(tuple(command))
        or SimpleNamespace(returncode=0),
    )

    assert ManagedEvaluationWorker().run() == 0
    assert endpoints == [
        ("/weka/.coop/run", "redis"),
        ("/weka/.coop/run", "gateway"),
    ]
    assert requirements == [("redis://redis:6379", {"terminal": 1})]
    assert spawned[0][0][1:] == ("-m", "jtc.eval.worker", "--config-json={}")
    assert spawned[0][1]["OPENAI_BASE_URL"] == "http://gateway:8080/v1"
    assert stopped == [("beaker", "experiment", "stop", "01SERVICES")]


def test_request_rejects_non_module_commands(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid Python module"):
        _request(tmp_path).model_copy(update={"module": "bash -lc eval"}).model_validate(
            _request(tmp_path).model_dump() | {"module": "bash -lc eval"}
        )
