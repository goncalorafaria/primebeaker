from pathlib import Path

from fastapi.testclient import TestClient

from primebeaker.watcher.app import (
    _training_hyperparameters,
    create_app,
)
from primebeaker.watcher.index import WatcherIndex


def _database(path: Path, output: Path) -> None:
    model = output / "weights/step_100"
    with WatcherIndex(path).connect() as db:
        model_id = db.execute(
            """INSERT INTO models(model_path, output_dir, checkpoint_step, indexed_at)
               VALUES (?, ?, 100, '2026-08-19T00:00:00+00:00')""",
            (str(model), str(output)),
        ).lastrowid
        rl_id = db.execute(
            """INSERT INTO training_runs(
                   model_id, stage, checkpoint_path, output_dir, config_path
               ) VALUES (?, 'rl', ?, ?, ?)""",
            (model_id, str(model), str(output), "/repo/examples/configs/train/rl/run.toml"),
        ).lastrowid
        sft_id = db.execute(
            """INSERT INTO training_runs(
                   model_id, stage, checkpoint_path, output_dir, config_path
               ) VALUES (?, 'sft', ?, ?, ?)""",
            (
                model_id,
                "/outputs/sft/weights/step_200",
                "/outputs/sft",
                "/repo/examples/configs/train/sft/run.toml",
            ),
        ).lastrowid
        db.execute(
            """INSERT INTO wandb_runs(
                   training_run_id, run_id, project, entity, name, url,
                   offline, local_run_path
               ) VALUES (?, 'rl-attempt-1', 'rubric-rl', 'graf', 'rl run',
                         'https://wandb.ai/graf/rubric-rl/runs/rl-attempt-1', 0,
                         '/outputs/rl/wandb/run-rl-attempt-1')""",
            (rl_id,),
        )
        db.execute(
            """INSERT INTO wandb_runs(
                   training_run_id, run_id, project, entity, name, offline
               ) VALUES (?, 'sft-attempt-1', 'rubric-sft', 'graf', 'sft run', 0)""",
            (sft_id,),
        )
        dataset_id = db.execute(
            """INSERT INTO datasets(
                   identity_key, dataset_name, manifest_path, config_uid, kind
               ) VALUES ('manifest:test', 'rl-mix', '/data/rl.manifest.json', 'abc123', 'rl')"""
        ).lastrowid
        db.execute(
            """INSERT INTO training_run_datasets(
                   training_run_id, dataset_id, role, configured_path
               ) VALUES (?, ?, 'train', '/data/rl_train.jsonl')""",
            (rl_id, dataset_id),
        )
        db.execute(
            """INSERT INTO dataset_sources(
                   dataset_id, source_name, source_path, weight,
                   train_quota, validation_quota, raw_source_json
               ) VALUES (?, 'science', '/records/science.parquet', 1.0, 5000, 128, '{}')""",
            (dataset_id,),
        )
        evaluation_id = db.execute(
            """INSERT INTO evaluations(
                   model_id, config_path, workflow, run_dir, status, raw_config_yaml
               ) VALUES (?, '/repo/examples/configs/eval/science.yaml', 'moss', '/evals/science',
                         'complete', 'evaluation: {}')""",
            (model_id,),
        ).lastrowid
        db.execute(
            """INSERT INTO evaluation_benchmarks(
                   evaluation_id, dataset, runs, mean_accuracy, stdev_accuracy,
                   min_accuracy, max_accuracy
               ) VALUES (?, 'frontier-science', 128, 0.625, 0.1, 0.0, 1.0)""",
            (evaluation_id,),
        )


def test_watcher_app_browses_checkpoint_lineage(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    output = outputs / "rl-run"
    database = tmp_path / "watcher.sqlite3"
    _database(database, output)
    client = TestClient(create_app(database=database))

    page = client.get("/")
    models = client.get("/api/models").json()
    detail = client.get("/api/models/1").json()

    assert page.status_code == 200
    assert "PrimeBeaker <i>Watcher</i>" in page.text
    assert models["count"] == 1
    assert "trace_available" not in models["models"][0]
    assert "trace_url" not in detail["model"]
    assert {run["stage"] for run in detail["training_runs"]} == {"rl", "sft"}
    assert {run["wandb_runs"][0]["run_id"] for run in detail["training_runs"]} == {
        "rl-attempt-1",
        "sft-attempt-1",
    }
    assert detail["datasets"][0]["sources"][0]["source_path"] == "/records/science.parquet"
    assert detail["datasets"][0]["role"] == "rl"
    assert detail["datasets"][0]["configured_paths"] == ["/data/rl_train.jsonl"]
    assert detail["evaluations"][0]["benchmarks"][0]["mean_accuracy"] == 0.625


def test_training_hyperparameters_cover_rl_and_sft_layouts() -> None:
    rl = _training_hyperparameters(
        """
max_steps = 800
[trainer.loss]
kl_tau = 0.001
[trainer.optim]
lr = 4e-6
weight_decay = 0.0
betas1 = 0.95
betas2 = 0.95
[orchestrator]
batch_size = 128
group_size = 16
seq_len = 32768
oversampling_factor = 1
"""
    )
    sft = _training_hyperparameters(
        """
max_steps = 200
[data]
seq_len = 32768
batch_size = 32
micro_batch_size = 2
[optim]
lr = 1e-5
weight_decay = 0.1
[scheduler]
type = "cosine"
warmup_steps = 10
min_lr = 1e-6
"""
    )

    assert rl == {
        "learning_rate": 4e-6,
        "weight_decay": 0.0,
        "beta1": 0.95,
        "beta2": 0.95,
        "max_steps": 800,
        "sequence_length": 32768,
        "batch_size": 128,
        "group_size": 16,
        "kl_tau": 0.001,
        "oversampling_factor": 1,
    }
    assert sft == {
        "learning_rate": 1e-5,
        "min_learning_rate": 1e-6,
        "weight_decay": 0.1,
        "max_steps": 200,
        "sequence_length": 32768,
        "batch_size": 32,
        "micro_batch_size": 2,
        "warmup_steps": 10,
        "scheduler": "cosine",
    }
    assert _training_hyperparameters("not = valid = toml") == {}



def test_watcher_loads_and_aggregates_wandb_metrics_only_on_demand(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    output = outputs / "rl-run"
    database = tmp_path / "watcher.sqlite3"
    _database(database, output)
    with WatcherIndex(database).connect() as db:
        sft_id = db.execute(
            "SELECT id FROM training_runs WHERE model_id = 1 AND stage = 'sft'"
        ).fetchone()[0]
        db.execute(
            """INSERT INTO wandb_runs(
                   training_run_id, run_id, project, entity, name, offline
               ) VALUES (?, 'sft-attempt-2', 'rubric-sft', 'graf', 'sft restart', 0)""",
            (sft_id,),
        )

    accuracy = "eval/jtc-tool-label-terminal-eval/all/metrics/correct_final_label/mean"

    class FakeRun:
        state = "finished"

        def __init__(self, path: str) -> None:
            self.path = path

        def history(self, **_: object) -> list[dict[str, float]]:
            if self.path.endswith("rl-attempt-1"):
                return [
                    {"step": 100, accuracy: 0.6},
                    {"step": 200, accuracy: 0.8},
                ]
            if self.path.endswith("sft-attempt-1"):
                return [{"step": 0, "val/loss": 0.5}, {"step": 100, "val/loss": 0.3}]
            return [{"step": 0, "val/loss": 0.7}, {"step": 150, "val/loss": 0.25}]

    class FakeApi:
        def __init__(self) -> None:
            self.paths: list[str] = []

        def run(self, path: str) -> FakeRun:
            self.paths.append(path)
            return FakeRun(path)

    fake_api = FakeApi()
    factory_calls: list[str] = []

    def factory(api_key: str) -> FakeApi:
        factory_calls.append(api_key)
        return fake_api

    client = TestClient(
        create_app(
            database=database,
            wandb_api_key="test-key",
            wandb_api_factory=factory,
        )
    )

    detail = client.get("/api/models/1")
    assert detail.status_code == 200
    assert factory_calls == []

    response = client.get("/api/models/1/metrics")
    assert response.status_code == 200
    data = response.json()
    assert factory_calls == ["test-key"]
    assert set(fake_api.paths) == {
        "graf/rubric-rl/rl-attempt-1",
        "graf/rubric-sft/sft-attempt-1",
        "graf/rubric-sft/sft-attempt-2",
    }
    charts = {chart["id"]: chart for chart in data["charts"]}
    validation = charts["sft_validation_loss"]
    assert validation["attempt_count"] == 2
    assert validation["points"][0] == {
        "step": 0,
        "value": 0.6,
        "min": 0.5,
        "max": 0.7,
        "samples": 2,
        "attempts": ["graf/rubric-sft/sft-attempt-1", "graf/rubric-sft/sft-attempt-2"],
    }
    eval_chart = charts["eval_accuracy"]
    assert [point["step"] for point in eval_chart["points"]] == [100, 200]
    assert eval_chart["metrics_used"] == [accuracy]
    page = client.get("/").text
    script = client.get("/watcher.js")
    assert "Live W&amp;B metrics" in script.text
    assert '<script src="/watcher.js"></script>' in page
    assert "loadMetrics(id)" not in page
    assert script.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert "loadMetrics(id)" in script.text


def test_watcher_app_search_and_missing_database_errors(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    database = tmp_path / "watcher.sqlite3"
    _database(database, outputs / "distinctive-run")
    client = TestClient(create_app(database=database))

    assert client.get("/api/models?q=distinctive").json()["count"] == 1
    assert client.get("/api/models?q=no-match").json()["count"] == 0
    assert client.get("/api/models/999").status_code == 404

    missing = TestClient(create_app(database=tmp_path / "missing.sqlite3"))
    response = missing.get("/api/models")
    assert response.status_code == 503
    assert "does not exist" in response.json()["detail"]


def test_watcher_app_groups_steps_by_run_and_sorts_evaluated_runs_first(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    evaluated = outputs / "evaluated-run"
    database = tmp_path / "watcher.sqlite3"
    _database(database, evaluated)
    with WatcherIndex(database).connect() as db:
        step_200_id = db.execute(
            """INSERT INTO models(model_path, output_dir, checkpoint_step, indexed_at)
               VALUES (?, ?, 200, '2026-08-19T01:00:00+00:00')""",
            (str(evaluated / "weights/step_200"), str(evaluated)),
        ).lastrowid
        db.execute(
            """INSERT INTO evaluations(
                   model_id, config_path, workflow, run_dir, status, raw_config_yaml
               ) VALUES (?, '/repo/examples/configs/eval/step200.yaml', 'moss', '/evals/step200',
                         'complete', 'evaluation: {}')""",
            (step_200_id,),
        )
        plain = outputs / "plain-run"
        db.execute(
            """INSERT INTO models(model_path, output_dir, checkpoint_step, indexed_at)
               VALUES (?, ?, 300, '2026-08-19T02:00:00+00:00')""",
            (str(plain / "weights/step_300"), str(plain)),
        )

    client = TestClient(create_app(database=database))
    page = client.get("/").text
    script = client.get("/watcher.js").text
    models = client.get("/api/models").json()["models"]
    detail = client.get(f"/api/models/{models[0]['id']}").json()

    assert [model["name"] for model in models] == ["evaluated-run", "plain-run"]
    assert models[0]["has_evaluations"] is True
    assert models[0]["evaluation_count"] == 2
    assert models[0]["checkpoint_count"] == 2
    assert {checkpoint["checkpoint_step"] for checkpoint in detail["checkpoints"]} == {100, 200}
    assert {evaluation["checkpoint_step"] for evaluation in detail["evaluations"]} == {100, 200}
    assert all("model_id" in evaluation for evaluation in detail["evaluations"])
    assert "model-card.has-evals" in page
    assert "checkpoint-strip" in page
    assert "checkpoint-menu" in page
    assert "eval-incomplete" in page
    assert "incomplete eval" in script
    assert "<th>Stage / step</th>" in script
    assert "evaluated first" in script


def test_rl_detail_includes_sft_evals_and_sft_is_a_standalone_entry(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    rl_output = outputs / "rl-run"
    database = tmp_path / "watcher.sqlite3"
    _database(database, rl_output)
    sft_model = Path("/outputs/sft/weights/step_200")
    with WatcherIndex(database).connect() as db:
        sft_id = db.execute(
            """INSERT INTO models(model_path, output_dir, checkpoint_step, indexed_at)
               VALUES (?, '/outputs/sft', 200, '2026-08-19T03:00:00+00:00')""",
            (str(sft_model),),
        ).lastrowid
        db.execute(
            """INSERT INTO training_runs(model_id, stage, checkpoint_path, output_dir)
               VALUES (?, 'sft', ?, '/outputs/sft')""",
            (sft_id, str(sft_model)),
        )
        db.execute(
            """INSERT INTO evaluations(
                   model_id, config_path, workflow, run_dir, status, raw_config_yaml
               ) VALUES (?, '/repo/examples/configs/eval/sft.yaml', 'moss', '/evals/sft',
                         'complete', 'evaluation: {}')""",
            (sft_id,),
        )

    client = TestClient(create_app(database=database))
    models = client.get("/api/models").json()["models"]
    rl_entry = next(model for model in models if model["stage"] == "rl")
    sft_entry = next(model for model in models if model["stage"] == "sft")
    rl_detail = client.get(f"/api/models/{rl_entry['id']}").json()
    sft_detail = client.get(f"/api/models/{sft_entry['id']}").json()

    assert rl_entry["evaluation_count"] == 2
    assert {evaluation["stage"] for evaluation in rl_detail["evaluations"]} == {"rl", "sft"}
    assert sft_detail["model"]["stage"] == "sft"
    assert [evaluation["stage"] for evaluation in sft_detail["evaluations"]] == ["sft"]
