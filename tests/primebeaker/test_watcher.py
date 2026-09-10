import json
from pathlib import Path
import sqlite3

from primebeaker.watcher import IndexAllRequest, IndexRequest, WatcherIndex, index_all, index_model


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_watcher_indexes_complete_rl_lineage_and_is_idempotent(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outputs = tmp_path / "outputs"
    sft_output = outputs / "sft-run"
    rl_output = outputs / "rl-run"
    sft_checkpoint = sft_output / "weights" / "step_200"
    model = rl_output / "weights" / "step_400"
    model.mkdir(parents=True)
    sft_checkpoint.mkdir(parents=True)
    (rl_output / "wandb/run-20260819_010203-rl-run-id").mkdir(parents=True)
    (rl_output / "wandb/run-20260818_010203-failed-rl-run-id").mkdir(parents=True)
    # Prime-RL may write the same distributed W&B run below run_default too;
    # that is one logical run, not another attempt.
    (rl_output / "run_default/wandb/run-20260819_010204-rl-run-id").mkdir(parents=True)
    (sft_output / "wandb/run-20260818_010203-sft-run-id").mkdir(parents=True)
    sft_dataset = tmp_path / "data" / "sft_datadev_c0e8854bb5143cac"
    rl_train = tmp_path / "data" / "rldata_datadev_161bc7548bdec992_train.jsonl"
    rl_validation = tmp_path / "data" / "rldata_datadev_161bc7548bdec992_validation.jsonl"

    sft_config = _write(
        repo / "examples/configs/train/sft/run.toml",
        f'''output_dir = "{sft_output}"
[data]
name = "{sft_dataset}"
[val.data]
name = "{sft_dataset}"
[wandb]
project = "rubric-sft"
entity = "graf"
name = "sft-run"
''',
    )
    rl_config = _write(
        repo / "examples/configs/train/rl/run.toml",
        f'''output_dir = "{rl_output}"
[trainer.model]
name = "{sft_checkpoint}"
[orchestrator.model]
name = "{sft_checkpoint}"
[[orchestrator.train.source]]
[orchestrator.train.source.legacy.args]
dataset = "{rl_train}"
[[orchestrator.eval.source]]
[orchestrator.eval.source.legacy.args]
dataset = "{rl_validation}"
[wandb]
project = "rubric-rl"
name = "rl-run"
''',
    )

    manifests = repo / "manifests"
    _write(
        manifests / "sft.manifest.json",
        json.dumps(
            {
                "builder": "datadev",
                "kind": "sft",
                "config_uid": "c0e8854bb5143cac",
                "hf_dataset_path": str(sft_dataset),
                "train_path": str(tmp_path / "data/sft_train.jsonl"),
                "validation_path": str(tmp_path / "data/sft_validation.jsonl"),
                "sources": [
                    {"name": "science", "parquet": "records/science.parquet", "weight": 0.75, "train_quota": 750},
                    {"name": "code", "parquet": "records/code.parquet", "weight": 0.25, "train_quota": 250},
                ],
            }
        ),
    )
    _write(
        manifests / "rl.manifest.json",
        json.dumps(
            {
                "builder": "datadev",
                "kind": "rl",
                "config_uid": "161bc7548bdec992",
                "train_path": str(rl_train),
                "validation_path": str(rl_validation),
                "sources": [
                    {
                        "name": "rubric-hub",
                        "parquet": "records/rubric-hub.parquet",
                        "weight": 1.0,
                        "train_quota": 5000,
                        "validation_quota": 128,
                    }
                ],
            }
        ),
    )

    launches = repo / "launches"
    _write(
        launches / "rl.json",
        json.dumps(
            {
                "tasks": [
                    {
                        "envVars": [
                            {"name": "CONFIG_PATH", "value": str(rl_config)},
                            {"name": "OUTPUT_DIR", "value": str(rl_output)},
                            {"name": "WANDB_SHARED_RUN_ID", "value": "rl-run-id"},
                            {"name": "WANDB_ENTITY", "value": "graf"},
                        ]
                    }
                ]
            }
        ),
    )
    _write(
        launches / "sft.json",
        json.dumps(
            {
                "tasks": [
                    {
                        "envVars": [
                            {"name": "CONFIG_PATH", "value": str(sft_config)},
                            {"name": "WANDB_RUN_ID", "value": "sft-run-id"},
                        ]
                    }
                ]
            }
        ),
    )
    _write(
        launches / "sft-reused-config-other-output.json",
        json.dumps(
            {
                "tasks": [
                    {
                        "envVars": [
                            {"name": "CONFIG_PATH", "value": str(sft_config)},
                            {"name": "OUTPUT_DIR", "value": str(outputs / "different-retry")},
                            {"name": "WANDB_RUN_ID", "value": "wrong-sft-run-id"},
                        ]
                    }
                ]
            }
        ),
    )

    eval_root = repo / "examples/configs/eval"
    complete_run = tmp_path / "evals/complete"
    _write(
        complete_run / "averages.jsonl",
        json.dumps(
            {
                "dataset": "frontier-science",
                "runs": 128,
                "mean_accuracy": 0.625,
                "stdev_accuracy": 0.1,
                "min_accuracy": 0.0,
                "max_accuracy": 1.0,
            }
        )
        + "\n",
    )
    _write(
        eval_root / "complete.yaml",
        f"evaluation:\n  model: {model}\n  run_dir: {complete_run}\n  datasets: [frontier-science]\n",
    )
    _write(
        eval_root / "pending.yaml",
        f"evaluation:\n  model: {model}\n  run_dir: {tmp_path / 'evals/pending'}\n  datasets: [rubricbench]\n",
    )
    _write(
        eval_root / "different-model.yaml",
        f"evaluation:\n  model: {outputs / 'other/weights/step_400'}\n  run_dir: {tmp_path / 'evals/other'}\n",
    )

    database = tmp_path / "watcher.sqlite3"
    request = IndexRequest(
        model=model,
        database=database,
        repo_root=repo,
        train_config_roots=(repo / "examples/configs/train",),
        eval_config_roots=(eval_root,),
        launch_spec_roots=(launches,),
        manifest_roots=(manifests,),
    )
    first = index_model(request)
    second = index_model(request)

    assert first["counts"] == second["counts"] == {
        "training_runs": 2,
        "wandb_runs": 3,
        "datasets": 2,
        "dataset_sources": 3,
        "evaluations": 2,
        "benchmarks": 1,
    }
    shown = WatcherIndex(database).show(model)
    assert shown["provenance"]["rl_config"] == str(rl_config)
    assert shown["provenance"]["sft_checkpoint"] == str(sft_checkpoint)
    assert shown["provenance"]["sft_config"] == str(sft_config)
    assert {run["run_id"] for run in shown["wandb_runs"]} == {
        "failed-rl-run-id",
        "rl-run-id",
        "sft-run-id",
    }
    assert [run["run_id"] for run in shown["wandb_runs"]].count("rl-run-id") == 1
    assert all(run["local_run_path"] for run in shown["wandb_runs"])
    assert {row["source_name"] for row in shown["dataset_mix"]} == {"science", "code", "rubric-hub"}
    assert len(shown["dataset_mix"]) == 3
    assert {row["role"] for row in shown["dataset_mix"]} == {"rl", "sft"}
    assert all(Path(row["source_path"]).is_absolute() for row in shown["dataset_mix"])
    assert {row["status"] for row in shown["evaluations"]} == {"complete", "not_started"}

    with sqlite3.connect(database) as db:
        raw_toml = db.execute("SELECT raw_config_toml FROM training_runs WHERE stage = 'rl'").fetchone()[0]
        raw_eval = db.execute("SELECT outcome_json FROM evaluations WHERE status = 'complete'").fetchone()[0]
    assert str(sft_checkpoint) in raw_toml
    assert json.loads(raw_eval)["benchmarks"]["frontier-science"]["mean_accuracy"] == 0.625

    bulk = index_all(
        IndexAllRequest(
            database=database,
            repo_root=repo,
            train_config_roots=(repo / "examples/configs/train",),
            eval_config_roots=(eval_root,),
            launch_spec_roots=(launches,),
            manifest_roots=(manifests,),
        )
    )
    assert bulk["discovered_checkpoints"] == 1
    assert bulk["indexed_checkpoints"] == 1
    assert bulk["unresolved"] == []


def test_watcher_keeps_old_wandb_metadata_when_run_id_is_unknown(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "rl"
    model = output / "weights/step_12"
    config = _write(
        repo / "run.toml",
        f'''output_dir = "{output}"
[wandb]
project = "rubric-rl"
name = "old-run"
''',
    )
    request = IndexRequest(
        model=model,
        database=tmp_path / "watcher.sqlite3",
        repo_root=repo,
        rl_config=config,
        train_config_roots=(),
        eval_config_roots=(),
        launch_spec_roots=(),
        manifest_roots=(),
        wandb_entity="graf",
    )

    index_model(request)
    shown = WatcherIndex(request.database).show(model)

    assert shown["wandb_runs"] == [
        {
            "rl_model": str(model),
            "stage": "rl",
            "run_id": None,
            "project": "rubric-rl",
            "entity": "graf",
            "name": "old-run",
            "url": None,
            "offline": 0,
            "launch_spec_path": None,
            "local_run_path": None,
        }
    ]


def test_watcher_imports_historical_moss_evals_from_launcher_model(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "outputs" / "rl"
    model = output / "weights" / "step_400"
    model.mkdir(parents=True)
    config = _write(
        repo / "run.toml",
        f'''output_dir = "{output}"
[trainer]
[orchestrator]
''',
    )
    historical = repo / "records" / "moss-evals" / "legacy-step400"
    _write(historical / "launcher.log", f"Launching LiteRegistry stack for {model}\n")
    _write(
        historical / "averages.jsonl",
        json.dumps(
            {
                "dataset": "frontier_science",
                "runs": 3,
                "mean_accuracy": 0.75,
                "stdev_accuracy": 0.01,
                "min_accuracy": 0.74,
                "max_accuracy": 0.76,
            }
        )
        + "\n",
    )
    database = tmp_path / "watcher.sqlite3"

    index_model(
        IndexRequest(
            model=model,
            database=database,
            repo_root=repo,
            rl_config=config,
            train_config_roots=(),
            eval_config_roots=(),
            launch_spec_roots=(),
            manifest_roots=(),
        )
    )
    shown = WatcherIndex(database).show(model)

    assert len(shown["evaluations"]) == 1
    assert shown["evaluations"][0]["workflow"] == "moss-archive"
    assert shown["evaluations"][0]["run_dir"] == str(historical)
    assert shown["evaluations"][0]["dataset"] == "frontier_science"
    assert shown["evaluations"][0]["mean_accuracy"] == 0.75


def test_bulk_index_reports_ambiguous_rl_configs_instead_of_guessing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "outputs/rl"
    (output / "weights/step_10").mkdir(parents=True)
    config_text = f'''output_dir = "{output}"
[trainer]
[orchestrator]
'''
    _write(repo / "examples/configs/train/rl/first.toml", config_text)
    _write(repo / "examples/configs/train/rl/resume.toml", config_text + "max_steps = 20\n")

    result = index_all(
        IndexAllRequest(
            database=tmp_path / "watcher.sqlite3",
            repo_root=repo,
            train_config_roots=(repo / "examples/configs/train",),
            eval_config_roots=(),
            launch_spec_roots=(),
            manifest_roots=(),
        )
    )

    assert result["discovered_checkpoints"] == 1
    assert result["indexed_checkpoints"] == 0
    assert result["unresolved"][0]["reason"] == "multiple RL TOMLs target this output directory"


def test_bulk_index_adds_evaluated_sft_checkpoint_as_standalone_model(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    outputs = tmp_path / "outputs"
    sft_output = outputs / "sft-run"
    sft_model = sft_output / "weights" / "step_200"
    rl_output = outputs / "rl-run"
    rl_model = rl_output / "weights" / "step_400"
    sft_model.mkdir(parents=True)
    rl_model.mkdir(parents=True)
    train_root = repo / "examples" / "configs" / "train"
    _write(
        train_root / "sft.toml",
        f'''output_dir = "{sft_output}"
[data]
type = "sft"
name = "sft-dataset"
''',
    )
    _write(
        train_root / "rl.toml",
        f'''output_dir = "{rl_output}"
[trainer.model]
name = "{sft_model}"
[orchestrator.model]
name = "{sft_model}"
''',
    )
    eval_root = repo / "examples" / "configs" / "eval"
    _write(
        eval_root / "sft.yaml",
        f"evaluation:\n  model: {sft_model}\n  run_dir: {repo / 'evals/sft'}\n",
    )
    database = tmp_path / "watcher.sqlite3"

    result = index_all(
        IndexAllRequest(
            database=database,
            repo_root=repo,
            train_config_roots=(train_root,),
            eval_config_roots=(eval_root,),
            launch_spec_roots=(),
            manifest_roots=(),
        )
    )

    assert result["indexed_checkpoints"] == 1
    assert result["indexed_sft_checkpoints"] == 1
    assert result["indexed_sft"] == [str(sft_model.absolute())]
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT count(*) FROM models").fetchone()[0] == 2
        standalone_stage = db.execute(
            """SELECT tr.stage FROM training_runs tr
               JOIN models m ON m.id = tr.model_id WHERE m.model_path = ?""",
            (str(sft_model.absolute()),),
        ).fetchone()[0]
    assert standalone_stage == "sft"
    assert WatcherIndex(database).show(sft_model)["evaluations"][0]["status"] == "not_started"
