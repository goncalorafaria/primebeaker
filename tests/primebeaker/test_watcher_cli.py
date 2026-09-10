from pathlib import Path
import sqlite3

import pytest

from primebeaker.watcher.cli import WatcherCLI
from primebeaker.watcher.index import WatcherIndex


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _seed_old_database(database: Path) -> None:
    with WatcherIndex(database).connect() as db:
        db.execute(
            "INSERT INTO models(model_path, output_dir, checkpoint_step, indexed_at) VALUES (?, ?, ?, ?)",
            ("/old/model/weights/step_1", "/old/model", 1, "old"),
        )


def test_fire_cli_rebuild_is_atomic_and_backs_up_previous_database(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "outputs" / "rl-run"
    checkpoint = output / "weights" / "step_10"
    checkpoint.mkdir(parents=True)
    train_root = repo / "examples" / "configs" / "train"
    _write(
        train_root / "rl" / "run.toml",
        f'''output_dir = "{output}"
[trainer]
[orchestrator]
[wandb]
project = "watcher-test"
name = "rl-run"
''',
    )
    database = tmp_path / "watcher.sqlite3"
    _seed_old_database(database)

    result = WatcherCLI().rebuild(
        database=database,
        repo_root=repo,
        train_config_roots=train_root,
        eval_config_roots=(),
        launch_spec_roots=(),
        manifest_roots=(),
    )

    assert result["operation"] == "rebuild"
    assert result["indexed_checkpoints"] == 1
    assert result["integrity"] == "ok"
    assert result["backup"] is not None
    backup = Path(result["backup"])
    assert backup.is_file()
    with sqlite3.connect(database) as db:
        assert db.execute("SELECT model_path FROM models").fetchall() == [(str(checkpoint.absolute()),)]
    with sqlite3.connect(backup) as db:
        assert db.execute("SELECT model_path FROM models").fetchall() == [
            ("/old/model/weights/step_1",)
        ]

    status = WatcherCLI().status(database)
    assert status["exists"] is True
    assert status["rl_checkpoints"] == 1
    assert status["rl_outputs"] == 1
    assert status["integrity"] == "ok"


def test_fire_cli_rebuild_keeps_database_when_scan_is_empty(tmp_path: Path) -> None:
    database = tmp_path / "watcher.sqlite3"
    _seed_old_database(database)

    with pytest.raises(RuntimeError, match="empty checkpoint database"):
        WatcherCLI().rebuild(
            database=database,
            repo_root=tmp_path,
            train_config_roots=tmp_path / "missing",
            eval_config_roots=(),
            launch_spec_roots=(),
            manifest_roots=(),
        )

    with sqlite3.connect(database) as db:
        assert db.execute("SELECT model_path FROM models").fetchall() == [
            ("/old/model/weights/step_1",)
        ]
    assert list(tmp_path.glob(".watcher.sqlite3.rebuild-*")) == []


def test_fire_cli_update_defaults_to_skipping_indexed_models(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "outputs" / "rl-run"
    checkpoint = output / "weights" / "step_20"
    checkpoint.mkdir(parents=True)
    train_root = repo / "examples" / "configs" / "train"
    _write(
        train_root / "rl.toml",
        f'''output_dir = "{output}"
[trainer]
[orchestrator]
''',
    )
    database = tmp_path / "watcher.sqlite3"
    cli = WatcherCLI()

    first = cli.update(
        database=database,
        repo_root=repo,
        train_config_roots=train_root,
        eval_config_roots=(),
        launch_spec_roots=(),
        manifest_roots=(),
    )
    second = cli.update(
        database=database,
        repo_root=repo,
        train_config_roots=train_root,
        eval_config_roots=(),
        launch_spec_roots=(),
        manifest_roots=(),
    )

    assert first["indexed_checkpoints"] == 1
    assert first["skipped_indexed"] == 0
    assert second["indexed_checkpoints"] == 0
    assert second["skipped_indexed"] == 1
