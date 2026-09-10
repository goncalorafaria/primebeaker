"""Python Fire command suite for the JTCEval watcher."""

from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Sequence

import fire

from primebeaker.watcher.app import (
    DEFAULT_DATABASE,
    create_app,
)
from primebeaker.watcher.index import IndexAllRequest, IndexRequest, WatcherIndex, index_all, index_model
from primebeaker.watcher.index import DEFAULT_JTC_ROOT


def _paths(value: str | Path | Sequence[str | Path] | None) -> tuple[Path, ...] | None:
    """Normalize Fire's scalar, comma-separated, and list flag values."""

    if value is None:
        return None
    values: Sequence[str | Path]
    if isinstance(value, (str, Path)):
        values = tuple(part.strip() for part in str(value).split(",") if part.strip())
    else:
        values = value
    return tuple(Path(item) for item in values)


def _request_overrides(
    *,
    train_config_roots: str | Path | Sequence[str | Path] | None,
    eval_config_roots: str | Path | Sequence[str | Path] | None,
    launch_spec_roots: str | Path | Sequence[str | Path] | None,
    manifest_roots: str | Path | Sequence[str | Path] | None,
) -> dict[str, tuple[Path, ...]]:
    overrides: dict[str, tuple[Path, ...]] = {}
    for name, value in (
        ("train_config_roots", train_config_roots),
        ("eval_config_roots", eval_config_roots),
        ("launch_spec_roots", launch_spec_roots),
        ("manifest_roots", manifest_roots),
    ):
        parsed = _paths(value)
        if parsed is not None:
            overrides[name] = parsed
    return overrides


def _bulk_summary(result: dict[str, Any], *, operation: str, details: bool = False) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "operation": operation,
        "database": result["database"],
        "rl_outputs_with_configs": result["rl_outputs_with_configs"],
        "discovered_checkpoints": result["discovered_checkpoints"],
        "indexed_checkpoints": result["indexed_checkpoints"],
        "indexed_sft_checkpoints": result.get("indexed_sft_checkpoints", 0),
        "skipped_indexed": len(result["skipped_indexed"]),
        "unresolved_outputs": len(result["unresolved"]),
        "failures": len(result["failures"]),
        "invalid_configs": len(result["invalid_configs"]),
    }
    if result["unresolved"]:
        summary["unresolved"] = result["unresolved"]
    if result["failures"]:
        summary["failure_details"] = result["failures"]
    if result["invalid_configs"]:
        summary["invalid_config_details"] = result["invalid_configs"]
    if details:
        summary["indexed"] = result["indexed"]
        summary["indexed_sft"] = result.get("indexed_sft", [])
        summary["skipped_indexed_paths"] = result["skipped_indexed"]
    return summary


def _read_status(database: Path) -> dict[str, Any]:
    database = database.absolute()
    if not database.is_file():
        return {"database": str(database), "exists": False}

    uri = database.as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
        counts = {
            "rl_checkpoints": db.execute(
                """SELECT count(DISTINCT m.id) FROM models m
                   JOIN training_runs tr ON tr.model_id = m.id WHERE tr.stage = 'rl'"""
            ).fetchone()[0],
            "sft_checkpoints": db.execute(
                """SELECT count(DISTINCT m.id) FROM models m
                   WHERE EXISTS (
                       SELECT 1 FROM training_runs tr
                       WHERE tr.model_id = m.id AND tr.stage = 'sft'
                   ) AND NOT EXISTS (
                       SELECT 1 FROM training_runs tr
                       WHERE tr.model_id = m.id AND tr.stage = 'rl'
                   )"""
            ).fetchone()[0],
            "rl_outputs": db.execute(
                """SELECT count(DISTINCT m.output_dir) FROM models m
                   JOIN training_runs tr ON tr.model_id = m.id WHERE tr.stage = 'rl'"""
            ).fetchone()[0],
            "sft_outputs": db.execute(
                """SELECT count(DISTINCT m.output_dir) FROM models m
                   WHERE EXISTS (
                       SELECT 1 FROM training_runs tr
                       WHERE tr.model_id = m.id AND tr.stage = 'sft'
                   ) AND NOT EXISTS (
                       SELECT 1 FROM training_runs tr
                       WHERE tr.model_id = m.id AND tr.stage = 'rl'
                   )"""
            ).fetchone()[0],
            "training_stages": db.execute("SELECT count(*) FROM training_runs").fetchone()[0],
            "wandb_attempts": db.execute("SELECT count(*) FROM wandb_runs").fetchone()[0],
            "datasets": db.execute("SELECT count(*) FROM datasets").fetchone()[0],
            "source_parquets": db.execute("SELECT count(*) FROM dataset_sources").fetchone()[0],
            "evaluations": db.execute("SELECT count(*) FROM evaluations").fetchone()[0],
            "benchmark_outcomes": db.execute("SELECT count(*) FROM evaluation_benchmarks").fetchone()[0],
        }
        indexed = db.execute("SELECT min(indexed_at), max(indexed_at) FROM models").fetchone()
        schema = db.execute("SELECT max(version) FROM schema_metadata").fetchone()[0]
    return {
        "database": str(database),
        "exists": True,
        "size_bytes": database.stat().st_size,
        "integrity": integrity,
        "schema_version": schema,
        "first_indexed_at": indexed[0],
        "last_indexed_at": indexed[1],
        **counts,
    }


def _finalize_database(database: Path) -> None:
    with sqlite3.connect(database) as db:
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        db.commit()
        mode = db.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise RuntimeError(f"could not finalize temporary database journal: {mode}")
        db.execute("PRAGMA optimize")
        foreign_keys = db.execute("PRAGMA foreign_key_check").fetchall()
        integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
    if foreign_keys:
        raise RuntimeError(f"rebuilt database has {len(foreign_keys)} foreign-key violations")
    if integrity != "ok":
        raise RuntimeError(f"rebuilt database failed integrity_check: {integrity}")


def _quiesce_database(database: Path) -> None:
    """Collapse any old WAL before backing up and replacing a live database."""

    if not database.is_file():
        return
    with sqlite3.connect(database, timeout=10) as db:
        db.execute("PRAGMA busy_timeout=10000")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
        db.commit()
        mode = db.execute("PRAGMA journal_mode=DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise RuntimeError(f"could not quiesce existing database journal: {mode}")


def _backup_database(database: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    backup = database.with_name(f"{database.name}.backup-{stamp}")
    source_uri = database.absolute().as_uri() + "?mode=ro"
    with sqlite3.connect(source_uri, uri=True) as source, sqlite3.connect(backup) as destination:
        source.backup(destination)
    return backup


def _cleanup_database_files(database: Path) -> None:
    for path in (database, Path(f"{database}-wal"), Path(f"{database}-shm"), Path(f"{database}-journal")):
        path.unlink(missing_ok=True)


class WatcherCLI:
    """Build, update, inspect, and browse the RL-checkpoint provenance database."""

    def rebuild(
        self,
        database: str | Path = DEFAULT_DATABASE,
        repo_root: str | Path = DEFAULT_JTC_ROOT,
        train_config_roots: str | Path | Sequence[str | Path] | None = None,
        eval_config_roots: str | Path | Sequence[str | Path] | None = None,
        launch_spec_roots: str | Path | Sequence[str | Path] | None = None,
        manifest_roots: str | Path | Sequence[str | Path] | None = None,
        manifests: str | Path | Sequence[str | Path] | None = None,
        wandb_entity: str | None = None,
        backup: bool = True,
        require_clean: bool = True,
        require_models: bool = True,
        details: bool = False,
    ) -> dict[str, Any]:
        """Recreate SQLite off to the side, validate it, then atomically install it.

        The previous database is snapshot-backed up by default. Indexing failures
        and invalid TOMLs prevent the swap when ``require_clean`` is true.
        Ambiguous resume configurations are reported but never guessed.
        """

        target = Path(database).absolute()
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.rebuild-",
            suffix=".sqlite3",
            dir=target.parent,
            delete=False,
        )
        temporary = Path(handle.name)
        handle.close()
        # SQLite should create the database itself, not inherit an empty file.
        temporary.unlink()

        try:
            request = IndexAllRequest.with_defaults(
                database=temporary,
                repo_root=repo_root,
                manifests=_paths(manifests) or (),
                wandb_entity=wandb_entity,
                skip_indexed=False,
                **_request_overrides(
                    train_config_roots=train_config_roots,
                    eval_config_roots=eval_config_roots,
                    launch_spec_roots=launch_spec_roots,
                    manifest_roots=manifest_roots,
                ),
            )
            result = index_all(request)
            errors = len(result["failures"]) + len(result["invalid_configs"])
            if require_clean and errors:
                raise RuntimeError(
                    "rebuild refused to replace the current database: "
                    f"{len(result['failures'])} indexing failures and "
                    f"{len(result['invalid_configs'])} invalid configs"
                )
            if require_models and result["indexed_checkpoints"] == 0:
                raise RuntimeError("rebuild refused to install an empty checkpoint database")
            _finalize_database(temporary)

            backup_path: Path | None = None
            if target.is_file():
                _quiesce_database(target)
                if backup:
                    backup_path = _backup_database(target)
            os.replace(temporary, target)

            summary = _bulk_summary(result, operation="rebuild", details=details)
            summary["database"] = str(target)
            summary["backup"] = str(backup_path) if backup_path is not None else None
            summary["integrity"] = _read_status(target)["integrity"]
            return summary
        finally:
            _cleanup_database_files(temporary)

    def update(
        self,
        database: str | Path = DEFAULT_DATABASE,
        repo_root: str | Path = DEFAULT_JTC_ROOT,
        train_config_roots: str | Path | Sequence[str | Path] | None = None,
        eval_config_roots: str | Path | Sequence[str | Path] | None = None,
        launch_spec_roots: str | Path | Sequence[str | Path] | None = None,
        manifest_roots: str | Path | Sequence[str | Path] | None = None,
        manifests: str | Path | Sequence[str | Path] | None = None,
        wandb_entity: str | None = None,
        skip_indexed: bool = True,
        details: bool = False,
    ) -> dict[str, Any]:
        """Discover saved checkpoints and incrementally update SQLite."""

        result = index_all(
            IndexAllRequest.with_defaults(
                database=database,
                repo_root=repo_root,
                manifests=_paths(manifests) or (),
                wandb_entity=wandb_entity,
                skip_indexed=skip_indexed,
                **_request_overrides(
                    train_config_roots=train_config_roots,
                    eval_config_roots=eval_config_roots,
                    launch_spec_roots=launch_spec_roots,
                    manifest_roots=manifest_roots,
                ),
            )
        )
        return _bulk_summary(result, operation="update", details=details)

    def index(
        self,
        model: str | Path,
        database: str | Path = DEFAULT_DATABASE,
        repo_root: str | Path = DEFAULT_JTC_ROOT,
        rl_config: str | Path | None = None,
        sft_config: str | Path | None = None,
        train_config_roots: str | Path | Sequence[str | Path] | None = None,
        eval_config_roots: str | Path | Sequence[str | Path] | None = None,
        launch_spec_roots: str | Path | Sequence[str | Path] | None = None,
        manifest_roots: str | Path | Sequence[str | Path] | None = None,
        manifests: str | Path | Sequence[str | Path] | None = None,
        wandb_entity: str | None = None,
        rl_wandb_run_id: str | None = None,
        sft_wandb_run_id: str | None = None,
    ) -> dict[str, Any]:
        """Index or refresh one end RL checkpoint and its full lineage."""

        return index_model(
            IndexRequest.with_defaults(
                model=model,
                database=database,
                repo_root=repo_root,
                rl_config=Path(rl_config) if rl_config else None,
                sft_config=Path(sft_config) if sft_config else None,
                manifests=_paths(manifests) or (),
                wandb_entity=wandb_entity,
                rl_wandb_run_id=rl_wandb_run_id,
                sft_wandb_run_id=sft_wandb_run_id,
                **_request_overrides(
                    train_config_roots=train_config_roots,
                    eval_config_roots=eval_config_roots,
                    launch_spec_roots=launch_spec_roots,
                    manifest_roots=manifest_roots,
                ),
            )
        )

    def show(
        self,
        model: str | Path,
        database: str | Path = DEFAULT_DATABASE,
        repo_root: str | Path = DEFAULT_JTC_ROOT,
    ) -> dict[str, Any]:
        """Print the indexed provenance graph for one checkpoint."""

        path = Path(model)
        if not path.is_absolute():
            path = Path(repo_root) / path
        return WatcherIndex(database).show(path)

    def status(self, database: str | Path = DEFAULT_DATABASE) -> dict[str, Any]:
        """Check database integrity and print compact inventory counts."""

        return _read_status(Path(database))

    def serve(
        self,
        database: str | Path = DEFAULT_DATABASE,
        host: str = "127.0.0.1",
        port: int = 8790,
        wandb_secret: str = "gfaria_WANDB_API_KEY",
        reload: bool = False,
    ) -> None:
        """Serve the local read-only checkpoint browser."""

        import uvicorn

        uvicorn.run(
            create_app(
                database=database,
                wandb_secret=wandb_secret,
            ),
            host=host,
            port=port,
            reload=reload,
        )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the watcher Fire CLI."""

    fire.Fire(WatcherCLI, command=list(argv) if argv is not None else None)


if __name__ == "__main__":
    main()
