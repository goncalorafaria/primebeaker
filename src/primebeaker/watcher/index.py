"""Index the provenance of an RL checkpoint into SQLite.

The index is deliberately artifact-driven: TOML files describe training,
Beaker launch specs bind those runs to W&B IDs, data manifests describe the
mixture actually materialized, and evaluation YAML/result directories describe
downstream measurements. Re-indexing a model replaces only that model's graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tomllib
from typing import Any, Iterable, Iterator, Mapping, Sequence
from urllib.parse import quote

import yaml

from jtc.eval.outcomes import parse_moss_evaluation


SCHEMA_VERSION = 1
DEFAULT_JTC_ROOT = Path(os.getenv("JTC_ROOT", "/weka/gfaria/jtc"))
DEFAULT_PRIMEBEAKER_ROOT = Path(
    os.getenv("PRIMEBEAKER_ROOT", "/weka/gfaria/primebeaker")
)
_CHECKPOINT_STEP = re.compile(r"(?:^|/)step[_-]?(\d+)$")
_UID = re.compile(r"(?<![0-9a-f])([0-9a-f]{16})(?![0-9a-f])", re.IGNORECASE)
_WANDB_RUN_DIR = re.compile(r"^(?:offline-)?run-\d{8}_\d{6}-(.+)$")


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_metadata (
    version INTEGER PRIMARY KEY,
    installed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY,
    model_path TEXT NOT NULL UNIQUE,
    output_dir TEXT NOT NULL,
    checkpoint_step INTEGER,
    indexed_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    stage TEXT NOT NULL CHECK (stage IN ('rl', 'sft')),
    checkpoint_path TEXT,
    output_dir TEXT,
    config_path TEXT,
    config_sha256 TEXT,
    raw_config_toml TEXT,
    UNIQUE(model_id, stage)
);

CREATE TABLE IF NOT EXISTS wandb_runs (
    id INTEGER PRIMARY KEY,
    training_run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    run_id TEXT,
    project TEXT,
    entity TEXT,
    name TEXT,
    url TEXT,
    offline INTEGER NOT NULL DEFAULT 0,
    launch_spec_path TEXT,
    local_run_path TEXT,
    raw_launch_json TEXT
);

CREATE TABLE IF NOT EXISTS datasets (
    id INTEGER PRIMARY KEY,
    identity_key TEXT NOT NULL UNIQUE,
    dataset_name TEXT NOT NULL,
    manifest_path TEXT,
    config_uid TEXT,
    kind TEXT,
    train_path TEXT,
    validation_path TEXT,
    hf_dataset_path TEXT,
    raw_manifest_json TEXT
);

CREATE TABLE IF NOT EXISTS training_run_datasets (
    training_run_id INTEGER NOT NULL REFERENCES training_runs(id) ON DELETE CASCADE,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    configured_path TEXT NOT NULL,
    PRIMARY KEY(training_run_id, dataset_id, role, configured_path)
);

CREATE TABLE IF NOT EXISTS dataset_sources (
    id INTEGER PRIMARY KEY,
    dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    source_name TEXT NOT NULL,
    source_path TEXT,
    weight REAL,
    train_quota INTEGER,
    validation_quota INTEGER,
    raw_source_json TEXT NOT NULL,
    UNIQUE(dataset_id, source_name)
);

CREATE TABLE IF NOT EXISTS evaluations (
    id INTEGER PRIMARY KEY,
    model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
    config_path TEXT NOT NULL,
    workflow TEXT NOT NULL,
    run_dir TEXT,
    status TEXT NOT NULL,
    error TEXT,
    raw_config_yaml TEXT NOT NULL,
    outcome_json TEXT,
    UNIQUE(model_id, config_path)
);

CREATE TABLE IF NOT EXISTS evaluation_benchmarks (
    id INTEGER PRIMARY KEY,
    evaluation_id INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
    dataset TEXT NOT NULL,
    runs INTEGER NOT NULL,
    mean_accuracy REAL NOT NULL,
    stdev_accuracy REAL NOT NULL,
    min_accuracy REAL NOT NULL,
    max_accuracy REAL NOT NULL,
    UNIQUE(evaluation_id, dataset)
);

CREATE INDEX IF NOT EXISTS idx_training_runs_model ON training_runs(model_id);
CREATE INDEX IF NOT EXISTS idx_wandb_training ON wandb_runs(training_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_id);
CREATE INDEX IF NOT EXISTS idx_source_dataset ON dataset_sources(dataset_id);

CREATE VIEW IF NOT EXISTS model_provenance AS
SELECT
    m.model_path AS rl_model,
    m.checkpoint_step AS rl_checkpoint_step,
    rl.config_path AS rl_config,
    rl.output_dir AS rl_output_dir,
    sft.checkpoint_path AS sft_checkpoint,
    sft.config_path AS sft_config,
    sft.output_dir AS sft_output_dir,
    m.indexed_at
FROM models AS m
LEFT JOIN training_runs AS rl ON rl.model_id = m.id AND rl.stage = 'rl'
LEFT JOIN training_runs AS sft ON sft.model_id = m.id AND sft.stage = 'sft';

CREATE VIEW IF NOT EXISTS model_wandb_runs AS
SELECT m.model_path AS rl_model, tr.stage, w.run_id, w.project, w.entity,
       w.name, w.url, w.offline, w.launch_spec_path, w.local_run_path
FROM models AS m
JOIN training_runs AS tr ON tr.model_id = m.id
JOIN wandb_runs AS w ON w.training_run_id = tr.id;

CREATE VIEW IF NOT EXISTS dataset_mix AS
SELECT m.model_path AS rl_model, tr.stage, trd.role, trd.configured_path,
       d.dataset_name, d.manifest_path, d.config_uid, d.kind,
       ds.source_name, ds.source_path, ds.weight, ds.train_quota,
       ds.validation_quota
FROM models AS m
JOIN training_runs AS tr ON tr.model_id = m.id
JOIN training_run_datasets AS trd ON trd.training_run_id = tr.id
JOIN datasets AS d ON d.id = trd.dataset_id
LEFT JOIN dataset_sources AS ds ON ds.dataset_id = d.id;

CREATE VIEW IF NOT EXISTS evaluation_results AS
SELECT m.model_path AS rl_model, e.config_path, e.workflow, e.run_dir,
       e.status, e.error, b.dataset, b.runs, b.mean_accuracy,
       b.stdev_accuracy, b.min_accuracy, b.max_accuracy
FROM models AS m
JOIN evaluations AS e ON e.model_id = m.id
LEFT JOIN evaluation_benchmarks AS b ON b.evaluation_id = e.id;
"""


@dataclass(frozen=True)
class IndexRequest:
    """Inputs and discovery roots for one model provenance import."""

    model: Path
    database: Path
    repo_root: Path
    rl_config: Path | None = None
    sft_config: Path | None = None
    train_config_roots: tuple[Path, ...] = ()
    eval_config_roots: tuple[Path, ...] = ()
    launch_spec_roots: tuple[Path, ...] = ()
    manifest_roots: tuple[Path, ...] = ()
    manifests: tuple[Path, ...] = ()
    rl_wandb_run_id: str | None = None
    sft_wandb_run_id: str | None = None
    wandb_entity: str | None = None

    @classmethod
    def with_defaults(
        cls,
        *,
        model: str | Path,
        database: str | Path,
        repo_root: str | Path,
        **overrides: Any,
    ) -> "IndexRequest":
        repo = Path(repo_root).absolute()
        jtc_root = Path(os.getenv("JTC_ROOT", str(repo)))
        primebeaker_root = DEFAULT_PRIMEBEAKER_ROOT
        defaults: dict[str, Any] = {
            "train_config_roots": (jtc_root / "examples" / "configs" / "train",),
            "eval_config_roots": (
                primebeaker_root / "examples" / "configs" / "eval",
            ),
            "launch_spec_roots": (
                primebeaker_root / "beaker_experiments",
            ),
            "manifest_roots": (
                jtc_root.parent / "prime_sft" / "data",
                jtc_root / "data",
            ),
        }
        defaults.update(overrides)
        return cls(
            model=Path(model),
            database=Path(database),
            repo_root=repo,
            **defaults,
        )


@dataclass(frozen=True)
class IndexAllRequest:
    """Discovery roots for indexing every unambiguous saved RL checkpoint."""

    database: Path
    repo_root: Path
    train_config_roots: tuple[Path, ...]
    eval_config_roots: tuple[Path, ...]
    launch_spec_roots: tuple[Path, ...]
    manifest_roots: tuple[Path, ...]
    manifests: tuple[Path, ...] = ()
    wandb_entity: str | None = None
    skip_indexed: bool = False

    @classmethod
    def with_defaults(
        cls,
        *,
        database: str | Path,
        repo_root: str | Path,
        **overrides: Any,
    ) -> "IndexAllRequest":
        single = IndexRequest.with_defaults(
            model=Path("unused"),
            database=database,
            repo_root=repo_root,
        )
        values: dict[str, Any] = {
            "database": single.database,
            "repo_root": single.repo_root,
            "train_config_roots": single.train_config_roots,
            "eval_config_roots": single.eval_config_roots,
            "launch_spec_roots": single.launch_spec_roots,
            "manifest_roots": single.manifest_roots,
            "manifests": single.manifests,
            "wandb_entity": single.wandb_entity,
            "skip_indexed": False,
        }
        values.update(overrides)
        return cls(**values)


@dataclass(frozen=True)
class _TrainingArtifact:
    stage: str
    checkpoint: Path | None
    output_dir: Path | None
    config_path: Path | None
    config: dict[str, Any] | None
    raw_toml: str | None


@dataclass(frozen=True)
class _Manifest:
    path: Path
    repo_root: Path
    raw: dict[str, Any]
    text: str
    matched_paths: tuple[Path, ...]
    uids: frozenset[str]


@dataclass(frozen=True)
class _Evaluation:
    config_path: Path
    raw_yaml: str
    workflow: str
    run_dir: Path | None
    status: str
    error: str | None
    outcome: dict[str, Any] | None
    benchmarks: tuple[dict[str, Any], ...] = field(default_factory=tuple)


class _ClosingConnection(sqlite3.Connection):
    """Make ``with WatcherIndex.connect()`` commit/rollback and actually close."""

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> bool:
        try:
            return bool(super().__exit__(exc_type, exc_value, traceback))
        finally:
            self.close()


class WatcherIndex:
    """SQLite persistence and readback for watcher provenance graphs."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, factory=_ClosingConnection)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(SCHEMA)
        connection.execute(
            "INSERT OR IGNORE INTO schema_metadata(version, installed_at) VALUES (?, ?)",
            (SCHEMA_VERSION, _now()),
        )
        connection.commit()
        return connection

    def replace(
        self,
        *,
        model: Path,
        artifacts: Sequence[_TrainingArtifact],
        wandb: Mapping[str, Sequence[dict[str, Any]]],
        datasets: Mapping[str, Sequence[tuple[str, Path, _Manifest | None]]],
        evaluations: Sequence[_Evaluation],
    ) -> dict[str, Any]:
        model = model.absolute()
        output_dir = _checkpoint_output_dir(model)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("DELETE FROM models WHERE model_path = ?", (str(model),))
            model_id = db.execute(
                "INSERT INTO models(model_path, output_dir, checkpoint_step, indexed_at) VALUES (?, ?, ?, ?)",
                (str(model), str(output_dir), _checkpoint_step(model), _now()),
            ).lastrowid
            assert model_id is not None

            run_ids: dict[str, int] = {}
            for artifact in artifacts:
                config_sha = (
                    hashlib.sha256(artifact.raw_toml.encode("utf-8")).hexdigest()
                    if artifact.raw_toml is not None
                    else None
                )
                run_id = db.execute(
                    """INSERT INTO training_runs(
                           model_id, stage, checkpoint_path, output_dir, config_path,
                           config_sha256, raw_config_toml
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model_id,
                        artifact.stage,
                        _str(artifact.checkpoint),
                        _str(artifact.output_dir),
                        _str(artifact.config_path),
                        config_sha,
                        artifact.raw_toml,
                    ),
                ).lastrowid
                assert run_id is not None
                run_ids[artifact.stage] = run_id
                for run in wandb.get(artifact.stage, ()):
                    db.execute(
                        """INSERT INTO wandb_runs(
                               training_run_id, run_id, project, entity, name, url,
                               offline, launch_spec_path, local_run_path,
                               raw_launch_json
                           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            run_id,
                            run.get("run_id"),
                            run.get("project"),
                            run.get("entity"),
                            run.get("name"),
                            run.get("url"),
                            int(bool(run.get("offline"))),
                            run.get("launch_spec_path"),
                            run.get("local_run_path"),
                            run.get("raw_launch_json"),
                        ),
                    )

            for stage, bindings in datasets.items():
                run_id = run_ids[stage]
                for role, configured_path, manifest in bindings:
                    dataset_id = self._upsert_dataset(db, configured_path, manifest)
                    db.execute(
                        """INSERT OR IGNORE INTO training_run_datasets(
                               training_run_id, dataset_id, role, configured_path
                           ) VALUES (?, ?, ?, ?)""",
                        (run_id, dataset_id, role, str(configured_path)),
                    )

            for evaluation in evaluations:
                evaluation_id = db.execute(
                    """INSERT INTO evaluations(
                           model_id, config_path, workflow, run_dir, status, error,
                           raw_config_yaml, outcome_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model_id,
                        str(evaluation.config_path),
                        evaluation.workflow,
                        _str(evaluation.run_dir),
                        evaluation.status,
                        evaluation.error,
                        evaluation.raw_yaml,
                        _json(evaluation.outcome) if evaluation.outcome is not None else None,
                    ),
                ).lastrowid
                assert evaluation_id is not None
                for benchmark in evaluation.benchmarks:
                    db.execute(
                        """INSERT INTO evaluation_benchmarks(
                               evaluation_id, dataset, runs, mean_accuracy,
                               stdev_accuracy, min_accuracy, max_accuracy
                           ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (
                            evaluation_id,
                            benchmark["dataset"],
                            benchmark["runs"],
                            benchmark["mean_accuracy"],
                            benchmark["stdev_accuracy"],
                            benchmark["min_accuracy"],
                            benchmark["max_accuracy"],
                        ),
                    )

            db.execute(
                "DELETE FROM datasets WHERE NOT EXISTS "
                "(SELECT 1 FROM training_run_datasets WHERE dataset_id = datasets.id)"
            )
            summary = self._summary(db, model_id)
            db.commit()
            return summary

    @staticmethod
    def _upsert_dataset(db: sqlite3.Connection, configured_path: Path, manifest: _Manifest | None) -> int:
        if manifest is None:
            identity = f"path:{configured_path}"
            name = configured_path.name
            values: tuple[Any, ...] = (identity, name, None, None, None, None, None, None, None)
        else:
            raw = manifest.raw
            identity = f"manifest:{manifest.path}"
            name = str(raw.get("name") or raw.get("mix_uid") or raw.get("config_uid") or configured_path.name)
            values = (
                identity,
                name,
                str(manifest.path),
                raw.get("config_uid"),
                raw.get("kind") or raw.get("format"),
                raw.get("train_path"),
                raw.get("validation_path"),
                raw.get("hf_dataset_path"),
                manifest.text,
            )
        db.execute(
            """INSERT INTO datasets(
                   identity_key, dataset_name, manifest_path, config_uid, kind,
                   train_path, validation_path, hf_dataset_path, raw_manifest_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(identity_key) DO UPDATE SET
                   dataset_name=excluded.dataset_name,
                   manifest_path=excluded.manifest_path,
                   config_uid=excluded.config_uid,
                   kind=excluded.kind,
                   train_path=excluded.train_path,
                   validation_path=excluded.validation_path,
                   hf_dataset_path=excluded.hf_dataset_path,
                   raw_manifest_json=excluded.raw_manifest_json""",
            values,
        )
        dataset_id = db.execute("SELECT id FROM datasets WHERE identity_key = ?", (identity,)).fetchone()[0]
        if manifest is not None:
            db.execute("DELETE FROM dataset_sources WHERE dataset_id = ?", (dataset_id,))
            for source in manifest.raw.get("sources", []):
                if not isinstance(source, dict):
                    continue
                source_name = source.get("name")
                if not isinstance(source_name, str) or not source_name:
                    continue
                db.execute(
                    """INSERT INTO dataset_sources(
                           dataset_id, source_name, source_path, weight,
                           train_quota, validation_quota, raw_source_json
                       ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        dataset_id,
                        source_name,
                        _manifest_source_path(source, manifest),
                        _number(source.get("weight")),
                        _integer(source.get("train_quota")),
                        _integer(source.get("validation_quota")),
                        _json(source),
                    ),
                )
        return int(dataset_id)

    @staticmethod
    def _summary(db: sqlite3.Connection, model_id: int) -> dict[str, Any]:
        model = dict(db.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone())
        counts = {
            "training_runs": db.execute("SELECT count(*) FROM training_runs WHERE model_id = ?", (model_id,)).fetchone()[0],
            "wandb_runs": db.execute(
                "SELECT count(*) FROM wandb_runs WHERE training_run_id IN (SELECT id FROM training_runs WHERE model_id = ?)",
                (model_id,),
            ).fetchone()[0],
            "datasets": db.execute(
                "SELECT count(DISTINCT dataset_id) FROM training_run_datasets WHERE training_run_id IN (SELECT id FROM training_runs WHERE model_id = ?)",
                (model_id,),
            ).fetchone()[0],
            "dataset_sources": db.execute(
                """SELECT count(*) FROM dataset_sources WHERE dataset_id IN (
                       SELECT dataset_id FROM training_run_datasets WHERE training_run_id IN (
                           SELECT id FROM training_runs WHERE model_id = ?))""",
                (model_id,),
            ).fetchone()[0],
            "evaluations": db.execute("SELECT count(*) FROM evaluations WHERE model_id = ?", (model_id,)).fetchone()[0],
            "benchmarks": db.execute(
                "SELECT count(*) FROM evaluation_benchmarks WHERE evaluation_id IN (SELECT id FROM evaluations WHERE model_id = ?)",
                (model_id,),
            ).fetchone()[0],
        }
        return {"database": str(db.execute("PRAGMA database_list").fetchone()[2]), "model": model, "counts": counts}

    def show(self, model: str | Path) -> dict[str, Any]:
        with self.connect() as db:
            path = str(Path(model).absolute())
            model_row = db.execute("SELECT * FROM models WHERE model_path = ?", (path,)).fetchone()
            if model_row is None:
                raise KeyError(f"model is not indexed: {path}")
            model_id = model_row["id"]
            provenance = dict(db.execute("SELECT * FROM model_provenance WHERE rl_model = ?", (path,)).fetchone())
            wandb = [dict(row) for row in db.execute("SELECT * FROM model_wandb_runs WHERE rl_model = ? ORDER BY stage, run_id", (path,))]
            mixes = [dict(row) for row in db.execute("SELECT * FROM dataset_mix WHERE rl_model = ? ORDER BY stage, role, source_name", (path,))]
            evaluations = [dict(row) for row in db.execute("SELECT * FROM evaluation_results WHERE rl_model = ? ORDER BY config_path, dataset", (path,))]
            return {
                "provenance": provenance,
                "wandb_runs": wandb,
                "dataset_mix": mixes,
                "evaluations": evaluations,
                "counts": self._summary(db, model_id)["counts"],
            }


def index_model(request: IndexRequest) -> dict[str, Any]:
    """Discover and persist one end-RL-model provenance graph."""

    repo = request.repo_root.absolute()
    model = _path(request.model, repo)
    rl_config_path = _find_config(
        checkpoint=model,
        explicit=request.rl_config,
        roots=request.train_config_roots,
        repo_root=repo,
        stage="rl",
    )
    rl = _training_artifact("rl", model, rl_config_path, repo)
    sft_checkpoint = _sft_checkpoint(rl.config or {}, repo)
    sft_config_path = _find_config(
        checkpoint=sft_checkpoint,
        explicit=request.sft_config,
        roots=request.train_config_roots,
        repo_root=repo,
        stage="sft",
        required=False,
    ) if sft_checkpoint is not None else None
    sft = _training_artifact("sft", sft_checkpoint, sft_config_path, repo)
    artifacts = (rl, sft)

    launch_specs = _launch_specs(request.launch_spec_roots)
    wandb = {
        "rl": _wandb_runs(
            rl,
            launch_specs,
            repo,
            explicit_run_id=request.rl_wandb_run_id,
            fallback_entity=request.wandb_entity,
        ),
        "sft": _wandb_runs(
            sft,
            launch_specs,
            repo,
            explicit_run_id=request.sft_wandb_run_id,
            fallback_entity=request.wandb_entity,
        ),
    }

    manifest_catalog = _manifest_catalog(request.manifests, request.manifest_roots, repo)
    datasets: dict[str, list[tuple[str, Path, _Manifest | None]]] = {}
    for artifact in artifacts:
        bindings: list[tuple[str, Path, _Manifest | None]] = []
        seen: set[tuple[str, str]] = set()
        for _, configured in _dataset_paths(artifact.stage, artifact.config or {}, repo):
            manifest = _match_manifest(configured, manifest_catalog)
            identity = (
                ("manifest", str(manifest.path))
                if manifest is not None
                else ("path", str(configured))
            )
            if identity in seen:
                continue
            seen.add(identity)
            # Train/validation are materialized views of the same mix whenever
            # they resolve to one manifest. The stage is the useful label.
            bindings.append((artifact.stage, configured, manifest))
        datasets[artifact.stage] = bindings

    evaluations = _evaluations(model, request.eval_config_roots, repo)
    return WatcherIndex(request.database).replace(
        model=model,
        artifacts=artifacts,
        wandb=wandb,
        datasets=datasets,
        evaluations=evaluations,
    )


def _index_sft_model(
    *,
    model: Path,
    config_path: Path | None,
    request: IndexAllRequest,
) -> dict[str, Any]:
    """Index one evaluated SFT checkpoint as a first-class standalone model."""

    repo = request.repo_root.absolute()
    sft = _training_artifact("sft", model, config_path, repo)
    launch_specs = _launch_specs(request.launch_spec_roots)
    wandb = {
        "sft": _wandb_runs(
            sft,
            launch_specs,
            repo,
            explicit_run_id=None,
            fallback_entity=request.wandb_entity,
        )
    }
    manifest_catalog = _manifest_catalog(request.manifests, request.manifest_roots, repo)
    bindings: list[tuple[str, Path, _Manifest | None]] = []
    seen: set[tuple[str, str]] = set()
    for _, configured in _dataset_paths("sft", sft.config or {}, repo):
        manifest = _match_manifest(configured, manifest_catalog)
        identity = ("manifest", str(manifest.path)) if manifest is not None else ("path", str(configured))
        if identity in seen:
            continue
        seen.add(identity)
        bindings.append(("sft", configured, manifest))
    return WatcherIndex(request.database).replace(
        model=model,
        artifacts=(sft,),
        wandb=wandb,
        datasets={"sft": bindings},
        evaluations=_evaluations(model, request.eval_config_roots, repo),
    )


def index_all(request: IndexAllRequest) -> dict[str, Any]:
    """Index every saved checkpoint belonging to an unambiguous RL TOML.

    Multiple RL TOMLs targeting one output directory are reported instead of
    guessed: resumed runs can legitimately have changed configuration, and a
    filename or modification-time heuristic would corrupt provenance.
    """

    repo = request.repo_root.absolute()
    configs_by_output: dict[Path, list[Path]] = {}
    sft_configs_by_output: dict[Path, list[Path]] = {}
    invalid_configs: list[dict[str, str]] = []
    for path in _files(request.train_config_roots, ("*.toml",)):
        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            invalid_configs.append({"config": str(path), "error": str(exc)})
            continue
        if not isinstance(config.get("output_dir"), str):
            continue
        output = config.get("output_dir")
        assert isinstance(output, str)
        output_path = _path(output, repo)
        if _is_rl_config(config):
            configs_by_output.setdefault(output_path, []).append(path)
        elif _is_sft_config(config):
            sft_configs_by_output.setdefault(output_path, []).append(path)

    indexed: list[str] = []
    indexed_sft: list[str] = []
    skipped_indexed: list[str] = []
    unresolved: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    discovered = 0
    existing: set[str] = set()
    if request.skip_indexed and request.database.is_file():
        with WatcherIndex(request.database).connect() as db:
            existing = {row[0] for row in db.execute("SELECT model_path FROM models")}
    for output_dir, config_paths in sorted(configs_by_output.items(), key=lambda item: str(item[0])):
        checkpoints = sorted(
            (
                path.absolute()
                for path in (output_dir / "weights").glob("step_*")
                if path.is_dir() and _checkpoint_step(path) is not None
            ),
            key=lambda path: (_checkpoint_step(path) or -1, str(path)),
        )
        discovered += len(checkpoints)
        unique_configs = _collapse_equivalent_configs(sorted(set(config_paths)))
        if len(unique_configs) != 1:
            if checkpoints:
                unresolved.append(
                    {
                        "output_dir": str(output_dir),
                        "checkpoints": [str(path) for path in checkpoints],
                        "reason": "multiple RL TOMLs target this output directory",
                        "configs": [str(path) for path in unique_configs],
                    }
                )
            continue
        for checkpoint in checkpoints:
            if str(checkpoint) in existing:
                skipped_indexed.append(str(checkpoint))
                continue
            try:
                index_model(
                    IndexRequest(
                        model=checkpoint,
                        database=request.database,
                        repo_root=repo,
                        rl_config=unique_configs[0],
                        train_config_roots=request.train_config_roots,
                        eval_config_roots=request.eval_config_roots,
                        launch_spec_roots=request.launch_spec_roots,
                        manifest_roots=request.manifest_roots,
                        manifests=request.manifests,
                        wandb_entity=request.wandb_entity,
                    )
                )
            except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
                failures.append({"checkpoint": str(checkpoint), "error": str(exc)})
            else:
                indexed.append(str(checkpoint))

    evaluated_models = _evaluation_catalog(
        tuple(Path(root).absolute() for root in request.eval_config_roots),
        repo,
    )
    for model_value in sorted(evaluated_models):
        model = Path(model_value).absolute()
        if not model.is_dir() or _checkpoint_step(model) is None:
            continue
        output_dir = _checkpoint_output_dir(model)
        if output_dir in configs_by_output:
            continue
        config_paths = _collapse_equivalent_configs(sft_configs_by_output.get(output_dir, []))
        looks_like_sft = "sft" in output_dir.name.lower()
        if not config_paths and not looks_like_sft:
            continue
        if len(config_paths) > 1:
            unresolved.append(
                {
                    "output_dir": str(output_dir),
                    "checkpoints": [str(model)],
                    "reason": "multiple SFT TOMLs target this output directory",
                    "configs": [str(path) for path in config_paths],
                }
            )
            continue
        if str(model) in existing:
            skipped_indexed.append(str(model))
            continue
        try:
            _index_sft_model(
                model=model,
                config_path=config_paths[0] if config_paths else None,
                request=request,
            )
        except (FileNotFoundError, ValueError, sqlite3.Error) as exc:
            failures.append({"checkpoint": str(model), "error": str(exc)})
        else:
            indexed_sft.append(str(model))
    return {
        "database": str(request.database.absolute()),
        "rl_outputs_with_configs": len(configs_by_output),
        "discovered_checkpoints": discovered,
        "indexed_checkpoints": len(indexed),
        "indexed": indexed,
        "indexed_sft_checkpoints": len(indexed_sft),
        "indexed_sft": indexed_sft,
        "skipped_indexed": skipped_indexed,
        "unresolved": unresolved,
        "failures": failures,
        "invalid_configs": invalid_configs,
    }


def _is_rl_config(config: Mapping[str, Any]) -> bool:
    return (
        isinstance(config.get("output_dir"), str)
        and isinstance(config.get("trainer"), Mapping)
        and isinstance(config.get("orchestrator"), Mapping)
    )


def _is_sft_config(config: Mapping[str, Any]) -> bool:
    data = config.get("data")
    return (
        isinstance(config.get("output_dir"), str)
        and isinstance(data, Mapping)
        and (data.get("type") == "sft" or isinstance(data.get("name"), str))
        and not _is_rl_config(config)
    )


def _collapse_equivalent_configs(paths: Sequence[Path]) -> list[Path]:
    """Treat copied TOMLs with identical parsed values as one artifact."""

    groups: dict[str, list[Path]] = {}
    for path in paths:
        try:
            parsed = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            groups.setdefault(f"invalid:{path}", []).append(path)
            continue
        signature = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
        groups.setdefault(signature, []).append(path)
    if len(groups) != 1:
        return sorted(paths)
    copies = next(iter(groups.values()))
    return [
        min(
            copies,
            key=lambda path: (
                0 if "examples/configs/train" in str(path) else 1,
                len(path.parts),
                str(path),
            ),
        )
    ]


def _find_config(
    *,
    checkpoint: Path | None,
    explicit: Path | None,
    roots: Sequence[Path],
    repo_root: Path,
    stage: str,
    required: bool = True,
) -> Path | None:
    if explicit is not None:
        path = _path(explicit, repo_root)
        if not path.is_file():
            raise FileNotFoundError(path)
        if checkpoint is not None:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
            output = config.get("output_dir")
            expected_output = _checkpoint_output_dir(checkpoint)
            if not isinstance(output, str) or _path(output, repo_root) != expected_output:
                raise ValueError(
                    f"{stage} TOML {path} has output_dir {output!r}, expected {expected_output}"
                )
        return path
    if checkpoint is None:
        return None
    expected_output = _checkpoint_output_dir(checkpoint)
    matches: list[Path] = []
    for path in _files(roots, ("*.toml",)):
        try:
            config = tomllib.loads(path.read_text(encoding="utf-8"))
        except (OSError, tomllib.TOMLDecodeError):
            continue
        output = config.get("output_dir")
        if isinstance(output, str) and _path(output, repo_root) == expected_output:
            matches.append(path.absolute())
    matches = _collapse_equivalent_configs(sorted(set(matches)))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        if required:
            raise FileNotFoundError(
                f"could not find the {stage} TOML whose output_dir is {expected_output}; "
                f"pass --{stage}-config"
            )
        return None
    rendered = "\n  ".join(str(path) for path in matches)
    raise ValueError(
        f"multiple {stage} TOMLs have output_dir {expected_output}; pass --{stage}-config:\n  {rendered}"
    )


def _training_artifact(stage: str, checkpoint: Path | None, config_path: Path | None, repo: Path) -> _TrainingArtifact:
    if config_path is None:
        return _TrainingArtifact(
            stage=stage,
            checkpoint=checkpoint,
            output_dir=_checkpoint_output_dir(checkpoint) if checkpoint is not None else None,
            config_path=None,
            config=None,
            raw_toml=None,
        )
    raw = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(raw)
    output = config.get("output_dir")
    return _TrainingArtifact(
        stage=stage,
        checkpoint=checkpoint,
        output_dir=_path(output, repo) if isinstance(output, str) else None,
        config_path=config_path,
        config=config,
        raw_toml=raw,
    )


def _sft_checkpoint(config: Mapping[str, Any], repo: Path) -> Path | None:
    candidates: set[Path] = set()
    for keys in (("trainer", "model", "name"), ("orchestrator", "model", "name"), ("inference", "model", "name")):
        value: Any = config
        for key in keys:
            value = value.get(key) if isinstance(value, Mapping) else None
        if isinstance(value, str) and (value.startswith("/") or "/" in value) and _checkpoint_step(Path(value)) is not None:
            candidates.add(_path(value, repo))
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(f"RL TOML refers to multiple predecessor checkpoints: {sorted(map(str, candidates))}")
    return next(iter(candidates))


def _dataset_paths(stage: str, config: Mapping[str, Any], repo: Path) -> list[tuple[str, Path]]:
    found: list[tuple[str, Path]] = []
    if stage == "sft":
        data = config.get("data")
        if isinstance(data, Mapping) and isinstance(data.get("name"), str):
            found.append(("train", _path(data["name"], repo)))
        val = config.get("val")
        val_data = val.get("data") if isinstance(val, Mapping) else None
        if isinstance(val_data, Mapping) and isinstance(val_data.get("name"), str):
            found.append(("validation", _path(val_data["name"], repo)))
    else:
        orchestrator = config.get("orchestrator")
        if isinstance(orchestrator, Mapping):
            for role in ("train", "eval"):
                subtree = orchestrator.get(role)
                for value in _values_for_key(subtree, "dataset"):
                    if isinstance(value, str):
                        found.append((role, _path(value, repo)))
    return list(dict.fromkeys(found))


def _values_for_key(value: Any, key: str) -> Iterator[Any]:
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if child_key == key:
                yield child
            yield from _values_for_key(child, key)
    elif isinstance(value, list):
        for child in value:
            yield from _values_for_key(child, key)


def _manifest_catalog(explicit: Sequence[Path], roots: Sequence[Path], repo: Path) -> list[_Manifest]:
    candidates = {_path(path, repo) for path in explicit}
    if not candidates:
        for path in _files(roots, ("*.json",)):
            name = path.name.lower()
            if "manifest" in name:
                candidates.add(path)
    manifests: list[_Manifest] = []
    for path in sorted(candidates):
        try:
            text = path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
            continue
        matched_paths = tuple(
            _path(value, repo)
            for key in ("train_path", "validation_path", "hf_dataset_path")
            if isinstance((value := raw.get(key)), str)
        )
        uid_text = " ".join(
            [str(path), *(str(raw.get(key, "")) for key in ("config_uid", "mix_uid")), *(str(p) for p in matched_paths)]
        )
        manifests.append(
            _Manifest(
                path=path,
                repo_root=repo,
                raw=raw,
                text=text,
                matched_paths=matched_paths,
                uids=frozenset(_UID.findall(uid_text)),
            )
        )
    return manifests


def _match_manifest(dataset: Path, manifests: Sequence[_Manifest]) -> _Manifest | None:
    dataset = dataset.absolute()
    dataset_uids = set(_UID.findall(str(dataset)))
    scored: list[tuple[int, _Manifest]] = []
    for manifest in manifests:
        score = 0
        if dataset in manifest.matched_paths:
            score = 100
        elif any(_same_dataset_family(dataset, path) for path in manifest.matched_paths):
            score = 20
        if dataset_uids & manifest.uids:
            score += 40
        if score:
            scored.append((score, manifest))
    if not scored:
        return None
    best = max(score for score, _ in scored)
    matches = [manifest for score, manifest in scored if score == best]
    if len(matches) > 1:
        signatures = {_manifest_mix_signature(manifest) for manifest in matches}
        if len(signatures) == 1:
            return max(
                matches,
                key=lambda manifest: (_shared_prefix_depth(dataset, manifest.path), str(manifest.path)),
            )
        rendered = ", ".join(str(manifest.path) for manifest in matches)
        raise ValueError(f"multiple data manifests match {dataset}: {rendered}; pass only the intended --manifest")
    return matches[0]


def _manifest_mix_signature(manifest: _Manifest) -> str:
    sources = []
    for source in manifest.raw.get("sources", []):
        if not isinstance(source, Mapping):
            continue
        sources.append(
            {
                key: source.get(key)
                for key in ("name", "parquet", "path", "weight", "train_quota", "validation_quota")
            }
        )
    return json.dumps(sources, sort_keys=True, separators=(",", ":"))


def _shared_prefix_depth(left: Path, right: Path) -> int:
    depth = 0
    for left_part, right_part in zip(left.absolute().parts, right.absolute().parts):
        if left_part != right_part:
            break
        depth += 1
    return depth


def _same_dataset_family(left: Path, right: Path) -> bool:
    def stem(path: Path) -> str:
        value = path.name
        for suffix in (".jsonl", ".parquet", "_train", "_validation", "_messages", "_prompt_split", "_harmony"):
            value = value.removesuffix(suffix)
        return value
    a, b = stem(left), stem(right)
    return bool(a and b and (a == b or a.startswith(b) or b.startswith(a)))


def _manifest_source_path(source: Mapping[str, Any], manifest: _Manifest) -> str | None:
    value = source.get("parquet") or source.get("path")
    if not isinstance(value, str) or not value:
        return None
    return str(_path(value, manifest.repo_root))


def _launch_specs(roots: Sequence[Path]) -> list[tuple[Path, dict[str, Any], str]]:
    specs: list[tuple[Path, dict[str, Any], str]] = []
    for path in _files(roots, ("*.json",)):
        try:
            text = path.read_text(encoding="utf-8")
            raw = json.loads(text)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(raw, dict) and isinstance(raw.get("tasks"), list):
            specs.append((path, raw, text))
    return specs


def _wandb_runs(
    artifact: _TrainingArtifact,
    specs: Sequence[tuple[Path, dict[str, Any], str]],
    repo: Path,
    *,
    explicit_run_id: str | None,
    fallback_entity: str | None,
) -> list[dict[str, Any]]:
    local_runs = _local_wandb_runs(artifact.output_dir)
    if artifact.config is None and explicit_run_id is None and not local_runs:
        return []
    wandb = artifact.config.get("wandb", {}) if artifact.config else {}
    if not isinstance(wandb, Mapping):
        wandb = {}
    project = _text(wandb.get("project"))
    entity = _text(wandb.get("entity")) or fallback_entity
    name = _text(wandb.get("name"))
    offline = bool(wandb.get("offline", False))
    matches: list[dict[str, Any]] = []
    for spec_path, spec, raw_text in specs:
        for task in spec.get("tasks", []):
            if not isinstance(task, dict):
                continue
            env = {
                item.get("name"): item.get("value")
                for item in task.get("envVars", [])
                if isinstance(item, dict) and isinstance(item.get("name"), str) and "value" in item
            }
            env_output = _env_path(env.get("OUTPUT_DIR"), repo)
            env_config = _env_path(env.get("CONFIG_PATH"), repo)
            # A TOML is sometimes reused for a renamed retry.  OUTPUT_DIR is
            # therefore authoritative whenever the launch spec contains it;
            # CONFIG_PATH is only the fallback for older specs without one.
            output_matches = artifact.output_dir is not None and env_output == artifact.output_dir
            config_matches = artifact.config_path is not None and env_config == artifact.config_path
            if not (output_matches if env_output is not None else config_matches):
                continue
            run_id = _text(env.get("WANDB_SHARED_RUN_ID")) or _text(env.get("WANDB_RUN_ID"))
            matches.append(
                _wandb_record(
                    run_id=run_id,
                    project=_text(env.get("WANDB_PROJECT")) or project,
                    entity=_text(env.get("WANDB_ENTITY")) or entity,
                    name=_text(env.get("WANDB_NAME")) or name,
                    offline=offline,
                    launch_spec_path=spec_path,
                    local_run_path=None,
                    raw_launch_json=raw_text,
                )
            )
    if explicit_run_id:
        matches.append(
            _wandb_record(
                run_id=explicit_run_id,
                project=project,
                entity=entity,
                name=name,
                offline=offline,
                launch_spec_path=None,
                local_run_path=None,
                raw_launch_json=None,
            )
        )
    if local_runs:
        concrete = [match for match in matches if match["run_id"] is not None]
        placeholders = [match for match in matches if match["run_id"] is None]
        for run_id, local_path, local_offline in local_runs:
            same_id = [match for match in concrete if match["run_id"] == run_id]
            if same_id:
                for match in same_id:
                    match["local_run_path"] = str(local_path)
                continue
            launch = placeholders[0] if len(placeholders) == 1 else None
            concrete.append(
                _wandb_record(
                    run_id=run_id,
                    project=project,
                    entity=entity,
                    name=name,
                    offline=offline or local_offline,
                    launch_spec_path=Path(launch["launch_spec_path"]) if launch and launch["launch_spec_path"] else None,
                    local_run_path=local_path,
                    raw_launch_json=launch["raw_launch_json"] if launch else None,
                )
            )
        matches = concrete
    if not matches and (project or entity or name):
        matches.append(
            _wandb_record(
                run_id=None,
                project=project,
                entity=entity,
                name=name,
                offline=offline,
                launch_spec_path=None,
                local_run_path=None,
                raw_launch_json=None,
            )
        )
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for match in matches:
        key = (match["run_id"], match["launch_spec_path"], match["local_run_path"])
        unique[key] = match
    return list(unique.values())


def _wandb_record(
    *,
    run_id: str | None,
    project: str | None,
    entity: str | None,
    name: str | None,
    offline: bool,
    launch_spec_path: Path | None,
    local_run_path: Path | None,
    raw_launch_json: str | None,
) -> dict[str, Any]:
    url = None
    if run_id and project and entity and not offline:
        url = f"https://wandb.ai/{quote(entity, safe='')}/{quote(project, safe='')}/runs/{quote(run_id, safe='')}"
    return {
        "run_id": run_id,
        "project": project,
        "entity": entity,
        "name": name,
        "url": url,
        "offline": offline,
        "launch_spec_path": _str(launch_spec_path),
        "local_run_path": _str(local_run_path),
        "raw_launch_json": raw_launch_json,
    }


def _local_wandb_runs(output_dir: Path | None) -> list[tuple[str, Path, bool]]:
    """Recover W&B IDs from run directories written by the training process."""

    if output_dir is None:
        return []
    found: dict[str, tuple[Path, bool]] = {}
    for root in (output_dir / "wandb", output_dir / "run_default" / "wandb"):
        if not root.is_dir():
            continue
        for path in sorted(root.iterdir()):
            if not path.is_dir():
                continue
            match = _WANDB_RUN_DIR.match(path.name)
            if match:
                found.setdefault(match.group(1), (path.absolute(), path.name.startswith("offline-run-")))
    return [(run_id, path, offline) for run_id, (path, offline) in sorted(found.items())]


def _evaluations(model: Path, roots: Sequence[Path], repo: Path) -> list[_Evaluation]:
    catalog = _evaluation_catalog(tuple(Path(root).absolute() for root in roots), repo.absolute())
    return list(catalog.get(str(model.absolute()), ()))


@lru_cache(maxsize=16)
def _evaluation_catalog(
    roots: tuple[Path, ...],
    repo: Path,
) -> dict[str, tuple[_Evaluation, ...]]:
    """Read current YAMLs and legacy MOSS archives once per indexing process."""

    catalog: dict[str, list[_Evaluation]] = {}
    known_run_dirs: set[tuple[str, str]] = set()
    for path in _files(roots, ("*.yaml", "*.yml")):
        try:
            raw_yaml = path.read_text(encoding="utf-8")
            raw = yaml.safe_load(raw_yaml)
        except (OSError, yaml.YAMLError):
            continue
        if not isinstance(raw, dict):
            continue
        workflow = str(raw.get("workflow") or "moss")
        payload = raw.get("search_agent") if workflow == "search-agent" else raw.get("evaluation", raw)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), str):
            continue
        model = _path(payload["model"], repo)
        run_value = payload.get("run_dir")
        run_dir = _path(run_value, repo) if isinstance(run_value, str) else None
        status, error, outcome, benchmarks = _evaluation_outcome(run_dir)
        catalog.setdefault(str(model), []).append(
            _Evaluation(
                config_path=path,
                raw_yaml=raw_yaml,
                workflow=workflow,
                run_dir=run_dir,
                status=status,
                error=error,
                outcome=outcome,
                benchmarks=benchmarks,
            )
        )
        if run_dir is not None:
            known_run_dirs.add((str(model), str(run_dir)))

    legacy_roots = (
        repo / "records" / "moss-evals",
        repo / "records" / "evalmoss",
        repo / "records" / "eval-moss",
    )
    for legacy_root in legacy_roots:
        if not legacy_root.is_dir():
            continue
        for run_dir in sorted(path for path in legacy_root.iterdir() if path.is_dir()):
            if not ((run_dir / "averages.jsonl").is_file() or (run_dir / "summary.jsonl").is_file()):
                continue
            try:
                parsed = parse_moss_evaluation(run_dir)
            except (FileNotFoundError, ValueError):
                continue
            if not parsed.model:
                continue
            model = _path(parsed.model, repo)
            if (str(model), str(run_dir)) in known_run_dirs:
                continue
            status, error, outcome, benchmarks = _evaluation_outcome(run_dir)
            launcher = run_dir / "launcher.log"
            catalog.setdefault(str(model), []).append(
                _Evaluation(
                    config_path=launcher,
                    raw_yaml=launcher.read_text(encoding="utf-8"),
                    workflow="moss-archive",
                    run_dir=run_dir,
                    status=status,
                    error=error,
                    outcome=outcome,
                    benchmarks=benchmarks,
                )
            )
    return {model: tuple(evaluations) for model, evaluations in catalog.items()}


def _evaluation_outcome(run_dir: Path | None) -> tuple[str, str | None, dict[str, Any] | None, tuple[dict[str, Any], ...]]:
    if run_dir is None:
        return "unconfigured", "evaluation config has no run_dir", None, ()
    if not run_dir.is_dir():
        return "not_started", None, None, ()
    if (run_dir / "averages.jsonl").is_file() or (run_dir / "summary.jsonl").is_file():
        try:
            parsed = parse_moss_evaluation(run_dir)
        except (FileNotFoundError, ValueError) as exc:
            return "incomplete", str(exc), None, ()
        outcome = parsed.model_dump(mode="json")
        benchmarks = tuple(result.model_dump(mode="json") for result in parsed.benchmarks.values())
        return "complete", None, outcome, benchmarks
    failure = run_dir / "failure.json"
    if failure.is_file():
        try:
            outcome = json.loads(failure.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            outcome = {"raw": failure.read_text(encoding="utf-8")}
        return "failed", None, outcome, ()
    artifacts: dict[str, Any] = {}
    for path in sorted(run_dir.glob("*.json")):
        try:
            artifacts[path.name] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
    if artifacts:
        return "complete", None, {"artifacts": artifacts}, ()
    return "incomplete", "run directory exists but has no recognized outcome files", None, ()


def _files(roots: Sequence[Path], patterns: Sequence[str]) -> Iterator[Path]:
    seen: set[Path] = set()
    for root in roots:
        root = Path(root).absolute()
        if root.is_file():
            candidates: Iterable[Path] = (root,)
        elif root.is_dir():
            candidates = (path for pattern in patterns for path in root.rglob(pattern))
        else:
            continue
        for path in candidates:
            path = path.absolute()
            if path not in seen:
                seen.add(path)
                yield path


def _checkpoint_output_dir(checkpoint: Path) -> Path:
    checkpoint = checkpoint.absolute()
    if checkpoint.parent.name == "weights" and _checkpoint_step(checkpoint) is not None:
        return checkpoint.parent.parent
    return checkpoint.parent


def _checkpoint_step(path: Path) -> int | None:
    match = _CHECKPOINT_STEP.search(str(path))
    return int(match.group(1)) if match else None


def _path(value: str | Path, repo: Path) -> Path:
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (repo / path).absolute()


def _env_path(value: Any, repo: Path) -> Path | None:
    return _path(value, repo) if isinstance(value, str) and value else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _str(value: Path | None) -> str | None:
    return str(value) if value is not None else None


def _text(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))
