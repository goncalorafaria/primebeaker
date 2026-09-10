"""Small read-only web browser for the watcher provenance database."""

from __future__ import annotations

import os
from pathlib import Path
import sqlite3
import tomllib
from typing import Any, Callable, Sequence
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response

from primebeaker.watcher.metrics import (
    WandbMetricsUnavailable,
    default_wandb_api_factory,
    fetch_wandb_charts,
    metric_attempts,
    resolve_wandb_api_key,
)


DEFAULT_DATABASE = Path("watcher.sqlite3")


def _watcher_assets() -> tuple[str, str]:
    """Split the embedded page into HTML and same-origin JavaScript assets."""

    page_prefix, script_start, remainder = PAGE.rpartition("<script>")
    script, script_end, page_suffix = remainder.partition("</script>")
    if not script_start or not script_end:
        raise RuntimeError("Watcher page must contain one inline script")
    page = f'{page_prefix}<script src="/watcher.js"></script>{page_suffix}'
    return page, script


def _connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise HTTPException(status_code=503, detail=f"Watcher database does not exist: {path}")
    uri = f"file:{quote(str(path.absolute()))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _models(db: sqlite3.Connection, query: str, limit: int) -> list[dict[str, Any]]:
    pattern = f"%{query}%"
    rows = db.execute(
        """WITH runs AS (
               SELECT m.output_dir,
                      count(DISTINCT m.id) AS checkpoint_count,
                      count(e.id) + (
                          SELECT count(DISTINCT linked_eval.id)
                          FROM models lineage_model
                          JOIN training_runs lineage
                            ON lineage.model_id = lineage_model.id AND lineage.stage = 'sft'
                          JOIN models sft_model ON sft_model.model_path = lineage.checkpoint_path
                          JOIN evaluations linked_eval ON linked_eval.model_id = sft_model.id
                          WHERE lineage_model.output_dir = m.output_dir
                            AND EXISTS (
                                SELECT 1 FROM training_runs rl_stage
                                WHERE rl_stage.model_id = lineage_model.id AND rl_stage.stage = 'rl'
                            )
                      ) AS evaluation_count,
                      CASE WHEN EXISTS (
                          SELECT 1 FROM models stage_model
                          JOIN training_runs stage_run ON stage_run.model_id = stage_model.id
                          WHERE stage_model.output_dir = m.output_dir AND stage_run.stage = 'rl'
                      ) THEN 'rl' ELSE 'sft' END AS stage,
                      max(m.indexed_at) AS last_indexed_at
               FROM models m
               LEFT JOIN evaluations e ON e.model_id = m.id
               WHERE m.model_path LIKE ? OR m.output_dir LIKE ?
               GROUP BY m.output_dir
           )
           SELECT representative.id, representative.model_path,
                  representative.output_dir, representative.checkpoint_step,
                  runs.last_indexed_at AS indexed_at,
                  runs.checkpoint_count, runs.evaluation_count, runs.stage,
                  (SELECT count(*) FROM wandb_runs w
                     JOIN training_runs tr ON tr.id = w.training_run_id
                     JOIN models wm ON wm.id = tr.model_id
                    WHERE wm.output_dir = runs.output_dir) AS wandb_count,
                  (SELECT count(DISTINCT trd.dataset_id) FROM training_run_datasets trd
                     JOIN training_runs tr ON tr.id = trd.training_run_id
                     JOIN models dm ON dm.id = tr.model_id
                    WHERE dm.output_dir = runs.output_dir) AS dataset_count
           FROM runs
           JOIN models representative ON representative.id = (
               SELECT candidate.id FROM models candidate
               WHERE candidate.output_dir = runs.output_dir
               ORDER BY EXISTS(
                            SELECT 1 FROM evaluations ce WHERE ce.model_id = candidate.id
                        ) DESC,
                        candidate.checkpoint_step DESC, candidate.id DESC
               LIMIT 1
           )
           ORDER BY (runs.evaluation_count > 0) DESC,
                    runs.evaluation_count DESC,
                    runs.last_indexed_at DESC,
                    runs.output_dir
           LIMIT ?""",
        (pattern, pattern, limit),
    )
    return [
        {
            **dict(row),
            "name": Path(row["output_dir"]).name,
            "has_evaluations": row["evaluation_count"] > 0,
        }
        for row in rows
    ]


def _training_runs(db: sqlite3.Connection, model_id: int) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for row in db.execute(
        "SELECT * FROM training_runs WHERE model_id = ? ORDER BY CASE stage WHEN 'rl' THEN 0 ELSE 1 END",
        (model_id,),
    ):
        item = dict(row)
        item["hyperparameters"] = _training_hyperparameters(item.get("raw_config_toml"))
        item["wandb_runs"] = [
            dict(wandb)
            for wandb in db.execute(
                """SELECT run_id, project, entity, name, url, offline,
                          launch_spec_path, local_run_path
                   FROM wandb_runs WHERE training_run_id = ? ORDER BY run_id""",
                (row["id"],),
            )
        ]
        runs.append(item)
    return runs


def _training_hyperparameters(raw_toml: str | None) -> dict[str, Any]:
    """Extract a compact, stage-agnostic summary from a stored training TOML."""

    if not raw_toml:
        return {}
    try:
        config = tomllib.loads(raw_toml)
    except (tomllib.TOMLDecodeError, TypeError):
        return {}

    def value(*paths: tuple[str, ...]) -> Any:
        for path in paths:
            current: Any = config
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    break
                current = current[key]
            else:
                return current
        return None

    fields = {
        "learning_rate": value(("trainer", "optim", "lr"), ("optim", "lr")),
        "min_learning_rate": value(("scheduler", "min_lr"),),
        "weight_decay": value(("trainer", "optim", "weight_decay"), ("optim", "weight_decay")),
        "beta1": value(("trainer", "optim", "betas1"), ("optim", "betas1")),
        "beta2": value(("trainer", "optim", "betas2"), ("optim", "betas2")),
        "max_steps": value(("max_steps",),),
        "sequence_length": value(
            ("orchestrator", "seq_len"),
            ("data", "seq_len"),
            ("model", "seq_len"),
            ("trainer", "model", "seq_len"),
        ),
        "batch_size": value(("orchestrator", "batch_size"), ("data", "batch_size")),
        "micro_batch_size": value(("data", "micro_batch_size"),),
        "group_size": value(("orchestrator", "group_size"),),
        "kl_tau": value(("trainer", "loss", "kl_tau"),),
        "warmup_steps": value(("scheduler", "warmup_steps"),),
        "scheduler": value(("scheduler", "type"),),
        "oversampling_factor": value(("orchestrator", "oversampling_factor"),),
    }
    return {key: item for key, item in fields.items() if item is not None}


def _datasets(db: sqlite3.Connection, model_id: int) -> list[dict[str, Any]]:
    rows = db.execute(
        """SELECT tr.stage, trd.role, trd.configured_path,
                  d.id AS dataset_id, d.dataset_name, d.manifest_path, d.config_uid,
                  d.kind, d.train_path, d.validation_path, d.hf_dataset_path,
                  ds.source_name, ds.source_path, ds.weight,
                  ds.train_quota, ds.validation_quota
           FROM training_runs tr
           JOIN training_run_datasets trd ON trd.training_run_id = tr.id
           JOIN datasets d ON d.id = trd.dataset_id
           LEFT JOIN dataset_sources ds ON ds.dataset_id = d.id
           WHERE tr.model_id = ?
           ORDER BY CASE tr.stage WHEN 'rl' THEN 0 ELSE 1 END,
                    d.dataset_name, trd.role, ds.source_name""",
        (model_id,),
    )
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = (row["stage"], row["dataset_id"])
        item = grouped.setdefault(
            key,
            {
                "stage": row["stage"],
                "role": row["stage"],
                "configured_path": row["configured_path"],
                "configured_paths": [],
                "dataset_id": row["dataset_id"],
                "dataset_name": row["dataset_name"],
                "manifest_path": row["manifest_path"],
                "config_uid": row["config_uid"],
                "kind": row["kind"],
                "train_path": row["train_path"],
                "validation_path": row["validation_path"],
                "hf_dataset_path": row["hf_dataset_path"],
                "sources": [],
            },
        )
        if row["configured_path"] not in item["configured_paths"]:
            item["configured_paths"].append(row["configured_path"])
        if row["source_name"] and not any(
            source["source_name"] == row["source_name"] for source in item["sources"]
        ):
            item["sources"].append(
                {
                    "source_name": row["source_name"],
                    "source_path": row["source_path"],
                    "weight": row["weight"],
                    "train_quota": row["train_quota"],
                    "validation_quota": row["validation_quota"],
                }
            )
    return list(grouped.values())


def _evaluations(db: sqlite3.Connection, model_id: int) -> list[dict[str, Any]]:
    evaluations: list[dict[str, Any]] = []
    for row in db.execute(
        """SELECT id, config_path, workflow, run_dir, status, error
           FROM evaluations WHERE model_id = ? ORDER BY config_path""",
        (model_id,),
    ):
        item = dict(row)
        item["benchmarks"] = [
            dict(benchmark)
            for benchmark in db.execute(
                """SELECT dataset, runs, mean_accuracy, stdev_accuracy,
                          min_accuracy, max_accuracy
                   FROM evaluation_benchmarks WHERE evaluation_id = ? ORDER BY dataset""",
                (row["id"],),
            )
        ]
        evaluations.append(item)
    return evaluations


def _run_evaluations(
    db: sqlite3.Connection,
    output_dir: str,
    model_stage: str,
) -> list[dict[str, Any]]:
    """Collect run evaluations and, for RL, its SFT predecessor evaluations."""

    evaluations: list[dict[str, Any]] = []
    seen: set[int] = set()

    def append_checkpoint(checkpoint: sqlite3.Row, stage: str) -> None:
        for evaluation in _evaluations(db, checkpoint["id"]):
            if evaluation["id"] in seen:
                continue
            seen.add(evaluation["id"])
            evaluations.append(
                {
                    **evaluation,
                    "model_id": checkpoint["id"],
                    "model_path": checkpoint["model_path"],
                    "checkpoint_step": checkpoint["checkpoint_step"],
                    "stage": stage,
                }
            )

    for checkpoint in db.execute(
        """SELECT id, model_path, checkpoint_step
           FROM models WHERE output_dir = ?
           ORDER BY checkpoint_step DESC, id DESC""",
        (output_dir,),
    ):
        append_checkpoint(checkpoint, model_stage)
    if model_stage == "rl":
        for checkpoint in db.execute(
            """SELECT DISTINCT sft_model.id, sft_model.model_path, sft_model.checkpoint_step
               FROM models rl_model
               JOIN training_runs lineage
                 ON lineage.model_id = rl_model.id AND lineage.stage = 'sft'
               JOIN models sft_model ON sft_model.model_path = lineage.checkpoint_path
               WHERE rl_model.output_dir = ?
               ORDER BY sft_model.checkpoint_step DESC, sft_model.id DESC""",
            (output_dir,),
        ):
            append_checkpoint(checkpoint, "sft")
    evaluations.sort(
        key=lambda item: (
            0 if item["stage"] == "rl" else 1,
            -(item["checkpoint_step"] or -1),
            item["config_path"],
        )
    )
    return evaluations


def create_app(
    *,
    database: str | Path = DEFAULT_DATABASE,
    wandb_secret: str = "gfaria_WANDB_API_KEY",
    wandb_api_key: str | None = None,
    wandb_api_factory: Callable[[str], Any] = default_wandb_api_factory,
) -> FastAPI:
    database_path = Path(database)
    app = FastAPI(title="JTCEval Watcher", docs_url=None, redoc_url=None)

    @app.get("/", response_class=HTMLResponse)
    def home() -> HTMLResponse:
        page, _script = _watcher_assets()
        return HTMLResponse(page, headers={"cache-control": "no-store"})

    @app.get("/watcher.js", response_class=Response)
    def watcher_javascript() -> Response:
        _page, script = _watcher_assets()
        return Response(
            content=script,
            media_type="text/javascript",
            headers={"cache-control": "no-store"},
        )

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"ok": database_path.is_file(), "database": str(database_path.absolute())}

    @app.get("/api/models")
    def models(
        q: str = Query(default="", max_length=300),
        limit: int = Query(default=1000, ge=1, le=1000),
    ) -> dict[str, Any]:
        with _connect(database_path) as db:
            results = _models(db, q, limit)
        return {"models": results, "count": len(results), "query": q}

    @app.get("/api/models/{model_id}")
    def model(model_id: int) -> dict[str, Any]:
        with _connect(database_path) as db:
            row = db.execute("SELECT * FROM models WHERE id = ?", (model_id,)).fetchone()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Unknown model id {model_id}")
            model_row = dict(row)
            model_stage = "rl" if db.execute(
                "SELECT 1 FROM training_runs WHERE model_id = ? AND stage = 'rl'",
                (model_id,),
            ).fetchone() else "sft"
            checkpoints = [
                dict(checkpoint)
                for checkpoint in db.execute(
                    """SELECT m.id, m.model_path, m.checkpoint_step, m.indexed_at,
                              count(e.id) AS evaluation_count
                       FROM models m
                       LEFT JOIN evaluations e ON e.model_id = m.id
                       WHERE m.output_dir = ?
                       GROUP BY m.id
                       ORDER BY m.checkpoint_step DESC, m.id DESC""",
                    (model_row["output_dir"],),
                )
            ]
            return {
                "model": {
                    **model_row,
                    "name": Path(model_row["output_dir"]).name,
                    "stage": model_stage,
                },
                "checkpoints": checkpoints,
                "training_runs": _training_runs(db, model_id),
                "datasets": _datasets(db, model_id),
                "evaluations": _run_evaluations(db, model_row["output_dir"], model_stage),
            }

    @app.get("/api/models/{model_id}/metrics")
    def metrics(model_id: int) -> dict[str, Any]:
        """Fetch linked W&B histories on demand; never persist them in SQLite."""

        with _connect(database_path) as db:
            if db.execute("SELECT 1 FROM models WHERE id = ?", (model_id,)).fetchone() is None:
                raise HTTPException(status_code=404, detail=f"Unknown model id {model_id}")
            attempts = metric_attempts(db, model_id)
        try:
            api_key = wandb_api_key or (resolve_wandb_api_key(wandb_secret) if attempts else "")
            return fetch_wandb_charts(
                attempts,
                api_key=api_key,
                api_factory=wandb_api_factory,
            )
        except WandbMetricsUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    return app


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>JTCEval Watcher</title>
<style>
:root{color-scheme:dark;--bg:#0b0e14;--side:#10151e;--panel:#151b26;--panel2:#1b2230;--line:#293346;--text:#edf3ff;--muted:#8c9ab0;--blue:#7aa2ff;--green:#67d3a0;--orange:#f1ae70;--red:#f08383;--purple:#a78bfa}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}button,input,a{font:inherit}header{height:62px;display:flex;align-items:center;justify-content:space-between;padding:0 22px;border-bottom:1px solid var(--line);background:#0e131c;position:sticky;top:0;z-index:4}header b{font-size:17px;letter-spacing:.01em}header b i{font-style:normal;color:var(--blue)}header span{color:var(--muted);font-size:12px}.layout{display:grid;grid-template-columns:360px minmax(0,1fr);height:calc(100vh - 62px)}aside{background:var(--side);border-right:1px solid var(--line);overflow:auto;padding:16px}.search{width:100%;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--text);padding:10px 12px;outline:none}.search:focus{border-color:var(--blue)}#model-meta{display:block;margin:12px 2px;color:var(--muted);font-size:11px}.model-list{display:grid;gap:7px}.model-card{position:relative;overflow:hidden;width:100%;text-align:left;border:1px solid var(--line);border-radius:11px;background:var(--panel);color:var(--text);padding:12px;cursor:pointer;transition:border-color .16s ease,background .16s ease,box-shadow .16s ease,transform .16s ease}.model-card:hover{border-color:#3b4b66;background:#18202d;transform:translateY(-1px)}.model-card.active{border-color:#5975aa;background:#192437;box-shadow:0 0 0 1px rgba(122,162,255,.08)}.model-card.has-evals{border-color:#343a4b;background:linear-gradient(180deg,#171d28,#151b26);padding-left:15px}.model-card.has-evals::before{content:"";position:absolute;left:0;top:10px;bottom:10px;width:2px;border-radius:0 3px 3px 0;background:linear-gradient(180deg,#a78bfa,#7aa2ff)}.model-card.has-evals:hover,.model-card.has-evals.active{border-color:#514e68;box-shadow:0 8px 24px rgba(0,0,0,.18),0 0 0 1px rgba(167,139,250,.06)}.model-card.has-evals.active{border-color:#677da7;background:#192331}.model-card h3{margin:0 0 8px;font-size:13px;line-height:1.35;overflow-wrap:anywhere}.model-card footer{display:flex;gap:7px;align-items:center;color:var(--muted);font-size:10px}.eval-signal{display:inline-flex;align-items:center;gap:5px;color:#c4b5fd;border:1px solid #3c3b4e;border-radius:999px;background:#1a1d28;padding:2px 7px;font-weight:600}.eval-signal::before{content:"";width:4px;height:4px;border-radius:50%;background:var(--purple);box-shadow:0 0 0 3px rgba(167,139,250,.09)}.pill,.stage,.status{display:inline-flex;align-items:center;border:1px solid var(--line);border-radius:999px;padding:3px 7px;font:10px ui-monospace,monospace}main{overflow:auto;padding:28px clamp(20px,4vw,60px) 80px}.empty{max-width:560px;margin:18vh auto;color:var(--muted)}.empty h1{color:var(--text);font-size:26px}.hero{border-bottom:1px solid var(--line);padding-bottom:23px;margin-bottom:22px}.eyebrow{color:var(--blue);font:11px ui-monospace,monospace;text-transform:uppercase;letter-spacing:.12em}.hero h1{margin:8px 0 7px;font-size:clamp(21px,3vw,32px);line-height:1.15;overflow-wrap:anywhere}.path{color:var(--muted);font:11px ui-monospace,monospace;overflow-wrap:anywhere}.checkpoint-strip{display:flex;gap:8px;flex-wrap:wrap;margin-top:16px}.checkpoint{display:inline-flex;align-items:center;gap:7px;border:1px solid var(--line);border-radius:8px;padding:8px 11px;color:var(--text);text-decoration:none;background:var(--panel);cursor:pointer}.checkpoint:hover{border-color:var(--blue)}.checkpoint.has-eval{border-color:#454258;background:#181c27}.checkpoint.active{border-color:#607eae;background:#1b2940;color:#edf3ff;box-shadow:0 0 0 1px rgba(122,162,255,.08)}.checkpoint small{color:#c4b5fd}.stats{display:grid;grid-template-columns:repeat(4,minmax(110px,1fr));gap:10px;margin:0 0 24px}.stat{border:1px solid var(--line);border-radius:10px;background:var(--panel);padding:13px}.stat b{display:block;font-size:20px;margin-bottom:2px}.stat span{color:var(--muted);font-size:11px}.section{margin-top:28px}.section-title{display:flex;align-items:baseline;justify-content:space-between;gap:12px;margin-bottom:10px}.section-title h2{font-size:15px;margin:0}.section-title span{color:var(--muted);font-size:11px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.card{border:1px solid var(--line);border-radius:11px;background:var(--panel);padding:14px;overflow:hidden}.card h3{margin:0 0 10px;font-size:13px}.stage.rl{color:var(--green)}.stage.sft{color:var(--purple)}.kv{display:grid;grid-template-columns:105px 1fr;gap:6px 11px;font-size:11px}.kv dt{color:var(--muted)}.kv dd{margin:0;font-family:ui-monospace,monospace;overflow-wrap:anywhere}.wandb-list{display:grid;gap:7px;margin-top:12px}.wandb{display:flex;justify-content:space-between;align-items:center;gap:10px;background:var(--panel2);border-radius:8px;padding:8px 9px}.wandb code{font-size:11px;overflow-wrap:anywhere}.wandb a{color:var(--blue);text-decoration:none;white-space:nowrap;font-size:11px}.dataset{margin-bottom:10px}.dataset-head{display:flex;justify-content:space-between;gap:12px;margin-bottom:9px}.dataset-head h3{margin:0;font-size:13px}.source-table,.eval-table{width:100%;border-collapse:collapse;font-size:11px}.source-table th,.source-table td,.eval-table th,.eval-table td{text-align:left;border-top:1px solid var(--line);padding:8px 7px;vertical-align:top}.source-table th,.eval-table th{color:var(--muted);font-weight:500}.source-table td.path-cell{font-family:ui-monospace,monospace;overflow-wrap:anywhere;max-width:460px}.weight{color:var(--green);font-variant-numeric:tabular-nums}.status.complete{color:var(--green)}.status.failed{color:var(--red)}.status.incomplete,.status.not_started{color:var(--orange)}.benchmarks{display:flex;gap:5px;flex-wrap:wrap}.benchmark{border:1px solid var(--line);border-radius:6px;padding:4px 6px;background:var(--panel2);white-space:nowrap}.benchmark b{color:var(--green)}.muted{color:var(--muted)}@media(max-width:900px){.layout{grid-template-columns:300px minmax(0,1fr)}.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}@media(max-width:680px){header{position:static}.layout{display:block;height:auto}aside{border-right:0;border-bottom:1px solid var(--line);max-height:42vh}.model-list{grid-template-columns:repeat(2,minmax(0,1fr))}main{min-height:58vh}.source-table{display:block;overflow-x:auto}}@media(max-width:440px){.model-list{grid-template-columns:1fr}.stats{grid-template-columns:1fr 1fr}}
.hyper-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:6px;margin:2px 0 14px}.hyper{min-width:0;border:1px solid #2b3749;border-radius:8px;background:var(--panel2);padding:8px}.hyper span{display:block;color:var(--muted);font-size:9px;text-transform:uppercase;letter-spacing:.06em}.hyper b{display:block;margin-top:3px;color:#d9e5ff;font:12px ui-monospace,monospace;overflow-wrap:anywhere}@media(max-width:440px){.hyper-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}
.checkpoint-menu{margin-top:14px}.checkpoint-menu>summary,.eval-incomplete>summary{display:flex;align-items:center;justify-content:space-between;gap:12px;border:1px solid var(--line);border-radius:9px;background:var(--panel);color:var(--text);padding:10px 12px;cursor:pointer;list-style-position:inside}.checkpoint-menu>summary:hover,.eval-incomplete>summary:hover{border-color:#465978}.checkpoint-menu>summary span,.eval-incomplete>summary span{color:var(--muted);font-size:11px}.checkpoint-menu[open]>summary,.eval-incomplete[open]>summary{border-color:#465978;background:#18202d}.checkpoint-menu .checkpoint-strip{max-height:164px;overflow:auto;margin-top:8px;padding:8px;border:1px solid var(--line);border-radius:9px;background:#101620}.eval-incomplete{margin-top:10px}.eval-incomplete>summary{border-color:#493f34;color:var(--orange)}.eval-incomplete .eval-table{margin-top:8px}.eval-incomplete[open]>summary{border-color:#6a533d}
.chart-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px}.chart-card{position:relative;min-height:330px;padding:16px;background:linear-gradient(180deg,#171e2a,#141a24);border-color:#2d384b}.chart-head{display:flex;justify-content:space-between;align-items:flex-start;gap:16px}.chart-head h3{font-size:14px;margin:0 0 5px}.chart-latest{text-align:right;font-variant-numeric:tabular-nums}.chart-latest b{display:block;font-size:23px;line-height:1}.chart-latest span,.chart-meta,.chart-metric{color:var(--muted);font-size:10px}.chart-metric{font-family:ui-monospace,monospace;overflow-wrap:anywhere;margin-top:5px;min-height:26px}.chart-svg{display:block;width:100%;height:auto;margin-top:10px;overflow:visible}.chart-gridline{stroke:#283346;stroke-width:1}.chart-axis-label{fill:#77869d;font:10px ui-monospace,monospace}.chart-band{fill:rgba(122,162,255,.09)}.chart-line{fill:none;stroke:#83a8ff;stroke-width:2.2;stroke-linecap:round;stroke-linejoin:round}.chart-card.accuracy .chart-band{fill:rgba(103,211,160,.08)}.chart-card.accuracy .chart-line{stroke:#67d3a0}.chart-dot{fill:#c7d7ff;stroke:#172033;stroke-width:2}.chart-card.accuracy .chart-dot{fill:#8ae4b7}.chart-empty,.chart-loading{display:grid;place-items:center;min-height:265px;text-align:center;color:var(--muted);font-size:12px}.chart-loading::before{content:"";width:18px;height:18px;border:2px solid #344056;border-top-color:var(--blue);border-radius:50%;animation:spin .8s linear infinite;margin-bottom:9px}.live-mark{display:inline-flex;align-items:center;gap:6px}.live-mark::before{content:"";width:6px;height:6px;border-radius:50%;background:var(--green);box-shadow:0 0 0 4px rgba(103,211,160,.08)}@keyframes spin{to{transform:rotate(360deg)}}@media(max-width:1100px){.chart-grid{grid-template-columns:1fr}}@media(max-width:680px){.chart-card{min-height:300px}}
</style></head><body>
<header><b>PrimeBeaker <i>Watcher</i></b><span id="global-status">Opening provenance database…</span></header>
<div class="layout"><aside><input id="search" class="search" placeholder="Filter runs…" aria-label="Filter runs"><small id="model-meta"></small><nav id="models" class="model-list"></nav></aside><main id="detail"><div class="empty"><h1>Choose a run</h1><p>Browse its checkpoints, training lineage, W&amp;B attempts, source parquet mixes, evaluations, and saved RL traces.</p></div></main></div>
<script>
const $=(id)=>document.getElementById(id);let selected=null;let selectedOutput=null;let timer=null;
function esc(value){const node=document.createElement('span');node.textContent=String(value??'');return node.innerHTML}
function shortPath(value){const parts=String(value||'').split('/');return parts.slice(-3).join('/')}
function num(value,digits=3){return typeof value==='number'?value.toFixed(digits):'—'}
const hyperLabels={learning_rate:'Learning rate',min_learning_rate:'Min LR',weight_decay:'Weight decay',beta1:'Beta 1',beta2:'Beta 2',max_steps:'Max steps',sequence_length:'Context',batch_size:'Batch size',micro_batch_size:'Microbatch',group_size:'Group size',kl_tau:'KL tau',warmup_steps:'Warmup',scheduler:'Scheduler',oversampling_factor:'Oversampling'};
function hyperValue(key,value){if(typeof value==='number'&&['learning_rate','min_learning_rate','kl_tau'].includes(key))return value.toExponential().replace('e-0','e-').replace('e+0','e+');if(typeof value==='number'&&Number.isInteger(value))return value.toLocaleString();return String(value)}
function metricValue(value,format){if(typeof value!=='number')return '—';return format==='percent'?(value*100).toFixed(1)+'%':value.toFixed(4)}
async function api(path){const response=await fetch(path);const data=await response.json();if(!response.ok)throw new Error(data.detail||data.error||response.statusText);return data}
function modelCard(model){return `<button class="model-card ${model.has_evaluations?'has-evals':''} ${selectedOutput===model.output_dir?'active':''}" data-id="${model.id}" data-output="${esc(model.output_dir)}"><h3>${esc(model.name)}</h3><footer><span class="stage ${model.stage}">${model.stage}</span><span class="pill">${model.checkpoint_count} checkpoint${model.checkpoint_count===1?'':'s'}</span>${model.evaluation_count?`<span class="eval-signal">${model.evaluation_count} eval${model.evaluation_count===1?'':'s'}</span>`:'<span>no evals</span>'}</footer></button>`}
async function loadModels(){const q=encodeURIComponent($('search').value);$('global-status').textContent='Loading runs…';const data=await api('/api/models?q='+q);$('model-meta').textContent=data.count+' indexed runs · evaluated first';$('models').innerHTML=data.models.map(modelCard).join('')||'<p class="muted">No runs match.</p>';document.querySelectorAll('.model-card[data-id]').forEach(button=>button.onclick=()=>loadModel(Number(button.dataset.id)));$('global-status').textContent='Watcher ready';if(selected===null&&data.models.length)loadModel(data.models[0].id)}
function runCard(run){const params=Object.entries(run.hyperparameters||{});const hyper=params.length?`<div class="hyper-grid">${params.map(([key,value])=>`<div class="hyper"><span>${esc(hyperLabels[key]||key)}</span><b>${esc(hyperValue(key,value))}</b></div>`).join('')}</div>`:'<p class="muted">No parseable hyperparameters in the stored config.</p>';const wandb=run.wandb_runs.map(item=>`<div class="wandb"><code>${esc(item.run_id||'run id unavailable')}</code>${item.url?`<a href="${esc(item.url)}" target="_blank" rel="noreferrer">Open W&amp;B ↗</a>`:'<span class="muted">local metadata</span>'}</div>`).join('')||'<p class="muted">No W&amp;B metadata found.</p>';return `<article class="card"><h3><span class="stage ${run.stage}">${run.stage}</span></h3>${hyper}<dl class="kv"><dt>Checkpoint</dt><dd>${esc(run.checkpoint_path||'—')}</dd><dt>Config</dt><dd>${esc(run.config_path||'—')}</dd><dt>Output</dt><dd>${esc(run.output_dir||'—')}</dd></dl><div class="wandb-list">${wandb}</div></article>`}
function datasetCard(dataset){const sources=dataset.sources.map(source=>`<tr><td>${esc(source.source_name)}</td><td class="weight">${num(source.weight,6)}</td><td>${source.train_quota??'—'}</td><td>${source.validation_quota??'—'}</td><td class="path-cell">${esc(source.source_path||'—')}</td></tr>`).join('');return `<article class="card dataset"><div class="dataset-head"><h3>${esc(dataset.dataset_name)}</h3><span class="stage ${dataset.stage}">${dataset.stage}</span></div>${sources?`<table class="source-table"><thead><tr><th>Source</th><th>Weight</th><th>Train</th><th>Val</th><th>Parquet</th></tr></thead><tbody>${sources}</tbody></table>`:'<p class="muted">No surviving source manifest.</p>'}</article>`}
function evaluationRows(evaluations){return evaluations.map(item=>{const benchmarks=item.benchmarks.map(row=>`<span class="benchmark">${esc(row.dataset)} <b>${num(row.mean_accuracy)}</b> · n=${row.runs}</span>`).join('')||'<span class="muted">No benchmark rows</span>';return `<tr><td><span class="stage ${item.stage}">${item.stage}</span> <span class="checkpoint ${item.model_id===selected?'active':''}">step ${item.checkpoint_step??'?'}</span></td><td><span class="status ${esc(item.status)}">${esc(item.status)}</span></td><td><div>${esc(shortPath(item.config_path))}</div><div class="path">${esc(item.run_dir||'')}</div></td><td><div class="benchmarks">${benchmarks}</div></td></tr>`}).join('')}
function evaluationTable(evaluations,emptyMessage){const rows=evaluationRows(evaluations)||`<tr><td colspan="4" class="muted">${esc(emptyMessage)}</td></tr>`;return `<table class="eval-table"><thead><tr><th>Stage / step</th><th>Status</th><th>Configuration</th><th>Outcomes</th></tr></thead><tbody>${rows}</tbody></table>`}
function evaluationSection(evaluations){const incomplete=evaluations.filter(item=>!['complete','failed'].includes(item.status));const current=evaluations.filter(item=>['complete','failed'].includes(item.status));const deferred=incomplete.length?`<details class="card eval-incomplete"><summary><b>${incomplete.length} incomplete eval${incomplete.length===1?'':'s'}</b><span>click to inspect</span></summary>${evaluationTable(incomplete,'')}</details>`:'';return `<article class="card">${evaluationTable(current,'No completed or failed evaluations yet.')}</article>${deferred}`}
function chartSvg(chart){const points=chart.points;if(!points.length)return '<div class="chart-empty">This metric was not logged by the linked W&amp;B attempts.</div>';const width=680,height=260,left=52,right=18,top=14,bottom=35;const values=points.flatMap(p=>[p.min,p.max]);let yMin=Math.min(...values),yMax=Math.max(...values);if(chart.format==='percent'){const observedSpan=yMax-yMin;const span=Math.max(observedSpan,.04);yMin=Math.max(0,yMin-span*.18);yMax=Math.min(1,yMax+span*.18);if(yMax-yMin<.08){const center=(yMin+yMax)/2;yMin=Math.max(0,center-.04);yMax=Math.min(1,center+.04)}}else{const span=Math.max(yMax-yMin,Math.abs(yMax)*.08,.01);yMin=Math.max(0,yMin-span*.15);yMax+=span*.15}let xMin=points[0].step,xMax=points[points.length-1].step;if(xMin===xMax){xMin-=1;xMax+=1}const sx=x=>left+(x-xMin)/(xMax-xMin)*(width-left-right);const sy=y=>top+(yMax-y)/(yMax-yMin)*(height-top-bottom);const line=points.map((p,i)=>`${i?'L':'M'}${sx(p.step).toFixed(2)},${sy(p.value).toFixed(2)}`).join(' ');const upper=points.map((p,i)=>`${i?'L':'M'}${sx(p.step).toFixed(2)},${sy(p.max).toFixed(2)}`).join(' ');const lower=[...points].reverse().map(p=>`L${sx(p.step).toFixed(2)},${sy(p.min).toFixed(2)}`).join(' ');const grid=Array.from({length:5},(_,i)=>{const fraction=i/4,y=top+fraction*(height-top-bottom),value=yMax-fraction*(yMax-yMin),percentDigits=yMax-yMin<.1?1:0;return `<line class="chart-gridline" x1="${left}" x2="${width-right}" y1="${y}" y2="${y}"/><text class="chart-axis-label" x="${left-8}" y="${y+3}" text-anchor="end">${chart.format==='percent'?(value*100).toFixed(percentDigits)+'%':value.toFixed(3)}</text>`}).join('');const ticks=Array.from({length:5},(_,i)=>{const fraction=i/4,x=left+fraction*(width-left-right),value=xMin+fraction*(xMax-xMin);return `<text class="chart-axis-label" x="${x}" y="${height-8}" text-anchor="middle">${Math.round(value)}</text>`}).join('');const dots=points.map(p=>`<circle class="chart-dot" cx="${sx(p.step)}" cy="${sy(p.value)}" r="3.6"><title>step ${p.step}: ${metricValue(p.value,chart.format)}${p.samples>1?` · ${p.samples} attempts · range ${metricValue(p.min,chart.format)}–${metricValue(p.max,chart.format)}`:''}</title></circle>`).join('');return `<svg class="chart-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(chart.title)} over training steps">${grid}<path class="chart-band" d="${upper}${lower}Z"/><path class="chart-line" d="${line}"/>${dots}${ticks}<text class="chart-axis-label" x="${(left+width-right)/2}" y="${height}" text-anchor="middle">training step</text></svg>`}
function chartCard(chart){const latest=chart.points.at(-1);const metric=chart.metrics_used.length?chart.metrics_used.join(' · '):chart.requested_metric;return `<article class="card chart-card ${chart.format==='percent'?'accuracy':''}"><div class="chart-head"><div><h3>${esc(chart.title)}</h3><div class="chart-meta">mean across ${chart.attempt_count} of ${chart.source_count} linked ${chart.stage.toUpperCase()} W&amp;B attempts</div></div><div class="chart-latest"><b>${metricValue(latest?.value,chart.format)}</b><span>${latest?`step ${latest.step}`:'no data'}</span></div></div><div class="chart-metric" title="${esc(metric)}">${esc(metric)}</div>${chartSvg(chart)}</article>`}
async function loadMetrics(id){const host=$('metric-charts');if(!host)return;try{const data=await api('/api/models/'+id+'/metrics');if(selected!==id)return;const failed=data.sources.filter(source=>source.error).length;$('metric-meta').innerHTML=`<span class="live-mark">live from ${data.sources.length} W&amp;B link${data.sources.length===1?'':'s'}${failed?` · ${failed} unavailable`:''}</span>`;host.innerHTML=data.charts.map(chartCard).join('')}catch(error){if(selected!==id)return;$('metric-meta').textContent='live fetch unavailable';host.innerHTML=`<article class="card chart-empty">Could not load W&amp;B metrics: ${esc(error.message)}</article>`}}
async function loadModel(id){
  selected=id;
  $('detail').innerHTML='<div class="empty"><h1>Loading lineage…</h1></div>';
  const data=await api('/api/models/'+id);
  if(selected!==id)return;
  const model=data.model;
  selectedOutput=model.output_dir;
  document.querySelectorAll('.model-card').forEach(card=>card.classList.toggle('active',card.dataset.output===model.output_dir));
  const sourceCount=data.datasets.reduce((total,item)=>total+item.sources.length,0);
  const benchmarkCount=data.evaluations.reduce((total,item)=>total+item.benchmarks.length,0);
  const checkpointNav=data.checkpoints.map(item=>`<button class="checkpoint ${item.id===id?'active':''} ${item.evaluation_count?'has-eval':''}" data-checkpoint-id="${item.id}">step ${item.checkpoint_step??'?'}${item.evaluation_count?` <small>${item.evaluation_count} eval${item.evaluation_count===1?'':'s'}</small>`:''}</button>`).join('');
  const checkpointMenu=`<details class="checkpoint-menu"><summary><b>${data.checkpoints.length} checkpoint${data.checkpoints.length===1?'':'s'}</b><span>selected step ${model.checkpoint_step??'unknown'}</span></summary><div class="checkpoint-strip">${checkpointNav}</div></details>`;
  $('detail').innerHTML=`<section class="hero"><div class="eyebrow">${model.stage.toUpperCase()} run · selected step ${model.checkpoint_step??'unknown'}</div><h1>${esc(model.name)}</h1><div class="path">${esc(model.model_path)}</div>${checkpointMenu}</section><section class="stats"><div class="stat"><b>${data.training_runs.length}</b><span>training stages</span></div><div class="stat"><b>${data.training_runs.reduce((n,r)=>n+r.wandb_runs.length,0)}</b><span>W&amp;B attempts</span></div><div class="stat"><b>${sourceCount}</b><span>source parquet rows</span></div><div class="stat"><b>${benchmarkCount}</b><span>benchmark outcomes across lineage</span></div></section><section class="section"><div class="section-title"><h2>Live W&amp;B metrics</h2><span id="metric-meta">fetching linked histories now…</span></div><div id="metric-charts" class="chart-grid"><article class="card chart-loading"><span>Loading validation and accuracy series</span></article></div></section><section class="section"><div class="section-title"><h2>Training lineage</h2><span>${model.stage==='rl'?'RL → SFT predecessor':'Standalone SFT checkpoint'}</span></div><div class="grid">${data.training_runs.map(runCard).join('')}</div></section><section class="section"><div class="section-title"><h2>Dataset mixes</h2><span>${data.datasets.length} bindings</span></div>${data.datasets.map(datasetCard).join('')||'<p class="muted">No datasets indexed.</p>'}</section><section class="section"><div class="section-title"><h2>Evaluations</h2><span>${data.evaluations.length} configs across run lineage</span></div>${evaluationSection(data.evaluations)}</section>`;
  document.querySelectorAll('[data-checkpoint-id]').forEach(button=>button.onclick=()=>loadModel(Number(button.dataset.checkpointId)));
  loadMetrics(id);
}
$('search').oninput=()=>{clearTimeout(timer);timer=setTimeout(loadModels,180)};loadModels().catch(error=>{$('global-status').textContent='Watcher unavailable';$('detail').innerHTML=`<div class="empty"><h1>Could not open watcher</h1><p>${esc(error.message)}</p></div>`});
</script></body></html>'''


app = create_app(
    database=os.environ.get("WATCHER_DATABASE", str(DEFAULT_DATABASE)),
)
