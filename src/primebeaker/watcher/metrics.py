"""Lazy W&B history loading and cross-attempt aggregation for Watcher."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import math
import os
import sqlite3
import subprocess
from typing import Any, Callable, Iterable, Mapping, Sequence


SFT_VALIDATION_METRIC = "val/loss"
REQUESTED_EVAL_ACCURACY_METRIC = (
    "val/jtc-tool-label-terminal-eval/all/metrics/correct_final_label/mean"
)
# Prime-RL currently logs this series below ``eval``. Keep the requested
# ``val`` spelling first so runs using either convention work automatically.
EVAL_ACCURACY_METRICS = (
    REQUESTED_EVAL_ACCURACY_METRIC,
    "eval/jtc-tool-label-terminal-eval/all/metrics/correct_final_label/mean",
)


class WandbMetricsUnavailable(RuntimeError):
    """Raised when the live W&B client cannot be authenticated or created."""


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def resolve_wandb_api_key(secret_name: str) -> str:
    """Resolve W&B auth without printing or persisting the credential."""

    configured = os.environ.get("WANDB_API_KEY", "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["beaker", "secret", "read", secret_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise WandbMetricsUnavailable(
            f"W&B authentication is unavailable; could not read Beaker secret {secret_name!r}"
        ) from exc
    api_key = result.stdout.strip()
    if not api_key:
        raise WandbMetricsUnavailable(f"Beaker secret {secret_name!r} is empty")
    return api_key


def default_wandb_api_factory(api_key: str) -> Any:
    try:
        import wandb
    except ImportError as exc:  # pragma: no cover - dependency failure is deployment-specific
        raise WandbMetricsUnavailable("Install the `wandb` package to load live metrics") from exc
    return wandb.Api(api_key=api_key, timeout=30)


def metric_attempts(db: sqlite3.Connection, model_id: int) -> list[dict[str, Any]]:
    """Return distinct W&B links attached to this checkpoint's RL/SFT lineage."""

    return [
        dict(row)
        for row in db.execute(
            """SELECT DISTINCT tr.stage, w.entity, w.project, w.run_id, w.name, w.url
               FROM training_runs tr
               JOIN wandb_runs w ON w.training_run_id = tr.id
               WHERE tr.model_id = ?
                 AND w.entity IS NOT NULL
                 AND w.project IS NOT NULL
                 AND w.run_id IS NOT NULL
               ORDER BY CASE tr.stage WHEN 'rl' THEN 0 ELSE 1 END,
                        w.entity, w.project, w.run_id""",
            (model_id,),
        )
    ]


def _history_rows(run: Any, metrics: Sequence[str]) -> list[Mapping[str, Any]]:
    """Fetch only selected history columns, preferring the real training step."""

    keys = ["step", *metrics]
    try:
        rows = run.history(samples=10_000, keys=keys, x_axis="step", pandas=False)
    except Exception:
        # Older SFT runs sometimes only have W&B's internal step.
        rows = run.history(samples=10_000, keys=list(metrics), x_axis="_step", pandas=False)
    if hasattr(rows, "to_dict"):
        rows = rows.to_dict("records")
    return list(rows or [])


def _extract_series(
    rows: Iterable[Mapping[str, Any]],
    metrics: Sequence[str],
) -> tuple[list[tuple[float, float]], set[str]]:
    points: list[tuple[float, float]] = []
    used: set[str] = set()
    for row in rows:
        step = _number(row.get("step"))
        if step is None:
            step = _number(row.get("_step"))
        if step is None:
            continue
        for metric in metrics:
            value = _number(row.get(metric))
            if value is None:
                continue
            points.append((step, value))
            used.add(metric)
            break
    return points, used


def _aggregate(
    series: Sequence[tuple[str, Sequence[tuple[float, float]]]],
) -> list[dict[str, Any]]:
    # A restarted attempt can write one step more than once. Retain its latest
    # value, then aggregate overlapping attempts into a mean and min/max band.
    per_attempt: dict[tuple[str, float], float] = {}
    for attempt, points in series:
        for step, value in points:
            per_attempt[(attempt, step)] = value
    buckets: dict[float, list[tuple[str, float]]] = defaultdict(list)
    for (attempt, step), value in per_attempt.items():
        buckets[step].append((attempt, value))
    result: list[dict[str, Any]] = []
    for step, values in sorted(buckets.items()):
        observed = [value for _, value in values]
        result.append(
            {
                "step": int(step) if step.is_integer() else step,
                "value": sum(observed) / len(observed),
                "min": min(observed),
                "max": max(observed),
                "samples": len(observed),
                "attempts": sorted(attempt for attempt, _ in values),
            }
        )
    return result


def fetch_wandb_charts(
    attempts: Sequence[Mapping[str, Any]],
    *,
    api_key: str,
    api_factory: Callable[[str], Any] = default_wandb_api_factory,
) -> dict[str, Any]:
    """Fetch live histories and merge every linked W&B attempt by step."""

    specs = {
        "sft_validation_loss": {
            "stage": "sft",
            "title": "SFT validation loss",
            "metrics": (SFT_VALIDATION_METRIC,),
            "format": "number",
        },
        "eval_accuracy": {
            "stage": "rl",
            "title": "Eval accuracy",
            "metrics": EVAL_ACCURACY_METRICS,
            "format": "percent",
        },
    }
    chart_series: dict[str, list[tuple[str, Sequence[tuple[float, float]]]]] = defaultdict(list)
    metrics_used: dict[str, set[str]] = defaultdict(set)
    sources: list[dict[str, Any]] = []
    api = api_factory(api_key) if attempts else None

    for attempt in attempts:
        stage = str(attempt["stage"])
        path = f"{attempt['entity']}/{attempt['project']}/{attempt['run_id']}"
        source = {
            **dict(attempt),
            "path": path,
            "state": None,
            "metrics": [],
            "error": None,
        }
        try:
            assert api is not None
            run = api.run(path)
            source["state"] = getattr(run, "state", None)
            for chart_id, spec in specs.items():
                if spec["stage"] != stage:
                    continue
                points: list[tuple[float, float]] = []
                used: set[str] = set()
                # W&B's sampled-history API treats requested keys as an
                # intersection on some run schemas. Probe aliases separately
                # so a missing ``val/...`` key cannot mask the ``eval/...``
                # series that is actually present.
                for metric in spec["metrics"]:
                    rows = _history_rows(run, (metric,))
                    points, used = _extract_series(rows, (metric,))
                    if points:
                        break
                if points:
                    chart_series[chart_id].append((path, points))
                    metrics_used[chart_id].update(used)
                    source["metrics"].append(chart_id)
        except Exception as exc:  # One failed attempt must not hide the others.
            source["error"] = f"{type(exc).__name__}: {str(exc)[:240]}"
        sources.append(source)

    charts: list[dict[str, Any]] = []
    for chart_id, spec in specs.items():
        series = chart_series.get(chart_id, [])
        charts.append(
            {
                "id": chart_id,
                "title": spec["title"],
                "stage": spec["stage"],
                "format": spec["format"],
                "requested_metric": (
                    REQUESTED_EVAL_ACCURACY_METRIC
                    if chart_id == "eval_accuracy"
                    else SFT_VALIDATION_METRIC
                ),
                "metrics_used": sorted(metrics_used.get(chart_id, set())),
                "attempt_count": len(series),
                "source_count": sum(1 for source in sources if source["stage"] == spec["stage"]),
                "points": _aggregate(series),
            }
        )
    return {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "charts": charts,
    }
