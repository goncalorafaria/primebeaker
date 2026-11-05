"""Read-only Beaker job source. Credentials remain in the configured CLI."""
from __future__ import annotations

import json
import subprocess
from typing import Any

from .store import utc_now


def experiments(command: list[str], ids: list[str], timeout: float = 30) -> dict[str, dict]:
    ids = list(dict.fromkeys(i for i in ids if i))
    if not ids:
        return {}
    result = subprocess.run([*command, "experiment", "get", *ids, "--format", "json"], capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(f"Beaker query failed (exit {result.returncode}); check CLI authentication")
    return {item["id"]: item for item in json.loads(result.stdout)}


def select_job(experiment: dict, job_id: str | None = None, task_name: str | None = None) -> dict:
    jobs = experiment.get("jobs", [])
    # Receipt identity takes precedence. Do not confuse a trainer and a judge
    # when they are tasks inside the same experiment.
    if job_id:
        match = [job for job in jobs if job["id"] == job_id]
        if match:
            return max(match, key=lambda j: j.get("status", {}).get("created", ""))
    if task_name:
        jobs = [j for j in jobs if j.get("name") == task_name]
    return max(jobs, key=lambda j: j.get("status", {}).get("created", ""), default={})


def job_snapshot(job: dict[str, Any], experiment_id: str, role: str, label: str) -> dict:
    status = job.get("status", {})
    if status.get("canceled"):
        state = "canceled"
    elif status.get("exitCode", 0):
        state = "failed"
    elif status.get("finalized") or status.get("exited"):
        state = "finished"
    elif status.get("started"):
        state = "running"
    elif status.get("scheduled"):
        state = "starting"
    elif status.get("created"):
        state = "queued"
    else:
        state = "unknown"
    env = job.get("execution", {}).get("spec", {}).get("envVars", [])
    host = next((entry.get("value") for entry in env if entry.get("name") == "BEAKER_NODE_HOSTNAME"), None)
    port = next((entry.get("value") for entry in env if entry.get("name") == "JUDGE_PORT_BASE"), None)
    endpoint = f"http://{host}:{port}" if host and port and str(port).isdigit() else None
    # Never persist a full Beaker spec or its environment/secrets.
    return {"id": job.get("id"), "label": label, "role": role, "task_name": job.get("name"), "state": state, "host": host, "metrics_endpoint": endpoint, "status": status, "experiment_id": experiment_id, "url": f"https://beaker.org/ex/{experiment_id}", "observed_at": utc_now()}
