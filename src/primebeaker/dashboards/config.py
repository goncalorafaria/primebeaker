"""Portable dashboard manifests; paths are relative to the manifest, not the cwd."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def dotted(value: dict[str, Any], key: str) -> Any:
    for part in key.split("."):
        value = value.get(part, {})
    return value or None


class DashboardConfig:
    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.data = read_json(self.path)
        if self.data.get("schema_version", 1) != 1:
            raise ValueError("Unsupported dashboard schema_version")
        self.kind = self.data.get("kind", "rubric-judging")
        self.title = self.data.get("title", "Rubric judging")
        self.runs = self.data.get("runs", [])
        ids = [r.get("id") for r in self.runs]
        if not ids or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("runs requires unique, nonempty string ids")
        self.judge = self.data.get("judge", {})
        self.state_dir = self.resolve(self.data.get("state_dir", ".dashboard-state"))
        self.interval = self.seconds("refresh_seconds", 15)
        self.jobs_interval = self.seconds("jobs_refresh_seconds", 60)
        self.wandb_interval = self.seconds("wandb_refresh_seconds", 60)
        self.timeout = self.seconds("request_timeout_seconds", 8)
        self.beaker_command = self.data.get("beaker_command", ["beaker"])
        if not isinstance(self.beaker_command, list) or not self.beaker_command or any(not isinstance(s, str) for s in self.beaker_command):
            raise ValueError("beaker_command must be a nonempty argument list; shell commands are not accepted")
        self.headers = {}
        # Store only environment-variable names in manifests; never credentials.
        for name, env in self.data.get("policy_headers_from_env", {}).items():
            import os
            if env in os.environ:
                self.headers[name] = os.environ[env]

    def seconds(self, key: str, default: float) -> float:
        value = float(self.data.get(key, default))
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        return value

    def resolve(self, path: str) -> Path:
        result = Path(path).expanduser()
        return result if result.is_absolute() else self.path.parent / result

    def run(self, spec: dict[str, Any]) -> dict[str, Any]:
        receipt = read_json(self.resolve(spec["receipt"])) if spec.get("receipt") else {}
        result = {**receipt, **spec}
        result["label"] = spec.get("label", spec["id"])
        run_dir = spec.get("run_dir") or receipt.get("run_directory")
        result["run_dir"] = str(self.resolve(run_dir)) if run_dir else None
        result["experiment_id"] = spec.get("experiment_id") or receipt.get("training_experiment_id")
        result["job_id"] = spec.get("job_id") or receipt.get("training_job_id")
        wandb = spec.get("wandb_run") or receipt.get("wandb_url")
        if not wandb and receipt.get("wandb_run_id") and spec.get("wandb_entity") and spec.get("wandb_project"):
            wandb = "/".join([spec["wandb_entity"], spec["wandb_project"], receipt["wandb_run_id"]])
        if wandb:
            from urllib.parse import urlparse
            if wandb.startswith("https://"):
                parts = urlparse(wandb).path.strip("/").split("/")
                wandb = "/".join([parts[0], parts[1], parts[-1]])
        result["wandb_run"] = wandb
        return result

    def judge_spec(self) -> dict[str, Any]:
        spec = dict(self.judge)
        receipt = read_json(self.resolve(spec["receipt"])) if spec.get("receipt") else {}
        experiments = []
        for entry in spec.get("experiments", []):
            item = dict(entry)
            item["experiment_id"] = item.get("experiment_id") or dotted(receipt, item.get("receipt_key", ""))
            if item["experiment_id"]:
                experiments.append(item)
        spec["experiments"] = experiments
        if spec.get("expected_replicas_key"):
            spec["expected_replicas"] = dotted(receipt, spec["expected_replicas_key"])
        return spec
