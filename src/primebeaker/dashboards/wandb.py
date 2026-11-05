"""Sparse, raw W&B histories shared by dashboard views; optional dependency."""
from __future__ import annotations

import math
import os
import subprocess

from .store import utc_now


class WandbHistory:
    def __init__(self, secret: str | None = None, beaker_command: list[str] | None = None):
        self.secret = secret
        self.beaker_command = beaker_command or ["beaker"]
        self.api = None

    def connect(self):
        if self.api is not None:
            return self.api
        import wandb
        if not hasattr(wandb, "Api"):
            raise RuntimeError("Install primebeaker[dashboards] in the collector environment")
        # Prefer the SDK's configured credentials (including WANDB_API_KEY).
        # A Beaker fallback is opt-in, not a hardcoded user or secret name.
        kwargs = {}
        if self.secret and not os.environ.get("WANDB_API_KEY"):
            result = subprocess.run([*self.beaker_command, "secret", "read", self.secret], capture_output=True, text=True, timeout=20)
            if result.returncode or not result.stdout.strip():
                raise RuntimeError("Configured W&B Beaker credential is unavailable")
            kwargs["api_key"] = result.stdout.strip()
        self.api = wandb.Api(timeout=25, **kwargs)
        return self.api

    def read(self, run_path: str, definitions: dict) -> dict:
        run = self.connect().run(run_path)
        curves = {}
        for name, spec in definitions.items():
            key = spec["metric"]
            requires = spec.get("requires", [])
            points = {}
            # Query independently: reward, token and trainer timing rows are
            # sparse and may be written by different shared-run producers.
            for row in run.scan_history(keys=["step", key, *requires], page_size=1000):
                step, value = row.get("step"), row.get(key)
                if isinstance(step, (int, float)) and not isinstance(step, bool) and isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(step) and math.isfinite(value):
                    points[step] = value
            curves[name] = {"metric": key, "unit": spec.get("unit", ""), "row_filter_requires": requires,
                            "points": [{"step": step, "value": value} for step,value in sorted(points.items())]}
        return {"available": True, "url": run.url, "state": run.state, "fetched_at": utc_now(), "curves": curves}
