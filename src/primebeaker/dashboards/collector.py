"""View-independent polling and persistence for live training dashboards."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import threading
import time

from . import beaker
from .config import DashboardConfig
from .prometheus import VLLMMetrics, discover
from .store import SnapshotStore, utc_now
from .training import TrainingProgress
from .wandb import WandbHistory, curve_matches_run


class Collector:
    def __init__(self, config: DashboardConfig, curves: dict):
        self.config = config
        self.curve_definitions = curves
        self.store = SnapshotStore(config.state_dir)
        self.metrics = VLLMMetrics(config.timeout)
        self.training = TrainingProgress()
        self.wandb = WandbHistory(config.data.get("wandb_beaker_secret"), config.beaker_command)
        self.pool = ThreadPoolExecutor(max_workers=12)
        self.last_jobs = 0.0
        self.jobs = []
        self.endpoints = []
        self.source_errors = {}
        self.jobs_observed_at = None
        self.previous_curves = {}
        try:
            cached = self.store.read("latest.json")
            self.jobs = cached.get("jobs", [])
            self.jobs_observed_at = cached.get("jobs_observed_at")
            self.endpoints = [e["endpoint"] for e in cached.get("inference", {}).get("judge", [])]
        except (OSError, ValueError):
            pass
        try:
            self.previous_curves = self.store.read("curves.json").get("runs", {})
        except (OSError, ValueError):
            pass

    def refresh_sources(self, runs: list[dict], judge: dict) -> None:
        if time.monotonic() - self.last_jobs < self.config.jobs_interval:
            return
        self.last_jobs = time.monotonic()
        ids = [r.get("experiment_id") for r in runs] + [e["experiment_id"] for e in judge.get("experiments", [])]
        try:
            experiments = beaker.experiments(self.config.beaker_command, ids)
            jobs = []
            for run in runs:
                exid = run.get("experiment_id")
                if not exid:
                    continue
                job = beaker.select_job(experiments.get(exid, {}), run.get("job_id"), run.get("task_name"))
                jobs.append(beaker.job_snapshot(job, exid, run["id"], run["label"]))
            seen = {j["id"] for j in jobs if j["id"]}
            for item in judge.get("experiments", []):
                exid = item["experiment_id"]
                names = item.get("task_names")
                for job in experiments.get(exid, {}).get("jobs", []):
                    if job["id"] in seen or (names and job.get("name") not in names):
                        continue
                    seen.add(job["id"])
                    jobs.append(beaker.job_snapshot(job, exid, "judge", job.get("name", "Judge")))
            self.jobs, self.jobs_observed_at = jobs, utc_now()
            self.source_errors.pop("beaker", None)
        except Exception as exc:
            self.source_errors["beaker"] = str(exc)
        try:
            endpoints = list(judge.get("endpoints", []))
            if judge.get("registry") and judge.get("model"):
                endpoints += discover(judge["registry"], judge["model"], self.config.timeout)
            self.endpoints = list(dict.fromkeys(e.rstrip("/").removesuffix("/v1") for e in endpoints))
            self.source_errors.pop("registry", None)
        except Exception as exc:
            self.source_errors["registry"] = f"Registry discovery unavailable ({type(exc).__name__}); check client/runtime connectivity"

    def sample(self) -> dict:
        runs = []
        for spec in self.config.runs:
            key = "configuration/" + spec["id"]
            try:
                runs.append(self.config.run(spec))
                self.source_errors.pop(key, None)
            except (OSError, ValueError) as exc:
                self.source_errors[key] = f"{type(exc).__name__}: {exc}"
                runs.append({"id": spec["id"], "label": spec.get("label", spec["id"]), "run_dir": None})
        judge = self.config.judge_spec()
        self.refresh_sources(runs, judge)
        endpoints = [("judge", url, {}) for url in self.endpoints]
        for run in runs:
            urls = list(run.get("endpoints", []))
            job = next((j for j in self.jobs if j["role"] == run["id"]), {})
            if not urls and job.get("state") == "running" and job.get("host"):
                urls = [f"http://{job['host']}:{run.get('inference_port', 8000)}"]
            endpoints += [(run["id"], url, self.config.headers) for url in dict.fromkeys(urls)]
        futures = [(role, self.pool.submit(self.metrics.scrape, url, headers)) for role,url,headers in endpoints]
        inference = {"judge": []} | {r["id"]: [] for r in runs}
        for role, future in futures:
            inference[role].append(future.result())
        snapshot = {"schema_version": 1, "timestamp": utc_now(), "refresh_seconds": self.config.interval,
                    "dashboard": {"title": self.config.title, "kind": self.config.kind}, "jobs": self.jobs, "jobs_observed_at": self.jobs_observed_at,
                    "source_errors": dict(self.source_errors), "inference": inference,
                    "runs": {r["id"]: {"label": r["label"], "trainer": self.training.read(r["id"], r.get("run_dir")), "rubrics": r.get("rubrics")} for r in runs},
                    "judge_pool": {"requested": judge.get("expected_replicas"), "discovered": len(inference["judge"]), "serving": sum(bool(s.get("available")) for s in inference["judge"])}}
        self.store.write("latest.json", snapshot, history=True)
        return snapshot

    def sample_curves(self) -> dict:
        result = {"schema_version": 1, "source": "W&B API", "fetched_at": utc_now(), "refresh_seconds": self.config.wandb_interval, "runs": {}}
        for spec in self.config.runs:
            run = {"id": spec["id"], "label": spec.get("label", spec["id"])}
            try:
                run = self.config.run(spec)
                if not run.get("wandb_run"):
                    result[run["id"]] = {"label": run["label"], "available": False, "error": "No W&B run configured"}
                    continue
                value = self.wandb.read(run["wandb_run"], self.config.data.get("curves", self.curve_definitions))
                value["label"] = run["label"]
                value["wandb_run"] = run["wandb_run"]
            except Exception as exc:
                previous = self.previous_curves.get(run["id"], {})
                value = dict(previous) if curve_matches_run(previous, run.get('wandb_run')) else {}
                value['wandb_run'] = run.get('wandb_run')
                value.update(label=run["label"], available=False, error=f"W&B fetch unavailable: {type(exc).__name__}", attempted_at=utc_now())
            result["runs"][run["id"]] = value
        self.previous_curves = result["runs"]
        self.store.write("curves.json", result)
        return result

    def loop(self, source: str, stop: threading.Event, *, once: bool = False) -> None:
        with (self.config.state_dir / f".{source}.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(f"Another {source} collector already owns this state directory") from exc
            while not stop.is_set():
                start = time.monotonic()
                try:
                    result = self.sample() if source == "metrics" else self.sample_curves()
                    print(json.dumps({"event": "snapshot", "source": source, "at": result.get("timestamp", result.get("fetched_at"))}), flush=True)
                except Exception as exc:
                    print(json.dumps({"event": "source_error", "source": source, "error": f"{type(exc).__name__}: {exc}"}), flush=True)
                    if once:
                        raise
                if once:
                    return
                interval = self.config.interval if source == "metrics" else self.config.wandb_interval
                stop.wait(max(1, interval - (time.monotonic() - start)))

    def close(self):
        self.pool.shutdown(wait=True, cancel_futures=True)
