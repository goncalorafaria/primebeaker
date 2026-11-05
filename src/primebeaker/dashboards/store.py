"""Atomic snapshots and append-only history shared by all dashboard views."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import tempfile


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SnapshotStore:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def write(self, name: str, value: dict, *, history: bool = False) -> None:
        data = json.dumps(value, allow_nan=False, indent=2) + "\n"
        fd, tmp = tempfile.mkstemp(prefix=f".{name}.", dir=self.directory)
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(data)
            os.replace(tmp, self.directory / name)
        finally:
            Path(tmp).unlink(missing_ok=True)
        if history:
            day = datetime.now(timezone.utc).date().isoformat()
            with (self.directory / f"history-{day}.jsonl").open("a") as stream:
                stream.write(json.dumps(value, allow_nan=False) + "\n")

    def read(self, name: str) -> dict:
        return json.loads((self.directory / name).read_text())

    def inference_history(self) -> dict:
        """Bounded last-hour history, including gaps; no private job metadata."""
        now = datetime.now(timezone.utc)
        cutoff = now.timestamp() - 3600
        snapshots = []
        reached_cutoff = False
        for day in (now.date(), (now - timedelta(days=1)).date()):
            path = self.directory / f"history-{day.isoformat()}.jsonl"
            try:
                with path.open('rb') as stream:
                    stream.seek(0, 2)
                    size = stream.tell()
                    start = max(0, size - 8 * 1024 * 1024)
                    stream.seek(start)
                    if start:
                        stream.readline()  # Skip an incomplete first record.
                    lines = stream.read().splitlines()
            except FileNotFoundError:
                continue
            for line in reversed(lines):
                try:
                    row = json.loads(line)
                    stamp = datetime.fromisoformat(row['timestamp']).timestamp()
                except (ValueError, KeyError, TypeError):
                    continue  # An append in progress can leave a partial line.
                if stamp < cutoff:
                    reached_cutoff = True
                    break
                snapshots.append({'timestamp': row['timestamp'], 'inference': {
                    role: [{k: endpoint.get(k) for k in (
                        'endpoint', 'sampled_at', 'available', 'active',
                        'request_latency_seconds_interval_mean', 'kv_cache_percent_mean')}
                        for endpoint in endpoints]
                    for role, endpoints in row.get('inference', {}).items()}})
                if len(snapshots) >= 300:
                    break
            if len(snapshots) >= 300 or reached_cutoff:
                break
        return {'window_seconds': 3600, 'snapshots': sorted(snapshots, key=lambda row: row['timestamp'])}
