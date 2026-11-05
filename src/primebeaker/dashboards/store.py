"""Atomic snapshots and append-only history shared by all dashboard views."""
from __future__ import annotations

from datetime import datetime, timezone
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
