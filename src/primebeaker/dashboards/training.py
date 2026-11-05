"""Read completed PrimeRL steps and rollout progress without loading ML packages."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
COMPLETED_STEP = re.compile(r"(?<!Micro )\bStep (\d+)\s*\|[^\n]*Loss[^\n]*")


def tail(path: Path, size: int = 256_000) -> str:
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - size))
        return ANSI.sub("", stream.read().decode(errors="replace"))


class TrainingProgress:
    def __init__(self):
        self.completed = {}
        self.timings = {}
        self.offsets = {}

    def read(self, run_id: str, directory: str | None) -> dict:
        result = {"completed_optimizer_step": None, "train_batch": None, "episodes_inflight": None, "buffered_episodes": None, "timing_seconds": None}
        if directory is None:
            return result
        root = Path(directory)
        identity = (run_id, str(root))
        if identity in self.completed:
            result.update(self.completed[identity])
        candidates = []
        for path in (root / "logs").glob("**/*.log"):
            if not any(part.startswith(("trainer", "orchestrator")) for part in path.parts):
                continue
            try:
                text = tail(path)
                age = time.time() - path.stat().st_mtime
            except OSError:
                continue
            if any(part.startswith("trainer") for part in path.parts):
                for match in COMPLETED_STEP.finditer(text):
                    candidates.append((int(match[1]), path, match[0], age))
            elif path.name == "orchestrator.log":
                batch = re.findall(r"Train batch (\d+)/(\d+)", text)
                inflight = re.findall(r"(\d+) inflight episodes", text)
                buffered = re.findall(r"\(\+(\d+) buffered\)", text)
                result.update(train_batch="/".join(batch[-1]) if batch else None, episodes_inflight=int(inflight[-1]) if inflight else None,
                              buffered_episodes=int(buffered[-1]) if buffered else None, orchestrator_log_age_seconds=age)
        if candidates:
            step, path, line, age = max(candidates, key=lambda c: c[0])
            if step >= (result["completed_optimizer_step"] or 0):
                self.completed[identity] = {"completed_optimizer_step": step, "step_source": str(path), "latest_completed_step_line": line, "latest_step_duration_reported": line.split("|")[1].strip(), "trainer_log_age_seconds": age}
                result.update(self.completed[identity])
        # Incremental reads keep sparse trainer timing rows even when frequent
        # inference telemetry quickly pushes them outside a short file tail.
        for path in (root / "monitors/file/metrics.jsonl", root / "trainer/monitors/file/metrics.jsonl"):
            if not path.is_file():
                continue
            try:
                size = path.stat().st_size
                offset = self.offsets.get(path, max(0, size - 4_000_000))
                if offset > size:
                    offset = 0
                with path.open("rb") as stream:
                    stream.seek(offset)
                    data = stream.read(8_000_000)
                complete = data.rfind(b"\n") + 1
                self.offsets[path] = offset + complete
                for line in data[:complete].splitlines():
                    try:
                        row = json.loads(line)
                    except (ValueError, UnicodeDecodeError):
                        continue
                    if "time/forward_backward" in row and "time/step" in row:
                        self.timings[identity] = {"timing_seconds": {k.removeprefix("time/"): v for k,v in row.items() if k.startswith("time/")}, "timing_step": row.get("step"), "timing_timestamp": row.get("time"), "timing_source": str(path)}
            except OSError:
                continue
        result.update(self.timings.get(identity, {}))
        return result
