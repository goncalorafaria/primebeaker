"""Supervise one Python evaluator and its PrimeBeaker-owned service stack."""

from __future__ import annotations

import atexit
import json
import os
import signal
import subprocess
import sys
from typing import Any

from literegistry.coop.endpoints import wait as wait_endpoint

from primebeaker.coordination import wait_for_services


class ManagedEvaluationWorker:
    """Resolve native endpoints, run a Python module, and clean up services."""

    def __init__(self) -> None:
        self.module = _required("PRIMEBEAKER_EVALUATION_MODULE")
        arguments = json.loads(
            os.environ.get("PRIMEBEAKER_EVALUATION_ARGUMENTS_JSON", "[]")
        )
        if not isinstance(arguments, list) or any(
            not isinstance(value, str) for value in arguments
        ):
            raise ValueError(
                "PRIMEBEAKER_EVALUATION_ARGUMENTS_JSON must encode strings"
            )
        self.arguments = tuple(arguments)
        requirements = json.loads(
            os.environ.get("PRIMEBEAKER_REQUIRED_SERVICES_JSON", "{}")
        )
        if not isinstance(requirements, dict):
            raise ValueError(
                "PRIMEBEAKER_REQUIRED_SERVICES_JSON must encode an object"
            )
        self.requirements = {str(name): int(count) for name, count in requirements.items()}
        self.coordination_root = _required("PRIMEBEAKER_COORDINATION_ROOT")
        self.service_experiment_id = _required(
            "PRIMEBEAKER_MANAGED_SERVICE_EXPERIMENT_ID"
        )
        self.keep_services = _boolean(
            os.environ.get("PRIMEBEAKER_KEEP_SERVICES", "False")
        )
        self.timeout = float(
            os.environ.get("PRIMEBEAKER_READINESS_TIMEOUT_SECONDS", "7200")
        )
        self.process: subprocess.Popen[bytes] | None = None
        self.cleaned = False

    def run(self) -> int:
        atexit.register(self.cleanup)
        previous = self._install_signal_handlers()
        try:
            registry = os.environ.get("PRIMEBEAKER_REGISTRY") or wait_endpoint(
                self.coordination_root,
                "redis",
                timeout=self.timeout,
                healthcheck="redis",
            )
            gateway = wait_endpoint(
                self.coordination_root,
                "gateway",
                timeout=self.timeout,
                healthcheck="http",
            )
            wait_for_services(
                registry,
                self.requirements,
                timeout=self.timeout,
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "REGISTRY": registry,
                    "REGISTRY_URL": registry,
                    "LITEREGISTRY_GATEWAY_URL": gateway,
                    "OPENAI_BASE_URL": gateway.rstrip("/") + "/v1",
                }
            )
            self.process = subprocess.Popen(
                (sys.executable, "-m", self.module, *self.arguments),
                env=environment,
                start_new_session=True,
            )
            return self.process.wait()
        finally:
            self._stop_child()
            self.cleanup()
            self._restore_signal_handlers(previous)
            atexit.unregister(self.cleanup)

    def cleanup(self) -> None:
        if self.cleaned:
            return
        self.cleaned = True
        if self.keep_services:
            return
        completed = subprocess.run(
            ("beaker", "experiment", "stop", self.service_experiment_id),
            check=False,
        )
        if completed.returncode:
            print(
                "warning: could not stop managed service experiment "
                f"{self.service_experiment_id}",
                file=sys.stderr,
            )
        else:
            print(f"STOPPED_EXPERIMENT_ID={self.service_experiment_id}")

    def _stop_child(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()

    def _install_signal_handlers(self) -> dict[signal.Signals, Any]:
        previous: dict[signal.Signals, Any] = {}

        def terminate(signum: int, _frame: Any) -> None:
            if self.process is not None and self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signum)
                except ProcessLookupError:
                    pass
            raise SystemExit(128 + signum)

        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous[signum] = signal.getsignal(signum)
            signal.signal(signum, terminate)
        return previous

    @staticmethod
    def _restore_signal_handlers(previous: dict[signal.Signals, Any]) -> None:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def _required(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"missing required environment variable {name}")
    return value


def _boolean(value: str) -> bool:
    normalized = value.strip().casefold()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def main() -> int:
    return ManagedEvaluationWorker().run()


if __name__ == "__main__":
    raise SystemExit(main())
