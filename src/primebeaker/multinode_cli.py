"""Fire commands used inside PrimeBeaker multi-node tasks."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fire


def role(kind: str) -> None:
    """Replace this process with the RL or SFT node-role script."""
    if kind not in {"rl", "sft"}:
        raise ValueError("kind must be rl or sft")
    from primebeaker.multinode import _role

    _role(kind)


def prepare_rl(config: str, config_dir: str, runtime_dir: str) -> None:
    """Materialize Prime-RL's per-role runtime configuration."""
    from primebeaker.multinode import prepare_rl_runtime

    prepare_rl_runtime(Path(config), Path(config_dir), Path(runtime_dir))


def prepare_sft(config: str, output: str, runtime_dir: str) -> None:
    """Materialize Prime-RL's multi-node SFT runtime configuration."""
    from primebeaker.multinode import prepare_sft_runtime

    prepare_sft_runtime(Path(config), Path(output), Path(runtime_dir))


def run(argv: Sequence[str] | None = None) -> Any:
    commands = {
        "role": role,
        "prepare-rl": prepare_rl,
        "prepare-sft": prepare_sft,
    }
    command = None if argv is None else list(argv)
    return fire.Fire(commands, command=command)


if __name__ == "__main__":
    run()
