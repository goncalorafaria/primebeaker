"""Fire CLI for inspecting bundled PrimeBeaker environments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import fire

from primebeaker.environments.registry import ENVIRONMENTS, environment_names


def environment(name: str | None = None) -> str:
    """List environments or resolve one name to its load function."""
    if name is None:
        return "\n".join(environment_names())
    normalized = name.removesuffix(".py").replace("_", "-")
    module = ENVIRONMENTS.get(normalized)
    if module is None:
        raise ValueError(f"unknown environment {name!r}")
    return f"primebeaker.environments.{module}:load_environment"


def main(argv: Sequence[str] | None = None) -> Any:
    command = None if argv is None else list(argv)
    return fire.Fire(environment, command=command)


if __name__ == "__main__":
    main()
