"""Provider-neutral helpers shared by the SFT and RL launchers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def image_source(image: str) -> dict[str, str]:
    """Render either a Beaker image reference or a Docker image reference."""

    if image.startswith("beaker://"):
        return {"beaker": image.removeprefix("beaker://")}
    return {"docker": image}


def require_under_mount(paths: Iterable[Path], mount_path: Path) -> None:
    """Ensure host paths will be visible inside the mounted Beaker dataset."""

    root = mount_path.resolve()
    for path in paths:
        candidate = path.resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(f"launch path must be under mounted path {root}: {candidate}") from error


def environment_entries(
    values: dict[str, str], secrets: dict[str, str]
) -> list[dict[str, str]]:
    """Render deterministic Beaker environment entries."""

    overlap = sorted(set(values) & set(secrets))
    if overlap:
        raise ValueError(f"environment names cannot be both values and secrets: {overlap}")
    return [
        *({"name": name, "value": value} for name, value in sorted(values.items())),
        *({"name": name, "secret": secret} for name, secret in sorted(secrets.items())),
    ]
