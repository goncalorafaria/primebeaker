"""Fire CLI for tested Prime-RL images on Beaker."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from typing import Any

import fire

from primebeaker.images import default_image_uri, image_locations


def list_images() -> list[dict[str, Any]]:
    """List cataloged images."""
    return [location.__dict__ for location in image_locations()]


def inspect(
    workspace: str | None = None,
    image: str | None = None,
) -> None:
    """Inspect one cataloged or explicitly selected Beaker image."""
    uri = image or default_image_uri(workspace=workspace)
    subprocess.run(
        ["beaker", "image", "inspect", uri.removeprefix("beaker://"), "--format", "json"],
        check=True,
    )


def pull(
    tag: str,
    workspace: str | None = None,
    image: str | None = None,
) -> None:
    """Pull one cataloged or explicitly selected Beaker image."""
    uri = image or default_image_uri(workspace=workspace)
    subprocess.run(
        ["beaker", "image", "pull", uri.removeprefix("beaker://"), tag],
        check=True,
    )


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return "" if value is None else str(value)


def main(argv: Sequence[str] | None = None) -> Any:
    commands = {
        "list": list_images,
        "inspect": inspect,
        "pull": pull,
    }
    command = None if argv is None else list(argv)
    return fire.Fire(commands, command=command, serialize=_serialize)


if __name__ == "__main__":
    main()
