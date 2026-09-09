"""Read the package-local catalog of tested Prime-RL training images."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


_CATALOG_PATH = Path(__file__).with_name("catalog.json")


@dataclass(frozen=True)
class ImageLocation:
    role: str
    immutable_uri: str
    name: str
    workspace: str
    size_bytes: int
    created: str
    committed: str

    @property
    def image_id(self) -> str:
        return self.immutable_uri.removeprefix("beaker://")


def load_catalog() -> dict[str, Any]:
    catalog = json.loads(_CATALOG_PATH.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 1:
        raise ValueError(f"unsupported image catalog schema in {_CATALOG_PATH}")
    return catalog


def image_locations(*, role: str | None = None) -> tuple[ImageLocation, ...]:
    catalog = load_catalog()
    selected = role or str(catalog["default_base_role"])
    role_data = catalog["roles"].get(selected)
    if not isinstance(role_data, dict):
        raise KeyError(f"unknown image role {selected!r}")
    return tuple(
        ImageLocation(role=selected, **location)
        for location in role_data["locations"]
    )


def default_image_uri(*, workspace: str | None = None) -> str:
    locations = image_locations()
    if workspace:
        for location in locations:
            if location.workspace == workspace:
                return location.immutable_uri
        raise KeyError(f"no cataloged Prime-RL image for workspace {workspace!r}")
    return locations[0].immutable_uri
