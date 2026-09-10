"""Resolve and snapshot runtime data referenced by evaluation YAMLs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def resolve_template_path(value: str | Path, *, config_path: str | Path) -> Path:
    """Resolve a template relative to its YAML, checkout, or installed JTC wheel."""

    requested = Path(value)
    source = Path(config_path).resolve()
    candidates: list[Path] = []
    if requested.is_absolute():
        candidates.append(requested)
    else:
        candidates.append(source.parent / requested)
        candidates.append(Path.cwd() / requested)
        candidates.extend(parent / requested for parent in source.parents)
        try:
            from jtc.common.templates import resolve_resource_path
        except ImportError:
            pass
        else:
            candidates.append(resolve_resource_path(requested))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file():
            _validate_json_template(resolved.read_bytes(), source=resolved)
            return resolved
    raise FileNotFoundError(
        f"template referenced by {source} was not found: {requested}"
    )


def snapshot_template(
    source: str | Path,
    *,
    run_dir: str | Path,
    materialize: bool,
) -> Path:
    """Return a stable mounted template path, optionally materializing it."""

    path = Path(source)
    if not path.is_file():
        return path
    contents = path.read_bytes()
    _validate_json_template(contents, source=path)
    digest = hashlib.sha256(contents).hexdigest()
    target = Path(run_dir) / ".jtceval-inputs" / "templates" / f"{digest}.json"
    if not materialize:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if target.read_bytes() != contents:
            raise RuntimeError(f"template snapshot digest collision: {target}")
        return target
    with tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def _validate_json_template(contents: bytes, *, source: Path) -> None:
    try:
        value = json.loads(contents)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON template {source}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"template must contain a JSON object: {source}")


__all__ = ["resolve_template_path", "snapshot_template"]
