"""Workspace-scoped Beaker dependency validation for launches."""

from __future__ import annotations

from dataclasses import dataclass
import json
import subprocess
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class BeakerResourceReport:
    """Images and secrets resolved for one rendered Beaker experiment."""

    workspace: str
    images: tuple[str, ...]
    secrets: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "workspace": self.workspace,
            "images": list(self.images),
            "secrets": list(self.secrets),
        }


def collect_beaker_resources(
    spec: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Collect Beaker image references and secret names from a rendered spec."""

    images: set[str] = set()
    secrets: set[str] = set()
    tasks = spec.get("tasks", ())
    if not isinstance(tasks, list):
        raise ValueError("Beaker spec tasks must be a list")
    for task in tasks:
        if not isinstance(task, dict):
            continue
        image = task.get("image")
        if isinstance(image, dict) and isinstance(image.get("beaker"), str):
            images.add(image["beaker"].removeprefix("beaker://"))
        env_vars = task.get("envVars", ())
        if not isinstance(env_vars, list):
            continue
        for item in env_vars:
            if isinstance(item, dict) and isinstance(item.get("secret"), str):
                secrets.add(item["secret"])
    return tuple(sorted(images)), tuple(sorted(secrets))


def _run_json(command: Sequence[str]) -> Any:
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as error:
        detail = (error.stderr or "").strip() or (error.stdout or "").strip()
        raise RuntimeError(f"Beaker resource preflight failed: {detail or 'no detail'}") from error
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("Beaker returned invalid JSON during resource preflight") from error


def validate_beaker_resources(
    *, workspace: str, spec: Mapping[str, Any]
) -> BeakerResourceReport:
    """Require every secret and immutable image to exist in ``workspace``."""

    images, secrets = collect_beaker_resources(spec)
    secret_payload = _run_json(
        ("beaker", "secret", "list", "--workspace", workspace, "--format", "json")
    )
    if not isinstance(secret_payload, list):
        raise RuntimeError(f"Beaker did not return the secret inventory for {workspace}")
    available_secrets = {
        item["name"]
        for item in secret_payload
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    missing_secrets = sorted(set(secrets) - available_secrets)

    misplaced_images: list[str] = []
    missing_images: list[str] = []
    uncommitted_images: list[str] = []
    for image in images:
        try:
            payload = _run_json(("beaker", "image", "get", image, "--format", "json"))
        except RuntimeError:
            missing_images.append(image)
            continue
        item = payload[0] if isinstance(payload, list) and payload else payload
        if not isinstance(item, dict):
            missing_images.append(image)
            continue
        workspace_ref = item.get("workspaceRef")
        owner = workspace_ref.get("fullName") if isinstance(workspace_ref, dict) else None
        if owner != workspace:
            misplaced_images.append(f"{image} (owned by {owner or 'unknown workspace'})")
        committed = item.get("committed")
        if not isinstance(committed, str) or committed.startswith("0001-"):
            uncommitted_images.append(image)

    failures: list[str] = []
    if missing_secrets:
        failures.append(f"missing secrets: {', '.join(missing_secrets)}")
    if missing_images:
        failures.append(f"missing images: {', '.join(missing_images)}")
    if misplaced_images:
        failures.append(f"images outside {workspace}: {', '.join(misplaced_images)}")
    if uncommitted_images:
        failures.append(f"uncommitted images: {', '.join(uncommitted_images)}")
    if failures:
        raise RuntimeError(
            f"Beaker workspace preflight failed for {workspace}: {'; '.join(failures)}"
        )
    return BeakerResourceReport(workspace=workspace, images=images, secrets=secrets)
