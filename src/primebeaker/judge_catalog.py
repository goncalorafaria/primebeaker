"""Packaged, model-keyed settings for PrimeBeaker rubric judges.

The catalog is independent of the judge HTTP service. A judge worker can use
load_judge_model_profile to select the correct template and rollout settings
from the exact LiteRegistry model path in a request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_RESOURCE_ROOT = Path(__file__).resolve().parent / "resources"


@dataclass(frozen=True)
class JudgeModelProfile:
    """One validated judge-model profile from the packaged catalog."""

    model: str
    source_path: Path
    prompt_template_path: Path
    settings: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view for the CLI and launch tooling."""
        return {
            **self.settings,
            "model": self.model,
            "source_path": str(self.source_path),
            "prompt_template_path": str(self.prompt_template_path),
        }


def judge_profile_catalog_dir() -> Path:
    """Return the installed catalog directory shipped with PrimeBeaker."""
    return _RESOURCE_ROOT / "judge_profiles"


def _load_profile_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid judge profile JSON {path}: {error.msg}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"judge profile {path} must contain a JSON object")
    return payload


def list_judge_model_profiles(
    profile_dir: str | Path | None = None,
) -> list[JudgeModelProfile]:
    """Load every profile from a catalog, rejecting invalid template paths."""
    directory = (
        judge_profile_catalog_dir()
        if profile_dir is None
        else Path(profile_dir).expanduser().resolve()
    )
    if not directory.is_dir():
        raise ValueError(f"judge profile catalog does not exist: {directory}")

    profiles: list[JudgeModelProfile] = []
    for source_path in sorted(directory.glob("*.json")):
        payload = _load_profile_payload(source_path)
        model = payload.get("model")
        if not isinstance(model, str) or not model.strip():
            raise ValueError(f"judge profile {source_path} has no non-empty model")
        template = payload.get("prompt_template_path")
        if not isinstance(template, str) or not template.strip():
            raise ValueError(
                f"judge profile {source_path} has no non-empty prompt_template_path"
            )
        prompt_template_path = Path(template)
        if not prompt_template_path.is_absolute():
            prompt_template_path = source_path.parent / prompt_template_path
        prompt_template_path = prompt_template_path.resolve()
        if not prompt_template_path.is_file():
            raise ValueError(
                f"judge profile template does not exist: {prompt_template_path}"
            )
        profiles.append(
            JudgeModelProfile(
                model=model.strip(),
                source_path=source_path,
                prompt_template_path=prompt_template_path,
                settings=dict(payload),
            )
        )
    return profiles


def load_judge_model_profile(
    model: str,
    profile_dir: str | Path | None = None,
) -> JudgeModelProfile:
    """Resolve one model path to exactly one packaged judge profile."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("judge model must be a non-empty model path")
    selected = [
        profile
        for profile in list_judge_model_profiles(profile_dir)
        if profile.model == model.strip()
    ]
    if not selected:
        directory = (
            judge_profile_catalog_dir()
            if profile_dir is None
            else Path(profile_dir).expanduser().resolve()
        )
        raise ValueError(
            f"no judge profile found for {model!r} in catalog {directory}"
        )
    if len(selected) > 1:
        paths = ", ".join(str(profile.source_path) for profile in selected)
        raise ValueError(f"multiple judge profiles found for {model!r}: {paths}")
    return selected[0]


class JudgeCatalogCLI:
    """Inspect the judge-profile catalog bundled with PrimeBeaker."""

    def path(self) -> str:
        """Print the packaged catalog directory for a judge worker."""
        return str(judge_profile_catalog_dir())

    def list(self) -> list[dict[str, Any]]:
        """List the catalog's registered model paths and source files."""
        return [
            {"model": profile.model, "source_path": str(profile.source_path)}
            for profile in list_judge_model_profiles()
        ]

    def show(self, model: str) -> dict[str, Any]:
        """Render the profile selected by an exact model path."""
        return load_judge_model_profile(model).as_dict()
