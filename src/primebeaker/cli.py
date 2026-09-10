"""Fire command-line interface for PrimeBeaker training launches."""

from __future__ import annotations

import json
from functools import partial
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import fire

from primebeaker.images import default_image_uri
from primebeaker.judge_catalog import JudgeCatalogCLI
from primebeaker.multinode import (
    BeakerMultiNodeRLBackend,
    BeakerMultiNodeSFTBackend,
    MultiNodeRLLaunchRequest,
    MultiNodeSFTLaunchRequest,
    deployment_type,
)
from primebeaker.rl import BeakerRLBackend, RLLaunchRequest
from primebeaker.resume import resume
from primebeaker.services import LiteRegistryServices
from primebeaker.sft import BeakerSFTBackend, SFTLaunchRequest


def _items(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _mapping(
    values: str | Sequence[str] | None,
    *,
    option: str,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in _items(values):
        name, separator, item = value.partition("=")
        if not separator or not name or not item:
            raise ValueError(f"{option} expects NAME=VALUE, got {value!r}")
        result[name] = item
    return result


def _counts(
    values: str | Sequence[str] | None,
    *,
    option: str,
) -> dict[str, int]:
    raw = _mapping(values, option=option)
    result: dict[str, int] = {}
    for name, value in raw.items():
        try:
            count = int(value)
        except ValueError as error:
            raise ValueError(
                f"{option} expects NAME=COUNT, got {name}={value!r}"
            ) from error
        if count < 1:
            raise ValueError(f"{option} counts must be positive")
        result[name] = count
    return result


def _launch(
    program: str,
    action: str,
    *,
    toml: str | Path,
    image: str | None = None,
    workspace: str = "ai2/oe-agents",
    cluster: str | Sequence[str] | None = None,
    priority: str = "high",
    min_runtime_hours: int = 0,
    mount_path: str | Path = "/weka",
    dataset: str = "oe-adapt-default",
    working_dir: str | Path = "/weka",
    scratch_dir: str | Path = "/weka",
    home_dir: str | Path = "/weka",
    wandb_secret: str | None = "WANDB_API_KEY",
    hf_secret: str | None = "HF_TOKEN",
    wandb_run_id: str | None = None,
    env: str | Sequence[str] | None = None,
    secret: str | Sequence[str] | None = None,
    setup_command: str | None = None,
    budget: str | None = None,
    description: str | None = None,
    rendezvous_timeout_seconds: int = 1800,
    registry: str | None = None,
    wandb_entity: str | None = None,
    gateway_port: int = 1212,
    gateway_workers: int = 8,
    required_service: str | Sequence[str] | None = None,
    write_spec: bool = False,
) -> dict[str, Any]:
    """Create, preview, or submit one SFT/RL launch request."""
    if priority not in {"normal", "high", "urgent"}:
        raise ValueError("priority must be normal, high, or urgent")
    toml_path = Path(toml)
    topology = deployment_type(toml_path)
    multi_node = topology == "multi_node"
    if multi_node:
        if program == "rl":
            if not registry:
                raise ValueError("multi-node RL requires --registry=redis://...")
            request_type = MultiNodeRLLaunchRequest
            backend = BeakerMultiNodeRLBackend()
        else:
            request_type = MultiNodeSFTLaunchRequest
            backend = BeakerMultiNodeSFTBackend()
    elif topology == "single_node":
        request_type = SFTLaunchRequest if program == "sft" else RLLaunchRequest
        backend = BeakerSFTBackend() if program == "sft" else BeakerRLBackend()
    else:
        raise ValueError(f"unsupported deployment.type {topology!r}")

    request_values: dict[str, Any] = {
        "image": image or default_image_uri(workspace=workspace),
        "workspace": workspace,
        "clusters": tuple(_items(cluster) or ("ai2/holmes",)),
        "priority": priority,
        "min_runtime_hours": int(min_runtime_hours),
        "mount_path": Path(mount_path),
        "dataset": dataset,
        "working_dir": Path(working_dir),
        "scratch_dir": Path(scratch_dir),
        "home_dir": Path(home_dir),
        "wandb_secret": wandb_secret or None,
        "hf_secret": hf_secret or None,
        "environment": _mapping(env, option="--env"),
        "secrets": _mapping(secret, option="--secret"),
        "setup_command": setup_command,
        "budget": budget,
        "description": description,
    }
    if wandb_run_id:
        request_values["wandb_run_id"] = wandb_run_id
    if multi_node:
        request_values["rendezvous_timeout_seconds"] = int(
            rendezvous_timeout_seconds
        )
    if program == "rl":
        request_values["registry"] = registry
        request_values["wandb_entity"] = wandb_entity
        if multi_node:
            request_values["gateway_port"] = int(gateway_port)
            request_values["gateway_workers"] = int(gateway_workers)
            request_values["required_services"] = _counts(
                required_service,
                option="--required-service",
            )

    request = request_type.from_toml(toml_path, **request_values)
    if action == "submit":
        result = backend.submit(request)
    else:
        result = backend.preview(request)
        if write_spec:
            backend.write_preview(result)
    return result.model_dump(mode="json")

class TrainingProgram:
    """Fire command group for one Prime-RL training mode."""

    def __init__(self, program: str) -> None:
        self.preview = partial(_launch, program, "preview")
        self.submit = partial(_launch, program, "submit")
        if program == "rl":
            self.resume = resume


class PrimeBeakerCLI:
    """Launch Prime-RL SFT or RL training on Beaker."""

    def __init__(self) -> None:
        self.sft = TrainingProgram("sft")
        self.rl = TrainingProgram("rl")
        self.services = LiteRegistryServices()
        self.judge = JudgeCatalogCLI()


def _serialize(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return "" if value is None else str(value)


def run(argv: Sequence[str] | None = None) -> Any:
    """Run the Fire CLI, optionally with an explicit argument vector."""
    command = None if argv is None else list(argv)
    return fire.Fire(PrimeBeakerCLI(), command=command, serialize=_serialize)


def main() -> None:
    run()


if __name__ == "__main__":
    main()
