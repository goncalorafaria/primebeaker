from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

from primebeaker import service_images
from primebeaker.cli import run


def test_installer_builds_and_publishes_podman_stack(
    monkeypatch, tmp_path: Path
) -> None:
    source = tmp_path / "literegistry_podman_beaker"
    script = source / "scripts" / "build-images.sh"
    script.parent.mkdir(parents=True)
    script.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    commands: list[list[str]] = []
    created = iter(["01REDIS", "01GATEWAY", "01PODMAN", "01MIRROR"])

    def fake_run(command: list[str], **kwargs) -> CompletedProcess[str]:
        commands.append(command)
        if command[:3] == ["docker", "image", "inspect"]:
            tag = command[-1]
            return CompletedProcess(command, 0, stdout=f"sha256:{tag}\n")
        if command[:3] == ["beaker", "image", "create"]:
            return CompletedProcess(
                command,
                0,
                stdout=json.dumps({"id": next(created)}),
            )
        return CompletedProcess(command, 0, stdout="")

    monkeypatch.setattr(service_images.shutil, "which", lambda name: f"/bin/{name}")
    monkeypatch.setattr(service_images, "_run", fake_run)
    monkeypatch.setattr(
        service_images, "_distribution_version", lambda name: "0.2.13"
    )

    result = service_images.install_literegistry_images(
        workspace="ai2/test",
        stack="podman",
        source_root=str(source),
    )

    assert result == {
        "workspace": "ai2/test",
        "stacks": {
            "podman": {
                "distribution": "literegistry-podman-beaker",
                "version": "0.2.13",
                "launcher_args": {
                    "redis_image": "01REDIS",
                    "gateway_image": "01GATEWAY",
                    "podman_image": "01PODMAN",
                    "docker_mirror_image": "01MIRROR",
                },
            }
        },
    }
    assert commands[:3] == [
        ["docker", "version"],
        ["beaker", "account", "whoami"],
        ["bash", str(script), "", "0.2.13"],
    ]
    create_commands = [
        command
        for command in commands
        if command[:3] == ["beaker", "image", "create"]
    ]
    assert len(create_commands) == 4
    assert create_commands[0][-6:] == [
        "--name",
        "literegistry-redis-podman-0-2-13",
        "--workspace",
        "ai2/test",
        "--format",
        "json",
    ]


def test_fire_exposes_literegistry_image_install(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_install(**options):
        seen.update(options)
        return {"installed": True}

    monkeypatch.setattr(service_images, "install_literegistry_images", fake_install)

    result = run(
        [
            "services",
            "images",
            "install",
            "--workspace=ai2/test",
            "--stack=podman",
        ]
    )

    assert result == {"installed": True}
    assert seen == {
        "workspace": "ai2/test",
        "stack": "podman",
        "docker_bin": "docker",
        "pull_bases": True,
        "source_root": None,
        "build_local_search": False,
        "jtc_build_context": None,
    }


def test_installer_rejects_missing_local_search_context() -> None:
    try:
        service_images.install_literegistry_images(
            workspace="ai2/test",
            stack="base",
            build_local_search=True,
        )
    except ValueError as error:
        assert str(error) == "jtc_build_context is required with build_local_search"
    else:
        raise AssertionError("missing JTC context should fail before external commands")
