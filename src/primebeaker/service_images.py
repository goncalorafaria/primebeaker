"""Build and publish the LiteRegistry images used by PrimeBeaker services.

Deployment companion wheels contain launchers but not Docker build contexts.
This installer downloads matching source distributions, builds the upstream
Dockerfiles, and imports the resulting images into Beaker.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
from typing import Any


@dataclass(frozen=True)
class _Stack:
    distribution: str
    source_directory: str
    images: tuple[tuple[str, str], ...]


_STACKS = {
    "base": _Stack(
        "literegistry-base-deployment",
        "literegistry_base_deployment",
        (
            ("redis_image", "literegistry-redis"),
            ("services_image", "literegistry-base-services"),
            ("terminal_image", "literegistry-base-terminal"),
            ("vllm_image", "literegistry-base-vllm"),
        ),
    ),
    "podman": _Stack(
        "literegistry-podman-beaker",
        "literegistry_podman_beaker",
        (
            ("redis_image", "literegistry-redis"),
            ("gateway_image", "literegistry-podman-gateway"),
            ("podman_image", "literegistry-podman-server"),
            ("docker_mirror_image", "literegistry-docker-mirror"),
        ),
    ),
}


def _run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, **kwargs)


def _distribution_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError as error:
        raise RuntimeError(
            f"{distribution} is not installed; install primebeaker[runtime] first"
        ) from error


def _safe_unpack(archive_path: Path, destination: Path) -> Path:
    destination = destination.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError(f"unsafe path in {archive_path.name}: {member.name}")
        archive.extractall(destination, members=members, filter="data")
    roots = [path for path in destination.iterdir() if path.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"expected one source root in {archive_path.name}")
    return roots[0]


def _download_source(stack: _Stack, package_version: str, destination: Path) -> Path:
    download = destination / "download"
    unpacked = destination / "source"
    download.mkdir(parents=True)
    unpacked.mkdir()
    pip = shutil.which("pip")
    if pip is None:
        raise RuntimeError(
            "a pip executable is required to download LiteRegistry source distributions; "
            "pass source_root to use a local checkout instead"
        )
    _run(
        [
            pip,
            "download",
            "--no-deps",
            "--no-binary",
            ":all:",
            "--dest",
            str(download),
            f"{stack.distribution}=={package_version}",
        ]
    )
    archives = list(download.glob("*.tar.gz"))
    if len(archives) != 1:
        raise RuntimeError(
            f"expected one source archive for {stack.distribution}, found {len(archives)}"
        )
    return _safe_unpack(archives[0], unpacked)


def _checkout_source(stack: _Stack, source_root: Path) -> Path:
    candidates = [source_root / stack.source_directory]
    if source_root.name == stack.source_directory:
        candidates.insert(0, source_root)
    for candidate in candidates:
        if (candidate / "scripts" / "build-images.sh").is_file():
            return candidate
    raise RuntimeError(
        f"could not find {stack.source_directory}/scripts/build-images.sh "
        f"under {source_root}"
    )


def _beaker_id(output: str) -> str:
    try:
        value: Any = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("beaker image create did not return JSON") from error

    def find_id(candidate: Any) -> str | None:
        if isinstance(candidate, dict):
            for key in ("id", "image_id", "imageId"):
                found = candidate.get(key)
                if isinstance(found, str) and found:
                    return found
            for nested in candidate.values():
                if found := find_id(nested):
                    return found
        elif isinstance(candidate, list):
            for nested in candidate:
                if found := find_id(nested):
                    return found
        return None

    if image_id := find_id(value):
        return image_id
    raise RuntimeError("beaker image create JSON did not contain an image ID")


def _image_name(stack_name: str, local_name: str, package_version: str) -> str:
    suffix = package_version.replace(".", "-").replace("+", "-")
    if local_name == "literegistry-redis":
        return f"literegistry-redis-{stack_name}-{suffix}"
    return f"{local_name}-{suffix}"


def _publish_stack(
    *,
    stack_name: str,
    stack: _Stack,
    package_version: str,
    source: Path,
    workspace: str,
    docker_bin: str,
    pull_bases: bool,
    build_local_search: bool,
    jtc_build_context: str | None,
) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        DOCKER_BIN=docker_bin,
        PULL_BASES="1" if pull_bases else "0",
        BUILD_LOCAL_SEARCH="1" if build_local_search else "0",
    )
    if jtc_build_context is not None:
        environment["JTC_BUILD_CONTEXT"] = str(Path(jtc_build_context).resolve())
    _run(
        ["bash", str(source / "scripts" / "build-images.sh"), "", package_version],
        cwd=source,
        env=environment,
    )

    images = list(stack.images)
    if stack_name == "base" and build_local_search:
        images.append(("local_search_image", "jtc-local-search-lucene-bm25"))
    published: dict[str, str] = {}
    for config_name, local_name in images:
        local_tag = f"{local_name}:{package_version}"
        inspected = _run(
            [docker_bin, "image", "inspect", "--format", "{{.Id}}", local_tag],
            capture_output=True,
        )
        docker_id = inspected.stdout.strip()
        if not docker_id:
            raise RuntimeError(f"{docker_bin} returned no image ID for {local_tag}")
        created = _run(
            [
                "beaker",
                "image",
                "create",
                docker_id,
                "--name",
                _image_name(stack_name, local_name, package_version),
                "--workspace",
                workspace,
                "--format",
                "json",
            ],
            capture_output=True,
        )
        published[config_name] = _beaker_id(created.stdout)
    return published


def install_literegistry_images(
    workspace: str,
    stack: str = "all",
    docker_bin: str = "docker",
    pull_bases: bool = True,
    source_root: str | None = None,
    build_local_search: bool = False,
    jtc_build_context: str | None = None,
) -> dict[str, Any]:
    """Build and upload the LiteRegistry Beaker service images."""

    if stack not in {"all", *_STACKS}:
        raise ValueError("stack must be one of: all, base, podman")
    if not workspace.strip():
        raise ValueError("workspace must be non-empty")
    if build_local_search and stack == "podman":
        raise ValueError("local search belongs to the base stack")
    if build_local_search and not jtc_build_context:
        raise ValueError("jtc_build_context is required with build_local_search")
    for executable in (docker_bin, "beaker"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable was not found: {executable}")
    _run([docker_bin, "version"], capture_output=True)
    _run(["beaker", "account", "whoami"], capture_output=True)

    selected = tuple(_STACKS) if stack == "all" else (stack,)
    result: dict[str, Any] = {"workspace": workspace, "stacks": {}}
    with tempfile.TemporaryDirectory(prefix="primebeaker-literegistry-images-") as raw:
        temporary_root = Path(raw)
        for stack_name in selected:
            stack_config = _STACKS[stack_name]
            package_version = _distribution_version(stack_config.distribution)
            if source_root is None:
                source = _download_source(
                    stack_config, package_version, temporary_root / stack_name
                )
            else:
                source = _checkout_source(stack_config, Path(source_root).resolve())
            images = _publish_stack(
                stack_name=stack_name,
                stack=stack_config,
                package_version=package_version,
                source=source,
                workspace=workspace,
                docker_bin=docker_bin,
                pull_bases=pull_bases,
                build_local_search=build_local_search and stack_name == "base",
                jtc_build_context=jtc_build_context,
            )
            result["stacks"][stack_name] = {
                "distribution": stack_config.distribution,
                "version": package_version,
                "launcher_args": images,
            }
    return result


class LiteRegistryImageInstaller:
    """Fire command group for LiteRegistry service-image provisioning."""

    @staticmethod
    def install(
        workspace: str,
        stack: str = "all",
        docker_bin: str = "docker",
        pull_bases: bool = True,
        source_root: str | None = None,
        build_local_search: bool = False,
        jtc_build_context: str | None = None,
    ) -> dict[str, Any]:
        """Build and upload LiteRegistry service images to a Beaker workspace."""

        return install_literegistry_images(
            workspace=workspace,
            stack=stack,
            docker_bin=docker_bin,
            pull_bases=pull_bases,
            source_root=source_root,
            build_local_search=build_local_search,
            jtc_build_context=jtc_build_context,
        )
