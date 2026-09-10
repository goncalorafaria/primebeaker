from __future__ import annotations

from pathlib import Path

from literegistry_tool_client import (
    FetchClient,
    JudgeClient,
    SearchClient,
    TerminalExecutionClient,
    WebTerminalExecutionClient,
)
from primebeaker.config import RLTrainingToml
from primebeaker.environments import ENVIRONMENTS, load_environment_module
from primebeaker.images import default_image_uri, image_locations
from primebeaker.runtime.templates import resolve_resource_path


def test_all_environment_implementations_import_from_primebeaker() -> None:
    modules = {load_environment_module(name).__name__ for name in ENVIRONMENTS}

    assert len(modules) == 16
    assert all(name.startswith("primebeaker.environments.") for name in modules)


def test_clients_are_owned_by_literegistry_tool_client() -> None:
    expected_modules = {
        FetchClient: "literegistry_tool_client.fetch",
        JudgeClient: "literegistry_tool_client.judge",
        SearchClient: "literegistry_tool_client.search",
        TerminalExecutionClient: "literegistry_tool_client.terminal",
        WebTerminalExecutionClient: "literegistry_tool_client.webterminal",
    }
    for client, module_name in expected_modules.items():
        assert client.__module__ == module_name


def test_primebeaker_does_not_package_client_facades() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "primebeaker"

    assert not (package_root / "client").exists()
    assert not (package_root / "clients.py").exists()


def test_packaged_templates_resolve_inside_primebeaker(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    path = resolve_resource_path("templates/search-agent.json")

    assert path.is_file()
    assert "primebeaker/resources/templates" in str(path)


def test_environment_namespace_migration_changes_only_bundled_modules(
    tmp_path: Path,
) -> None:
    source = RLTrainingToml(
        template_path=tmp_path / "source.toml",
        values={
            "orchestrator": {
                "train": {
                    "source": [
                        {
                            "legacy": {
                                "id": "old.package.jtc_tool_label_env",
                                "args": {},
                            }
                        }
                    ]
                },
                "eval": {
                    "source": [
                        {
                            "legacy": {
                                "id": "third_party.environments.custom_env",
                                "args": {},
                            }
                        }
                    ]
                },
            }
        },
    )

    migrated = source.with_environment_namespace()

    assert (
        migrated.values["orchestrator"]["train"]["source"][0]["legacy"]["id"]
        == "primebeaker.environments.jtc_tool_label_env"
    )
    assert (
        migrated.values["orchestrator"]["eval"]["source"][0]["legacy"]["id"]
        == "third_party.environments.custom_env"
    )
    assert source.values["orchestrator"]["train"]["source"][0]["legacy"][
        "id"
    ].startswith("old.package")


def test_catalog_selects_immutable_image_by_workspace() -> None:
    locations = image_locations()

    assert len(locations) >= 2
    assert (
        default_image_uri(workspace="ai2/oe-agents")
        == "beaker://01KZVDND2PYP538F5JSGJ469EC"
    )
    assert (
        default_image_uri(workspace="ai2/oe-agents-holmes")
        == "beaker://01M0E15WYCV7T0J1CMCFPBQ21P"
    )
    assert all(location.immutable_uri.startswith("beaker://") for location in locations)


def test_non_evaluation_runtime_has_no_jtc_imports() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "primebeaker"
    forbidden = "jtc.common"

    offenders = []
    evaluation_boundary = {"jtc_evaluation.py", "evaluation_assets.py"}
    for path in package_root.rglob("*.py"):
        if path.name in evaluation_boundary:
            continue
        if forbidden in path.read_text(encoding="utf-8"):
            offenders.append(path.relative_to(package_root))
    assert offenders == []
