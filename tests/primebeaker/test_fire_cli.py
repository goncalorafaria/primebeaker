from __future__ import annotations

from pathlib import Path

from primebeaker import coordination, gateway, multinode
from primebeaker.environments.__main__ import main as environments_main
from primebeaker.images.__main__ import main as images_main
from primebeaker.cli import run as cli_run


def test_primebeaker_has_no_argparse_implementation() -> None:
    package_root = Path(__file__).resolve().parents[2] / "src" / "primebeaker"
    offenders = [
        path.relative_to(package_root)
        for path in package_root.rglob("*.py")
        if "argparse" in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_environment_fire_command_resolves_an_environment(
    capsys,
) -> None:
    result = environments_main(["tool-label"])

    assert result == (
        "primebeaker.environments.jtc_tool_label_env:load_environment"
    )
    assert result in capsys.readouterr().out


def test_image_fire_list_command(capsys) -> None:
    result = images_main(["list"])

    assert isinstance(result, list)
    assert result
    assert "immutable_uri" in capsys.readouterr().out


def test_coordination_fire_command_accepts_json(capsys) -> None:
    exit_code = coordination.main(
        [
            "wait-services",
            "--registry=redis://unused:6379",
            "--requirements-json={}",
        ]
    )

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "{}"


def test_multinode_fire_role_dispatch(monkeypatch) -> None:
    called: list[str] = []
    monkeypatch.setattr(multinode, "_role", called.append)

    assert multinode.main(["role", "rl"]) == 0
    assert called == ["rl"]


def test_gateway_fire_root_function(monkeypatch) -> None:
    called: dict[str, object] = {}

    def fake_serve(registry: str, port: int) -> None:
        called.update(registry=registry, port=port)

    monkeypatch.setattr(gateway, "serve", fake_serve)

    gateway.main(["--registry=redis://registry:6379", "--port=1212"])
    assert called == {"registry": "redis://registry:6379", "port": 1212}


def test_watcher_fire_command_is_nested_under_primebeaker(
    tmp_path: Path,
) -> None:
    result = cli_run(
        [
            "watcher",
            "status",
            f"--database={tmp_path / 'watcher.sqlite3'}",
        ]
    )

    assert result["exists"] is False
