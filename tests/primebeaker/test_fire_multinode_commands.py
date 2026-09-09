from pathlib import Path

from primebeaker import multinode


def test_fire_dispatches_multinode_prepare_commands(
    tmp_path: Path, monkeypatch
) -> None:
    rl_call: list[tuple[Path, Path, Path]] = []
    sft_call: list[tuple[Path, Path, Path]] = []
    monkeypatch.setattr(
        multinode,
        "prepare_rl_runtime",
        lambda config, config_dir, runtime_dir: rl_call.append(
            (config, config_dir, runtime_dir)
        ),
    )
    monkeypatch.setattr(
        multinode,
        "prepare_sft_runtime",
        lambda config, output, runtime_dir: sft_call.append(
            (config, output, runtime_dir)
        ),
    )

    assert multinode.main(
        [
            "prepare-rl",
            f"--config={tmp_path / 'rl.toml'}",
            f"--config-dir={tmp_path / 'config'}",
            f"--runtime-dir={tmp_path / 'runtime'}",
        ]
    ) == 0
    assert multinode.main(
        [
            "prepare-sft",
            f"--config={tmp_path / 'sft.toml'}",
            f"--output={tmp_path / 'rendered.toml'}",
            f"--runtime-dir={tmp_path / 'runtime'}",
        ]
    ) == 0

    assert rl_call == [
        (tmp_path / "rl.toml", tmp_path / "config", tmp_path / "runtime")
    ]
    assert sft_call == [
        (tmp_path / "sft.toml", tmp_path / "rendered.toml", tmp_path / "runtime")
    ]


def test_role_scripts_use_fire_visible_prepare_commands() -> None:
    package_root = Path(multinode.__file__).parent

    assert "primebeaker.multinode prepare-rl" in (
        package_root / "multinode_rl_role.sh"
    ).read_text(encoding="utf-8")
    assert "primebeaker.multinode prepare-sft" in (
        package_root / "multinode_sft_role.sh"
    ).read_text(encoding="utf-8")
