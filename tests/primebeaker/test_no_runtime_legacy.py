from pathlib import Path

import primebeaker


def test_runtime_has_no_legacy_client_or_submission_facades() -> None:
    package_root = Path(primebeaker.__file__).parent

    assert not (package_root / "runtime/code_execution.py").exists()
    assert not (package_root / "runtime/submission.py").exists()


def test_primebeaker_has_no_jtcflow_dependency_or_import() -> None:
    package_root = Path(primebeaker.__file__).parent
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.rglob("*.py")
    )
    project = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")

    assert "jtcflow" not in source
    assert "jtcflow" not in project
