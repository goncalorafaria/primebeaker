import inspect
from pathlib import Path

import primebeaker
from literegistry_tool_client import RemoteCodeExecutionClient


def test_legacy_record_schema_and_processors_are_not_packaged() -> None:
    package_root = Path(primebeaker.__file__).parent

    for relative_path in (
        "runtime/schema.py",
        "runtime/process.py",
        "runtime/rubric_execution.py",
    ):
        assert not (package_root / relative_path).exists()


def test_legacy_record_types_and_execution_helpers_are_absent() -> None:
    package_root = Path(primebeaker.__file__).parent
    forbidden = (
        "ExecutableRubric",
        "ExecutableRubricJudgment",
        "JRecord",
        "JRow",
        "CodeExecutionMap",
        "execute_record_async",
        "process_executable_record_async",
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in package_root.rglob("*.py")
    )

    assert all(name not in source for name in forbidden)


def test_code_client_has_only_direct_execution_interface() -> None:
    constructor = inspect.signature(RemoteCodeExecutionClient)

    assert "code_output_truncation" not in constructor.parameters
    assert not hasattr(RemoteCodeExecutionClient, "invoke")
