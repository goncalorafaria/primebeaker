"""Adapt a Sloth submit-terminated episode to JudgeAPI's per-rubric result contract."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def bundle_rows(rows: list[Any]) -> Any:
    """Keep original rubric IDs when retrying only missing submissions."""
    if not rows:
        raise ValueError("cannot bundle an empty rubric list")
    row = deepcopy(rows[0])
    row.source.misc = {
        **(row.source.misc or {}),
        "indexed_rubrics": [
            {"rubric_id": str(item.rubric_index), "criteria": item.rubric.criteria}
            for item in rows
        ],
    }
    return row


def expand_submissions(results: list[dict[str, Any]], rows: list[Any]) -> list[dict[str, Any]]:
    """Never mistake the last submission for the verdict on every rubric."""
    finals = [result["final_row"] for result in results if result.get("final_row") is not None]
    if len(finals) != 1:
        raise RuntimeError(f"expected one Sloth harness episode, received {len(finals)}")
    final = finals[0]
    metadata = final.rubric.workflow_metadata or {}
    submissions = metadata.get("submission_judgments") or {}
    expected = {str(row.rubric_index) for row in rows}
    if not isinstance(submissions, dict) or set(submissions) - expected:
        raise RuntimeError("Sloth returned submissions for unexpected rubric IDs")
    expanded = []
    for row in rows:
        result = deepcopy(final)
        result.rubric_index = row.rubric_index
        result.rubric.criteria = row.rubric.criteria
        submission = submissions.get(str(row.rubric_index), {})
        result.rubric.label = submission.get("label")
        result.rubric.feedback = submission.get("feedback")
        expanded.append({"final_row": result})
    return expanded
