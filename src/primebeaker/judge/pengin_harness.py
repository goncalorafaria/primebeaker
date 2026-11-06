"""Pengin's benchmark harness exposed through the LiteRegistry JudgeAPI.

Each rubric gets an independent episode. The evaluated response stays in tool
stdin, and only an accepted submit tool result can become an API judgment.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from jtc.common.templates import format_rubric_dict, render_chat_template
from jtc.common.workflows import verifier_tool_use_workflow as base
from jtc.common.workflows.verifier_submit_tool_use_workflow import SubmitToolUseWorkflow


PROMPT_SHA256 = "c4c138d6bd91b2800a53eeaaee7f454078f410fa3156733133aeedae891dcf30"
FINAL_WARNING = (
    "This was your last allowed tool call. Record any remaining rubric judgment "
    "using submit; no final JSON answer is required."
)


def validate_pengin_template(path: str | Path) -> None:
    """Reject accidental prompt/tool-schema drift from the evaluated release."""
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != PROMPT_SHA256:
        raise ValueError("Pengin prompt/tool fingerprint differs from the benchmark harness")


class PenginToolUseWorkflow(SubmitToolUseWorkflow):
    """One criterion per rollout, with strict accepted-submit finalization."""

    def _seed(self, data: dict[str, Any]) -> dict[str, Any]:
        row = data["row"]
        rubric_id = str(row.rubric_index if row.rubric_index is not None else row.id)
        messages = render_chat_template(
            self.prompt_chat_template_spec,
            inputs=row.input or "",
            indexed_rubrics=json.dumps(
                [{"rubric_id": rubric_id, "criteria": row.rubric.criteria}],
                ensure_ascii=False,
                indent=2,
            ),
            rubric=format_rubric_dict(row.rubric.scores),
        )
        return {**data, base.MESSAGES_KEY: messages, "expected_rubric_ids": [rubric_id]}

    def _attach_final_row(self, data: dict[str, Any]) -> dict[str, Any]:
        rollout = data.get("__jtc_rollout") or {}
        rubric_id = data["expected_rubric_ids"][0]
        submissions = rollout.get("jtc_submit_submissions") or {}
        submission = submissions.get(rubric_id) if isinstance(submissions, dict) else None
        if (
            not rollout.get("error")
            and rollout.get("jtc_submit_complete")
            and isinstance(submissions, dict)
            and set(submissions) == {rubric_id}
            and isinstance(submission, dict)
            and submission.get("label") in (data["row"].rubric.scores or {})
            and isinstance(submission.get("feedback"), str)
            and submission["feedback"].strip()
        ):
            return super()._attach_final_row(data)

        # Do not promote final JSON or unaccepted tool-call arguments to labels.
        result = base.ToolUseWorkflow._attach_final_row(self, data)
        final = result["final_row"]
        final.rubric.label = None
        final.rubric.feedback = None
        final.rubric.workflow_metadata = {
            **(final.rubric.workflow_metadata or {}),
            "route": "invalid",
            "submission_complete": False,
        }
        result.update(route="invalid", label=None, feedback=None)
        return result


def configure_pengin_verifier(verifier: Any, tool_defs: list[dict[str, Any]]) -> None:
    """Match benchmark tool names, final warning, and transient rollout retries."""
    if verifier.tool_defs != tool_defs or set(verifier.tool_names) != {"trace_inspection", "submit"}:
        raise ValueError("Pengin verifier tools do not match the SFT contract")
    verifier.final_warning = FINAL_WARNING
    native_rollout = verifier.rollout

    async def reliable_rollout(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(3):
            state = await native_rollout(*args, **kwargs)
            error = state.get("error")
            chain: list[str] = []
            seen: set[int] = set()
            while error is not None and id(error) not in seen:
                seen.add(id(error))
                chain.append(repr(error))
                error = getattr(error, "__cause__", None) or getattr(error, "__context__", None)
            detail = " -> ".join(chain)
            transient = any(
                token in detail.lower()
                for token in ("timeout", "connectionerror", "connecterror", "connection reset", "serviceunavailable")
            )
            if not transient or attempt == 2:
                if detail:
                    state["error"] = detail
                return state
            await asyncio.sleep(2 * (attempt + 1))

    verifier.rollout = reliable_rollout
