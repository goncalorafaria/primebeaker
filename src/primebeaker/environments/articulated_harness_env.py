"""Submit-terminated Verifiers harness for articulated rubric judging."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import verifiers as vf

from literegistry_tool_client.submission import SubmissionStore
from .jtc_tool_label_env import JTCToolLabelEnv
from . import jtc_tool_label_env as _base


class JTCArticulatedHarnessEnv(JTCToolLabelEnv):
    """End an episode as soon as every expected rubric has been submitted.

    JTCToolLabelEnv remains the ordinary final-JSON harness. This subclass uses
    the same terminal/search/submit dispatch, but the final successful submit
    response is the environment's final response; no redundant model turn or
    final JSON object is requested.
    """

    def __init__(
        self,
        *,
        rubric_completion_reward_coef: float = 1.0,
        missing_rubric_penalty_coef: float = 1.0,
        **kwargs: Any,
    ) -> None:
        if rubric_completion_reward_coef < 0:
            raise ValueError("rubric_completion_reward_coef must be non-negative")
        if missing_rubric_penalty_coef < 0:
            raise ValueError("missing_rubric_penalty_coef must be non-negative")
        self.rubric_completion_reward_coef = float(rubric_completion_reward_coef)
        self.missing_rubric_penalty_coef = float(missing_rubric_penalty_coef)
        # A submit-complete episode deliberately has no final JSON response.
        kwargs.setdefault("final_response_required", False)
        super().__init__(**kwargs)

        def gold_labels(state: vf.State) -> dict[str, str]:
            task = state.get("task")
            raw = task.get("gold_submissions") if isinstance(task, Mapping) else None
            if not isinstance(raw, list):
                return {}
            result: dict[str, str] = {}
            for item in raw:
                if not isinstance(item, Mapping):
                    continue
                rubric_id = str(item.get("rubric_id") or "").strip()
                label = str(item.get("label") or "").strip().casefold()
                if rubric_id and label in {"pass", "fail"}:
                    result[rubric_id] = label
            return result

        async def rubric_completion_reward(state: vf.State) -> float:
            """Give +1/N only for each submission matching its hidden gold label."""
            store = state.get("jtc_submit_session")
            if not isinstance(store, SubmissionStore):
                return 0.0
            expected = len(store.expected_rubric_ids)
            if expected == 0:
                return 0.0
            gold = gold_labels(state)
            if not gold:
                # Retain the prior behavior for old evaluation-only rows that
                # have no per-rubric gold labels.
                return self.rubric_completion_reward_coef * len(store.completed_rubric_ids) / expected
            submissions = store.as_dict()
            correct = sum(
                str(submissions.get(rubric_id, {}).get("label") or "").strip().casefold() == gold.get(rubric_id)
                for rubric_id in store.expected_rubric_ids
            )
            return self.rubric_completion_reward_coef * correct / expected

        async def missing_rubric_penalty(state: vf.State) -> float:
            """Subtract -1/N for every rubric the model did not submit."""
            store = state.get("jtc_submit_session")
            if not isinstance(store, SubmissionStore) or not gold_labels(state):
                return 0.0
            expected = len(store.expected_rubric_ids)
            if expected == 0:
                return 0.0
            return -self.missing_rubric_penalty_coef * len(store.waiting_rubric_ids) / expected

        self.rubric.add_reward_func(rubric_completion_reward)
        self.rubric.add_reward_func(missing_rubric_penalty)

    @vf.stop(priority=75)
    async def all_rubrics_submitted(self, state: vf.State) -> bool:
        store = state.get("jtc_submit_session")
        return isinstance(store, SubmissionStore) and store.all_completed

    async def env_response(
        self,
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> vf.Messages:
        responses = await super().env_response(messages, state, **kwargs)
        store = state.get("jtc_submit_session")
        if isinstance(store, SubmissionStore) and store.all_completed:
            # MultiTurnEnv checks stop conditions before producing a model
            # completion, but env_response runs afterward while assembling the
            # next prompt. final_env_response tells that loop to skip the next
            # model call; all_rubrics_submitted then supplies the stop reason.
            state["jtc_submit_complete"] = True
            state["jtc_submit_submissions"] = store.as_dict()
            state["final_env_response"] = responses
        return responses


def load_environment(
    *args: Any,
    rubric_completion_reward_coef: float = 1.0,
    missing_rubric_penalty_coef: float = 1.0,
    **kwargs: Any,
) -> JTCArticulatedHarnessEnv:
    """PrimeRL factory for gold-scored submit-terminated rubric episodes."""
    original = _base.JTCToolLabelEnv

    def factory(**environment_kwargs: Any) -> JTCArticulatedHarnessEnv:
        return JTCArticulatedHarnessEnv(
            **environment_kwargs,
            rubric_completion_reward_coef=rubric_completion_reward_coef,
            missing_rubric_penalty_coef=missing_rubric_penalty_coef,
        )

    _base.JTCToolLabelEnv = factory  # type: ignore[assignment]
    try:
        return _base.load_environment(*args, **kwargs)
    finally:
        _base.JTCToolLabelEnv = original


__all__ = ["JTCArticulatedHarnessEnv", "load_environment"]
