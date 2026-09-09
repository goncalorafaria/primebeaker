"""Metric-only single-turn validation for multiple-choice datasets.

This environment deliberately contributes no reward.  It reuses the proven
last-standalone-option A--J metric from ``jtc_reward_model_env`` so Prime-RL
can report ground-truth accuracy independently of the training judge reward.
"""

from __future__ import annotations

from pathlib import Path

from datasets import load_dataset
import verifiers as vf

from .jtc_reward_model_env import MultipleChoiceAccuracyRubric, _load_dataset


def load_environment(
    dataset: str,
    split: str = "validation",
) -> vf.SingleTurnEnv:
    """Load rows containing ``prompt`` and gold ``answer`` fields."""
    path = Path(dataset)
    rows = (
        load_dataset(str(path), split=split)
        if path.is_dir()
        else _load_dataset(dataset, split)
    )
    return vf.SingleTurnEnv(
        dataset=rows,
        rubric=MultipleChoiceAccuracyRubric(),
    )


__all__ = ["load_environment"]
