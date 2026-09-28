"""verifiers v1 taskset plugin entry point: ``taskset.id = "primebeaker-mc-accuracy"`` (metric-only eval)."""

from primebeaker.environments.rubrichub_judge_v1 import (  # noqa: F401
    MultipleChoiceAccuracyTask,
    MultipleChoiceAccuracyTaskset,
    MultipleChoiceAccuracyTasksetConfig,
)

__all__ = ["MultipleChoiceAccuracyTask", "MultipleChoiceAccuracyTaskset", "MultipleChoiceAccuracyTasksetConfig"]
