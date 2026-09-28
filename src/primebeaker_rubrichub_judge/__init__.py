"""verifiers v1 taskset plugin entry point: ``taskset.id = "primebeaker-rubrichub-judge"``."""

from primebeaker.environments.rubrichub_judge_v1 import (  # noqa: F401
    RubricHubJudgeState,
    RubricHubJudgeTask,
    RubricHubJudgeTaskConfig,
    RubricHubJudgeTaskset,
    RubricHubJudgeTasksetConfig,
)

# v1 requires exactly one Taskset subclass in __all__ (the MC eval taskset is primebeaker_mc_accuracy).
__all__ = [
    "RubricHubJudgeState",
    "RubricHubJudgeTask",
    "RubricHubJudgeTaskConfig",
    "RubricHubJudgeTaskset",
    "RubricHubJudgeTasksetConfig",
]
