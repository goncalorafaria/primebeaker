"""Single-turn Verifiers environment scored by a remote reward model.

The policy produces one assistant response. The sole rubric sends the original
OpenAI-style prompt plus that response to a vLLM sequence-classification model
through the LiteRegistry classification gateway route.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import verifiers as vf
from verifiers.rubrics.math_rubric import MathRubric
from verifiers.rubrics.rubric_group import RubricGroup
from verifiers.utils.data_utils import extract_boxed_answer

from literegistry_tool_client.reward_model import (
    DEFAULT_CLASSIFY_SERVER_URL as CLASSIFY_SERVER_URL,
    RewardModelClient,
)


REWARD_MODEL_NORMALIZATION_STATS: dict[str, dict[str, float]] = {
    (
        "graf/math_1b_verifymetropolis16_1b_b05-"
        "metropolis16-62e32005-1-195-on-holmes"
    ): {
        "mean": 2.715330360208,
        "std": 1.205339307144,
    },
    (
        "graf/math_1b_verifymetropolis16_1b_b10-"
        "metropolis16-87611899-1-195-on-holmes"
    ): {
        "mean": 2.964056748472,
        "std": 2.285506791875,
    },
    (
        "graf/math_1b_verifymetropolis32_1b_b05-"
        "metropolis32-73352b4f-1-195-on"
    ): {
        "mean": -0.486580200731,
        "std": 1.096135139494,
    },
    (
        "graf/math_1b_verifymetropolis32_1b-"
        "metropolis32-3d7129ad-1-195-on"
    ): {
        "mean": -1.253313497364,
        "std": 1.467700644083,
    },
    (
        "graf/math_1b_verifybt_oracle_1b-"
        "bt_oracle-0609ce76-1-195-on"
    ): {
        "mean": -0.404234390727,
        "std": 1.424969515115,
    },
    (
        "graf/math_1b_verifymetroplis8_1b_b05-"
        "metropolis8-d2ccb3ff-1-195-on"
    ): {
        "mean": -0.376511091230,
        "std": 0.919229167584,
    },
    (
        "graf/math_1b_verifymetroplis8_1b_b10-"
        "metropolis8-39d57079-1-195-on"
    ): {
        "mean": -0.888339553527,
        "std": 1.433448760506,
    },
    "Skywork/Skywork-Reward-V2-Qwen3-4B": {
        "mean": 12.440304755197,
        "std": 2.974110652071,
    },
    (
        "graf/science_4b_mix_bt_oracle_4b-"
        "bt_oracle-a6794831-1-188-on"
    ): {
        "mean": 1.266270907189,
        "std": 1.651864508258,
    },
    (
        "graf/science_4b_mix_bt_8b_solid-"
        "bt-a577c3ef-1-104-on"
    ): {
        "mean": 0.037037406137,
        "std": 1.068665851990,
    },
    (
        "graf/science_4b_mix_metropolis16_8b_solid-"
        "metropolis16-6cbbcf35-1-104-on"
    ): {
        "mean": 2.105808611216424,
        "std": 2.671662943543117,
    },
    (
        "graf/science_4b_mix_metropolis32_4b-"
        "metropolis32-9bb21907-1-188-on"
    ): {
        "mean": 0.687109787629,
        "std": 1.418616492905,
    },
    (
        "graf/science_4b_mix_metropolis32_4b-"
        "metropolis32-3dded240-1-188-on"
    ): {
        "mean": 1.411928435203,
        "std": 2.316247683237,
    },
}


def reward_model_normalization_stats(model_path: str) -> dict[str, float]:
    """Return frozen calibration stats, or an identity transform when unknown."""
    return dict(
        REWARD_MODEL_NORMALIZATION_STATS.get(
            model_path,
            {"mean": 0.0, "std": 1.0},
        )
    )


def reward_model_standard_deviation(model_path: str) -> float:
    """Return the calibrated reward std, or identity scaling when unknown."""
    return reward_model_normalization_stats(model_path)["std"]


class RewardModelOutputClient(Protocol):
    """Minimal classifier contract consumed by the reward rubric."""

    model_path: str

    async def classify(
        self,
        *,
        messages: Sequence[Mapping[str, Any]],
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]: ...


def _message_field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _openai_message(message: Any, *, default_role: str) -> dict[str, Any]:
    """Convert a Verifiers or mapping message to OpenAI chat shape."""
    role = _message_field(message, "role", default_role)
    content = _message_field(message, "content")
    if not isinstance(role, str) or not role.strip():
        role = default_role
    if content is None:
        content = ""
    normalized: dict[str, Any] = {
        "role": role.strip(),
        "content": content,
    }
    name = _message_field(message, "name")
    if isinstance(name, str) and name.strip():
        normalized["name"] = name.strip()
    return normalized


def reward_conversation(prompt: Any, completion: list[Any]) -> list[dict[str, Any]]:
    """Build the exact chat conversation classified by the reward model."""
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    elif isinstance(prompt, Mapping):
        messages = [_openai_message(prompt, default_role="user")]
    elif isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        messages = [
            _openai_message(message, default_role="user")
            for message in prompt
        ]
    else:
        raise TypeError("prompt must be text or OpenAI-style messages")
    if not messages:
        raise ValueError("prompt must not be empty")
    if not completion:
        raise ValueError("completion must contain an assistant message")
    candidate = _openai_message(completion[-1], default_role="assistant")
    candidate["role"] = "assistant"
    return [*messages, candidate]


def positive_class_probability(
    response: Mapping[str, Any],
    positive_class_index: int,
) -> float:
    """Extract one finite class probability from a vLLM classify response."""
    data = response.get("data")
    if not isinstance(data, list) or len(data) != 1:
        raise RuntimeError(
            "reward model response must contain exactly one data item"
        )
    item = data[0]
    if not isinstance(item, Mapping):
        raise RuntimeError("reward model data item must be an object")
    probabilities = item.get("probs")
    if not isinstance(probabilities, list) or not probabilities:
        raise RuntimeError("reward model data item must contain non-empty probs")
    if positive_class_index >= len(probabilities):
        raise RuntimeError(
            "positive_class_index is outside the reward model probability vector"
        )
    raw_score = probabilities[positive_class_index]
    if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
        raise RuntimeError("reward model probability must be numeric")
    score = float(raw_score)
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise RuntimeError("reward model probability must be finite and in [0, 1]")
    return score


def probability_to_logit(probability: float, epsilon: float = 1e-7) -> float:
    """Recover a finite pre-sigmoid score from a classifier probability."""
    if not 0.0 < epsilon < 0.5:
        raise ValueError("epsilon must be between zero and 0.5")
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be finite and in [0, 1]")
    clipped = min(max(probability, epsilon), 1.0 - epsilon)
    return math.log(clipped) - math.log1p(-clipped)


class RewardModelRubric(vf.Rubric):
    """Use a remote sequence classifier as the rollout's only reward."""

    def __init__(
        self,
        *,
        reward_client: RewardModelOutputClient,
        positive_class_index: int = 0,
        score_transform: Literal["identity", "logit"] = "identity",
        logit_epsilon: float = 1e-7,
        standardization_std: float | None = None,
        center_rewards: bool = False,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(positive_class_index, int) or isinstance(
            positive_class_index, bool
        ):
            raise TypeError("positive_class_index must be an integer")
        if positive_class_index < 0:
            raise ValueError("positive_class_index must be non-negative")
        if score_transform not in {"identity", "logit"}:
            raise ValueError("score_transform must be 'identity' or 'logit'")
        if not 0.0 < logit_epsilon < 0.5:
            raise ValueError("logit_epsilon must be between zero and 0.5")
        if standardization_std is not None and (
            not math.isfinite(standardization_std) or standardization_std <= 0.0
        ):
            raise ValueError("standardization_std must be finite and positive")
        if chat_template_kwargs is not None and not isinstance(
            chat_template_kwargs, Mapping
        ):
            raise TypeError("chat_template_kwargs must be an object")
        self.reward_client = reward_client
        self.positive_class_index = positive_class_index
        self.score_transform = score_transform
        self.logit_epsilon = logit_epsilon
        model_path = getattr(reward_client, "model_path", "")
        normalization_stats = reward_model_normalization_stats(model_path)
        self.standardization_mean = (
            normalization_stats["mean"] if center_rewards else 0.0
        )
        self.standardization_std = (
            standardization_std
            if standardization_std is not None
            else normalization_stats["std"]
        )
        self.center_rewards = center_rewards
        self.chat_template_kwargs = (
            dict(chat_template_kwargs)
            if chat_template_kwargs is not None
            else None
        )
        super().__init__(funcs=[self.reward_model_score])

    async def reward_model_score(
        self,
        completion: list[Any],
        prompt: Any,
        state: dict[str, Any] | None = None,
    ) -> float:
        messages = reward_conversation(prompt, completion)
        response = await self.reward_client.classify(
            messages=messages,
            chat_template_kwargs=self.chat_template_kwargs,
        )
        probability = positive_class_probability(
            response,
            self.positive_class_index,
        )
        raw_score = (
            probability_to_logit(probability, self.logit_epsilon)
            if self.score_transform == "logit"
            else probability
        )
        score = (
            (raw_score - self.standardization_mean) / self.standardization_std
            if self.standardization_std is not None
            else raw_score
        )
        if state is not None:
            existing_info = state.get("info")
            info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
            request: dict[str, Any] = {
                "messages": messages,
                "add_generation_prompt": False,
            }
            model_path = getattr(self.reward_client, "model_path", None)
            if isinstance(model_path, str) and model_path:
                request["model"] = model_path
            if self.chat_template_kwargs is not None:
                request["chat_template_kwargs"] = dict(self.chat_template_kwargs)
            info["reward_model"] = {
                "request": request,
                "response": response,
                "positive_class_index": self.positive_class_index,
                "probability": probability,
                "score_transform": self.score_transform,
                "raw_score": raw_score,
                "standardization": (
                    {
                        "mean": self.standardization_mean,
                        "std": self.standardization_std,
                        "centered": self.center_rewards,
                    }
                    if self.standardization_std is not None
                    else None
                ),
                "score": score,
            }
            state["info"] = info
        return score


class MathAccuracyRubric(MathRubric):
    """Log boxed-answer mathematical equivalence without changing reward."""

    def __init__(self) -> None:
        super().__init__()
        self.funcs.clear()
        self.weights.clear()
        self.add_metric(self.accuracy)

    async def accuracy(
        self,
        parser: vf.Parser,
        completion: Any,
        answer: Any,
        **kwargs: Any,
    ) -> float:
        answer_text = answer if isinstance(answer, str) else str(answer or "")
        gold = extract_boxed_answer(answer_text, strict=True) or answer_text
        return await self.correct_answer(
            parser=parser,
            completion=completion,
            answer=gold,
            **kwargs,
        )


def get_last_option(text: str) -> str | None:
    """Return the last standalone multiple-choice option letter (A-J)."""
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, Sequence) and not isinstance(
        completion, (str, bytes)
    ):
        if not completion:
            return ""
        content = _message_field(completion[-1], "content", "")
        return content if isinstance(content, str) else str(content or "")
    content = _message_field(completion, "content", "")
    return content if isinstance(content, str) else str(content or "")


class MultipleChoiceAccuracyRubric(vf.Rubric):
    """Log last-option exact-match accuracy without changing reward."""

    def __init__(self) -> None:
        super().__init__(funcs=[])
        self.add_metric(self.accuracy)

    async def accuracy(
        self,
        completion: Any,
        answer: Any,
        **kwargs: Any,
    ) -> float:
        predicted = get_last_option(_completion_text(completion))
        answer_text = answer if isinstance(answer, str) else str(answer or "")
        gold = get_last_option(answer_text)
        return float(
            predicted is not None
            and gold is not None
            and predicted == gold
        )


def _with_prompt(row: dict[str, Any]) -> dict[str, Any]:
    """Accept prompt, messages, or a plain input field as dataset input."""
    normalized = dict(row)
    prompt = normalized.get("prompt")
    if prompt is None:
        prompt = normalized.get("messages")
    if prompt is None and isinstance(normalized.get("input"), str):
        prompt = [{"role": "user", "content": normalized["input"]}]
    if prompt is None:
        raise ValueError(
            "reward-model dataset row needs prompt, messages, or text input"
        )
    normalized["prompt"] = prompt
    return normalized


def _load_dataset(dataset: str, split: str) -> Dataset:
    path = Path(dataset)
    if path.is_dir():
        if (path / "dataset_dict.json").is_file() or (
            (path / "dataset_info.json").is_file()
            and (path / "state.json").is_file()
        ):
            saved = load_from_disk(str(path))
            if isinstance(saved, DatasetDict):
                if split not in saved:
                    raise ValueError(
                        f"{dataset}: split {split!r} not found; available: {list(saved)}"
                    )
                rows = saved[split]
            elif isinstance(saved, Dataset):
                rows = saved
            else:
                raise TypeError(f"{dataset}: unsupported saved dataset type")
        else:
            # local Hugging Face datasets are directories of
            # conventionally named JSONL files (train.jsonl, validation.jsonl,
            # and so on), not Arrow datasets produced by save_to_disk().
            rows = load_dataset(str(path), split=split)
    elif path.is_file():
        with path.open(encoding="utf-8") as handle:
            local_rows = [
                _with_prompt({**dict(json.loads(line)), "split": split})
                for line in handle
                if line.strip()
            ]
        return Dataset.from_list(local_rows)
    else:
        rows = load_dataset(dataset, split=split)
    if "split" not in rows.column_names:
        rows = rows.map(lambda _: {"split": split})
    return rows.map(_with_prompt)


def load_environment(
    dataset: str,
    reward_model_path: str,
    split: str = "train",
    reward_server_url: str = CLASSIFY_SERVER_URL,
    reward_timeout: float = 240,
    reward_max_retries: int = 3,
    positive_class_index: int = 0,
    score_transform: Literal["identity", "logit"] = "identity",
    logit_epsilon: float = 1e-7,
    standardization_std: float | None = None,
    center_rewards: bool = False,
    log_math_accuracy: bool = False,
    log_science_accuracy: bool = False,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> vf.SingleTurnEnv:
    """Prime-RL factory for a single-turn classifier-scored environment."""
    reward_client = RewardModelClient(
        model_path=reward_model_path,
        server_url=reward_server_url,
        timeout=reward_timeout,
        max_retries=reward_max_retries,
    )
    reward_rubric = RewardModelRubric(
        reward_client=reward_client,
        positive_class_index=positive_class_index,
        score_transform=score_transform,
        logit_epsilon=logit_epsilon,
        standardization_std=standardization_std,
        center_rewards=center_rewards,
        chat_template_kwargs=chat_template_kwargs,
    )
    metric_rubrics: list[vf.Rubric] = []
    if log_math_accuracy:
        metric_rubrics.append(MathAccuracyRubric())
    if log_science_accuracy:
        metric_rubrics.append(MultipleChoiceAccuracyRubric())
    rubric: vf.Rubric = (
        RubricGroup([reward_rubric, *metric_rubrics])
        if metric_rubrics
        else reward_rubric
    )
    return vf.SingleTurnEnv(
        dataset=_load_dataset(dataset, split),
        rubric=rubric,
    )


__all__ = [
    "RewardModelOutputClient",
    "RewardModelRubric",
    "MathAccuracyRubric",
    "MultipleChoiceAccuracyRubric",
    "REWARD_MODEL_NORMALIZATION_STATS",
    "get_last_option",
    "load_environment",
    "probability_to_logit",
    "positive_class_probability",
    "reward_model_normalization_stats",
    "reward_model_standard_deviation",
    "reward_conversation",
]
