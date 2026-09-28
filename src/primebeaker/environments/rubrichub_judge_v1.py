"""verifiers v1 port of ``jtc_rubrichub_judge_env`` (single-turn RubricHub RL scored by the JTC judge).

prime-rl >= 0.9 accepts only v1 sources and the v0 module cannot be imported under verifiers
v1 (it subclasses ``vf.Rubric``), so this module copies its pure helpers verbatim (row
normalization, reproducible signed rubric selection, judge-response validation) and rebuilds
the scoring contract:

* policy output that hit the token limit scores ``non_termination_penalty`` (+ orphan
  ``</think>`` penalty) without a judge call;
* empty/missing final content scores ``invalid_output_penalty`` without a judge call;
* otherwise one judge request with the sampled rubrics; score = weighted pass fraction over
  the sampled weights + orphan ``</think>`` penalty. A judge failure errors the rollout.

The judge runs once per trace, in the ``judge`` metric hook (v1 runs every metric before any
reward); ``jtc_weighted_score`` returns the value it stored on the trace state. The shadow
judge of the v0 module is not ported.

``MultipleChoiceAccuracyTaskset`` ports the metric-only ``jtc_multiple_choice_accuracy_env``
(last standalone A-J option exact match) for evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from literegistry_tool_client import JudgeClient
from primebeaker.environments.signed_rubrics import (
    judge_criterion,
    nonzero_weight,
    signed_sample_indexes,
    weight_normalizer,
)

__all__ = [
    "MultipleChoiceAccuracyTask",
    "MultipleChoiceAccuracyTaskset",
    "MultipleChoiceAccuracyTasksetConfig",
    "RubricHubJudgeTask",
    "RubricHubJudgeTaskConfig",
    "RubricHubJudgeTaskset",
    "RubricHubJudgeTasksetConfig",
    "normalize_rubrichub_row",
    "select_rubrics",
]

# ----------------------------------------------------------------------------
# Pure helpers copied verbatim from jtc_rubrichub_judge_env (lines 52-351).
# ----------------------------------------------------------------------------

def _message_field(message: Any, name: str, default: Any = None) -> Any:
    if isinstance(message, Mapping):
        return message.get(name, default)
    return getattr(message, name, default)


def _prompt_text(prompt: Any) -> str:
    """Render the original policy prompt deterministically for the judge."""
    if isinstance(prompt, str):
        if not prompt.strip():
            raise ValueError("policy prompt must not be empty")
        return prompt
    if isinstance(prompt, Mapping):
        prompt = [prompt]
    if isinstance(prompt, Sequence) and not isinstance(prompt, (str, bytes)):
        messages: list[dict[str, str]] = []
        for index, message in enumerate(prompt):
            role = _message_field(message, "role", "user")
            content = _message_field(message, "content", "")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"prompt[{index}].role must be non-empty")
            if not isinstance(content, str):
                raise TypeError(f"prompt[{index}].content must be text")
            messages.append({"role": role.strip(), "content": content})
        if not messages:
            raise ValueError("policy prompt must not be empty")
        if len(messages) == 1 and messages[0]["role"] == "user":
            return messages[0]["content"]
        return "\n\n".join(f"{message['role']}:\n{message['content']}" for message in messages)
    raise TypeError("policy prompt must be text or OpenAI-style messages")


def _completion_text(completion: Sequence[Any]) -> str:
    if not completion:
        raise ValueError("completion must contain an assistant message")
    message = completion[-1]
    if _message_field(message, "role", "assistant") != "assistant":
        raise ValueError("completion must end with an assistant message")
    content = _message_field(message, "content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("final assistant content must be non-empty text")
    return content


def _has_orphan_think_close_tag(output: str) -> bool:
    """Return whether visible output contains a ``</think>`` without an opener."""
    cursor = 0
    open_tags = 0
    while True:
        next_open = output.find("<think>", cursor)
        next_close = output.find("</think>", cursor)
        if next_close < 0:
            return False
        if 0 <= next_open < next_close:
            open_tags += 1
            cursor = next_open + len("<think>")
            continue
        if open_tags == 0:
            return True
        open_tags -= 1
        cursor = next_close + len("</think>")


def _positive_weight(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    weight = float(value)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return weight


def _canonical_rubrics(payload: Any) -> list[dict[str, Any]]:
    """Validate the strict runtime ``text``/``weight`` rubric payload."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    rubrics = payload.get("rubrics") if isinstance(payload, Mapping) else payload
    if isinstance(rubrics, (str, bytes)) or not isinstance(rubrics, Sequence):
        raise ValueError("RubricHub payload must contain a rubric list")
    normalized: list[dict[str, Any]] = []
    for index, rubric in enumerate(rubrics):
        if not isinstance(rubric, Mapping):
            raise TypeError(f"rubrics[{index}] must be an object")
        text = rubric.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"rubrics[{index}].text must be non-empty")
        original_index = rubric.get("original_index", index)
        if (
            isinstance(original_index, bool)
            or not isinstance(original_index, int)
            or original_index < 0
        ):
            raise ValueError(
                f"rubrics[{index}].original_index must be a non-negative integer"
            )
        normalized.append(
            {
                "text": text.strip(),
                "weight": nonzero_weight(
                    rubric.get("weight"), field=f"rubrics[{index}].weight"
                ),
                "original_index": original_index,
            }
        )
    if not normalized:
        raise ValueError("RubricHub row must contain at least one rubric")
    if len({rubric["original_index"] for rubric in normalized}) != len(normalized):
        raise ValueError("rubric original_index values must be unique")
    return normalized


def _raw_rubrichub_rubrics(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Adapt the public Hugging Face schema into the strict runtime schema."""
    canonical = row.get("rubrics")
    if canonical is not None:
        return _canonical_rubrics({"rubrics": canonical})

    raw = row.get("Rubrics")
    if raw is None:
        reward_model = row.get("reward_model")
        if isinstance(reward_model, Mapping):
            raw = reward_model.get("rubrics")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("RubricHub source row is missing Rubrics")

    adapted: list[dict[str, Any]] = []
    for index, rubric in enumerate(raw):
        if not isinstance(rubric, Mapping):
            raise TypeError(f"Rubrics[{index}] must be an object")
        criterion = rubric.get("criterion")
        if not isinstance(criterion, str) or not criterion.strip():
            raise ValueError(f"Rubrics[{index}].criterion must be non-empty")
        adapted.append(
            {
                "text": criterion.strip(),
                "weight": _positive_weight(
                    rubric.get("points"), field=f"Rubrics[{index}].points"
                ),
                "original_index": index,
            }
        )
    return _canonical_rubrics({"rubrics": adapted})


def _source_record_id(
    row: Mapping[str, Any],
    *,
    prompt: Any,
    rubrics: Sequence[Mapping[str, Any]],
    dataset_name: str | None = None,
    split: str | None = None,
    row_index: int | None = None,
) -> str:
    explicit = next(
        (
            row.get(key)
            for key in ("record_id", "id")
            if row.get(key) is not None and str(row.get(key)).strip()
        ),
        None,
    )
    namespace = ":".join(part for part in (dataset_name, split) if part)
    if explicit is not None:
        # Canonical materialized IDs are already globally namespaced and must
        # remain stable when the artifact is moved to another local path.
        return str(explicit).strip()
    source_index = row.get("__index_level_0__")
    if source_index is not None and str(source_index).strip():
        value = str(source_index).strip()
        return f"{namespace}:{value}" if namespace else value
    if row_index is not None:
        value = str(row_index)
        return f"{namespace}:{value}" if namespace else value
    canonical = json.dumps(
        {"prompt": prompt, "rubrics": list(rubrics)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def normalize_rubrichub_row(
    row: Mapping[str, Any],
    row_index: int | None = None,
    *,
    dataset_name: str | None = None,
    split: str | None = None,
) -> dict[str, Any]:
    """Adapt one raw or already-canonical RubricHub row for Verifiers."""
    normalized = dict(row)
    prompt = normalized.get("prompt")
    if prompt is None:
        extra_info = normalized.get("extra_info")
        if isinstance(extra_info, Mapping):
            prompt = extra_info.get("prompt")
    if prompt is None:
        raise ValueError("RubricHub row is missing prompt")
    _prompt_text(prompt)

    rubrics = _raw_rubrichub_rubrics(normalized)
    record_id = _source_record_id(
        normalized,
        prompt=prompt,
        rubrics=rubrics,
        dataset_name=dataset_name,
        split=split,
        row_index=row_index,
    )
    normalized["prompt"] = prompt
    normalized["answer"] = json.dumps(
        {"record_id": record_id, "rubrics": rubrics},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    existing_info = normalized.get("info")
    info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
    info["rubrichub_record_id"] = record_id
    normalized["info"] = info
    return normalized


def _answer_payload(answer: Any) -> tuple[str, list[dict[str, Any]]]:
    payload = json.loads(answer) if isinstance(answer, str) else answer
    if not isinstance(payload, Mapping):
        raise ValueError("RubricHub answer must be an object")
    record_id = payload.get("record_id")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError("RubricHub answer.record_id must be non-empty")
    return record_id.strip(), _canonical_rubrics(payload)


def select_rubrics(
    rubrics: Sequence[Mapping[str, Any]],
    *,
    max_rubrics: int,
    sample_seed: int,
    record_id: str,
) -> list[dict[str, Any]]:
    """Select a reproducible uniform subset without process-global RNG state."""
    if (
        isinstance(max_rubrics, bool)
        or not isinstance(max_rubrics, int)
        or max_rubrics < 1
    ):
        raise ValueError("max_rubrics must be a positive integer")
    if isinstance(sample_seed, bool) or not isinstance(sample_seed, int):
        raise TypeError("rubric_sample_seed must be an integer")
    normalized = [dict(rubric) for rubric in rubrics]
    if len(normalized) <= max_rubrics:
        return normalized
    seed_material = f"{sample_seed}\\0{record_id}".encode("utf-8")
    derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest(), "big")
    indexes = signed_sample_indexes(normalized, max_rubrics, random.Random(derived_seed))
    return [normalized[index] for index in indexes]


def _validated_judgments(
    response: Any, selected: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if not isinstance(response, Mapping):
        raise RuntimeError("judge response must be an object")
    judgments = response.get("judgments")
    if not isinstance(judgments, list) or len(judgments) != len(selected):
        raise RuntimeError("judge response must contain exactly one judgment per rubric")
    by_request_index: dict[int, dict[str, Any]] = {}
    for judgment in judgments:
        if not isinstance(judgment, Mapping):
            raise RuntimeError("each judge judgment must be an object")
        request_index = judgment.get("rubric_index")
        if (
            isinstance(request_index, bool)
            or not isinstance(request_index, int)
            or not 0 <= request_index < len(selected)
        ):
            raise RuntimeError(f"invalid judge rubric_index: {request_index!r}")
        if request_index in by_request_index:
            raise RuntimeError(f"duplicate judge rubric_index: {request_index}")
        label = judgment.get("label")
        normalized_label = label.strip().casefold() if isinstance(label, str) else ""
        if normalized_label not in {"pass", "fail"}:
            raise RuntimeError(f"unsupported judge label: {label!r}")
        selected_rubric = selected[request_index]
        weight = float(selected_rubric["weight"])
        by_request_index[request_index] = {
            "request_index": request_index,
            "original_index": int(selected_rubric["original_index"]),
            "text": str(selected_rubric["text"]),
            "weight": weight,
            "label": normalized_label,
            "feedback": judgment.get("feedback"),
            "trace": judgment.get("trace"),
            "weighted_pass": weight if normalized_label == "pass" else 0.0,
        }
    expected = set(range(len(selected)))
    if set(by_request_index) != expected:
        raise RuntimeError("judge response has missing rubric_index values")
    return [by_request_index[index] for index in range(len(selected))]



# ----------------------------------------------------------------------------
# RubricHub judge task
# ----------------------------------------------------------------------------


class RubricHubJudgeData(vf.TaskData):
    answer: str = ""
    """JSON ``{record_id, rubrics}`` payload produced by ``normalize_rubrichub_row``."""


class RubricHubJudgeState(vf.State):
    jtc_weighted_score: float | None = None
    judge_audit: dict[str, Any] = Field(default_factory=dict)


class RubricHubJudgeTaskConfig(vf.TaskConfig):
    judge_server_url: str = "http://127.0.0.1:1213/judge"
    judge_model_path: str = ""
    judge_service_model_path: str = "judge"
    judge_timeout: float = 240
    judge_max_retries: int = 3
    max_rubrics: int = Field(8, ge=1)
    rubric_sample_seed: int = 0
    non_termination_penalty: float = Field(-0.25, lt=0)
    invalid_output_penalty: float = Field(0.0, le=0)
    orphan_think_close_penalty: float = Field(-0.2, le=0)


class RubricHubJudgeTasksetConfig(vf.TasksetConfig):
    dataset: str = ""
    """HF ``save_to_disk`` directory (DatasetDict/Dataset), local JSON/JSONL, or hub id."""
    split: str = "train"
    task: RubricHubJudgeTaskConfig = RubricHubJudgeTaskConfig()


_JUDGE_CLIENTS: dict[tuple, JudgeClient] = {}


def _judge_client(cfg: RubricHubJudgeTaskConfig) -> JudgeClient:
    key = (cfg.judge_server_url, cfg.judge_service_model_path, cfg.judge_timeout, cfg.judge_max_retries)
    if key not in _JUDGE_CLIENTS:
        _JUDGE_CLIENTS[key] = JudgeClient(
            cfg.judge_server_url,
            service_model_path=cfg.judge_service_model_path,
            timeout=cfg.judge_timeout,
            max_retries=cfg.judge_max_retries,
        )
    return _JUDGE_CLIENTS[key]


def _assistant_completion(trace: vf.Trace) -> list[dict[str, Any]]:
    """The trace's assistant turns as v0 completion messages (visible content only)."""
    return [
        {"role": "assistant", "content": message.content}
        for message in trace.messages
        if isinstance(message, vf.AssistantMessage)
    ]


class RubricHubJudgeTask(vf.Task[RubricHubJudgeData, RubricHubJudgeState, RubricHubJudgeTaskConfig]):
    @property
    def key(self) -> str:
        record_id, _ = _answer_payload(self.data.answer)
        return f"{record_id}:{self.data.idx}"

    def _selected(self) -> tuple[str, list[dict[str, Any]]]:
        record_id, rubrics = _answer_payload(self.data.answer)
        cfg = self.config
        return record_id, select_rubrics(
            rubrics, max_rubrics=cfg.max_rubrics, sample_seed=cfg.rubric_sample_seed, record_id=record_id
        )

    @vf.metric
    async def judge(self, trace: vf.Trace) -> dict[str, float]:
        cfg = self.config
        record_id, selected = self._selected()
        completion = _assistant_completion(trace)
        try:
            output_text = _completion_text(completion)
        except ValueError as error:
            output_text, invalid_error = "", str(error)
        else:
            invalid_error = None
        orphan = _has_orphan_think_close_tag(output_text) if output_text else False
        format_penalty = cfg.orphan_think_close_penalty if orphan else 0.0
        audit: dict[str, Any] = {"source_record_id": record_id, "selected_rubrics": selected}
        non_terminated = invalid = 0.0
        rubric_score = 0.0

        if trace.is_truncated:
            non_terminated = 1.0
            score = cfg.non_termination_penalty + format_penalty
            audit["status"] = "non_terminated_policy_output"
        elif invalid_error is not None:
            if invalid_error not in {
                "final assistant content must be non-empty text",
                "completion must contain an assistant message",
            }:
                raise ValueError(invalid_error)
            invalid = 1.0
            format_penalty = 0.0
            score = cfg.invalid_output_penalty
            audit.update(status="invalid_policy_output", error=invalid_error)
        else:
            response = await _judge_client(cfg).verify_output(
                input=_prompt_text(self.data.prompt),
                output=output_text,
                rubrics=[judge_criterion(rubric) for rubric in selected],
                model=cfg.judge_model_path,
                service_model_path=cfg.judge_service_model_path,
            )
            judgments = _validated_judgments(response, selected)
            weight_sum = weight_normalizer(judgments)
            if not math.isfinite(weight_sum) or weight_sum <= 0:
                raise RuntimeError("selected rubric weight sum must be positive")
            rubric_score = sum(float(item["weighted_pass"]) for item in judgments) / weight_sum
            score = rubric_score + format_penalty
            audit.update(
                status="judged",
                judge_record_id=response.get("record_id"),
                judge_profile=response.get("profile"),
                judgments=judgments,
            )

        audit["score"] = score
        trace.state.jtc_weighted_score = score
        trace.state.judge_audit = audit
        return {
            "invalid_final_output": invalid,
            "non_terminated_policy_output": non_terminated,
            "orphan_think_close_tag": float(orphan),
            "orphan_think_close_penalty": format_penalty,
            "rubric_score_before_penalties": rubric_score,
            "num_selected_rubrics": float(len(selected)),
        }

    @vf.reward
    async def jtc_weighted_score(self, trace: vf.Trace) -> float:
        if trace.state.jtc_weighted_score is None:
            raise RuntimeError("judge metric did not run before jtc_weighted_score")
        return float(trace.state.jtc_weighted_score)


def _saved_rows(path: Path, split: str) -> Iterable[Mapping[str, Any]]:
    import datasets

    if path.is_dir():
        if (path / "dataset_dict.json").is_file() or (
            (path / "dataset_info.json").is_file() and (path / "state.json").is_file()
        ):
            loaded = datasets.load_from_disk(str(path))
            if isinstance(loaded, datasets.DatasetDict):
                if split not in loaded:
                    raise ValueError(f"{path}: split {split!r} not found; available: {list(loaded)}")
                return loaded[split]
            return loaded
        return datasets.load_dataset(str(path), split=split)
    if path.is_file():
        rows = []
        with path.open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return rows
    return datasets.load_dataset(str(path), split=split)


class RubricHubJudgeTaskset(vf.Taskset[RubricHubJudgeTask, RubricHubJudgeTasksetConfig]):
    def load(self) -> Iterable[RubricHubJudgeTask]:
        cfg = self.config
        for idx, row in enumerate(_saved_rows(Path(cfg.dataset), cfg.split)):
            normalized = normalize_rubrichub_row(row, idx, dataset_name=cfg.dataset, split=cfg.split)
            data = RubricHubJudgeData(idx=idx, prompt=normalized["prompt"], answer=normalized["answer"])
            yield RubricHubJudgeTask(data, cfg.task)


# ----------------------------------------------------------------------------
# Multiple-choice accuracy (metric-only evaluation)
# ----------------------------------------------------------------------------


def get_last_option(text: str) -> str | None:
    """Return the last standalone multiple-choice option letter (A-J)."""
    match = re.search(r"\b[A-J]\b(?!.*\b[A-J]\b)", text, re.DOTALL)
    return match.group(0) if match else None


class MultipleChoiceAccuracyData(vf.TaskData):
    answer: str = ""


class MultipleChoiceAccuracyTasksetConfig(vf.TasksetConfig):
    dataset: str = ""
    split: str = "validation"


class MultipleChoiceAccuracyTask(vf.Task[MultipleChoiceAccuracyData, vf.State, vf.TaskConfig]):
    @vf.metric
    async def accuracy(self, trace: vf.Trace) -> float:
        completion = _assistant_completion(trace)
        content = completion[-1]["content"] if completion else ""
        predicted = get_last_option(content if isinstance(content, str) else str(content or ""))
        gold = get_last_option(str(self.data.answer or ""))
        return float(predicted is not None and gold is not None and predicted == gold)


def _mc_prompt(row: Mapping[str, Any]) -> Any:
    prompt = row.get("prompt")
    if prompt is None:
        prompt = row.get("messages")
    if prompt is None and isinstance(row.get("input"), str):
        prompt = [{"role": "user", "content": row["input"]}]
    if prompt is None:
        raise ValueError("multiple-choice row needs prompt, messages, or text input")
    return prompt


class MultipleChoiceAccuracyTaskset(vf.Taskset[MultipleChoiceAccuracyTask, MultipleChoiceAccuracyTasksetConfig]):
    def load(self) -> Iterable[MultipleChoiceAccuracyTask]:
        cfg = self.config
        for idx, row in enumerate(_saved_rows(Path(cfg.dataset), cfg.split)):
            answer = row.get("answer")
            data = MultipleChoiceAccuracyData(
                idx=idx, prompt=_mc_prompt(row), answer=answer if isinstance(answer, str) else str(answer or "")
            )
            yield MultipleChoiceAccuracyTask(data, vf.TaskConfig())
