"""Single-turn RubricHub environment scored by the deployed JTC judge pool.

Raw ``sojuL/RubricHub_v1`` rows are adapted once at dataset load time. The
runtime scorer consumes canonical lower-case ``rubrics`` with ``text`` and
positive ``weight`` fields, samples at most ``max_rubrics`` criteria
reproducibly, and sends the complete sample in one judge request.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
import verifiers as vf

from literegistry_tool_client import JudgeClient


DEFAULT_RUBRICHUB_DATASET = "sojuL/RubricHub_v1"
JTC4B_STEP400 = (
    "/weka/gfaria/prime_sft/outputs/"
    "qwen35_4b_glm52_sft_c0e8854bb5143cac_rl_5k_161bc7548bdec992_"
    "step400-nooversamp-renderer-eval-inflight512-kl1e-3-lr1e-6/weights/step_400"
)


class JudgeOutputClient(Protocol):
    """Judge transport consumed by :class:`JTCJudgeRubric`."""

    async def verify_output(
        self,
        *,
        input: str,
        output: str,
        rubrics: Sequence[str],
        model: str,
        service_model_path: str | None = None,
    ) -> dict[str, Any]: ...


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
        return json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
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
                "weight": _positive_weight(
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
    indexes = sorted(
        random.Random(derived_seed).sample(range(len(normalized)), max_rubrics)
    )
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


class JTCJudgeRubric(vf.Rubric):
    """The deployed JTC judge is the environment's only reward function."""

    def __init__(
        self,
        *,
        judge_client: JudgeOutputClient,
        judge_model_path: str,
        judge_service_model_path: str = "judge",
        max_rubrics: int = 8,
        rubric_sample_seed: int = 0,
        non_termination_penalty: float = -0.25,
        orphan_think_close_penalty: float = -0.2,
        shadow_judge_client: JudgeOutputClient | None = None,
        shadow_judge_model_path: str | None = None,
        shadow_judge_service_model_path: str = "judge",
        shadow_judge_sample_rate: float = 0.0,
        shadow_judge_sample_seed: int = 0,
        shadow_judge_audit_dir: str | None = None,
    ) -> None:
        if not isinstance(judge_model_path, str) or not judge_model_path.strip():
            raise ValueError("judge_model_path must be non-empty")
        if (
            not isinstance(judge_service_model_path, str)
            or not judge_service_model_path.strip()
        ):
            raise ValueError("judge_service_model_path must be non-empty")
        if (
            isinstance(non_termination_penalty, bool)
            or not isinstance(non_termination_penalty, (int, float))
            or not math.isfinite(float(non_termination_penalty))
            or float(non_termination_penalty) >= 0
        ):
            raise ValueError("non_termination_penalty must be finite and negative")
        if (
            isinstance(orphan_think_close_penalty, bool)
            or not isinstance(orphan_think_close_penalty, (int, float))
            or not math.isfinite(float(orphan_think_close_penalty))
            or float(orphan_think_close_penalty) > 0
        ):
            raise ValueError(
                "orphan_think_close_penalty must be finite and non-positive"
            )
        if isinstance(shadow_judge_sample_rate, bool) or not isinstance(
            shadow_judge_sample_rate, (int, float)
        ):
            raise TypeError("shadow_judge_sample_rate must be numeric")
        shadow_sample_rate = float(shadow_judge_sample_rate)
        if not math.isfinite(shadow_sample_rate) or not 0 <= shadow_sample_rate <= 1:
            raise ValueError("shadow_judge_sample_rate must be between zero and one")
        shadow_enabled = shadow_judge_client is not None or shadow_judge_model_path is not None
        if shadow_enabled and (shadow_judge_client is None or shadow_judge_model_path is None):
            raise ValueError(
                "shadow_judge_client and shadow_judge_model_path must be provided together"
            )
        if shadow_judge_model_path is not None and not shadow_judge_model_path.strip():
            raise ValueError("shadow_judge_model_path must be non-empty when provided")
        if not shadow_judge_service_model_path.strip():
            raise ValueError("shadow_judge_service_model_path must be non-empty")
        select_rubrics(
            [{"text": "validation", "weight": 1.0, "original_index": 0}],
            max_rubrics=max_rubrics,
            sample_seed=rubric_sample_seed,
            record_id="validation",
        )
        self.judge_client = judge_client
        self.judge_model_path = judge_model_path.strip()
        self.judge_service_model_path = judge_service_model_path.strip()
        self.max_rubrics = max_rubrics
        self.rubric_sample_seed = rubric_sample_seed
        self.non_termination_penalty = float(non_termination_penalty)
        self.orphan_think_close_penalty = float(orphan_think_close_penalty)
        self.shadow_judge_client = shadow_judge_client
        self.shadow_judge_model_path = (
            shadow_judge_model_path.strip() if shadow_judge_model_path is not None else None
        )
        self.shadow_judge_service_model_path = shadow_judge_service_model_path.strip()
        self.shadow_judge_sample_rate = shadow_sample_rate
        self.shadow_judge_sample_seed = shadow_judge_sample_seed
        self.shadow_judge_audit_dir = (
            str(Path(shadow_judge_audit_dir)) if shadow_judge_audit_dir else None
        )
        super().__init__(funcs=[self.jtc_weighted_score])

    @staticmethod
    def _policy_version(state: Mapping[str, Any]) -> str:
        sampling = state.get("sampling_args")
        extra_body = _message_field(sampling, "extra_body", {})
        if isinstance(extra_body, Mapping):
            cache_salt = extra_body.get("cache_salt")
            if cache_salt is not None:
                return str(cache_salt)
        return "unknown"

    def _shadow_sample_selected(
        self, state: Mapping[str, Any], *, source_record_id: str, output_text: str
    ) -> tuple[bool, str, str]:
        policy_version = self._policy_version(state)
        trajectory_id = str(
            state.get("trajectory_id")
            or state.get("example_id")
            or hashlib.sha256(output_text.encode("utf-8")).hexdigest()
        )
        sample_key = (
            f"{self.shadow_judge_sample_seed}:{policy_version}:"
            f"{trajectory_id}:{source_record_id}"
        )
        sample_value = int.from_bytes(
            hashlib.sha256(sample_key.encode("utf-8")).digest()[:8], "big"
        ) / 2**64
        return sample_value < self.shadow_judge_sample_rate, policy_version, trajectory_id

    def _persist_shadow_audit(self, audit: Mapping[str, Any]) -> None:
        if self.shadow_judge_audit_dir is None:
            return
        policy_version = str(audit.get("policy_version", "unknown"))
        safe_version = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in policy_version
        )
        trajectory_id = str(audit.get("trajectory_id", uuid4().hex))
        safe_trajectory = "".join(
            character if character.isalnum() or character in "._-" else "_"
            for character in trajectory_id
        )
        directory = Path(self.shadow_judge_audit_dir) / f"policy_version={safe_version}"
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{safe_trajectory}.json"
        temporary = destination.with_name(
            f".{destination.name}.{os.getpid()}.{uuid4().hex}.tmp"
        )
        temporary.write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, destination)

    async def _maybe_run_shadow_judge(
        self,
        *,
        state: dict[str, Any],
        request: Mapping[str, Any],
        source_record_id: str,
        selected: Sequence[Mapping[str, Any]],
        primary_judgments: Sequence[Mapping[str, Any]],
        primary_rubric_score: float,
    ) -> None:
        if self.shadow_judge_client is None or self.shadow_judge_model_path is None:
            return
        sampled, policy_version, trajectory_id = self._shadow_sample_selected(
            state,
            source_record_id=source_record_id,
            output_text=str(request["output"]),
        )
        existing_info = state.get("info")
        info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
        if not sampled:
            info["jtc_rubrichub_shadow_judge"] = {
                "status": "not_sampled",
                "sample_rate": self.shadow_judge_sample_rate,
                "sample_seed": self.shadow_judge_sample_seed,
                "policy_version": policy_version,
                "trajectory_id": trajectory_id,
            }
            state["info"] = info
            return

        audit: dict[str, Any] = {
            "status": "failed",
            "sample_rate": self.shadow_judge_sample_rate,
            "sample_seed": self.shadow_judge_sample_seed,
            "policy_version": policy_version,
            "trajectory_id": trajectory_id,
            "source_record_id": source_record_id,
            "primary_judge_model_path": self.judge_model_path,
            "shadow_judge_model_path": self.shadow_judge_model_path,
            "shadow_judge_service_model_path": self.shadow_judge_service_model_path,
            "primary_rubric_score": primary_rubric_score,
        }
        try:
            response = await self.shadow_judge_client.verify_output(
                input=str(request["input"]),
                output=str(request["output"]),
                rubrics=list(request["rubrics"]),
                model=self.shadow_judge_model_path,
                service_model_path=self.shadow_judge_service_model_path,
            )
            shadow_judgments = _validated_judgments(response, selected)
            selected_weight_sum = sum(float(item["weight"]) for item in shadow_judgments)
            shadow_weighted_pass_sum = sum(
                float(item["weighted_pass"]) for item in shadow_judgments
            )
            shadow_rubric_score = shadow_weighted_pass_sum / selected_weight_sum
            pairs = []
            for primary, shadow in zip(primary_judgments, shadow_judgments, strict=True):
                agrees = str(primary["label"]) == str(shadow["label"])
                pairs.append(
                    {
                        "request_index": int(primary["request_index"]),
                        "original_index": int(primary["original_index"]),
                        "text": str(primary["text"]),
                        "weight": float(primary["weight"]),
                        "agreement": agrees,
                        "primary": dict(primary),
                        "shadow": dict(shadow),
                    }
                )
            agreements = sum(float(pair["agreement"]) for pair in pairs)
            weighted_agreements = sum(
                float(pair["weight"]) * float(pair["agreement"]) for pair in pairs
            )
            primary_passes = sum(
                float(pair["primary"]["label"] == "pass") for pair in pairs
            )
            shadow_passes = sum(
                float(pair["shadow"]["label"] == "pass") for pair in pairs
            )
            primary_pass_shadow_fail = sum(
                float(
                    pair["primary"]["label"] == "pass"
                    and pair["shadow"]["label"] == "fail"
                )
                for pair in pairs
            )
            primary_fail_shadow_pass = sum(
                float(
                    pair["primary"]["label"] == "fail"
                    and pair["shadow"]["label"] == "pass"
                )
                for pair in pairs
            )
            count = float(len(pairs))
            audit.update(
                {
                    "status": "ok",
                    "shadow_judge_record_id": response.get("record_id"),
                    "shadow_judge_profile": response.get("profile"),
                    "pairs": pairs,
                    "num_judgments": len(pairs),
                    "label_agreement": agreements / count,
                    "weighted_label_agreement": weighted_agreements / selected_weight_sum,
                    "primary_pass_rate": primary_passes / count,
                    "shadow_pass_rate": shadow_passes / count,
                    "primary_pass_shadow_fail_rate": primary_pass_shadow_fail / count,
                    "primary_fail_shadow_pass_rate": primary_fail_shadow_pass / count,
                    "shadow_rubric_score": shadow_rubric_score,
                    "shadow_minus_primary_score": shadow_rubric_score
                    - primary_rubric_score,
                }
            )
        except Exception as error:
            # Shadow judging is observational. It must never fail or alter a
            # rollout whose primary Lion judgment succeeded.
            audit["error"] = f"{type(error).__name__}: {error}"
        info["jtc_rubrichub_shadow_judge"] = audit
        state["info"] = info
        try:
            self._persist_shadow_audit(audit)
        except Exception as error:
            audit["audit_persistence_error"] = f"{type(error).__name__}: {error}"

    @staticmethod
    def _add_shadow_metrics(state: dict[str, Any]) -> None:
        info = state.get("info")
        if not isinstance(info, Mapping):
            return
        audit = info.get("jtc_rubrichub_shadow_judge")
        if not isinstance(audit, Mapping):
            return
        metrics = state.setdefault("metrics", {})
        status = audit.get("status")
        metrics["shadow_judge_sampled"] = float(status != "not_sampled")
        metrics["shadow_judge_error"] = float(status == "failed")
        if status == "ok":
            for key in (
                "label_agreement",
                "weighted_label_agreement",
                "primary_pass_rate",
                "shadow_pass_rate",
                "primary_pass_shadow_fail_rate",
                "primary_fail_shadow_pass_rate",
                "primary_rubric_score",
                "shadow_rubric_score",
                "shadow_minus_primary_score",
            ):
                metrics[f"shadow_judge_{key}"] = float(audit[key])

    async def jtc_weighted_score(
        self,
        completion: list[Any],
        prompt: Any,
        answer: Any,
        state: dict[str, Any] | None = None,
    ) -> float:
        source_record_id, rubrics = _answer_payload(answer)
        selected = select_rubrics(
            rubrics,
            max_rubrics=self.max_rubrics,
            sample_seed=self.rubric_sample_seed,
            record_id=source_record_id,
        )
        output_text = _completion_text(completion)
        has_orphan_think_close = _has_orphan_think_close_tag(output_text)
        applied_format_penalty = (
            self.orphan_think_close_penalty if has_orphan_think_close else 0.0
        )
        request = {
            "input": _prompt_text(prompt),
            "output": output_text,
            "rubrics": [str(rubric["text"]) for rubric in selected],
            "model": self.judge_model_path,
            "service_model_path": self.judge_service_model_path,
        }
        response = await self.judge_client.verify_output(**request)
        judgments = _validated_judgments(response, selected)
        selected_weight_sum = sum(float(item["weight"]) for item in judgments)
        if not math.isfinite(selected_weight_sum) or selected_weight_sum <= 0:
            raise RuntimeError("selected rubric weight sum must be positive")
        weighted_pass_sum = sum(float(item["weighted_pass"]) for item in judgments)
        rubric_score = weighted_pass_sum / selected_weight_sum
        score = rubric_score + applied_format_penalty

        if state is not None:
            existing_info = state.get("info")
            info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
            info["jtc_rubrichub_judge"] = {
                "source_record_id": source_record_id,
                "judge_record_id": response.get("record_id"),
                "judge_model_path": self.judge_model_path,
                "judge_service_model_path": self.judge_service_model_path,
                "judge_profile": response.get("profile"),
                "max_rubrics": self.max_rubrics,
                "rubric_sample_seed": self.rubric_sample_seed,
                "selected_rubrics": selected,
                "judgments": judgments,
                "weighted_pass_sum": weighted_pass_sum,
                "selected_weight_sum": selected_weight_sum,
                "rubric_score_before_format_penalties": rubric_score,
                "orphan_think_close_tag": has_orphan_think_close,
                "orphan_think_close_penalty": self.orphan_think_close_penalty,
                "applied_orphan_think_close_penalty": applied_format_penalty,
                "score": score,
            }
            state["info"] = info
            await self._maybe_run_shadow_judge(
                state=state,
                request=request,
                source_record_id=source_record_id,
                selected=selected,
                primary_judgments=judgments,
                primary_rubric_score=rubric_score,
            )
        return score

    async def score_rollout(self, state: dict[str, Any]) -> None:
        """Penalize truncation, score empty output as zero, and judge the rest."""
        if bool(state.get("is_truncated", False)):
            source_record_id, rubrics = _answer_payload(state.get("answer", ""))
            selected = select_rubrics(
                rubrics,
                max_rubrics=self.max_rubrics,
                sample_seed=self.rubric_sample_seed,
                record_id=source_record_id,
            )
            selected_weight_sum = sum(float(item["weight"]) for item in selected)
            existing_info = state.get("info")
            info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
            try:
                output_text = _completion_text(state.get("completion", []))
            except ValueError:
                output_text = ""
            has_orphan_think_close = _has_orphan_think_close_tag(output_text)
            applied_format_penalty = (
                self.orphan_think_close_penalty if has_orphan_think_close else 0.0
            )
            score = self.non_termination_penalty + applied_format_penalty
            info["jtc_rubrichub_judge"] = {
                "status": "non_terminated_policy_output",
                "error": "policy generation reached its token limit before termination",
                "source_record_id": source_record_id,
                "judge_record_id": None,
                "judge_model_path": self.judge_model_path,
                "judge_service_model_path": self.judge_service_model_path,
                "judge_profile": None,
                "judge_called": False,
                "max_rubrics": self.max_rubrics,
                "rubric_sample_seed": self.rubric_sample_seed,
                "selected_rubrics": selected,
                "judgments": [],
                "weighted_pass_sum": 0.0,
                "selected_weight_sum": selected_weight_sum,
                "non_termination_penalty": self.non_termination_penalty,
                "orphan_think_close_tag": has_orphan_think_close,
                "orphan_think_close_penalty": self.orphan_think_close_penalty,
                "applied_orphan_think_close_penalty": applied_format_penalty,
                "score": score,
            }
            state["info"] = info
            state["reward"] = score
            state["metrics"] = {
                "jtc_weighted_score": score,
                "invalid_final_output": 0.0,
                "non_terminated_policy_output": 1.0,
                "orphan_think_close_tag": float(has_orphan_think_close),
                "orphan_think_close_penalty": applied_format_penalty,
            }
            return

        try:
            score = await self.jtc_weighted_score(
                completion=state["completion"],
                prompt=state["prompt"],
                answer=state.get("answer", ""),
                state=state,
            )
        except ValueError as error:
            if str(error) != "final assistant content must be non-empty text":
                raise
            source_record_id, rubrics = _answer_payload(state.get("answer", ""))
            selected = select_rubrics(
                rubrics,
                max_rubrics=self.max_rubrics,
                sample_seed=self.rubric_sample_seed,
                record_id=source_record_id,
            )
            selected_weight_sum = sum(float(item["weight"]) for item in selected)
            existing_info = state.get("info")
            info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
            score = 0.0
            info["jtc_rubrichub_judge"] = {
                "status": "invalid_policy_output",
                "error": str(error),
                "source_record_id": source_record_id,
                "judge_record_id": None,
                "judge_model_path": self.judge_model_path,
                "judge_service_model_path": self.judge_service_model_path,
                "judge_profile": None,
                "judge_called": False,
                "max_rubrics": self.max_rubrics,
                "rubric_sample_seed": self.rubric_sample_seed,
                "selected_rubrics": selected,
                "judgments": [],
                "weighted_pass_sum": 0.0,
                "selected_weight_sum": selected_weight_sum,
                "orphan_think_close_tag": False,
                "orphan_think_close_penalty": self.orphan_think_close_penalty,
                "applied_orphan_think_close_penalty": 0.0,
                "score": score,
            }
            state["info"] = info
            invalid_final_output = 1.0
        else:
            invalid_final_output = 0.0
        audit = state["info"]["jtc_rubrichub_judge"]
        has_orphan_think_close = bool(audit["orphan_think_close_tag"])
        applied_format_penalty = float(
            audit["applied_orphan_think_close_penalty"]
        )
        state["reward"] = score
        state["metrics"] = {
            "jtc_weighted_score": score,
            "invalid_final_output": invalid_final_output,
            "non_terminated_policy_output": 0.0,
            "orphan_think_close_tag": float(has_orphan_think_close),
            "orphan_think_close_penalty": applied_format_penalty,
        }
        self._add_shadow_metrics(state)


def _load_rubrichub_dataset(dataset: str, split: str) -> Dataset:
    path = Path(dataset)
    if path.is_dir():
        loaded = load_from_disk(str(path))
        if isinstance(loaded, DatasetDict):
            if split not in loaded:
                raise ValueError(
                    f"{dataset}: split {split!r} not found; available: {list(loaded)}"
                )
            rows = loaded[split]
        elif isinstance(loaded, Dataset):
            rows = loaded
        else:
            raise TypeError(f"{dataset}: unsupported saved dataset type")
    elif path.is_file():
        suffix = path.suffix.casefold()
        if suffix in {".json", ".jsonl"}:
            rows = load_dataset("json", data_files=str(path), split="train")
        else:
            raise ValueError(f"unsupported local dataset file: {dataset}")
    else:
        rows = load_dataset(dataset, split=split)
    return rows.map(
        normalize_rubrichub_row,
        with_indices=True,
        fn_kwargs={"dataset_name": dataset, "split": split},
        desc="Normalizing RubricHub rows",
    )


def load_environment(
    dataset: str = DEFAULT_RUBRICHUB_DATASET,
    split: str = "train",
    judge_model_path: str = JTC4B_STEP400,
    judge_server_url: str = "http://127.0.0.1:1212/judge",
    judge_service_model_path: str = "judge",
    judge_timeout: float = 240,
    judge_max_retries: int = 3,
    max_rubrics: int = 8,
    rubric_sample_seed: int = 0,
    non_termination_penalty: float = -0.25,
    orphan_think_close_penalty: float = -0.2,
    shadow_judge_model_path: str | None = None,
    shadow_judge_server_url: str | None = None,
    shadow_judge_service_model_path: str = "judge",
    shadow_judge_timeout: float = 300,
    shadow_judge_max_retries: int = 1,
    shadow_judge_sample_rate: float = 0.0,
    shadow_judge_sample_seed: int = 0,
    shadow_judge_audit_dir: str | None = None,
) -> vf.SingleTurnEnv:
    """Prime-RL factory for single-turn RubricHub RL with a JTC judge."""
    judge_client = JudgeClient(
        judge_server_url,
        service_model_path=judge_service_model_path,
        timeout=judge_timeout,
        max_retries=judge_max_retries,
    )
    shadow_judge_client = None
    if shadow_judge_model_path is not None:
        shadow_judge_client = JudgeClient(
            shadow_judge_server_url or judge_server_url,
            service_model_path=shadow_judge_service_model_path,
            timeout=shadow_judge_timeout,
            max_retries=shadow_judge_max_retries,
        )
    return vf.SingleTurnEnv(
        dataset=_load_rubrichub_dataset(dataset, split),
        rubric=JTCJudgeRubric(
            judge_client=judge_client,
            judge_model_path=judge_model_path,
            judge_service_model_path=judge_service_model_path,
            max_rubrics=max_rubrics,
            rubric_sample_seed=rubric_sample_seed,
            non_termination_penalty=non_termination_penalty,
            orphan_think_close_penalty=orphan_think_close_penalty,
            shadow_judge_client=shadow_judge_client,
            shadow_judge_model_path=shadow_judge_model_path,
            shadow_judge_service_model_path=shadow_judge_service_model_path,
            shadow_judge_sample_rate=shadow_judge_sample_rate,
            shadow_judge_sample_seed=shadow_judge_sample_seed,
            shadow_judge_audit_dir=shadow_judge_audit_dir,
        ),
    )


__all__ = [
    "DEFAULT_RUBRICHUB_DATASET",
    "JTC4B_STEP400",
    "JTCJudgeRubric",
    "JudgeOutputClient",
    "load_environment",
    "normalize_rubrichub_row",
    "select_rubrics",
]
