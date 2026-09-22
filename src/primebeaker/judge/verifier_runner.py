"""In-memory JRecord-to-verifier execution for :mod:`judge_server`."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from concurrent.futures import Executor
from typing import Any, Awaitable, Callable

from openai import AsyncOpenAI
from jtcflow.model.utils import extract_json_from_response
from jtc.common.workflows.inference_workflow import build_workflow_results, workflow_inputs_from_rows
from jtc.common.jrow_sampling import sample_jrows
from jtc.common.schema import (
    BINARY_RUBRIC_SCORES,
    JRecord,
    RubricDefinition,
    RubricType,
    SourceInfo,
)
from jtc.common.templates import load_prompt_template, render_chat_template
from primebeaker.judge.model_profiles import load_model_profile
from jtc.common.workflows.verifier_tool_use_workflow import (
    DEFAULT_ENVIRONMENT_ID,
    ToolUseWorkflow,
    _openai_api_base,
    bind_advertised_tools,
    build_rollout_client,
    build_verifier_environment,
    build_tool_clients,
    normalize_tools,
)


logger = logging.getLogger(__name__)

GROUPED_JUDGMENTS_TOOL_NAME = "submit_grouped_judgments"


def build_grouped_judgments_tool(
    rubric_indices: list[int],
) -> dict[str, Any]:
    """Build the forced response tool for one grouped-judgment attempt."""
    indices = list(rubric_indices)
    if not indices:
        raise ValueError("grouped judgment tool requires at least one rubric_id")
    if len(indices) != len(set(indices)):
        raise ValueError("grouped judgment tool rubric_ids must be unique")
    return {
        "type": "function",
        "function": {
            "name": GROUPED_JUDGMENTS_TOOL_NAME,
            "description": (
                "Submit exactly one final pass/fail judgment for every requested "
                "rubric. This must be the final response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "judgments": {
                        "type": "array",
                        "minItems": len(indices),
                        "maxItems": len(indices),
                        "items": {
                            "type": "object",
                            "properties": {
                                "rubric_id": {"type": "integer", "enum": indices},
                                "label": {
                                    "type": "string",
                                    "enum": ["pass", "fail"],
                                },
                                "feedback": {"type": "string", "minLength": 1},
                            },
                            "required": ["rubric_id", "label", "feedback"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["judgments"],
                "additionalProperties": False,
            },
        },
    }


def _jsonable_tool_calls(message: Any) -> list[Any]:
    """Serialize SDK tool-call objects for traces and failure diagnostics."""
    tool_calls = getattr(message, "tool_calls", None) or []
    serialized: list[Any] = []
    for tool_call in tool_calls:
        if hasattr(tool_call, "model_dump"):
            serialized.append(tool_call.model_dump(mode="json"))
        elif isinstance(tool_call, dict):
            serialized.append(tool_call)
        else:
            function = getattr(tool_call, "function", None)
            serialized.append(
                {
                    "id": getattr(tool_call, "id", None),
                    "type": getattr(tool_call, "type", None),
                    "function": {
                        "name": getattr(function, "name", None),
                        "arguments": getattr(function, "arguments", None),
                    },
                }
            )
    return serialized


def parse_grouped_judgments_tool_call(message: Any) -> dict[str, Any]:
    """Parse the sole required ``submit_grouped_judgments`` function call."""
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        raise RuntimeError(
            f"grouped judge did not call {GROUPED_JUDGMENTS_TOOL_NAME}"
        )
    if len(tool_calls) != 1:
        raise RuntimeError(
            "grouped judge must return exactly one final tool call; "
            f"received {len(tool_calls)}"
        )
    tool_call = tool_calls[0]
    call_type = (
        tool_call.get("type")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "type", None)
    )
    if call_type not in (None, "function"):
        raise RuntimeError(
            f"grouped judge returned unsupported tool-call type {call_type!r}"
        )
    function = (
        tool_call.get("function")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "function", None)
    )
    name = (
        function.get("name")
        if isinstance(function, dict)
        else getattr(function, "name", None)
    )
    if name != GROUPED_JUDGMENTS_TOOL_NAME:
        raise RuntimeError(
            f"grouped judge called {name!r}; expected "
            f"{GROUPED_JUDGMENTS_TOOL_NAME!r}"
        )
    arguments = (
        function.get("arguments")
        if isinstance(function, dict)
        else getattr(function, "arguments", None)
    )
    if isinstance(arguments, str):
        try:
            payload = json.loads(arguments)
        except json.JSONDecodeError:
            try:
                # Some judge models copy literal control characters (for
                # example a tab from LaTeX) into otherwise valid tool-call
                # arguments. Accept those while keeping strict JSON as the
                # normal parsing path.
                payload = json.loads(arguments, strict=False)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "grouped judge tool arguments are not valid JSON"
                ) from exc
    elif isinstance(arguments, dict):
        payload = arguments
    else:
        raise RuntimeError("grouped judge tool arguments must be a JSON object")
    if not isinstance(payload, dict):
        raise RuntimeError("grouped judge tool arguments must be a JSON object")
    return payload



def build_judge_record(request: Any):
    """Create a single-output JRecord with one binary definition per rubric."""
    record_id = f"judge-{uuid.uuid4()}"
    return JRecord(
        id=record_id,
        input=request.input,
        output=request.output,
        rubrics=[
            RubricDefinition(
                rubric_type=RubricType.BINARY,
                criteria=rubric,
                scores=dict(BINARY_RUBRIC_SCORES),
            )
            for rubric in request.rubrics
        ],
        source=SourceInfo(
            name="judge_server",
            source_id=record_id,
            misc={"judging_model": request.model},
        ),
    )


def build_grouped_rubric_messages(
    record: JRecord,
    prompt_template_path: str,
    rubric_indices: list[int] | None = None,
) -> list[dict[str, str]]:
    """Render one chat containing the candidate and selected stable rubric IDs."""
    indices = (
        list(range(len(record.rubrics)))
        if rubric_indices is None
        else list(rubric_indices)
    )
    lines: list[str] = []
    for rubric_index in indices:
        rubric = record.rubrics[rubric_index]
        allowed_labels = ", ".join((rubric.scores or {}).keys()) or "unspecified"
        lines.extend(
            [
                f'<rubric rubric_id="{rubric_index}">',
                f"<criterion>{rubric.criteria or ''}</criterion>",
                f"<allowed_labels>{allowed_labels}</allowed_labels>",
                "</rubric>",
            ]
        )
    return render_chat_template(
        load_prompt_template(prompt_template_path),
        inputs="" if record.input is None else record.input,
        output="" if record.output is None else record.output,
        indexed_rubrics="\n".join(lines),
    )


def serialize_grouped_judgments(
    raw_text: str,
    record: JRecord,
    *,
    trace: dict[str, Any],
) -> list[dict[str, Any]]:
    """Map one strict grouped response back to the judge API schema."""
    if not (raw_text or "").strip():
        raise RuntimeError("empty model response")
    payload = extract_json_from_response(raw_text)
    raw_judgments = payload.get("judgments") if isinstance(payload, dict) else None
    if not isinstance(raw_judgments, list):
        raise RuntimeError(
            "response did not contain an object with a judgments list"
        )

    expected_indices = set(range(len(record.rubrics)))
    judgments: dict[int, dict[str, Any]] = {}
    received_indices: list[int] = []
    for position, item in enumerate(raw_judgments):
        if not isinstance(item, dict):
            raise RuntimeError(f"judgment {position} is not an object")
        rubric_index = item.get("rubric_id")
        if isinstance(rubric_index, bool) or not isinstance(rubric_index, int):
            raise RuntimeError(
                f"judgment {position} has a non-integer rubric_id"
            )
        label = item.get("label")
        feedback = item.get("feedback")
        if not isinstance(label, str) or not label.strip():
            raise RuntimeError(f"judgment {position} has an empty label")
        if not isinstance(feedback, str):
            raise RuntimeError(
                f"judgment {position} has non-string feedback"
            )

        normalized_label = label.strip().casefold()
        if rubric_index in expected_indices:
            allowed = set(record.rubrics[rubric_index].scores or {})
            if allowed and normalized_label not in allowed:
                raise RuntimeError(
                    f"rubric_id {rubric_index} label {label!r} "
                    f"is not one of {sorted(allowed)!r}"
                )
        received_indices.append(rubric_index)
        judgments[rubric_index] = {
            "rubric_index": rubric_index,
            "label": normalized_label,
            "feedback": feedback.strip(),
            "trace": {**trace, "rubric_index": rubric_index},
        }

    if len(received_indices) != len(set(received_indices)):
        raise RuntimeError("response contains duplicate rubric_id values")
    if set(received_indices) != expected_indices:
        raise RuntimeError(
            f"response rubric_ids {sorted(received_indices)} do not exactly match "
            f"expected {sorted(expected_indices)}"
        )
    return [judgments[index] for index in range(len(record.rubrics))]


async def collect_grouped_judgments_with_retries(
    record: JRecord,
    call_model: Callable[
        [int, list[int]], Awaitable[tuple[dict[str, Any], dict[str, Any]]]
    ],
    *,
    max_retries: int,
    evaluation_timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Retry only missing grouped judgments while retaining valid ones."""
    if evaluation_timeout is not None:
        if evaluation_timeout <= 0:
            raise ValueError("evaluation_timeout must be positive")
        try:
            async with asyncio.timeout(evaluation_timeout):
                return await collect_grouped_judgments_with_retries(
                    record,
                    call_model,
                    max_retries=max_retries,
                    evaluation_timeout=None,
                )
        except TimeoutError as exc:
            raise RuntimeError(
                f"judge evaluation exceeded {evaluation_timeout:g} seconds"
            ) from exc

    if max_retries < 0:
        raise ValueError("rubric max_retries cannot be negative")

    expected_indices = set(range(len(record.rubrics)))
    complete: dict[int, dict[str, Any]] = {}
    last_failures: dict[int, str] = {}
    pending_indices = expected_indices
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        selected_indices = sorted(pending_indices)
        try:
            payload, trace = await call_model(attempt, selected_indices)
            raw_judgments = payload.get("judgments") if isinstance(payload, dict) else None
            if not isinstance(raw_judgments, list):
                raise RuntimeError(
                    "grouped judge response must contain a judgments list"
                )

            seen: set[int] = set()
            attempt_complete: dict[int, dict[str, Any]] = {}
            attempt_failures: dict[int, str] = {}
            selected_set = set(selected_indices)
            for item in raw_judgments:
                if not isinstance(item, dict):
                    continue
                rubric_index = item.get("rubric_id")
                if (
                    isinstance(rubric_index, bool)
                    or not isinstance(rubric_index, int)
                    or rubric_index not in selected_set
                ):
                    continue
                if rubric_index in seen:
                    attempt_complete.pop(rubric_index, None)
                    attempt_failures[rubric_index] = (
                        f"duplicate rubric_id={rubric_index}"
                    )
                    continue
                seen.add(rubric_index)
                label = item.get("label")
                normalized_label = (
                    label.strip().casefold() if isinstance(label, str) else ""
                )
                feedback = item.get("feedback")
                if normalized_label not in BINARY_RUBRIC_SCORES:
                    attempt_failures[rubric_index] = (
                        f"unsupported judge label: {label!r}"
                    )
                    continue
                if not isinstance(feedback, str):
                    attempt_failures[rubric_index] = (
                        f"invalid feedback for rubric_id={rubric_index}"
                    )
                    continue
                attempt_complete[rubric_index] = {
                    "rubric_index": rubric_index,
                    "label": normalized_label,
                    "feedback": feedback,
                    "trace": {**trace, "rubric_index": rubric_index},
                }

            for rubric_index in selected_set.difference(seen):
                attempt_failures[rubric_index] = (
                    "grouped judge returned no judgment for "
                    f"rubric_id={rubric_index}"
                )
            complete.update(attempt_complete)
            last_failures.update(attempt_failures)
        except Exception as exc:
            for rubric_index in selected_indices:
                last_failures[rubric_index] = str(exc)

        pending_indices = expected_indices.difference(complete)
        if not pending_indices:
            return [complete[index] for index in range(len(record.rubrics))]
        if attempt < max_retries:
            logger.warning(
                "Retrying %d missing/invalid grouped rubric judgment(s), "
                "retry %d/%d: indices=%s",
                len(pending_indices),
                attempt + 1,
                max_retries,
                sorted(pending_indices),
            )

    details = "; ".join(
        f"rubric_id={index}: {last_failures.get(index, 'unknown failure')}"
        for index in sorted(pending_indices)
    )
    raise RuntimeError(
        "grouped rubric judge failed after "
        f"{total_attempts} attempt(s): {details}"
    )


def _partition_complete_judgments(
    results: list[dict[str, Any]], *, expected_indices: set[int]
) -> tuple[dict[int, dict[str, Any]], dict[int, str]]:
    """Return complete judgments and retryable failures keyed by rubric index."""
    by_index: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    seen: set[int] = set()
    for result in results:
        final_row = result.get("final_row")
        if final_row is None:
            continue
        rubric_index = final_row.rubric_index
        if (
            isinstance(rubric_index, bool)
            or not isinstance(rubric_index, int)
            or rubric_index not in expected_indices
        ):
            raise RuntimeError(f"judge returned invalid rubric_index: {rubric_index!r}")
        if rubric_index in seen:
            raise RuntimeError(f"judge returned duplicate rubric_index: {rubric_index}")
        seen.add(rubric_index)

        rubric = final_row.rubric
        label = rubric.label
        normalized_label = label.strip().casefold() if isinstance(label, str) else ""
        if normalized_label not in BINARY_RUBRIC_SCORES:
            metadata = getattr(rubric, "workflow_metadata", {}) or {}
            failures[rubric_index] = (
                "judge produced an unfinished judgment: "
                f"rubric_index={rubric_index}, label={label!r}, "
                f"route={metadata.get('route')!r}, "
                f"stop_condition={metadata.get('stop_condition')!r}, "
                f"error={metadata.get('error')!r}"
            )
            continue
        by_index[rubric_index] = {
            "rubric_index": rubric_index,
            "label": normalized_label,
            "feedback": rubric.feedback,
            "trace": rubric.to_dict(),
        }

    for rubric_index in expected_indices.difference(seen):
        failures[rubric_index] = (
            "judge workflow returned no final row: "
            f"rubric_index={rubric_index}"
        )
    return by_index, failures


def serialize_complete_judgments(
    results: list[dict[str, Any]], *, expected_count: int
) -> list[dict[str, Any]]:
    """Serialize complete binary judgments, rejecting unfinished results."""
    expected = set(range(expected_count))
    by_index, failures = _partition_complete_judgments(
        results, expected_indices=expected
    )
    if failures:
        first_failed_index = min(failures)
        raise RuntimeError(failures[first_failed_index])
    return [by_index[index] for index in range(expected_count)]


async def collect_judgments_with_retries(
    rows: list[Any],
    run_rows: Callable[[list[Any]], Awaitable[list[dict[str, Any]]]],
    *,
    max_retries: int,
    evaluation_timeout: float | None = None,
) -> list[dict[str, Any]]:
    """Retry only unfinished rubric rows while retaining completed judgments."""
    if evaluation_timeout is not None:
        if evaluation_timeout <= 0:
            raise ValueError("evaluation_timeout must be positive")
        try:
            async with asyncio.timeout(evaluation_timeout):
                return await collect_judgments_with_retries(
                    rows,
                    run_rows,
                    max_retries=max_retries,
                    evaluation_timeout=None,
                )
        except TimeoutError as exc:
            raise RuntimeError(
                f"judge evaluation exceeded {evaluation_timeout:g} seconds"
            ) from exc

    if max_retries < 0:
        raise ValueError("rubric max_retries cannot be negative")

    rows_by_index = {row.rubric_index: row for row in rows}
    expected_indices = set(range(len(rows)))
    if set(rows_by_index) != expected_indices:
        raise RuntimeError(
            "judge rows must contain each rubric_index exactly once: "
            f"expected={sorted(expected_indices)}, actual={sorted(rows_by_index)}"
        )

    complete: dict[int, dict[str, Any]] = {}
    last_failures: dict[int, str] = {}
    pending_indices = expected_indices
    total_attempts = max_retries + 1
    for attempt in range(total_attempts):
        selected_indices = sorted(pending_indices)
        selected_rows = [rows_by_index[index] for index in selected_indices]
        results = await run_rows(selected_rows)
        attempt_complete, attempt_failures = _partition_complete_judgments(
            results, expected_indices=set(selected_indices)
        )
        complete.update(attempt_complete)
        last_failures.update(attempt_failures)
        pending_indices = expected_indices.difference(complete)
        if not pending_indices:
            return [complete[index] for index in range(len(rows))]
        if attempt < max_retries:
            logger.warning(
                "Retrying %d unfinished rubric judgment(s), retry %d/%d: indices=%s",
                len(pending_indices),
                attempt + 1,
                max_retries,
                sorted(pending_indices),
            )

    details = "; ".join(
        last_failures.get(index, f"rubric_index={index}: unknown failure")
        for index in sorted(pending_indices)
    )
    raise RuntimeError(
        "judge produced unfinished judgments after "
        f"{total_attempts} attempt(s): {details}"
    )


async def run_grouped_rubric_judge(
    record: JRecord,
    request: Any,
    *,
    model_server: str,
    profile: Any,
) -> list[dict[str, Any]]:
    """Request one forced function call covering all rubrics in ``record``."""
    if profile.tools or profile.max_turns != 1 or profile.max_tool_calls != 0:
        raise ValueError("grouped_rubrics profiles must be single-turn and tool-free")

    async with AsyncOpenAI(
        api_key="EMPTY",
        base_url=_openai_api_base(model_server),
        timeout=profile.timeout,
        max_retries=profile.max_retries,
    ) as client:

        async def call_grouped_model(
            attempt: int, rubric_indices: list[int]
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            messages = build_grouped_rubric_messages(
                record, profile.prompt_template_path, rubric_indices
            )
            response_tool = build_grouped_judgments_tool(rubric_indices)
            response = await client.chat.completions.create(
                model=request.model,
                messages=messages,
                max_tokens=profile.max_tokens,
                temperature=profile.temperature,
                tools=[response_tool],
                tool_choice={
                    "type": "function",
                    "function": {"name": GROUPED_JUDGMENTS_TOOL_NAME},
                },
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False}
                },
            )
            if not response.choices:
                raise RuntimeError("grouped judge returned no choices")
            choice = response.choices[0]
            content = choice.message.content or ""
            reasoning_content = getattr(choice.message, "reasoning_content", None)
            tool_calls = _jsonable_tool_calls(choice.message)
            usage = (
                response.usage.model_dump(mode="json")
                if response.usage is not None
                else None
            )
            trace = {
                "rubric_mode": "grouped_rubrics",
                "attempt": attempt + 1,
                "requested_rubric_indices": rubric_indices,
                "messages": messages,
                "response": {
                    "id": response.id,
                    "model": response.model,
                    "finish_reason": choice.finish_reason,
                    "content": content,
                    "reasoning_content": reasoning_content,
                    "tool_calls": tool_calls,
                    "usage": usage,
                },
            }
            try:
                payload = parse_grouped_judgments_tool_call(choice.message)
            except Exception:
                logger.exception(
                    "Invalid grouped judge tool response: finish_reason=%r "
                    "content=%r reasoning_content=%r tool_calls=%r",
                    choice.finish_reason,
                    content[:2000],
                    (reasoning_content or "")[:2000],
                    tool_calls,
                )
                raise
            return payload, trace

        return await collect_grouped_judgments_with_retries(
            record,
            call_grouped_model,
            max_retries=profile.rubric_max_retries,
            evaluation_timeout=profile.evaluation_timeout,
        )


async def run_verifier_tool_judge(
    request: Any,
    *,
    config: Any,
    executor: Executor | None = None,
) -> dict[str, Any]:
    """Run the verifier tool-use workflow entirely in memory.

    The workflow's callable boundary is synchronous, while its verifier maps
    execute asynchronous rollouts. Running that boundary in a worker thread
    prevents a Starlette request from blocking the application event loop.
    """
    # The local LiteRegistry gateway discovers and selects model replicas.
    model_server = config.model_gateway_url

    profile = load_model_profile(request.model, config)
    record = build_judge_record(request)
    if profile.rubric_mode == "grouped_rubrics":
        judgments = await run_grouped_rubric_judge(
            record,
            request,
            model_server=model_server,
            profile=profile,
        )
        return {
            "record_id": record.id,
            "model": request.model,
            "profile": str(profile.source_path),
            "rubric_mode": profile.rubric_mode,
            "judgments": judgments,
        }

    backend_tools = normalize_tools(profile.tools)
    prompt_template = load_prompt_template(profile.prompt_template_path)
    tool_clients = build_tool_clients(
        tools=backend_tools,
        server_url=profile.tool_server_url,
        local_search_model_path=profile.local_search_model_path,
    )
    tools, tool_clients = bind_advertised_tools(
        backend_tools=backend_tools,
        tool_clients=tool_clients,
        tool_defs=prompt_template.get("tools"),
    )
    rollout_client = build_rollout_client(
        api_base_url=model_server,
        api_key=None,
        timeout=profile.timeout,
        max_retries=profile.max_retries,
        connect_timeout=min(5.0, profile.timeout),
    )
    verifier = build_verifier_environment(
        DEFAULT_ENVIRONMENT_ID,
        tool_clients=tool_clients.tools,
        tools=tools,
        tool_defs=prompt_template.get("tools"),
        max_tool_calls=profile.max_tool_calls if tools else 0,
        max_turns=profile.max_turns if tools else 1,
        timeout_seconds=profile.rollout_timeout,
        score_rollouts=False,
    )
    workflow = ToolUseWorkflow(
        prompt_chat_template_spec=prompt_template,
        verifier=verifier,
        rollout_client=rollout_client,
        rollout_model=request.model,
        sampling_args={
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "tool_choice": "auto" if tools else "none",
        },
        batch_size=len(request.rubrics),
        stream_output_jsonl_path=None,
        show_progress=False,
        response_format="json",
    )

    rows, _ = sample_jrows(
        [record],
        max_outputs_per_record=1,
        max_rubrics_per_record=0,## no limit

        keep_empty_outputs=True,
    )
    async def run_rows(selected_rows: list[Any]) -> list[dict[str, Any]]:
        frozen_rows = tuple(selected_rows)
        raw_results = await asyncio.get_running_loop().run_in_executor(
            executor,
            lambda: list(workflow(workflow_inputs_from_rows(frozen_rows))),
        )
        return build_workflow_results(
            rows=list(frozen_rows),
            code_results=raw_results,
            direct_feedback_results=[],
            invalid_results=[],
        )

    judgments = await collect_judgments_with_retries(
        rows,
        run_rows,
        max_retries=profile.rubric_max_retries,
        evaluation_timeout=profile.evaluation_timeout,
    )
    return {
        "record_id": record.id,
        "model": request.model,
        "profile": str(profile.source_path),
        "rubric_mode": profile.rubric_mode,
        "judgments": judgments,
    }
