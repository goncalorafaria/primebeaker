"""Judge-scored search-agent environment with local-corpus document visits.

This is deliberately separate from :mod:`jtc_search_visit_agent_env`: the
existing environment retains deterministic exact-match scoring, while this
variant sends the complete final assistant message to the registered judge
pool and uses its binary pass/fail label as the answer-correctness reward.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import verifiers as vf

from literegistry_tool_client import FetchClient, JudgeClient, SearchClient
from .jtc_search_agent_env import (
    _has_tool_calls,
    _load_dataset,
    _message_content,
)
from .search_visit_agent_env import SearchVisitAgentEnv


DEFAULT_JUDGE_RUBRIC_TEMPLATE = "The final answer the model concluded is: {answer}"
JUDGE_FINAL_TOOL_CALL_WARNING = """IMPORTANT: This was your last allowed tool call.
Do not call another tool. Give your final evidence-based answer now, with
supporting source URLs. There is no required JSON format."""


class JudgeOutputClient(Protocol):
    """Minimal client contract consumed by :class:`JudgeAnswerRubric`."""

    async def verify_output(
        self,
        *,
        input: str,
        output: str,
        rubrics: Sequence[str],
        model: str,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class GatewayJudgeClient:
    """Route judge requests through the local LiteRegistry HTTP gateway.

    ``JudgeClient`` sends the gateway-only ``model_path`` selector for the
    replicated CPU judge pool while preserving ``model`` as the LLM that the
    judge server should use. The gateway owns Redis discovery and rotation, so
    a rollout worker never connects to the registry directly.
    """

    def __init__(
        self,
        server_url: str,
        *,
        service_model_path: str = "judge",
        timeout: float = 240,
        max_retries: int = 3,
    ) -> None:
        if not isinstance(server_url, str) or not server_url.strip():
            raise ValueError("judge server_url must be a non-empty string")
        if not isinstance(service_model_path, str) or not service_model_path.strip():
            raise ValueError("judge service_model_path must be non-empty")
        if timeout <= 0:
            raise ValueError("judge timeout must be positive")
        if max_retries < 1:
            raise ValueError("judge max_retries must be at least 1")
        self.server_url = server_url.rstrip("/")
        self.service_model_path = service_model_path.strip()
        self._client = JudgeClient(
            self.server_url,
            service_model_path=self.service_model_path,
            timeout=timeout,
            max_retries=max_retries,
        )

    async def verify_output(
        self,
        *,
        input: str,
        output: str,
        rubrics: Sequence[str],
        model: str,
    ) -> dict[str, Any]:
        if not isinstance(input, str):
            raise TypeError("judge input must be a string")
        if not isinstance(output, str):
            raise TypeError("judge output must be a string")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("judge model must be a non-empty model path")
        if isinstance(rubrics, (str, bytes)) or not isinstance(rubrics, Sequence):
            raise TypeError("judge rubrics must be a sequence")
        normalized = list(rubrics)
        if not normalized or any(
            not isinstance(rubric, str) or not rubric.strip()
            for rubric in normalized
        ):
            raise ValueError("judge rubrics must contain non-empty text")
        return await self._client.verify_output(
            input=input,
            output=output,
            rubrics=normalized,
            model=model.strip(),
        )

    async def close(self) -> None:
        # JudgeClient creates and closes its HTTP session per request.
        return None


def _prompt_text(prompt: Any, question: Any) -> str:
    if isinstance(question, str) and question.strip():
        return question.strip()
    if isinstance(prompt, str):
        return prompt.strip()
    if isinstance(prompt, list):
        parts = [
            _message_content(message).strip()
            for message in prompt
            if _message_content(message).strip()
        ]
        return "\n\n".join(parts)
    return str(prompt or "").strip()


class JudgeAnswerRubric(vf.Rubric):
    """Use the external judge as the environment's only reward function."""

    def __init__(
        self,
        *,
        judge_client: JudgeOutputClient,
        judge_model_path: str,
        rubric_template: str = DEFAULT_JUDGE_RUBRIC_TEMPLATE,
    ) -> None:
        if not isinstance(judge_model_path, str) or not judge_model_path.strip():
            raise ValueError("judge_model_path must be a non-empty model path")
        if not isinstance(rubric_template, str) or "{answer}" not in rubric_template:
            raise ValueError("judge rubric template must contain {answer}")
        # Validate all format fields now, before a training rollout starts.
        try:
            rubric_template.format(answer="example")
        except (KeyError, ValueError) as exc:
            raise ValueError("invalid judge rubric template") from exc
        self.judge_client = judge_client
        self.judge_model_path = judge_model_path.strip()
        self.rubric_template = rubric_template
        super().__init__(funcs=[self.judge_answer])

    async def judge_answer(
        self,
        completion: list[Any],
        answer: Any,
        prompt: Any,
        question: Any = "",
        state: dict[str, Any] | None = None,
    ) -> float:
        """Return one only when the judge passes the full final message.

        The legacy Verifiers bridge copies ``state["info"]`` into the durable
        Prime-RL ``Trace.info`` field. Preserve the exact judge request and
        response there so the judge's messages/tool transcript remain
        auditable next to the policy rollout that they scored.
        """
        if (
            not completion
            or _has_tool_calls(completion[-1])
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            return 0.0
        final_message = _message_content(completion[-1]).strip()
        if not final_message:
            return 0.0
        request = {
            "input": _prompt_text(prompt, question),
            "output": final_message,
            "rubrics": [self.rubric_template.format(answer=answer.strip())],
            "model": self.judge_model_path,
        }
        response = await self.judge_client.verify_output(**request)
        if state is not None:
            existing_info = state.get("info")
            info = dict(existing_info) if isinstance(existing_info, Mapping) else {}
            info["judge"] = {"request": request, "response": response}
            state["info"] = info
        judgments = response.get("judgments")
        if not isinstance(judgments, list) or len(judgments) != 1:
            raise RuntimeError("judge response must contain exactly one judgment")
        judgment = judgments[0]
        if not isinstance(judgment, Mapping):
            raise RuntimeError("judge judgment must be an object")
        label = judgment.get("label")
        normalized_label = label.strip().casefold() if isinstance(label, str) else ""
        if normalized_label == "pass":
            return 1.0
        if normalized_label == "fail":
            return 0.0
        raise RuntimeError(f"judge returned unsupported label: {label!r}")

    @vf.teardown
    async def close_judge_client(self) -> None:
        await self.judge_client.close()


def load_environment(
    dataset: str,
    judge_model_path: str,
    split: str = "train",
    search_server_url: str = "http://127.0.0.1:1212/search",
    local_search_model_path: str = "localsearch:bc-v2-72k",
    judge_server_url: str | None = None,
    judge_registry: str | None = None,
    judge_service_model_path: str = "judge",
    judge_timeout: float = 240,
    judge_max_retries: int = 3,
    judge_rubric_template: str = DEFAULT_JUDGE_RUBRIC_TEMPLATE,
    timeout: float = 65,
    max_retries: int = 3,
    visit_max_content_length: int | None = 64000,
    max_turns: int | None = None,
    max_tool_calls: int = 150,
    template_path: str = "templates/search-agent-visit.json",
    **_: Any,
) -> SearchVisitAgentEnv:
    """Build the judge-scored search + direct-visit environment."""
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be at least 1")
    resolved_max_turns = max_turns or max_tool_calls + 1
    if resolved_max_turns < max_tool_calls + 1:
        raise ValueError("max_turns must reserve a final-answer turn")
    del judge_registry  # Legacy compatibility; judge routing is gateway-only.
    resolved_judge_url = judge_server_url
    if resolved_judge_url is None:
        if not search_server_url.rstrip("/").endswith("/search"):
            raise ValueError("judge_server_url is required when search_server_url is not a /search URL")
        resolved_judge_url = search_server_url.rstrip("/")[:-len("/search")] + "/judge"
    judge_client = GatewayJudgeClient(
        resolved_judge_url,
        service_model_path=judge_service_model_path,
        timeout=judge_timeout,
        max_retries=judge_max_retries,
    )
    return SearchVisitAgentEnv(
        dataset=_load_dataset(dataset, split, template_path),
        search_client=SearchClient(
            search_server_url,
            model_path=local_search_model_path,
            timeout=timeout,
            max_retries=max_retries,
        ),
        fetch_client=FetchClient(
            local_search_server_url=search_server_url,
            local_search_model_path=local_search_model_path,
            timeout=timeout,
            max_retries=max_retries,
            max_content_length=visit_max_content_length,
        ),
        max_turns=resolved_max_turns,
        max_tool_calls=max_tool_calls,
        final_tool_call_warning=JUDGE_FINAL_TOOL_CALL_WARNING,
        rubric=JudgeAnswerRubric(
            judge_client=judge_client,
            judge_model_path=judge_model_path,
            rubric_template=judge_rubric_template,
        ),
    )


__all__ = [
    "DEFAULT_JUDGE_RUBRIC_TEMPLATE",
    "JUDGE_FINAL_TOOL_CALL_WARNING",
    "JudgeAnswerRubric",
    "GatewayJudgeClient",
    "load_environment",
]
