"""PrimeBeaker-owned scheduling for JTC evaluation workloads.

JTC supplies Python workload models and worker modules.  This module owns the
YAML schema, LiteRegistry service capacity, Beaker placement, readiness, and
service shutdown through :mod:`primebeaker.evaluation`.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml

from jtc.eval.launcher import EvaluationWorkloadConfig
from jtc.eval.rubric_judge_audit import (
    AuditOutputConfig,
    InferenceConfig,
    RubricJudgeAuditWorkloadConfig,
    SampleConfig,
    atomic_json,
    audit_complete,
    finish_audit,
    jsonl_count,
    prepare_sample,
    sample_complete,
)
from jtc.eval.search_agent import SearchAgentWorkloadConfig
from jtc.eval.rejection_sampling import RejectionSamplingWorkloadConfig
from jtc.eval.workflow import WorkflowWorkloadConfig
from jtc.common.workflows.registry import workflow_names

from primebeaker.evaluation import PythonEvaluationRequest, PythonEvaluationScheduler
from primebeaker.evaluation_assets import resolve_template_path, snapshot_template


Priority = Literal["normal", "high", "urgent"]


def registered_workflow_names() -> list[str]:
    """Return workflow names from the installed optional JTC dependency."""

    return list(workflow_names())


def _workload(model: type[BaseModel], source: BaseModel, **updates: Any) -> BaseModel:
    values = {name: getattr(source, name) for name in model.model_fields}
    values.update(updates)
    return model.model_validate(values)


class StandardEvaluationConfig(EvaluationWorkloadConfig):
    """JTC benchmark inputs plus PrimeBeaker-owned infrastructure settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cluster: str = "ai2/holmes"
    service_cluster: str | None = "ai2/jupiter"
    omit_service_resources: bool = True
    priority: Priority = "high"
    min_runtime_hours: int = Field(default=1, ge=0)
    service_min_runtime_hours: int | None = Field(default=None, ge=0)
    model_replicas: int = Field(default=8, gt=0)
    max_num_batched_tokens: int = Field(default=65536, gt=0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    terminal_replicas: int = Field(default=8, ge=0)
    terminal_image: str | None = None
    python_replicas: int = Field(default=0, ge=0)
    search_replicas: int = Field(default=0, ge=0)
    local_search_replicas: int = Field(default=0, ge=0)
    local_search_corpus_jsonl: Path | None = None
    local_search_index_dir: Path | None = None
    local_search_image: str | None = None
    service_startup_timeout_seconds: float = Field(default=120, gt=0)
    readiness_timeout_seconds: int = Field(default=7200, gt=0)
    tags: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _complete_local_search_pool(self) -> "StandardEvaluationConfig":
        if self.local_search_replicas:
            missing = [
                name
                for name, value in (
                    ("local_search_corpus_jsonl", self.local_search_corpus_jsonl),
                    ("local_search_index_dir", self.local_search_index_dir),
                    ("local_search_model_path", self.local_search_model_path),
                )
                if value is None or (isinstance(value, str) and not value.strip())
            ]
            if missing:
                raise ValueError(f"local search replicas require: {', '.join(missing)}")
        return self


class StandardEvaluationDescription(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: Literal["standard"] = "standard"
    evaluation: StandardEvaluationConfig
    image: str | None = None
    existing_registry: str | None = None
    registry: str | None = None
    workspace: str = "ai2/oe-agents-holmes"
    budget: str | None = "ai2/oe-omai"
    tensor_parallelism: int = Field(default=1, gt=0)
    max_model_len: int = Field(default=32768, gt=0)
    gateway_workers: int = Field(default=8, gt=0)
    evaluation_cluster: str | None = None
    keep_stack: bool = False
    beaker_token_secret: str = "BEAKER_TOKEN"
    secrets: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StandardEvaluationDescription":
        source = Path(path).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"evaluation YAML must be a mapping: {source}")
        evaluation = data.get("evaluation", data)
        if not isinstance(evaluation, dict):
            raise ValueError("evaluation must be a mapping")
        if "verifier_prompt_chat_template_path" not in evaluation:
            raise ValueError(
                "evaluation.verifier_prompt_chat_template_path is required"
            )
        evaluation = dict(evaluation)
        evaluation["verifier_prompt_chat_template_path"] = resolve_template_path(
            evaluation["verifier_prompt_chat_template_path"], config_path=source
        )
        top = {
            key: value
            for key, value in data.items()
            if key not in {"evaluation", "workflow"}
        }
        return cls.model_validate({**top, "evaluation": evaluation})

    def workload(self, *, materialize: bool) -> EvaluationWorkloadConfig:
        template = snapshot_template(
            self.evaluation.verifier_prompt_chat_template_path,
            run_dir=self.evaluation.run_dir,
            materialize=materialize,
        )
        return _workload(
            EvaluationWorkloadConfig,
            self.evaluation,
            verifier_prompt_chat_template_path=template,
        )  # type: ignore[return-value]

    def request(
        self, *, image: str | None = None, materialize: bool = False
    ) -> PythonEvaluationRequest:
        evaluation_image = image or self.image
        if not evaluation_image:
            raise ValueError("evaluation image is required; pass --image or set image")
        evaluation = self.evaluation
        service_cluster = evaluation.service_cluster or evaluation.cluster
        reuse = self.existing_registry is not None
        services: dict[str, Any] = {
            "registry": self.existing_registry or self.registry,
            "python_replicas": 0 if reuse else evaluation.python_replicas,
            "terminal_replicas": 0 if reuse else evaluation.terminal_replicas,
            "web_search_replicas": 0 if reuse else evaluation.search_replicas,
            "local_search_replicas": 0 if reuse else evaluation.local_search_replicas,
            "local_search_corpus_jsonl": (
                str(evaluation.local_search_corpus_jsonl)
                if not reuse and evaluation.local_search_corpus_jsonl else None
            ),
            "local_search_index_dir": (
                str(evaluation.local_search_index_dir)
                if not reuse and evaluation.local_search_index_dir else None
            ),
            "local_search_service_name": evaluation.local_search_model_path,
            "generation_model": None if reuse else str(evaluation.model),
            "generation_replicas": 0 if reuse else evaluation.model_replicas,
            "generation_tp": self.tensor_parallelism,
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": evaluation.max_num_batched_tokens,
            "max_num_seqs": evaluation.max_num_seqs,
            "gateway_workers": self.gateway_workers,
            "service_clusters": (service_cluster,),
            "model_cluster": evaluation.cluster,
            "service_priority": evaluation.priority,
            "model_priority": evaluation.priority,
            "min_runtime_hours": (
                evaluation.service_min_runtime_hours
                if evaluation.service_min_runtime_hours is not None
                else evaluation.min_runtime_hours
            ),
            "omit_service_resources": evaluation.omit_service_resources,
            "name_prefix": "jtceval",
            "shared_dir": "/weka/gfaria",
        }
        if evaluation.terminal_image:
            services["terminal_image"] = evaluation.terminal_image
        if evaluation.local_search_image:
            services["local_search_image"] = evaluation.local_search_image
        workload = self.workload(materialize=materialize)
        return PythonEvaluationRequest(
            module="jtc.eval.worker",
            arguments=("run",),
            image=evaluation_image,
            run_dir=evaluation.run_dir,
            services=services,
            required_services=self.required_services(),
            workspace=self.workspace,
            evaluation_cluster=self.evaluation_cluster or service_cluster,
            priority=evaluation.priority,
            min_runtime_hours=evaluation.min_runtime_hours,
            environment={"JTCEVAL_EVALUATION_JSON": workload.model_dump_json()},
            secrets=self.secrets,
            beaker_token_secret=self.beaker_token_secret,
            keep_services=self.keep_stack,
            readiness_timeout_seconds=max(
                evaluation.readiness_timeout_seconds,
                evaluation.service_startup_timeout_seconds,
            ),
            experiment_name=evaluation.run_dir.name or "jtceval",
            budget=self.budget,
        )

    def required_services(self) -> dict[str, int]:
        evaluation = self.evaluation
        requirements = {str(evaluation.model): 1}
        tools = {part.strip().casefold() for part in evaluation.verifier_tools.split(",")}
        if tools & {"terminal", "webterminal"}:
            requirements["terminal"] = 1
        if "python" in tools:
            requirements["python"] = 1
        if "search" in tools and evaluation.local_search_model_path is None:
            requirements["search"] = 1
        if evaluation.local_search_model_path:
            requirements[evaluation.local_search_model_path] = 1
        return requirements

    def preview(self, *, image: str | None = None) -> dict[str, Any]:
        request = self.request(image=image)
        return PythonEvaluationScheduler().preview(request).model_dump(mode="json")

    def launch(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().submit(
            self.request(image=image, materialize=True)
        )


class RejectionSamplingDescription(BaseModel):
    """PrimeBeaker-owned lifecycle for iterative JTC rejection sampling."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: Literal["rejection-sampling"] = "rejection-sampling"
    rejection: RejectionSamplingWorkloadConfig
    image: str | None = None
    services: dict[str, Any] = Field(default_factory=dict)
    required_services: dict[str, int] = Field(default_factory=dict)
    registry: str | None = None
    workspace: str = "ai2/oe-agents"
    evaluation_cluster: str = "ai2/jupiter"
    priority: Priority = "high"
    min_runtime_hours: int = Field(default=1, ge=0)
    keep_stack: bool = False
    readiness_timeout_seconds: float = Field(default=7200, gt=0)
    beaker_token_secret: str = "BEAKER_TOKEN"
    secrets: dict[str, str] = Field(default_factory=dict)
    budget: str | None = "ai2/oe-omai"

    @model_validator(mode="after")
    def _valid_services(self) -> "RejectionSamplingDescription":
        if any(not name or count < 1 for name, count in self.required_services.items()):
            raise ValueError("required services need names and positive counts")
        configured_registry = self.services.get("registry")
        if self.registry and configured_registry and configured_registry != self.registry:
            raise ValueError("registry conflicts with services.registry")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RejectionSamplingDescription":
        source = Path(path).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"evaluation YAML must be a mapping: {source}")
        rejection = data.get("rejection")
        if not isinstance(rejection, dict):
            raise ValueError("rejection must be a mapping")
        values = dict(rejection)
        template = values.get("prompt_chat_template_path")
        if template is None:
            raise ValueError("rejection.prompt_chat_template_path is required")
        values["prompt_chat_template_path"] = resolve_template_path(
            template, config_path=source
        )
        return cls.model_validate({**data, "rejection": values})

    def workload(self, *, materialize: bool) -> RejectionSamplingWorkloadConfig:
        template = snapshot_template(
            self.rejection.prompt_chat_template_path,
            run_dir=self.rejection.output_dir,
            materialize=materialize,
        )
        return self.rejection.model_copy(
            update={"prompt_chat_template_path": template}
        )

    def request(
        self, *, image: str | None = None, materialize: bool = False
    ) -> PythonEvaluationRequest:
        evaluation_image = image or self.image
        if not evaluation_image:
            raise ValueError("evaluation image is required; pass --image or set image")
        services = dict(self.services)
        if self.registry:
            services["registry"] = self.registry
        services.setdefault("name_prefix", "jtceval-rejection-sampling")
        services.setdefault("shared_dir", "/weka/gfaria")
        workload = self.workload(materialize=materialize)
        return PythonEvaluationRequest(
            module="jtc.eval.rejection_sampling_worker",
            arguments=("run",),
            image=evaluation_image,
            run_dir=workload.output_dir,
            services=services,
            required_services=self.required_services,
            workspace=self.workspace,
            evaluation_cluster=self.evaluation_cluster,
            priority=self.priority,
            min_runtime_hours=self.min_runtime_hours,
            environment={
                "JTCEVAL_REJECTION_SAMPLING_JSON": workload.model_dump_json()
            },
            secrets=self.secrets,
            beaker_token_secret=self.beaker_token_secret,
            keep_services=self.keep_stack,
            readiness_timeout_seconds=self.readiness_timeout_seconds,
            experiment_name=workload.output_dir.name or "rejection-sampling",
            budget=self.budget,
        )

    def preview(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().preview(
            self.request(image=image)
        ).model_dump(mode="json")

    def launch(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().submit(
            self.request(image=image, materialize=True)
        )


class WorkflowEvaluationDescription(BaseModel):
    """PrimeBeaker lifecycle for any registered JTC Python workflow."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: str = Field(min_length=1)
    run_dir: Path
    arguments: dict[str, Any] = Field(default_factory=dict)
    image: str | None = None
    services: dict[str, Any] = Field(default_factory=dict)
    required_services: dict[str, int] = Field(default_factory=dict)
    registry: str | None = None
    workspace: str = "ai2/oe-agents"
    evaluation_cluster: str = "ai2/jupiter"
    priority: Priority = "high"
    min_runtime_hours: int = Field(default=1, ge=0)
    keep_stack: bool = False
    readiness_timeout_seconds: float = Field(default=7200, gt=0)
    beaker_token_secret: str = "BEAKER_TOKEN"
    secrets: dict[str, str] = Field(default_factory=dict)
    budget: str | None = "ai2/oe-omai"

    @model_validator(mode="after")
    def _registered_workflow(self) -> "WorkflowEvaluationDescription":
        if self.workflow not in workflow_names():
            choices = ", ".join(workflow_names())
            raise ValueError(
                f"unsupported JTC workflow {self.workflow!r}; choose one of: {choices}"
            )
        if any(not name or count < 1 for name, count in self.required_services.items()):
            raise ValueError("required services need names and positive counts")
        configured_registry = self.services.get("registry")
        if self.registry and configured_registry and configured_registry != self.registry:
            raise ValueError("registry conflicts with services.registry")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "WorkflowEvaluationDescription":
        source = Path(path).resolve()
        data = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"evaluation YAML must be a mapping: {source}")
        arguments = data.get("arguments", {})
        if not isinstance(arguments, dict):
            raise ValueError("arguments must be a mapping")
        resolved = dict(arguments)
        for name, value in arguments.items():
            if name.endswith("template_path") and isinstance(value, (str, Path)):
                resolved[name] = str(
                    resolve_template_path(value, config_path=source)
                )
        return cls.model_validate({**data, "arguments": resolved})

    def workload(self, *, materialize: bool) -> WorkflowWorkloadConfig:
        arguments = dict(self.arguments)
        for name, value in self.arguments.items():
            if name.endswith("template_path") and isinstance(value, (str, Path)):
                arguments[name] = str(
                    snapshot_template(
                        value,
                        run_dir=self.run_dir,
                        materialize=materialize,
                    )
                )
        return WorkflowWorkloadConfig(
            workflow=self.workflow,
            run_dir=self.run_dir,
            arguments=arguments,
        )

    def request(
        self, *, image: str | None = None, materialize: bool = False
    ) -> PythonEvaluationRequest:
        evaluation_image = image or self.image
        if not evaluation_image:
            raise ValueError("evaluation image is required; pass --image or set image")
        services = dict(self.services)
        if self.registry:
            services["registry"] = self.registry
        services.setdefault("name_prefix", f"jtceval-{self.workflow}")
        services.setdefault("shared_dir", "/weka/gfaria")
        workload = self.workload(materialize=materialize)
        return PythonEvaluationRequest(
            module="jtc.eval.workflow_worker",
            arguments=("run",),
            image=evaluation_image,
            run_dir=self.run_dir,
            services=services,
            required_services=self.required_services,
            workspace=self.workspace,
            evaluation_cluster=self.evaluation_cluster,
            priority=self.priority,
            min_runtime_hours=self.min_runtime_hours,
            environment={"JTCEVAL_WORKFLOW_JSON": workload.model_dump_json()},
            secrets=self.secrets,
            beaker_token_secret=self.beaker_token_secret,
            keep_services=self.keep_stack,
            readiness_timeout_seconds=self.readiness_timeout_seconds,
            experiment_name=self.run_dir.name or f"jtceval-{self.workflow}",
            budget=self.budget,
        )

    def preview(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().preview(
            self.request(image=image)
        ).model_dump(mode="json")

    def launch(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().submit(
            self.request(image=image, materialize=True)
        )


class SearchAgentEvaluationDescription(SearchAgentWorkloadConfig):
    """Search workload plus PrimeBeaker-owned topology and placement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: Literal["search-agent"] = "search-agent"
    image: str | None = None
    cluster: str = "ai2/holmes"
    evaluation_cluster: str | None = None
    search_replicas: int = Field(default=0, ge=0)
    service_cluster: str = "ai2/jupiter"
    workspace: str = "ai2/oe-agents"
    budget: str | None = "ai2/oe-omai"
    priority: Priority = "high"
    model_replicas: int = Field(default=1, ge=0)
    required_model_replicas: int | None = Field(default=None, gt=0)
    tensor_parallelism: int = Field(default=1, gt=0)
    max_model_len: int = Field(default=131072, gt=0)
    max_num_batched_tokens: int = Field(default=131072, gt=0)
    max_num_seqs: int | None = Field(default=None, gt=0)
    local_search_replicas: int = Field(default=1, ge=0)
    terminal_replicas: int = Field(default=1, ge=0)
    local_search_corpus_jsonl: Path | None = None
    local_search_index_dir: Path | None = None
    terminal_image: str | None = None
    local_search_image: str | None = None
    judge_model_replicas: int = Field(default=0, ge=0)
    judge_tensor_parallelism: int = Field(default=1, gt=0)
    judge_max_model_len: int = Field(default=32768, gt=0)
    judge_max_num_batched_tokens: int = Field(default=32768, gt=0)
    judge_replicas: int = Field(default=0, ge=0)
    min_runtime_hours: int = Field(default=0, ge=0)
    model_ready_timeout: int = Field(default=7200, gt=0)
    registry: str | None = None
    keep_stack: bool = False
    beaker_token_secret: str = "BEAKER_TOKEN"
    secrets: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid_topology(self) -> "SearchAgentEvaluationDescription":
        if not self.dataset.is_relative_to("/weka"):
            raise ValueError("dataset must be under /weka")
        if not self.run_dir.is_relative_to("/weka"):
            raise ValueError("run_dir must be under /weka")
        if self.local_search_replicas:
            for field in ("local_search_corpus_jsonl", "local_search_index_dir"):
                value = getattr(self, field)
                if value is None or not value.is_relative_to("/weka"):
                    raise ValueError(f"{field} must be under /weka")
            if not self.local_search_model_path:
                raise ValueError("local_search_model_path is required")
        if self.max_tokens >= self.max_model_len:
            raise ValueError("max_tokens must be smaller than max_model_len")
        if self.model_replicas == 0 and (
            self.registry is None or self.required_model_replicas is None
        ):
            raise ValueError(
                "model_replicas=0 requires registry and required_model_replicas"
            )
        if self.harness in {"visit-judge", "webterminal-judge"} and not self.registry:
            raise ValueError("judge-backed search requires an existing registry")
        if self.registry is None and self.harness != "visit" and self.terminal_replicas < 1:
            raise ValueError("tool-using evaluation requires terminal_replicas >= 1")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SearchAgentEvaluationDescription":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("workflow") != "search-agent":
            raise ValueError("search-agent YAML must set workflow: search-agent")
        payload = raw.get("search_agent", raw)
        if not isinstance(payload, dict):
            raise ValueError("search_agent must be a mapping")
        values = {**payload, "workflow": "search-agent"}
        if values.get("image") is None and raw.get("image") is not None:
            values["image"] = raw["image"]
        if values.get("template_path") is not None:
            values["template_path"] = resolve_template_path(
                values["template_path"], config_path=source
            )
        return cls.model_validate(values)

    def workload(self, *, materialize: bool) -> SearchAgentWorkloadConfig:
        template = (
            snapshot_template(
                self.template_path, run_dir=self.run_dir, materialize=materialize
            )
            if self.template_path is not None else None
        )
        return _workload(
            SearchAgentWorkloadConfig,
            self,
            template_path=template,
            rollout_timeout=self.model_ready_timeout,
        )  # type: ignore[return-value]

    def required_services(self) -> dict[str, int]:
        requirements = {self.model: self.required_model_replicas or 1}
        if self.harness != "visit":
            requirements["terminal"] = 1
        if self.local_search_model_path:
            requirements[self.local_search_model_path] = 1
        if self.harness in {"visit-judge", "webterminal-judge"}:
            assert self.judge_model_path is not None
            requirements[self.judge_model_path] = self.judge_model_replicas or 1
            requirements[self.judge_service_model_path] = self.judge_replicas or 1
        return requirements

    def request(
        self, *, image: str | None = None, materialize: bool = False
    ) -> PythonEvaluationRequest:
        evaluation_image = image or self.image
        if not evaluation_image:
            raise ValueError("evaluation image is required; pass --image or set image")
        reuse = self.registry is not None
        model_replicas = 0 if reuse else self.model_replicas
        services: dict[str, Any] = {
            "registry": self.registry,
            "python_replicas": 0,
            "terminal_replicas": 0 if reuse or self.harness == "visit" else self.terminal_replicas,
            "web_search_replicas": 0 if reuse else self.search_replicas,
            "local_search_replicas": 0 if reuse else self.local_search_replicas,
            "local_search_corpus_jsonl": str(self.local_search_corpus_jsonl) if not reuse and self.local_search_corpus_jsonl else None,
            "local_search_index_dir": str(self.local_search_index_dir) if not reuse and self.local_search_index_dir else None,
            "local_search_service_name": self.local_search_model_path,
            "generation_model": self.model if model_replicas else None,
            "generation_replicas": model_replicas,
            "generation_tp": self.tensor_parallelism,
            "max_model_len": self.max_model_len,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "service_clusters": (self.service_cluster,),
            "model_cluster": self.cluster,
            "service_priority": self.priority,
            "model_priority": self.priority,
            "min_runtime_hours": self.min_runtime_hours,
            "omit_service_resources": True,
            "name_prefix": "jtceval-search",
            "shared_dir": "/weka/gfaria",
        }
        if self.terminal_image:
            services["terminal_image"] = self.terminal_image
        if self.local_search_image:
            services["local_search_image"] = self.local_search_image
        workload = self.workload(materialize=materialize)
        return PythonEvaluationRequest(
            module="jtc.eval.search_worker",
            arguments=("run",), image=evaluation_image, run_dir=self.run_dir,
            services=services, required_services=self.required_services(),
            workspace=self.workspace,
            evaluation_cluster=self.evaluation_cluster or self.service_cluster,
            priority=self.priority, min_runtime_hours=self.min_runtime_hours,
            environment={"JTCEVAL_SEARCH_JSON": workload.model_dump_json()},
            secrets=self.secrets, beaker_token_secret=self.beaker_token_secret,
            keep_services=self.keep_stack,
            readiness_timeout_seconds=self.model_ready_timeout,
            experiment_name=self.run_dir.name or "jtceval-search", budget=self.budget,
        )

    def preview(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().preview(
            self.request(image=image)
        ).model_dump(mode="json")

    def launch(self, *, image: str | None = None) -> dict[str, Any]:
        return PythonEvaluationScheduler().submit(
            self.request(image=image, materialize=True)
        )


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_replicas: int = Field(default=2, gt=0)
    tensor_parallelism: int = Field(default=2, gt=0)
    max_model_len: int = Field(default=131072, gt=0)
    max_num_batched_tokens: int = Field(default=32768, gt=0)
    max_num_seqs: int = Field(default=64, gt=0)
    gateway_workers: int = Field(default=8, gt=0)
    cluster: str = "ai2/holmes"
    service_cluster: str | None = None
    workspace: str = "ai2/oe-agents-holmes"
    budget: str | None = "ai2/oe-omai"
    priority: Priority = "high"
    min_runtime_hours: int = Field(default=1, ge=0)
    readiness_timeout_seconds: int = Field(default=5400, gt=0)


class RubricJudgeAuditDescription(RubricJudgeAuditWorkloadConfig):
    model_config = ConfigDict(extra="forbid", frozen=True)

    workflow: Literal["rubric-judge-audit"] = "rubric-judge-audit"
    image: str | None = None
    existing_registry: str | None = None
    registry: str | None = None
    keep_stack: bool = False
    beaker_token_secret: str = "BEAKER_TOKEN"
    secrets: dict[str, str] = Field(default_factory=dict)
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)

    @model_validator(mode="after")
    def _valid_context(self) -> "RubricJudgeAuditDescription":
        requested = self.inference.max_prompt_length + self.inference.max_new_tokens
        if requested > self.deployment.max_model_len:
            raise ValueError("max_prompt_length + max_new_tokens exceeds max_model_len")
        return self

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RubricJudgeAuditDescription":
        source = Path(path).resolve()
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(raw, dict) or raw.get("workflow") != "rubric-judge-audit":
            raise ValueError("audit YAML must set workflow: rubric-judge-audit")
        inference = raw.get("inference")
        if not isinstance(inference, dict) or "template_path" not in inference:
            raise ValueError("audit YAML must set inference.template_path")
        raw = dict(raw)
        raw["inference"] = {
            **inference,
            "template_path": resolve_template_path(
                inference["template_path"], config_path=source
            ),
        }
        return cls.model_validate(raw)

    def workload(self, *, materialize: bool) -> RubricJudgeAuditWorkloadConfig:
        template = snapshot_template(
            self.inference.template_path,
            run_dir=self.outputs.audit_dir,
            materialize=materialize,
        )
        inference = self.inference.model_copy(update={"template_path": template})
        return RubricJudgeAuditWorkloadConfig(
            sample=self.sample, inference=inference, outputs=self.outputs
        )

    def request(
        self, *, image: str | None = None, force: bool = False,
        materialize: bool = False,
    ) -> PythonEvaluationRequest:
        evaluation_image = image or self.image
        if not evaluation_image:
            raise ValueError("evaluation image is required; pass --image or set image")
        deploy = self.deployment
        service_cluster = deploy.service_cluster or deploy.cluster
        reuse = self.existing_registry is not None
        workload = self.workload(materialize=materialize)
        services = {
            "registry": self.existing_registry or self.registry,
            "python_replicas": 0, "terminal_replicas": 0,
            "web_search_replicas": 0, "local_search_replicas": 0,
            "generation_model": None if reuse else self.inference.model,
            "generation_replicas": 0 if reuse else deploy.model_replicas,
            "generation_tp": deploy.tensor_parallelism,
            "max_model_len": deploy.max_model_len,
            "max_num_batched_tokens": deploy.max_num_batched_tokens,
            "max_num_seqs": deploy.max_num_seqs,
            "gateway_workers": deploy.gateway_workers,
            "service_clusters": (service_cluster,), "model_cluster": deploy.cluster,
            "service_priority": deploy.priority, "model_priority": deploy.priority,
            "min_runtime_hours": deploy.min_runtime_hours,
            "omit_service_resources": True, "name_prefix": "jtceval-audit",
            "shared_dir": "/weka/gfaria",
        }
        return PythonEvaluationRequest(
            module="jtc.eval.rubric_judge_audit_worker", arguments=("run",),
            image=evaluation_image, run_dir=self.outputs.audit_dir,
            services=services,
            required_services={self.inference.model: deploy.model_replicas},
            workspace=deploy.workspace, evaluation_cluster=service_cluster,
            priority=deploy.priority, min_runtime_hours=deploy.min_runtime_hours,
            environment={
                "JTCEVAL_RUBRIC_AUDIT_JSON": workload.model_dump_json(),
                "JTCEVAL_RUBRIC_AUDIT_FORCE": json.dumps(force),
            },
            secrets=self.secrets, beaker_token_secret=self.beaker_token_secret,
            keep_services=self.keep_stack,
            readiness_timeout_seconds=deploy.readiness_timeout_seconds,
            experiment_name=self.outputs.audit_dir.name or "jtceval-audit",
            budget=deploy.budget,
        )


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class RubricJudgeAuditCoordinator:
    def __init__(self, config: RubricJudgeAuditDescription) -> None:
        self.config = config

    def preview(self, *, image: str | None = None) -> dict[str, Any]:
        request = self.config.request(image=image)
        return {
            "workflow": self.config.workflow,
            "expected_records": self.config.expected_records,
            "sample_complete": sample_complete(self.config.workload(materialize=False)),
            "audit_complete": audit_complete(self.config.workload(materialize=False)),
            "python_module": request.module,
            "primebeaker": PythonEvaluationScheduler().preview(request).model_dump(mode="json"),
        }

    def run(self, *, image: str | None = None, force: bool = False) -> dict[str, Any]:
        config = self.config
        workload = config.workload(materialize=False)
        if audit_complete(workload) and not force:
            summary = json.loads((config.outputs.audit_dir / "summary.json").read_text(encoding="utf-8"))
            result = {
                "workflow": config.workflow, "status": "complete_existing",
                "compute_launched": False, "checked_at": _utc_now(),
                "expected_records": config.expected_records,
                "audit_dir": str(config.outputs.audit_dir), "overall": summary["overall"],
            }
            atomic_json(config.outputs.receipt_path, result)
            return result
        prepare_sample(workload, force=force)
        response_count = jsonl_count(config.inference.responses_path)
        if response_count == config.expected_records and not force:
            summary = finish_audit(workload)
            result = {
                "workflow": config.workflow, "status": "complete_recovered",
                "compute_launched": False, "completed_at": _utc_now(),
                "expected_records": config.expected_records,
                "audit_dir": str(config.outputs.audit_dir), "overall": summary["overall"],
            }
            atomic_json(config.outputs.receipt_path, result)
            return result
        if response_count and not force:
            raise RuntimeError(
                f"partial response stream has {response_count} rows; pass --force to rerun inference"
            )
        launch = PythonEvaluationScheduler().submit(
            config.request(image=image, force=force, materialize=True)
        )
        result = {
            "workflow": config.workflow, "status": "scheduled",
            "compute_launched": True, "scheduled_at": _utc_now(),
            "expected_records": config.expected_records, **launch,
        }
        launch_receipt = config.outputs.receipt_path.with_name(
            f"{config.outputs.receipt_path.stem}.launch.json"
        )
        atomic_json(launch_receipt, result)
        return result


EvaluationDescription = (
    StandardEvaluationDescription
    | RejectionSamplingDescription
    | WorkflowEvaluationDescription
    | SearchAgentEvaluationDescription
    | RubricJudgeAuditDescription
)


def load_evaluation(path: str | Path) -> EvaluationDescription:
    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"evaluation YAML must be a mapping: {source}")
    workflow = raw.get("workflow")
    if workflow == "search-agent":
        return SearchAgentEvaluationDescription.from_yaml(source)
    if workflow == "rubric-judge-audit":
        return RubricJudgeAuditDescription.from_yaml(source)
    if workflow == "rejection-sampling":
        return RejectionSamplingDescription.from_yaml(source)
    if workflow in workflow_names():
        return WorkflowEvaluationDescription.from_yaml(source)
    if workflow not in (None, "standard"):
        raise ValueError(f"unsupported evaluation workflow: {workflow!r}")
    return StandardEvaluationDescription.from_yaml(source)


def run_evaluation(
    config: str | Path, image: str | None = None, registry: str | None = None,
    dry_run: bool = False, force: bool = False,
) -> dict[str, Any]:
    description = load_evaluation(config)
    if registry:
        field = "existing_registry" if isinstance(
            description, (StandardEvaluationDescription, RubricJudgeAuditDescription)
        ) else "registry"
        description = description.model_copy(update={field: registry})
    if isinstance(description, RubricJudgeAuditDescription):
        coordinator = RubricJudgeAuditCoordinator(description)
        return coordinator.preview(image=image) if dry_run else coordinator.run(image=image, force=force)
    return description.preview(image=image) if dry_run else description.launch(image=image)


class JTCEvaluationCLI:
    """Preview and schedule JTC evaluation YAMLs from PrimeBeaker."""

    run = staticmethod(run_evaluation)
    submit = staticmethod(run_evaluation)

    @staticmethod
    def workflows() -> list[str]:
        return registered_workflow_names()

    @staticmethod
    def preview(config: str | Path, image: str | None = None, registry: str | None = None) -> dict[str, Any]:
        return run_evaluation(config, image=image, registry=registry, dry_run=True)


__all__ = [
    "DeploymentConfig", "JTCEvaluationCLI", "RejectionSamplingDescription",
    "RubricJudgeAuditCoordinator",
    "RubricJudgeAuditDescription", "SearchAgentEvaluationDescription",
    "StandardEvaluationConfig", "StandardEvaluationDescription",
    "WorkflowEvaluationDescription",
    "load_evaluation", "registered_workflow_names", "run_evaluation",
]
