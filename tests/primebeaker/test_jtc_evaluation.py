from __future__ import annotations

import json
from pathlib import Path

import pytest

from jtc.eval.launcher import EvaluationWorkloadConfig
from jtc.eval.search_agent import SearchAgentWorkloadConfig
from jtc.eval.workflow import WorkflowWorkloadConfig
from jtc.eval.rejection_sampling import RejectionSamplingWorkloadConfig
from jtc.common.workflows.registry import workflow_names
from primebeaker.jtc_evaluation import (
    SearchAgentEvaluationDescription,
    RejectionSamplingDescription,
    StandardEvaluationConfig,
    StandardEvaluationDescription,
    WorkflowEvaluationDescription,
    load_evaluation,
)
from primebeaker.cli import PythonEvaluationCLI


IMAGE = "beaker://01JTCEVAL"


def test_standard_yaml_and_request_keep_scheduler_fields_out_of_worker(
    tmp_path: Path,
) -> None:
    template = tmp_path / "new-template.json"
    template.write_text(json.dumps({"chat_template": []}), encoding="utf-8")
    config = tmp_path / "eval.yaml"
    config.write_text(
        "image: beaker://01JTCEVAL\n"
        "evaluation:\n"
        "  model: /weka/gfaria/models/policy\n"
        "  run_dir: /weka/gfaria/evals/policy\n"
        "  cluster: ai2/holmes\n"
        "  service_cluster: ai2/jupiter\n"
        "  terminal_replicas: 6\n"
        "  verifier_tools: terminal\n"
        "  verifier_prompt_chat_template_path: new-template.json\n",
        encoding="utf-8",
    )

    description = StandardEvaluationDescription.from_yaml(config)
    request = description.request()
    workload = EvaluationWorkloadConfig.model_validate_json(
        request.environment["JTCEVAL_EVALUATION_JSON"]
    )

    assert description.evaluation.verifier_prompt_chat_template_path == template
    assert request.module == "jtc.eval.worker"
    assert request.services["terminal_replicas"] == 6
    assert request.services["model_cluster"] == "ai2/holmes"
    assert request.evaluation_cluster == "ai2/jupiter"
    assert "cluster" not in EvaluationWorkloadConfig.model_fields
    assert "terminal_replicas" not in EvaluationWorkloadConfig.model_fields


def test_existing_registry_reuses_standard_capacity() -> None:
    description = StandardEvaluationDescription(
        image=IMAGE,
        existing_registry="redis://registry.example:6379",
        evaluation=StandardEvaluationConfig(
            model=Path("/weka/gfaria/models/policy"),
            run_dir=Path("/weka/gfaria/evals/policy"),
            verifier_prompt_chat_template_path=Path("templates/example.json"),
            terminal_replicas=8,
            python_replicas=3,
        ),
    )
    request = description.request()
    assert request.services["generation_replicas"] == 0
    assert request.services["terminal_replicas"] == 0
    assert request.services["python_replicas"] == 0


def test_search_request_serializes_only_jtc_workload_fields() -> None:
    description = SearchAgentEvaluationDescription(
        image=IMAGE,
        model="Qwen/Qwen3.5-4B",
        dataset=Path("/weka/gfaria/records/tasks.parquet"),
        run_dir=Path("/weka/gfaria/evals/search"),
        local_search_corpus_jsonl=Path("/weka/gfaria/records/corpus.jsonl"),
        local_search_index_dir=Path("/weka/gfaria/records/index"),
        model_replicas=4,
        model_ready_timeout=86400,
    )
    request = description.request()
    workload = SearchAgentWorkloadConfig.model_validate_json(
        request.environment["JTCEVAL_SEARCH_JSON"]
    )

    assert request.module == "jtc.eval.search_worker"
    assert request.services["generation_replicas"] == 4
    assert workload.rollout_timeout == 86400
    assert "cluster" not in SearchAgentWorkloadConfig.model_fields
    assert "registry" not in SearchAgentWorkloadConfig.model_fields


def test_all_owned_lifecycle_yamls_parse() -> None:
    root = Path(__file__).resolve().parents[2] / "examples/configs/eval"
    paths = sorted(root.rglob("*.yaml"))
    assert len(paths) == 141
    assert all(load_evaluation(path) is not None for path in paths)


def test_every_registered_jtc_workflow_has_generic_scheduler_contract(
    tmp_path: Path,
) -> None:
    for workflow in workflow_names():
        description = WorkflowEvaluationDescription(
            workflow=workflow,
            image=IMAGE,
            run_dir=Path(f"/weka/gfaria/evals/{workflow}"),
            arguments={"output_path": f"/weka/gfaria/evals/{workflow}/output.jsonl"},
            services={"registry": "redis://registry.example:6379"},
        )
        request = description.request()
        workload = WorkflowWorkloadConfig.model_validate_json(
            request.environment["JTCEVAL_WORKFLOW_JSON"]
        )

        assert request.module == "jtc.eval.workflow_worker"
        assert workload.workflow == workflow


def test_fire_cli_lists_every_registry_backed_workflow() -> None:
    assert PythonEvaluationCLI.workflows() == list(workflow_names())


def test_generic_verifier_yaml_resolves_and_snapshots_new_template(
    tmp_path: Path,
) -> None:
    template = tmp_path / "brand-new-template.json"
    template.write_text(json.dumps({"chat_template": []}), encoding="utf-8")
    config = tmp_path / "verifier.yaml"
    config.write_text(
        "workflow: verifier-articulated-submit-tool-use\n"
        "image: beaker://01JTCEVAL\n"
        "run_dir: /weka/gfaria/evals/articulated\n"
        "arguments:\n"
        "  input_path: /weka/gfaria/records/input.parquet\n"
        "  output_path: /weka/gfaria/evals/articulated/output.parquet\n"
        "  prompt_chat_template_path: brand-new-template.json\n"
        "services:\n"
        "  registry: redis://registry.example:6379\n",
        encoding="utf-8",
    )

    description = load_evaluation(config)
    assert isinstance(description, WorkflowEvaluationDescription)
    assert description.arguments["prompt_chat_template_path"] == str(template)
    workload = description.workload(materialize=False)
    snapshot = Path(workload.arguments["prompt_chat_template_path"])
    assert snapshot.parent.name == "templates"
    assert snapshot.suffix == ".json"


def test_rejection_sampling_request_separates_workflow_from_infrastructure(
    tmp_path: Path,
) -> None:
    template = tmp_path / "rejection-template.json"
    template.write_text(json.dumps({"chat_template": []}), encoding="utf-8")
    config = tmp_path / "rejection.yaml"
    config.write_text(
        "workflow: rejection-sampling\n"
        "image: beaker://01JTCEVAL\n"
        "rejection:\n"
        "  source_path: /weka/gfaria/records/source.parquet\n"
        "  output_dir: /weka/gfaria/evals/rejection\n"
        "  model: /weka/gfaria/models/policy\n"
        "  prompt_chat_template_path: rejection-template.json\n"
        "services:\n"
        "  registry: redis://registry.example:6379\n"
        "  terminal_replicas: 4\n"
        "required_services:\n"
        "  terminal: 1\n",
        encoding="utf-8",
    )

    description = load_evaluation(config)
    assert isinstance(description, RejectionSamplingDescription)
    request = description.request()
    workload = RejectionSamplingWorkloadConfig.model_validate_json(
        request.environment["JTCEVAL_REJECTION_SAMPLING_JSON"]
    )
    assert request.module == "jtc.eval.rejection_sampling_worker"
    assert request.services["terminal_replicas"] == 4
    assert request.required_services == {"terminal": 1}
    assert workload.rounds == 8
    assert "services" not in RejectionSamplingWorkloadConfig.model_fields
    assert "evaluation_cluster" not in RejectionSamplingWorkloadConfig.model_fields
    assert workload.prompt_chat_template_path.parent.name == "templates"


def test_search_rejects_judge_stack_without_existing_registry() -> None:
    with pytest.raises(ValueError, match="existing registry"):
        SearchAgentEvaluationDescription(
            image=IMAGE,
            model="Qwen/Qwen3.5-4B",
            dataset=Path("/weka/gfaria/records/tasks.parquet"),
            run_dir=Path("/weka/gfaria/evals/search"),
            local_search_replicas=0,
            harness="visit-judge",
            judge_model_path="/weka/gfaria/models/judge",
        )
