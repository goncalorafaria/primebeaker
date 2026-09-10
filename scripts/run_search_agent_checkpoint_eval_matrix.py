#!/usr/bin/env python3
"""Run comparable serial Search Agent evaluations for one or more checkpoints."""

from __future__ import annotations

import fire
from collections.abc import Sequence
import time
from pathlib import Path

from primebeaker.jtc_evaluation import SearchAgentEvaluationDescription


DEFAULT_ROOT = Path("/weka/gfaria/prime_sft/outputs/qwen35_4b_base_search_webterminal_agent_step400-from-scratch-quokka9b-judge-local-bc-v2-browsecomp-oversampling2-ctx128k-multinode-train4-infer24")
DEFAULT_CONFIG = Path("/weka/gfaria/primebeaker/examples/configs/eval/qwen35_4b_browsecomp_plus_search_agent_webterminal_judge_step240_ctx128k.yaml")


def parse_context(value: str) -> int:
    normalized = value.strip().lower()
    multiplier = 1024 if normalized.endswith("k") else 1
    try:
        result = int(normalized.removesuffix("k")) * multiplier
    except ValueError as error:
        raise ValueError(f"invalid context length: {value}") from error
    if result <= 32768:
        raise ValueError("context length must exceed max_tokens (32768)")
    return result


def run(
    steps: Sequence[int],
    contexts: Sequence[str | int] = ("128k",),
    root: str | Path = DEFAULT_ROOT,
    base_config: str | Path = DEFAULT_CONFIG,
    force: bool = False,
    wait_for_checkpoints: bool = False,
    checkpoint_poll_seconds: float = 60.0,
    launch_retries: int = 5,
    retry_delay_seconds: float = 30.0,
) -> None:
    """Run a serial matrix; pass list values using Fire list syntax."""

    step_values = [steps] if isinstance(steps, (str, int)) else steps
    context_values = [contexts] if isinstance(contexts, (str, int)) else contexts
    steps = [int(step) for step in step_values]
    contexts = [parse_context(str(context)) for context in context_values]
    root = Path(root)
    base_config = Path(base_config)

    base = SearchAgentEvaluationDescription.from_yaml(base_config)
    for step in steps:
        weights = root / "weights" / f"step_{step}"
        stable_marker = weights / "STABLE"
        while not stable_marker.is_file():
            if not wait_for_checkpoints:
                raise FileNotFoundError(f"checkpoint is not stable: {weights}")
            print(
                f"WAIT step={step}: checkpoint not stable yet; retrying in "
                f"{checkpoint_poll_seconds:.0f}s",
                flush=True,
            )
            time.sleep(checkpoint_poll_seconds)
        print(f"CHECKPOINT_READY step={step} weights={weights}", flush=True)
        for context in contexts:
            context_name = f"{context // 1024}k" if context % 1024 == 0 else str(context)
            run_dir = root / f"search_agent_eval_browsecomp_plus_webterminal_judge_step{step}_ctx{context_name}"
            summary = run_dir / "rollouts/step_0/eval/all/search_eval_summary.json"
            if summary.is_file() and not force:
                print(f"SKIP step={step} context={context_name}: {summary}", flush=True)
                continue
            description = base.model_copy(
                update={
                    "model": str(weights),
                    "run_dir": run_dir,
                    "max_model_len": context,
                    "max_num_batched_tokens": context,
                    "max_total_completion_tokens": context,
                }
            )
            print(f"START step={step} context={context_name} run_dir={run_dir}", flush=True)
            for attempt in range(1, launch_retries + 1):
                try:
                    result = description.launch()
                    break
                except RuntimeError as error:
                    retryable = "retryable database conflict" in str(error).lower()
                    if not retryable or attempt == launch_retries:
                        raise
                    delay_seconds = retry_delay_seconds * attempt
                    print(
                        f"RETRY step={step} context={context_name} "
                        f"attempt={attempt}/{launch_retries} in {delay_seconds:.0f}s: {error}",
                        flush=True,
                    )
                    time.sleep(delay_seconds)
            print(f"DONE step={step} context={context_name} result={result}", flush=True)


def main(argv: list[str] | None = None) -> None:
    fire.Fire(run, command=argv)


if __name__ == "__main__":
    main()
