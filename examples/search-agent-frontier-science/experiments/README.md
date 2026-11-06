# Search-agent training and Frontier Science evaluation

This is the maintained recipe for the experiments run in November 2026. It
packages launch code, training TOML, evaluation harness, web-fetch recovery,
GPT-5 grading, image recipes and a trainer-started Pengin trigger in one place.
`prepare.py` writes a **fresh** experiment directory; `submit.py` is the explicit
submission boundary. Neither script changes an already-submitted experiment.

## Recipes

| Recipe | Model and layout | Precision | Placement |
| --- | --- | --- | --- |
| Training resume | Qwen3.5-4B, checkpoint 450 → 600; 4 trainer GPUs in one CP4 group | BF16 optimization weights and reductions, LR 1e-6 | Holmes, high, minimum 4h |
| Policy inference | Three TP4 engines, 12 GPUs; four API processes per engine | BF16 activations, FP8 per-block weights, FP8 KV | Holmes, high, minimum 4h |
| Fixed reward judge | Pengin 9B RL step 400; four TP4 engines, four separately schedulable 4-GPU tasks | BF16; 128 sequences/engine, 128K context | Jupiter, urgent, minimum 4h |
| Frontier Science eval | Exported checkpoint 430; seeds 1–5 concurrently on one TP4 engine | BF16 weights, **no FP8 weight/KV override** | Jupiter, urgent, minimum 4h |
| Benchmark grading | GPT-5, high reasoning, 40,000 output tokens, concurrency 512 | Original partial-credit 10-point rubric; ≥7 passes | CPU job on Jupiter, no CPU/memory reservations |

Training preserves MC32 score centering, IPO group-token-mean loss, batch 256,
rollout group 32, max inflight 512, off-policy limit 32, 128K context, 100K per
completion, checkpoint every 10 steps and validation every 25. CP4 on the trainer
is context parallelism, not tensor parallelism. The 12 inference GPUs are TP4.
The resumed optimizer retains its history; the packaged trainer override resets
the loaded checkpoint LR to the requested LR. It also checks BF16 parameter dtype.

The benchmark uses 59 **reformatted** research questions, training-matched
`search(query)` / `webterminal(command)` schemas, temperature 1, 262,144-token
context and cumulative output budget, 100,000 tokens per completion, 150 tool
calls and 151 turns. GPT-5 grades the original question/rubric against the final
answer. This is distinct from training's 200-question validation and Pengin reward.

## Dependencies and paths

These are **Weka-mounted recipes**, not standalone images. All jobs need `/weka`
mounted at the same absolute location. Before preparing, retain:

- `/weka/gfaria/search-agent-repr/data`: 1,724 train, 200 validation, six smoke rows.
- The training checkpoint in `training.toml` and exported eval model in
  `evaluation.toml`; checkpoint weights are not committed here.
- `/weka/gfaria/search-agent-repr/datastore`: the copied corpus and BM25 index.
- `/weka/gfaria/search-agent-repr/policy-resume450-to600-bf16-cp4-12infer-20261105/runtime`:
  exact PrimeRL/Verifiers/renderer snapshot. `training/runtime-manifest.json`
  verifies its source hashes before preparation. Only the run-specific trainer
  override is vendored here, not the whole third-party runtime.
- The mounted Python environments under
  `/weka/gfaria/experiments/reverse-text-online-20261013`, MC32 dependencies under
  `/weka/gfaria/prime-rl-mc32/experiments/mc32`, `/weka/gfaria/vllm-mc32`,
  `/weka/gfaria/search-agent-repr/mc128-runtime/extra-deps`, and the source/deps
  under `/weka/gfaria/launches/quokka9b-verifiable-score-centering-lag16-20261013`.
- This checkout at `/weka/gfaria/primebeaker`, JTC at `/weka/gfaria/jtc`, canonical
  clients in the mounted `runtime/tool-client`, LiteRegistry at
  `/weka/gfaria/search-agent-repr/services/runtime`, and eval compatibility deps
  at `examples/frontier-science-live-search/runtime-web-v2/deps`.

The launcher scripts are the authoritative full `PYTHONPATH` specification.
Benchmark questions and original grading rubrics are read from the existing
workspace paths in `data-manifest.json`, hash-checked, and copied at preparation
time; the dataset itself is not committed in this example.

Copying only this directory to a machine without those mounts will not work.
`provenance.json` records the original source paths and hashes before packaging;
packaged files add relocation and the optional Jina ablation.

Required Beaker secrets: `HF_TOKEN`, `gfaria_WANDB_API_KEY` for training;
`SERPER_API_KEY`, `JINA_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY` for benchmark eval
and grading. Secrets are referenced by name, never copied into image layers.
Use Python 3.11+ for preparation. The launchers select the pinned runtime Python.

## Shared service stack

```bash
export HEAD_REGISTRY=sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3
export REGISTRY=head+sqlite:///weka/gfaria/search-agent-repr/registry/control.sqlite3
```

SQLite stores Redis discovery; Redis holds live model/tool registrations.
Reuse the existing Redis, terminal, local corpus and JudgeAPI workers. Inspect
`services/check_ready.py` and `services/services.beaker.json` before provisioning
another stack: the latter is an archived deployment specification with explicit
Weka persistence paths and should be adapted for an independent registry.
The schema in `services/services.yaml` documents the service roles; the corpus
uses the SQLite-offset adapter, not a stock JSONL corpus server.

Every worker retains the `head+sqlite` URI. Policy gateway is localhost:1212;
JudgeAPI's tools use its gateway. Pengin's model identity must match the profile
and checkpoint path exactly. The configured Jupiter model workers reuse the
existing JudgeAPI workers, so their local service startup is disabled. They keep
`--enable-prefix-caching --mamba-cache-mode=align`.

Training uses the stored science corpus and signed rubric weights, including
negative criteria. Up to 20% failed rubric episodes are excluded from reward
normalization; larger failures remain errors. The eval instead uses Serper/Jina
live search with retries, redirect resolution, challenge detection, successful
response caching and final `Page not found.` presentation after exhausted fetches.

## Resume training and arm the judge trigger

From the repository root:

```bash
EXAMPLE=examples/search-agent-frontier-science/experiments
python "$EXAMPLE/prepare.py" --config "$EXAMPLE/training.toml" \
  --out /weka/gfaria/search-agent-repr/runs/resume450-bf16-example
# Inspect generated train.toml and policy.beaker.json before submitting.
python "$EXAMPLE/submit.py" \
  --run-dir /weka/gfaria/search-agent-repr/runs/resume450-bf16-example
```

Edit `training.toml` before preparation to change checkpoint, final step, learning
rate or placement. The detailed native training configuration is
`training/train.toml`. The layout is intentionally fixed at 4 train + 12 infer.
Preparation assigns fresh W&B, rendezvous and output paths and refuses a nonempty
output directory. Existing receipts make submission idempotent; an unresolved
submission intent requires checking Beaker for the named experiment before retry.

`submit.py` starts `watch-judges.py` as a detached local process. Keep the submission
host alive: this is not a managed Beaker automation. It checks every 30 seconds,
submits Jupiter judges when the trainer task starts, and stops its own judge
experiment when any policy task exits/cancels. A model task can start before the
trainer has completed initialization; readiness still gates actual rollout work.
State is written to `judge-trigger.json`, logs to `judge-trigger.log`.

Use an authenticated `beaker` CLI, or set `BEAKER_COMMAND` to an explicit command:

```bash
export BEAKER_COMMAND='/weka/gfaria/primebeaker/.venv/bin/python /weka/gfaria/search-agent-repr/policy-16train-16infer-cp4-mc32-resume130/beaker-auth.py'
```

The trigger inherits it. Training and inference keep running through final
validation; the packaged peer-watcher includes the clean-trainer-exit fix.

## Run checkpoint evaluation and grading

```bash
EXAMPLE=examples/search-agent-frontier-science/experiments
python "$EXAMPLE/prepare.py" --config "$EXAMPLE/evaluation.toml" \
  --out /weka/gfaria/search-agent-repr/evals/frontier430-example
python "$EXAMPLE/submit.py" \
  --run-dir /weka/gfaria/search-agent-repr/evals/frontier430-example
python /weka/gfaria/search-agent-repr/evals/frontier430-example/watch.py --interval 10
```

The evaluator starts one BF16 TP4 model, web worker and gateway, performs live
Serper/Jina/terminal checks and one smoke rollout, then runs all configured seeds
concurrently. GPT-5 grading is a separate CPU job that consumes completed answers
as they arrive. Outputs are `step430-seedN/full/{traces.jsonl,answers.jrecords.jsonl,
gpt5-research-judgments.jsonl,summary-gpt5-research.json}`.

To compare Jina formats, copy `evaluation.toml`, set `jina_format = "default"`,
and prepare a **different** output directory. `markdown` explicitly requests
Markdown; `default` removes both format headers. Header-sensitive cache keys
separate these page responses. Search query caching remains shared.

The watcher reports valid-grade-only means/pass rates and explicit errors.
The historical grading summary divides by 59 and counts errors as zero; inspect
`request_errors` and `parse_errors` before calling a summary fully graded. All
59 valid grades are required for a complete benchmark score. Truncation is not
itself a tool error. No-tool percentages count actual model tool calls.

## Container images

`images/manifest.json` records the exact base images used. To rebuild images with
this recipe bundled (Docker and authenticated Beaker required):

```bash
bash examples/search-agent-frontier-science/experiments/images/build.sh
beaker image create search-agent-policy:local --workspace ai2/oe-agents --name search-agent-policy
beaker image create search-agent-judge:local --workspace ai2/oe-agents --name search-agent-judge
beaker image create search-agent-services:local --workspace ai2/oe-agents --name search-agent-services
```

Update generated specs' `image` objects to the returned Beaker IDs to use those
rebuilds. The policy base is CUDA 13.0.1; the judge base supplies both the vLLM and
JudgeAPI Python environments. These Dockerfiles preserve verified base binaries
and add code under `/opt/search-agent-experiments`; launch commands still use the
prepared Weka copy. They do **not** rebuild CUDA/vLLM from scratch or bake the
mounted environments into a portable image. The repository's
`src/primebeaker/images/build-score-centering.sh` is a separate source-build recipe,
not a verified replacement for this experiment's exact MC32 runtime.

## Validation

```bash
python examples/search-agent-frontier-science/experiments/check.py
```

This renders both variants in temporary directories, checks Python/shell syntax,
GPU layout, priority, precision, relocated paths and error-on-overwrite. It submits
nothing. GPU scheduling, live web access and image builds are separate checks.
