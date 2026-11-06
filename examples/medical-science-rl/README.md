# Medical/science policy RL with MoE, Qwen9B or Pengin9B feedback

These examples reproduce the current RubricHub **answer-policy** recipe. A 4B
policy answers medical/science questions; a separately served judge supplies
binary rubric labels for its reward. This trains the policy, not the 9B/35B judge.
The existing training dataset is mixed **medical + science**, not medical-only.

The launcher renders inspectable Beaker JSON and a resolved TOML first. Submission
is a separate command. Nothing is submitted by importing the example or installing
PrimeBeaker. This directory does not alter existing runs or their environments.

## Files and images

| Component | Actual runtime/image |
| --- | --- |
| Policy trainer + policy inference | `docker://nvidia/cuda:13.0.1-cudnn-devel-ubuntu24.04` |
| Judge vLLM and isolated JudgeAPI | `beaker://01M3WPHYKDAZ4533VCDZGNZGHZ` |
| Judge serving version | vLLM **0.29.0**; TP=4; EP additionally enabled for MoE |
| Judge API interpreter | `/app/judge-venv/bin/python` inside the judge image |
| Policy interpreter | Mounted `/weka/gfaria/experiments/reverse-text-online-20261013/venv/bin/python` |
| Redis/head and optional tools | Existing private LiteRegistry head; discovered at startup |

**The CUDA base image does not contain PrimeRL, vLLM or MC32.** The exact mounted
Python paths used by the current jobs are listed in [runtime.json](runtime.json).
That file also records source HEADs and whether their tracked working trees were
dirty when captured; a HEAD alone does not pin uncommitted runtime changes.
The launcher exposes this PrimeBeaker checkout's `src` first and imports the ML
stack and legacy API dependencies from those mounted paths. They must exist on
all allocated nodes. This is the Weka reproduction path, not a claim that a
fresh CUDA container is a standalone training image.

`runtime.json` is editable for another installation: supply an equivalent mounted
environment with native MC32 and the listed JTC/LiteRegistry interfaces. Do not
install into a Python environment used by ongoing jobs. For an image containing
the native ML stack, see the [pinned MC32 image builder](../../src/primebeaker/images/README.md#standard-score-centering-and-mc32-image-build).
That build is separate: it has no published Beaker ID assumed by this example,
and replacing the mounted runtime requires adapting the interpreter/Python paths
in a copy of `runtime.json` and building the application dependencies into it.

The source configs come from these operational launches:

- `rh-base-moe35b-medscience-uniform8-8k-mc32-bf16-fp8-20261104`
- `rh-base-moe35b-medscience-resume400-to800-32k-20261105`
- `rh-instruct-moe35b-medscience-fresh-64kjudge-tp4-20261104`

The templates incorporate the latest requested **1,024 max inflight** for future
starts. Running earlier processes used 256 until restarted. Pengin applies its
existing step-400 judge harness to this medical/science recipe; it is a prepared
variant, not a claim that a Pengin-judged medical run has completed.

## Policy configs

| Phase / TOML | Initial policy | Steps | Output / context | Rubrics | Train / infer GPUs | Checkpoint |
| --- | --- | --- | --- | --- | --- | --- |
| [base-0-400](configs/base-0-400.toml) | `Qwen/Qwen3-4B-Base` | 0 → 400 | 8,192 / 32,768 | 8 | 2 / 6 | every 100 |
| [base-400-800](configs/base-400-800.toml) | same run, checkpoint 400 | 400 → 800 | 32,768 / 65,536 | 8 | 4 / 4 | every 100 |
| [instruct-0-800](configs/instruct-0-800.toml) | `Qwen/Qwen3.5-4B` | 0 → 800 | 57,344 / 65,536 | 16 | 2 / 6 | every 25 |

All three use LR **4e-6**, IcePop token-mean loss (ratio band 0.2…5.0), BF16
optimizer/model and gradient reduction, MC32 score centering with 32 samples,
128-token score-centering chunks, and maximum off-policy lag **32**. Policy
inference uses FP8 per-block weight quantization and FP8 KV cache, TP=1 with
DP/API workers matching its inference GPU count. Weight broadcast is NCCL.
Batch size is 256; group size is 8; sampling is temperature 1, top-p 1.
Policy thinking remains enabled. Judge thinking is configured separately.

These differ from the historical JTC9B/Qwen9B baselines, which used LR1e-6,
FP32 training/reduction, no MC32 and lag4. Their curves are not a controlled
comparison of judge choice alone. [historical-qwen9b-reference.json](historical-qwen9b-reference.json) records those original settings; selecting Qwen9B in this launcher changes the feedback source within the current MC32 recipe.

## Judge selection and exact prompt contracts

| `--judge` | Model | Workflow | Output budget | Thinking |
| --- | --- | --- | --- | --- |
| `moe` | Original `Qwen/Qwen3.5-35B-A3B` | grouped rubrics, one `submit_grouped_judgments` call | 65,536 | **off** |
| `qwen9b` | Original `Qwen/Qwen3.5-9B` | same grouped-rubric prompt | 16,384 | off |
| `pengin9b` | Pengin9B RL step400, override with `--judge-model` | one tool episode per rubric | 16,384 per response | on |

Judge context defaults to **131,072** for all modes. Qwen9B preserves the
historical batched prompt and output budget but raises the historical 32k
serving context explicitly: known overflow/truncation failures should not be
silently reproduced. This does not remove the need to monitor finish reasons.
The model workers allow 1,024 sequences each; this differs from policy episode
`max_inflight`. Add model replicas using separate allocations and the same
`judge_model` registry identity. TP4 means four GPUs per replica, not four replicas.

Pengin uses the exact [hint-free prompt and tool schema](profiles/pengin-hint-free-submit.json):
`trace_inspection(command)` and `submit(rubric_id, feedback, label)`. The evaluated
response is supplied in inspection-tool stdin, not inserted into the user prompt.
Terminal output is truncated to **2,000 characters**. The rubric labels remain
pass/fail; no numeric 1–10 scales, hints, grouped Qwen prompt, or separate final
JSON are substituted. An accepted `submit` is required; an unfinished/tool-limit
trace does not become a valid label. The native package validates the template's
SHA256 and fixes the tool naming in its Pengin harness.

## Registry, gateway and tools

Every worker resolves Redis from `--head-registry` using LiteRegistry's endpoint
discovery. No fixed remote gateway hostname is stored in the policy config. The
policy starts and supervises its **own local gateway** at `127.0.0.1:1213`;
`judge_server_url` is always `http://127.0.0.1:1213/judge`. The separate CPU JudgeAPI
also owns a local gateway and registers its run-specific service alias in Redis.
Model routing uses the same Redis-backed least-outstanding namespace.

MoE and Qwen grouped judging require Redis, gateway, model replicas and JudgeAPI;
they do not use terminal/local-search tools. Pengin additionally requires the
usual private **terminal + local-search** deployment registered in that same
head, with the matching corpus (default `localsearch:bc-rl-v1`). These workers
reuse that warmed tool deployment; they do not launch a Python tool adapter or
replay response snippets in place of stdin.

Before Pengin submission, make sure the private head has live `terminal` and
`localsearch:bc-rl-v1` services and the workers have the required terminal/corpus
images cached. A different local-search corpus changes what `browse` can verify.
For a new tool deployment, use the repository's
[service image installation](../README.md#install-the-required-images) and
[Podman service-stack example](../podman-terminal/services.yaml), substituting
your built immutable image IDs and matching local-search corpus/index. The old
Podman/gateway base IDs alone are insufficient: the operational deployment had
startup discovery/cache fixes and preloaded tools. This example reuses its head
instead of claiming those fixes are baked into a newly published tool image.

## Public experiment dataset

The exact input slices are published at
[`graf/rubrichub-medical-science-rl-slice`](https://huggingface.co/datasets/graf/rubrichub-medical-science-rl-slice).
Release revision: `667c5d18eb9182c34919d427e71acec3b99c764e`.
There are 6,437 training questions with 229,632 signed rubric items, and 256
multiple-choice validation questions. The two schemas have separate Hub configs:
`rubric_rl` (train) and `mc_validation` (validation). Negative signs, original
rubric indices, prompt ordering and validation provenance are preserved.

Download the original files consumed by the tasksets, with a pinned Hub revision
and SHA256 checks against the release manifest:

```bash
uv run --with huggingface_hub python examples/medical-science-rl/download_data.py \
  --directory /weka/gfaria/datasets/rubrichub-medical-science-rl-slice
```

Then add these arguments to any `prepare` command below:

```bash
--train-data /weka/gfaria/datasets/rubrichub-medical-science-rl-slice/original/rubric_rl \
--eval-data /weka/gfaria/datasets/rubrichub-medical-science-rl-slice/original/validation.jsonl
```

The original Weka paths remain defaults for reproducing existing jobs. Hub Parquet
exports are also available via `load_dataset`; do not substitute the dataset
viewer schema for the on-disk directory path expected by this taskset. The source
training revision/selection recipe was not retained; this release preserves the
exact staged experiment slice rather than inventing reconstruction provenance.
See the Hub card and `LICENSES.md` for upstream attribution and component licenses.

## Prepare a fresh Base run

Run from a mounted PrimeBeaker checkout. Python3.11+ and a configured Beaker CLI
are sufficient to render specs; the host does not need the GPU libraries.
The target workspaces must have access to the Weka dataset and named secrets.

```bash
beaker account whoami
python examples/medical-science-rl/launch.py prepare \
  --directory /weka/gfaria/launches/example-medscience-moe-base-phase1 \
  --run-name example-medscience-moe-base \
  --judge moe --phase base-0-400
```

This prints the output run directory and a fresh W&B run ID. Keep that ID for
resume. Inspect `train.toml`, `launch.json`, `profiles/`, `runtime.json` (the source
file), `model.beaker.json`, `api.beaker.json` and `trainer.beaker.json` before
submitting. Images, paths, secrets and workspaces are explicit in those files.
Optional arguments include `--runtime`, `--train-data`, `--eval-data`,
`--output-dir`, `--head-registry`, `--judge-model`, `--judge-context`,
`--local-search-model`, `--weka-dataset`, `--wandb-entity`, `--wandb-secret`
and `--hf-secret`.

Submission is deliberate:

```bash
python examples/medical-science-rl/launch.py submit \
  --directory /weka/gfaria/launches/example-medscience-moe-base-phase1
```

The launcher submits model, API and trainer experiments immediately. It does
not wait for every planned replica. The trainer's native readiness gate requires
one registered model and one API (plus terminal/search for Pengin). The three
submission receipts contain the actual Beaker experiment IDs. A persisted intent
prevents duplicate submission after an uncertain network failure; inspect the
workspace before clearing an intent manually.

Placement defaults:

- Trainer: `ai2/oe-agents-holmes`, `ai2/holmes`, 8 GPUs, **high**, minimum 4h.
- Judge model: `ai2/oe-agents`, `ai2/jupiter`, TP4 (4 GPUs), **high**, minimum 4h.
- Judge API: `ai2/oe-agents`, Jupiter, CPU-only, **no CPU/GPU resource requests**,
  high, minimum 4h.

`--judge-cluster ai2/holmes` moves only model serving to Holmes, keeps it high,
and selects `oe-agents-holmes`; the CPU API remains on Jupiter. To submit model
capacity only when a queued trainer receives compute, use `submit --part trainer`
and then `--part model` and `--part api` once Beaker reports `scheduled`. This
example does not imply that an unscheduled model experiment is serving.

## Qwen9B and Pengin9B variants

```bash
python examples/medical-science-rl/launch.py prepare \
  --directory /weka/gfaria/launches/example-medscience-qwen-base \
  --run-name example-medscience-qwen-base --judge qwen9b --phase base-0-400

python examples/medical-science-rl/launch.py prepare \
  --directory /weka/gfaria/launches/example-medscience-pengin-base \
  --run-name example-medscience-pengin-base --judge pengin9b --phase base-0-400 \
  --judge-model /weka/gfaria/launches/pengin-9b-rl-benchmarks-100step-20261025/9b/step_400/hf-checkpoint
```

Then use `submit --directory ...` for the selected directory. Pengin is a
prepared new feedback-source variant with the deployed harness; its medical
training outcome and the new example launcher have not been GPU-validated.

For the Instruct companion, prepare with `--phase instruct-0-800` and a distinct
run name/directory. The model/context/prompt switch is in its full TOML, not just
a label. All modes use the medical/science data unless explicitly overridden.

## Resume Base at 400 with 32k output and 4 train / 4 infer

Wait for the successful phase-one checkpoint to contain both
`checkpoints/step_400/trainer/.metadata` and
`checkpoints/step_400/orchestrator/progress.pt`. Keep the **same** run name,
output directory, judge model/alias and W&B ID:

```bash
python examples/medical-science-rl/launch.py prepare \
  --directory /weka/gfaria/launches/example-medscience-moe-base-phase2 \
  --run-name example-medscience-moe-base --judge moe --phase base-400-800 \
  --wandb-run-id ORIGINAL_WANDB_RUN_ID

python examples/medical-science-rl/launch.py submit \
  --directory /weka/gfaria/launches/example-medscience-moe-base-phase2 \
  --part trainer
```

Reuse the already-running judge/API pool for the same alias; submit a replacement
model/API from phase two only if the old pool is no longer running. The launch
checks checkpoint completeness instead of silently restarting from base weights.
It resumes trainer/optimizer and orchestrator state, including source progress.
The four-trainer phase relies on native distributed-checkpoint resharding from
its two-trainer checkpoint. Keep compatible PrimeRL/Torch versions.

Base phase one leaves the judge pool alive for continuation. After a successful
step 800, the worker writes the completion marker and its model/API supervisors
exit. A failed/preempted policy does not mark completion. Stop leftover service
experiments explicitly with `beaker experiment stop EXPERIMENT_ID` if abandoning
a run; the launcher never cancels an unrelated shared deployment.

## Data, rubric signs and diagnostics

The defaults point at the current private HF-on-disk training directory and the
256-question MC validation JSONL. These data and model checkpoints are not
bundled in Git. Supply compatible paths for another installation. The taskset
loads source questions and signed rubric criteria; the candidate answer comes
from the live policy. Data is shuffled, rubrics are sampled uniformly without
replacement using the configured seed, and scoring is `sign_only`.

Negative rubric bodies get this native instruction; positive rubrics are unchanged:

> This is a penalty condition. Return pass ONLY if the described error or bad behavior is present in the response; return fail if it is absent. Pass here means the penalty applies, not that the response is good. Condition:

A negative rubric's pass applies its negative sign. A policy hitting its output
limit receives **−0.2 without a judge call**. Empty output receives−0.2; an orphan
`</think>` gets its configured format penalty. A failed judge request errors the
rollout; it is not silently accepted as a valid judgment. Judge labels are pass/fail.

Use the [shared dashboard](../dashboards/README.md) to inspect current optimizer
step, batch waiting, active/queued requests, interval mean request latency, KV
cache, W&B reward and episode token length. Track producer/run identity when
comparing resumed histories. `time/step` means one 256-rollout optimizer update
including waiting, not one model request. `num_total_tokens` includes policy input
and generated output, not judge reasoning. Compare raw gradient norms cautiously
across LR/MC32/precision changes, and report historical judge context-overflow /
truncation limitations when plotting the older Qwen9B/MoE experiments.

No new training/evaluation jobs, image builds or tests were run to add these
examples. Source configs and image IDs come from the operational runs listed
above; the newly packaged launcher and Pengin medical variant still need an
explicit GPU run before any throughput or quality claim.
