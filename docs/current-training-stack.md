# Current fork stack and Podman RL setup

The standard recipe pins the merged native MC32 fork revisions below. These
source pins do not imply that a new combined image has passed GPU validation.

## GitHub sources

| Component | Fork / branch | Published revision observed |
| --- | --- | --- |
| PrimeBeaker | [goncalorafaria/primebeaker](https://github.com/goncalorafaria/primebeaker) | This checkout; local changes may not be published |
| PrimeRL | [main](https://github.com/goncalorafaria/prime-rl/tree/main) | [`243cbb7`](https://github.com/goncalorafaria/prime-rl/commit/243cbb7d77702017b5f0e247afd7df35b07ee4e9) |
| Renderers | [feat/packed-sampler-heads](https://github.com/goncalorafaria/renderers/tree/feat/packed-sampler-heads) | [`a56bb9d`](https://github.com/goncalorafaria/renderers/commit/a56bb9d046ada1c7a1f6fdb183549e0b0f5a7e1f) |
| Verifiers | [feat/upstream-score-centering](https://github.com/goncalorafaria/verifiers/tree/feat/upstream-score-centering) | [`8a44b61`](https://github.com/goncalorafaria/verifiers/commit/8a44b6145905b19a94ec777dd7d8b06d4e4c9b0f) |
| vLLM compact extraction | [releases/v0.29.0](https://github.com/goncalorafaria/vllm/tree/releases/v0.29.0) | [`409c9ef`](https://github.com/goncalorafaria/vllm/commit/409c9efd1aeb0fdd3f32d06554e33f520534e273) |
| LiteRegistry / Podman services | [master](https://github.com/goncalorafaria/literegistry/tree/master) | [`ba51669`](https://github.com/goncalorafaria/literegistry/commit/ba516695577ffb4f897a706c0e64b5226b9022f1) |

Use the vLLM **release branch above** for the CUDA 13 / vLLM 0.29 build.
The fork's default `main` is a different revision; a default clone does not select
this compact extraction recipe.

### Latest source versus reproducible build

The [image builder](../src/primebeaker/images/build-score-centering.sh) pins
PrimeRL and vLLM to the full revisions above. PrimeRL itself pins these submodules:

| Dependency | Revision used by the build |
| --- | --- |
| Renderers | [`a56bb9d`](https://github.com/goncalorafaria/renderers/commit/a56bb9d046ada1c7a1f6fdb183549e0b0f5a7e1f) |
| Verifiers | [`8a44b61`](https://github.com/goncalorafaria/verifiers/commit/8a44b6145905b19a94ec777dd7d8b06d4e4c9b0f) |
| Prime environments | [`1213b73`](https://github.com/PrimeIntellect-ai/prime-envs/commit/1213b7336d33bdb7e1abd5d61f1e0e2295d001e1) |
| Pydantic config | [`65b15df`](https://github.com/PrimeIntellect-ai/pydantic-config/commit/65b15dffba82d4be19efdaf8b2b9705cc1756be8) |
| Prime kernels | [`3fb83eb`](https://github.com/PrimeIntellect-ai/prime-kernels/commit/3fb83eb7aa639693876034d92a8df29245372cdf) |

The Renderers and Verifiers commits include the merged MC32/packed-transport
changes. Preserve the gitlink pins for reproduction; updating submodules is a
separate image change requiring integration validation.

## Fresh terminal training with MC32 and FP8

The [Qwen3.5-4B native solver recipe](../examples/podman-terminal/MC32-FP8.md)
provides a taskset, direct bash harness, complete TOML, and preview/submission
launcher. It starts from an initial container with no reference prefix. It uses
the pinned native MC32 application image and requires compatible CUDA 13 binaries.

## Choose the environment before choosing the image

There are three different workflows:

1. **Terminal task solving:** `examples/podman-terminal/rl.toml` creates a task
   container, accepts `bash`, and grades private tests after `echo TERMINAL_COMPLETE`.
2. **Prefix continuation:** `PodmanTerminalReplayEnv` replays a successful prefix,
   then trains a policy to finish the task using the same private-test grader.
3. **Terminal Replay rubric judging:** the current 2B run replays an existing
   trajectory, exposes inspection tools, and accepts rubric labels through
   `submit`. Its reward compares those labels to Sol. It does not train a policy
   to solve the original terminal task.

The first two examples use the historical `source.legacy` schema and pinned
runtime wheels. Current PrimeRL uses `source.env.taskset` and native Verifiers
registration. Merely selecting the new image for an old TOML is insufficient.
The current rubric-judge harness is launch-local, not shipped in PrimeBeaker.

## Build the published score-centering image

From the PrimeBeaker repository:

```bash
bash src/primebeaker/images/build-score-centering.sh \
  prime-rl-score-centering:243cbb7-409c9ef-mc32
```

This builds PrimeRL and its pinned Renderers/Verifiers submodules, then installs
the matching native vLLM fork. All MC32 implementation changes are committed in
the forks; the image recipe applies no source patches or runtime overlays. The
recipe requires CUDA 13.0, Torch 2.13 and vLLM 0.29.0. Publish the resulting image
to your Beaker workspace and supply its immutable `beaker://...` URI explicitly.
The historical default image catalog is not this image.

Install the PrimeBeaker application and required tool-client packages in the
resulting environment without replacing its PrimeRL/Renderers/Verifiers stack.
In particular, the historical `primebeaker[runtime]` extra pins Verifiers 0.2.1;
installing it unchanged into the new training image can replace incompatible
parts of the stack. Resolve the application dependencies for the chosen image
and record them in the image receipt. The build above supplies the ML stack,
not the launch-local rubric harness or a new complete Podman application image.

For Holmes, use the native CUDA driver. For Jupiter, check the allocated node's
driver against CUDA 13; use the matching compatibility setup when necessary.
The cluster name alone does not establish driver compatibility. Keep vLLM Python
source and compiled extensions from the same build/version.

## Packed logprobs and score centering

Merge these tables into a **native** RL config; this is a fragment, not a complete
training TOML:

```toml
[trainer]
score_centering_token_chunk_size = 128

[trainer.loss]
type = "icepop"
score_centering = true
ratio_low = 0.2
ratio_high = 5.0
aggregation = "token_mean"

[trainer.model]
optimization_dtype = "float32"
reduce_dtype = "float32"
fused_lm_head_token_chunk_size = "disabled"

[orchestrator.train.sampling]
score_centering_top_k = 128
top_k = -1
top_p = 1.0
temperature = 1.0

[inference.vllm]
logprobs_mode = "processed_logprobs"
max_logprobs = 128

[inference.env_vars]
VLLM_USE_V2_MODEL_RUNNER = "1"
VLLM_USE_COMPACT_LOGPROBS = "1"
VLLM_PRIME_COMPACT_LOGPROBS = "0"
VLLM_PRIME_FLASHINFER_SAMPLER_HEAD = "0"
```

This selects native compact extraction from the vLLM fork, avoiding simultaneous
activation of the older PrimeRL extraction patches. `score_centering_top_k`
controls the recorded head, not sampling truncation. The normalizer still covers
the full distribution. Keep the packed token IDs/logprobs through the renderer,
verifier, and trainer; a new vLLM kernel alone does not fix Python-list unpacking
elsewhere. The current production jobs additionally use the local NumPy packing
and activation-offload overlay listed below. Do not assume the published image
includes every local overlay.

See [PrimeRL's score-centering implementation and options](https://github.com/goncalorafaria/prime-rl/blob/243cbb7d77702017b5f0e247afd7df35b07ee4e9/docs/score-centering.md).
MC32 is included natively by these fork commits; see the
[merged fork PRs and source pins](../src/primebeaker/images/mc32/README.md).
Merge the [config fragment](../src/primebeaker/images/mc32/rl-fragment.toml)
into a complete native RL config to select `score_centering_estimator="mc32"`,
exactly 32 `score_centering_samples`, and `VLLM_SCORE_CENTERING_MC32=1`. Omit
`top_k` and `score_centering_top_k`; use positive temperature and `top_p=1.0`.
The packed transport retains duplicate IID draws. The prior experimental source
verification does not validate a newly assembled image or establish a speedup.
No new Docker image build, tests or GPU run were performed for this integration.

## Start Podman services

Use the [service-stack instructions](../examples/podman-terminal/README.md#install-the-runtime-and-service-images)
to install the service images, insert their immutable IDs in the YAML, preview,
and launch the Podman workers and gateway. Training and service images are
separate artifacts.

```bash
primebeaker services podman yaml preview --config=/path/to/services.yaml
primebeaker services podman yaml launch --config=/path/to/services.yaml
```

For a head registry, the service publisher uses `sqlite:///shared/head.sqlite3`
and clients use `head+sqlite:///shared/head.sqlite3`. The head resolves Redis;
it is not the gateway URL. Resolve the gateway through discovery and configure
the environment to use it. Verify readiness for the actual Podman route, worker
capacity, and a real session before submitting training. A healthy HTTP server
alone does not establish that replay containers can start.

Reuse an existing service stack when appropriate. In multi-node PrimeBeaker,
`PRIMEBEAKER_EXTERNAL_GATEWAY_URL` plus `PRIMEBEAKER_REQUIRED_GATEWAY_PATHS`
select the existing gateway and its required routes. For a single-node custom
harness, configure its gateway explicitly; the launcher does not automatically
start or own external services.

## Configure and launch native training

Use explicit GPU placement and retain intermediate checkpoints:

```toml
[deployment]
type = "single_node"
gpus_per_node = 8
num_train_gpus = 2
num_infer_gpus = 6

[ckpt]
interval = 50
keep_last = 1
keep_interval = 100
```

For six TP=1 inference replicas, the native `[inference.vllm]` table also needs
`tensor_parallel_size = 1`, `data_parallel_size = 6`, and `api_server_count = 6`.
Use `[monitors.wandb]`, `[run]`, `clean = false`, and native `[resume]` configuration.
Set `shuffle = true` on the training source; it shuffles the finite source once
with a fixed seed, rather than reshuffling at every epoch.

Keep the following budgets distinct:

- trainer packing length: token capacity of a packed training sequence;
- model context: prompt, tool responses, and generated tokens together;
- `max_completion_tokens`: maximum generated tokens in one response;
- agent `max_output_tokens`: cumulative generated output budget;
- harness `token_budget`, tool calls, and agent turns: additional stopping rules.

The current rubric-judge recipe uses packing 524288, inference context 262144,
output 131072, 128 tool calls and 129 turns. A cumulative budget checked only
between responses can overshoot within the final response. These are the current
experiment's settings, not universal defaults or validated settings for every model.

With a complete native TOML and an application image containing its registered
harness, preview the single-node launch:

```bash
primebeaker rl preview \
  --toml=/weka/your-project/train.toml \
  --image=beaker://YOUR_APPLICATION_IMAGE_ID \
  --registry=head+sqlite:///weka/your-project/head.sqlite3 \
  --workspace=ai2/oe-agents-holmes \
  --cluster=ai2/holmes \
  --priority=urgent --min-runtime-hours=1 \
  --working-dir=/weka/your-project \
  --scratch-dir=/weka/your-project \
  --home-dir=/weka/your-project
```

Change `preview` to `submit` after reviewing the resolved spec. Preview checks
launch metadata, not GPU execution or the full native environment schema.
For a fresh run use a unique output directory and W&B identity. For a resume,
retain the original run directory and identity, choose a complete trainer and
orchestrator checkpoint, and set the desired total `max_steps`. Checkpoint
retention does not itself launch benchmark evaluations; the export/eval watcher
must be running separately.

## Current Weka reference deployment

The deployed Terminal Replay 2B setup is preserved at:

- Launch: `/weka/gfaria/launches/terminal-replay-output128k-lr5e7-20261025`
- Native TOML: `train-cp2-infer6-output128k.toml`
- Beaker spec: `experiment.beaker.json`
- Entrypoints: `start-rl-acfix-cp2-infer6-output128k.sh`,
  `start-inner-cp2-infer6-output128k.sh`, `run-runtime-acfix.sh`
- Harness: `src/tmax_sol_judge/`; dependency receipt: `wheel-manifest.json`
- NumPy packing / activation-offload runtime:
  `/weka/gfaria/launches/terminal-replay-holmes-packed-acfix-20261024/runtime`
  with hashes in the sibling `manifest.json`.

This launch uses shared Weka code, an explicit Python environment, and overlays;
it is not reproduced by a Beaker image ID alone. Copying the launch requires
rebinding its embedded paths, dataset, run identity, gateway, and output directory.
Its harness attempts session cleanup in `finally`; gateway/affinity failures or
process termination can still prevent cleanup. Inspect cleanup failures alongside
reward, truncation, submission completion, and pass/fail accuracy.

The fail-weighted correctness reward in this experiment is a deliberate reward
choice, not a runtime feature. Preserve or change it explicitly when preparing a
new run. Shorter traces and higher weighted reward alone do not establish better
all-rubric accuracy.
