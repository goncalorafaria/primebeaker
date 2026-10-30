# Fresh terminal solving: Qwen3.5-4B, MC32, FP8

This recipe trains **Qwen/Qwen3.5-4B to solve terminal tasks from the initial
container state**. It never runs a reference prefix or asks the policy to judge
rubric labels. It uses the same task/image/private-test data contract as the
prefix-continuation workflow, but reconstructs only the original user prompt.
Reference assistant/tool turns and row-level tool schemas are not passed to the model.

## Training settings

| Setting | Value |
| --- | --- |
| Starting model | Qwen/Qwen3.5-4B, not a Pengin judge checkpoint |
| Loss | IcePop with MC32 score centering |
| Probes | 32 IID draws with replacement; NumPy packed transport |
| Inference | FP8 per-block weights, BF16 activation dtype, V2 runner |
| Optimizer / reduction | FP32 / FP32 |
| Learning rate | 5e-7 |
| Placement | 2 trainer + 6 inference GPUs on one node |
| Batch / group | 256 rollouts / 16 per task |
| Inflight / off-policy limit | 64 / 4 steps |
| Context / packed sequence | 65,536 / 65,536 tokens |
| Per-response / cumulative output | 16,384 / 61,440 tokens |
| Tool calls / turns | 128 / 129 |
| Duration | 400 optimizer steps |
| Save / retain | save every 50; retain every 100 plus latest |
| Beaker | urgent, minimum runtime 1 hour |

See [mc32-fp8-4b.toml](mc32-fp8-4b.toml). These are prepared defaults, not a
throughput claim or a completed GPU smoke result for this new environment.
Sampling is untruncated (`top_p=1.0`, no `top_k`) with positive temperature.
MC32 is explicitly requested on training responses; evaluation does not request
MC probes. FP8 applies to inference, not optimizer state or gradient reduction.

## Native task and reward contract

- `primebeaker-podman-solver` loads JSONL rows with `task_id` (or `record_id`),
  `prompt`, `original_image`, and `test_final_state`.
- `primebeaker-podman-solver-harness` starts one fresh Podman session per rollout.
- The only model tool is `bash(command: str)`. Cwd and exported variables persist
  between calls; aliases, shell functions, and interactive process state do not.
- Exactly `echo TERMINAL_COMPLETE`, with successful execution and matching output,
  submits the task. No further model calls occur after submission.
- The host then sends private pytest source through stdin to a random path and
  executes it. Private tests and grader output are not part of the model prompt.
- Reward: **+1 for passing private tests**, **0 completion bonus**, **−0.1 per
  invalid tool call**. Incomplete tasks earn no test reward. No Sol/Flash labels,
  class-frequency weights, or rubric rewards are used.
- Reward comes from pytest exit status, not a model-writable reward file. Exit 0
  passes; exit 1 fails; other exits and grader timeouts are episode errors.
- Transport/setup failures propagate as errors rather than being scored as a
  policy failure. Cleanup is attempted in `finally`; cleanup failures are logged.

The grader runs in the task container after submission, matching the underlying
terminal-task workflow. It is not a separate trusted grading sandbox against a
policy that deliberately tampers with Python, pytest, or background processes.
Task images must already have their runtime dependencies and pytest available.

The bundled default data is the original 512-task train / 32-task validation
subset. To use the built GLM task images, supply that preparation's train and
validation JSONL files through the data arguments below. Their `original_image`
must identify an initial-state image; a snapshot taken after a prefix is not a
fresh task even if the reference turns are omitted. Train/validation overlap is
rejected by task ID.

## Runtime requirements

Build and publish the [standard native MC32 image](../../src/primebeaker/images/README.md#standard-score-centering-and-mc32-image-build).
It pins the merged PrimeRL, Renderers, Verifiers and vLLM fork commits; no local
MC32 source overlay or patched Python environment is required. Supply its
immutable Beaker URI to the launcher. The default worker interpreter is
`/app/.venv/bin/python`, with `uv` at `/usr/local/bin/uv`.

The image must also include PrimeBeaker's native solver and Podman client
application dependencies. `primebeaker[terminal-native]` supplies those client
and HTTP dependencies; do not install the historical `runtime` extra over the
native MC32 ML stack. The launcher exposes this checkout's application `src`
directory, while PrimeRL/Renderers/Verifiers/vLLM come from the installed image.

`mc32-preflight.py` checks native MC32 imports, the pinned vLLM VCS revision,
and the native RL configuration before GPU startup. It does not run tests or
claim a successful GPU validation of the assembled image. Validate the newly
built image and solver integration before full training.

## Prepare and preview

Start or reuse the [Podman service stack](README.md#install-the-runtime-and-service-images).
Pass its reachable gateway root URL; do not pass the Redis or SQLite registry URI
as the HTTP gateway. From the PrimeBeaker checkout, with PrimeBeaker installed in
your launcher environment:

```bash
uv run --no-project --python /path/to/launcher-venv/bin/python \
  examples/podman-terminal/launch-mc32.py \
  --image beaker://YOUR_NATIVE_MC32_APPLICATION_IMAGE \
  --gateway-url http://YOUR_GATEWAY:PORT \
  --output-root /weka/your-project/terminal-mc32
```

This writes a unique resolved TOML and preview receipt, without submitting a job.
Use `--train-data` and `--validation-data` for alternate task files. Use
`--runtime-python` and `--uv-executable` only when those executables live at
different paths inside the worker image. `--podman-client-source` optionally
selects a shared application-client checkout; the installed client is used by
default. No PrimeRL, Renderers, Verifiers or vLLM source paths are injected.

The default target is Jupiter / `ai2/oe-agents`. Use
`--cluster ai2/holmes --workspace ai2/oe-agents-holmes` for Holmes. If the Jupiter
node needs CUDA compatibility libraries, explicitly add
`--cuda-compat-dir /usr/local/cuda/compat`; no compatibility path is forced on Holmes.
Select the image after checking the node driver against its CUDA 13 binary requirements.

Add **`--submit`** to launch. Every invocation prepares a fresh run identity;
this helper does not resume an earlier run. Resume by using the saved TOML/run
identity with a complete native `[resume]` checkpoint configuration. The launcher
keeps the original checkpoint-retention settings; it does not schedule benchmark
evaluations automatically.
