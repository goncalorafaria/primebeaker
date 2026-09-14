# BrowseComp+ search-agent eval example

This runs an actual 32-question eval using the webterminal harness and Quokka
judge, then shows how to use its recorded calls for throughput tuning. It starts
only the evaluator: policy inference, local search, terminal and judge services
must already be registered in a reachable LiteRegistry stack.

## Configure the eval

Install PrimeBeaker with its runtime/evaluation dependencies, JTC with the
search-agent eval workflow, and Rexs on the submission host. The runtime image
must contain PrimeBeaker, JTC, and their eval dependencies. The dataset is a JTC
JTask parquet export of BrowseComp+, not arbitrary upstream parquet. Stage it and
the checkpoint on shared storage visible under `/weka` in the runtime image.

Copy `browsecomp-subset.yaml` and replace:

- `image`: your immutable eval image or Rexs image alias.
- `registry`: the existing registry reachable from the evaluator allocation.
- `model`, `judge_model_path`: exact model identities registered by the stack.
- `dataset`, `run_dir`: the staged JTask parquet and a fresh output directory.
- `local_search_model_path`, `judge_service_model_path`: registered service names.

The stack must have a healthy policy endpoint, BrowseComp local search, terminal,
and Quokka judge service. The judge's own search/terminal dependencies also need
to be available. `keep_stack: true` preserves the shared stack after the eval.
Engine limits in this YAML **do not reconfigure an existing model server**; set
128K context/token budget on that server before running. KV offloading is not
required. Start with batch size 8; raising it is a separate workload decision.

## Preview and submit through Rexs

From the repository root, activate the environment containing Rexs and put the
included Beaker shim first on PATH:

```bash
export PATH="$PWD/examples/search-agent-eval/bin:$PATH"
export REXS_PROFILE=/absolute/path/to/your-evaluator-profile.yaml
primebeaker evaluation preview --config=/absolute/path/to/browsecomp-subset.yaml
```

The Rexs profile must supply your account/partition, image mapping, runtime and
shared `/weka` mounts. Use an evaluator profile with CPU resources; the existing
policy/judge allocations supply the GPUs. Inspect the preview before submission.
Check the queue, counting pending and running interactive jobs together, and keep
at most two interactive submissions active. Unlike `vllm-sweep rexs`, the native
eval command does not itself implement this limit check.

```bash
squeue --user="$USER" --format='%.18i %.30P %.10T %.40j'
primebeaker evaluation submit --config=/absolute/path/to/browsecomp-subset.yaml
```

The shim routes Beaker experiment commands through Rexs. No new GPU experiment is
needed if the policy/judge stack already exists. Use `max_rows: 830` only when you
want the full dataset and have suitable wall time; begin with the small subset.

## Inspect results and replay a subset

Completed rollouts are saved beneath the configured run directory:

```text
rollouts/step_0/eval/all/traces.jsonl
```

These include per-turn prompts/tool results, output lengths and judge rewards.
A time-limited eval may have only partial traces; replaying them measures serving
throughput, not the full benchmark score.

Export tool schemas from the same environment version used for evaluation and
prepare the replay manifest:

```bash
python examples/search-agent-eval/export_tools.py > /shared/tools.json
primebeaker vllm-sweep prepare-replay \
  --traces=/shared/eval/rollouts/step_0/eval/all/traces.jsonl \
  --tools=/shared/tools.json --output=/shared/replay-manifest.json \
  --conversations=32 --turns=4 --runtime_traces=/data/traces.jsonl
```

Adjust the host/runtime paths to your mounts. Then copy
[`../vllm-sweep/replay.json`](../vllm-sweep/replay.json), set its model and manifest
paths, and follow the [throughput sweep guide](../vllm-sweep/README.md). Keep the
eval results as a baseline; use a fresh directory for every new sweep bundle.
