# vLLM eval throughput tuning

`primebeaker vllm-sweep` measures inference throughput using recorded search-agent
turns or vLLM 0.22 synthetic benchmarks. Run on allocated GPUs with the same model
and runtime as the eval deployment. It runs trials sequentially and saves partial
measurements; request completion is not required for throughput analysis.

## Start with actual eval traces

Export the eval environment's tool definitions as an OpenAI function-tool JSON
list. Use traces whose `info.search_agent.trajectory` entries contain `prompt`
and `response.usage.completion_tokens`, as produced by the JTC search-agent eval.

```bash
primebeaker vllm-sweep prepare-replay \
  --traces=/shared/traces.jsonl --tools=/shared/tools.json \
  --output=/shared/replay-manifest.json --conversations=32 --turns=4 \
  --runtime_traces=/data/traces.jsonl
```

This selects up to four consecutive turns from each of 32 sampled conversations,
retaining the full recorded prompt history, including search and terminal output.
The manifest records offsets, source checksum, tool schemas and seed; keep the
source file unchanged. `runtime_traces` is the path visible to the benchmark
container; omit it for local runs using the same path. No conversation text is
copied into the manifest. Existing manifests cannot be overwritten.

Edit `replay.json` to set the checkpoint and manifest paths. To test concurrency
128, select at least 128 distinct conversations. The supplied example compares
8, 16 and 32 at a fixed 128K token budget and max sequences 128, twice each.

```bash
primebeaker vllm-sweep plan --config=replay.json
primebeaker vllm-sweep run --config=replay.json --output=/shared/results
primebeaker vllm-sweep report --output=/shared/results
```

Each conversation advances sequentially, while separate conversations run
concurrently. The client converts stored tool calls to the OpenAI wire format,
uses the saved output-token count with EOS ignored, and streams responses.
Subsequent turns use recorded history, not the newly generated answer. This
isolates model-serving throughput; it does not execute tools or judges or measure
end-to-end eval latency. Completed traces can underrepresent interrupted eval
conversations. The source checksum is recorded for provenance, not rechecked at
every request.

## Parameters and sampling

- `server`: fixed vLLM engine arguments, with underscore names.
- `grid`: candidate values overriding fixed arguments. `null` omits a flag.
- `concurrency`: actual client requests/conversations in flight, separate from
  `max_num_seqs` and `max_num_batched_tokens`.
- `sample_settings`: optionally select this many engine/client combinations
  uniformly without replacement, keeping all workloads and repeats.
- `repeats`, `seed`: reproducible trial ordering and request ordering. Compare
  configurations with the same workload and seeds.
- `measurement_seconds`: bounds the measured load window after warm-up (180 in
  `replay.json`). Partial work remains usable when this window expires.
- `startup_timeout`, `benchmark_timeout`: startup and fallback benchmark limits.
- `latency_limits_ms`: optional p99 constraints for the strict native ranking.

`example.json` is an optional synthetic exploration grid: 16 sampled settings,
two workloads and two repeats (64 trials). It uses vLLM `bench serve` with random
prompts. It is not evidence of the best settings for a real agent workload and
will generally exceed one interactive allocation's wall time.

## Saved data and fitting

Each trial saves settings, exact commands, server/client logs, raw benchmark
results, request-level replay events, and Prometheus snapshots. Every two seconds
it records token counters and GPU utilization, memory use and power. GPU CSV
fields are index, GPU utilization %, memory-controller utilization %, used MiB,
total MiB and watts. Session environment snapshots record runtime versions.

- `compute-samples.json`, `compute.json`: raw samples and observed token rates,
  including in-flight work. Rates include load-generator startup/shutdown;
  inspect interior intervals for steady-state analysis.
- `compute-ranking.json`: per-workload median output throughput, with durations,
  statuses and repeat counts. Positive partial measurements are included; inspect
  short or failed runs before drawing conclusions.
- `analysis.csv`, `observations.json`: individual repeats, failures and raw metrics
  for later analysis; these do not force logarithms or aggregation.
- `ranking.json`: a separate strict native benchmark ranking requiring all
  requested responses/repeats and configured latency limits.

JSON checkpoints and the analysis table use atomic replacement. `report` rebuilds
exports and can recover interrupted trials from their saved counter samples.
Native detailed benchmark JSON is available after that benchmark finishes;
interrupted clients may have only incremental samples/logs.

On an analysis host with NumPy installed:

```bash
primebeaker vllm-sweep fit --output=/shared/results
```

This fits a regularized quadratic surface to log throughput, including
interactions and leave-one-setting-out cross-validation. It needs at least eight
measured settings per workload and supports numeric dimensions. `null` offload
maps to zero. `fit.json` contains predicted unmeasured candidates; predictions
require measurement and do not predict OOM feasibility. Cross-validation error
is not a confidence interval. Use `--source=completed` for the strict ranking.
Raw exports also support later log-linear/log-log or other models.

## Rexs submission and resuming

```bash
primebeaker vllm-sweep rexs \
  --config=/path/replay.json --profile=/path/rexs-profile.yaml \
  --output=/shared/sweep-bundle --image=eval-image \
  --mounts='{"/shared":"/data","/shared/models":"/models"}' \
  --gpus=1 --cpus=16 --memory=192GiB
```

This validates and renders only. Add `--submit=True` to submit through Rexs.
The profile maps the image alias to an existing SIF containing vLLM and Fire.
Use a single-node profile without an unrelated `completion_task`. Reserve host
RAM for the runtime and any CPU cache. GPU count must cover TP × PP × DP.

Submission checks all of the user's interactive partitions, counting queued as
well as running jobs; it refuses if two slots are occupied. A host lock serializes
this command's submissions. Other submitters do not share that lock, so Slurm is
the final authority. No dependent follow-up jobs are queued.

Trials use a fresh loopback-only server. Development endpoints are enabled for
cache reset. Warm-up is separate and excluded from measured throughput. Process
groups are cleaned up on timeout, Ctrl-C and SIGTERM. Invalid replay warm-up
requests stop the sweep, preserving the server response for diagnosis.

After the previous allocation ends, repeat the same `rexs` command and config
with `--resume=True --submit=True`. Finished/failed trials are skipped; interrupted
attempts are archived and retried. `max_trials` bounds remaining trials per
invocation. Direct `run --resume=True` uses the same results directory. Do not
resume while another allocation is using that bundle. Worker/config snapshots
are immutable for a bundle; new code needs a fresh bundle.

## Offloading limitation observed during validation

The tested vLLM 0.22/Qwen3.5 runtime failed with
`External KV connector is not verified yet` when the real replay reused
CPU-offloaded KV. Synthetic workloads did not expose this. The real-trace
example therefore keeps offloading disabled. No assertion is bypassed. Offloading
also does not remove GPU activation-memory costs. Compare a working offload
configuration against a measured baseline before recommending it.

[vLLM benchmark CLI](https://docs.vllm.ai/en/v0.22.0/cli/bench/serve/)
· [Engine arguments](https://docs.vllm.ai/en/v0.22.0/configuration/engine_args/)
