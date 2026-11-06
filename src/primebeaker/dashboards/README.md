# PrimeBeaker dashboards

Reusable live monitoring, with **rubric judging** as the first view. This package
is separate from `primebeaker.watcher`, which indexes checkpoint/evaluation
provenance into SQLite.

## Layout

```text
dashboards/
  config.py             portable manifests and receipt resolution
  collector.py          shared polling, worker locks and source isolation
  beaker.py             read-only experiment/task status
  prometheus.py         vLLM counters, engine gauges and registry discovery
  training.py           completed optimizer steps and rollout batches
  wandb.py              sparse W&B history, fetched metric by metric
  store.py              atomic JSON snapshots and daily JSONL history
  server.py             shared HTTP shell and versioned snapshot API
  assets/               shared CSS, formatting and SVG chart primitives
  rubric_judging/       view-specific metric defaults, HTML and JS
```

The manifest contains all run paths, model identity, head-registry URI,
experiment IDs, expected replica count and display labels. There are no private
cluster addresses, user names or experiment IDs in the implementation. For a
new view, add its assets/default metric definitions and register it in
`views.get_view`; reuse the existing sources, snapshot store, API and
`assets/core.js` chart renderer. Additional data sources belong in shared
providers; view JS consumes the snapshot API.

## Start and deploy

```bash
primebeaker dashboards serve --config=/path/to/dashboard.json --port=8791
```

Serving defaults to loopback. Use the existing port-forwarding setup for a
remote workspace. `--collect=False` serves a shared state directory produced
by independent `collect --source=metrics` and `collect --source=wandb` workers.
`--source=metrics` needs no W&B credentials. W&B and LiteRegistry imports are
lazy, so serving existing snapshots needs neither SDK.

The supported sources are read-only: Beaker CLI queries, native LiteRegistry
discovery, HTTP `/metrics`, PrimeRL log/file-monitor reads, and the W&B API.
They never make generation requests, cancel jobs, or modify training settings.
Use `WANDB_API_KEY` or the configured SDK login; Beaker uses the configured CLI.
`beaker_command` is an argument-list prefix for an existing authentication
wrapper if the worker uses one. It is invoked without a shell. Credentials are
not part of snapshots, histories, `/api/config`, or application logs.

## Snapshot contract (schema_version 1)

- `latest.json`: timestamp, dashboard metadata, minimized `jobs`, per-run
  `trainer` progress, per-role `inference` endpoints, source observation times,
  `source_errors`, and requested/discovered/serving judge counts.
- `curves.json`: W&B source, per-run availability, exact metric names,
  required row columns, source fetch timestamps, and raw `{step, value}` points.
- `history-YYYY-MM-DD.jsonl`: append-only metrics snapshots in UTC.

HTTP endpoints are `/api/config` (display metadata only), `/api/snapshot`, and
`/api/curves`. `/latest.json` and `/wandb.json` are aliases for existing clients.
Only an explicit asset/snapshot allowlist is served, with no-cache headers.
Assets are included in wheels and source distributions.

## Metric meanings

- **Active / queued:** vLLM engine execution/waiting gauges. These exclude any
  queue inside an upstream grouped API. A healthy idle gauge is zero; an
  unavailable endpoint is null with an error.
- **Throughput:** prompt/output/success counter differences between successive
  scrapes. First scrapes and resets have no rate. A multi-worker policy frontend
  is scraped once; engine labels are aggregated rather than multiplying its
  metrics by the number of API workers.
- **Latency:** interval mean for completed requests, queue observations or
  first-token observations. It is not the age of an active request. No observed
  completions means unavailable, not zero.
- **KV cache:** mean and maximum occupied percentage across engine gauges.
- **Trainer steps:** only completed `Step N | ... Loss ...` log records count.
  Micro-steps, start messages, batch collection and checkpoint presence do not
  prove completion. Partial batch/inflight/buffered progress is shown separately.
- **Timing:** the file monitor is read incrementally so sparse trainer timing
  rows survive frequent inference telemetry. `time/forward_backward` identifies
  trainer rows even in runtimes without a producer tag.
- **W&B time/step:** the rubric view requires `time/forward_backward` on the same
  history row, separating it from the orchestrator's colliding `time/step` key.
  The view displays minutes; exported raw values remain seconds.
- **W&B reward/tokens:** `train/agg/all/agent/reward/mean` and
  `train/agg/all/num_total_tokens/mean`. The token mean is the episode mean,
  not cumulative training tokens. A manifest's `curves` map can replace these
  defaults (`metric`, `unit`, optional `requires` columns).

Each curve is queried independently to preserve sparse shared-run rows. The
x axis is the actual logged `step`; there is no manual shifting, smoothing,
or interpolation. Missing series are not filled with zeros. Failed refreshes
retain the last good series and visibly mark it cached/unavailable.

Job identity is resolved by receipt job ID and/or task name, so a trainer and
judge in the same Beaker experiment are not confused. Submitted replicas appear
before registry discovery; a running container is labeled initializing until
its metric endpoint is healthy. API errors retain last-good status with its
observation timestamp, and stale snapshot ages remain visible.

The rubric view plots active requests, interval mean request latency, and engine-mean KV cache per server over the last hour. `/api/inference-history` returns a compact history projection (up to 300 snapshots, bounded to the final 8 MiB of each of two UTC daily logs). The HTTP server caches it for 10 seconds. Missing/unreachable/stale samples leave gaps. The shared plot component accepts `xMode: "time"` and `zeroFloor: true`; training plots keep their step axes. Detailed inference rows appear at the bottom of the view.

### Historical training baselines

Add `baselines` entries containing `id`, `label`, `wandb_run` (`entity/project/run`), and `curves_file` (a saved W&B curve export). Optional `color` applies consistently across plots. Exports use the same `{curves: {metric_name: {points: [{step, value}]}}}` format as live runs and must contain their matching `wandb_run` or original W&B `url`. Baselines appear only in training plots, never as live jobs or inference servers. Their status is labeled saved baseline.

`curve_max_step` restricts displayed training points without altering history; `curve_caveat` adds a visible comparison note. Run-level `color` controls live training series too. Saved baseline files are read by the HTTP server, so they do not require extra W&B polling.

Set `curve_smoothing_steps` to an integer greater than one (for example, `10`) to draw a trailing step-window mean over faint raw W&B values. Early windows use the available points, and missing values reset the window. Raw histories and inference time-series plots are unchanged. The default is `1` (no smoothing).
