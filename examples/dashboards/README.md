# Live dashboards

Copy `rubric-judging.json`, replace its receipt/registry/model paths, and run:

```bash
python -m pip install 'primebeaker[dashboards]'
primebeaker dashboards serve --config=./rubric-judging.json --port=8791
```

Open `http://localhost:8791/`. Paths in the manifest are relative to the manifest
file, so a dashboard can move to another checkout without changing its code.

Each run can use a launch receipt or explicit fields:

```json
{
  "id": "policy",
  "label": "My policy",
  "experiment_id": "BEAKER_EXPERIMENT_ID",
  "task_name": "trainer",
  "run_dir": "/path/to/prime-rl-run",
  "wandb_run": "entity/project/run-id",
  "inference_port": 8000
}
```

Receipts supply `training_experiment_id`, `training_job_id`, `run_directory`,
and `wandb_url`. A receipt with only `wandb_run_id` can be paired with
`wandb_entity` and `wandb_project` in the run entry. Explicit manifest fields
take precedence. Receipt files are re-read during polling, so replacing a
training experiment does not require changing dashboard code.

Static inference endpoints can be configured as `endpoints` on any run or on
`judge`. Registry discovery adds endpoints to that list. It uses the native
LiteRegistry client; run collection in the serving deployment's compatible
LiteRegistry environment. Static-endpoint collection needs no registry SDK.

The Beaker CLI and W&B SDK use their configured credentials. In Beaker, inject
`WANDB_API_KEY` with the existing secret binding. An optional
`wandb_beaker_secret` selects a Beaker-secret fallback; it has no user-specific
default. Never put API keys in this JSON. `policy_headers_from_env` reads an
authorization header from the environment without persisting it.

For a CPU worker and a separate browser process sharing storage:

```bash
primebeaker dashboards collect --config=./rubric-judging.json --source=metrics
primebeaker dashboards collect --config=./rubric-judging.json --source=wandb
primebeaker dashboards serve --config=./rubric-judging.json --collect=False
```

Each command is a long-running read-only process. `--once=True` on `collect`
captures one production snapshot. Separate locks prevent two workers from
writing the same source; metrics and W&B may run independently. Inference
servers, training jobs and model weights are never changed by collection.

The package also supports `primebeaker-dashboard` and
`python -m primebeaker.dashboards` with the same commands. See the
[dashboard architecture and metric definitions](../../src/primebeaker/dashboards/README.md).
