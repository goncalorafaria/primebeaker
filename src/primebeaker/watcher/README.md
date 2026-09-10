# Watcher

`primebeaker watcher` creates a queryable SQLite provenance graph rooted at saved
RL checkpoints and evaluated standalone SFT checkpoints. It records the RL and
SFT TOMLs, their W&B run IDs, the SFT predecessor checkpoint, every configured
data set and its source mixture, and all matching evaluation YAMLs and outcomes.

The watcher lives in PrimeBeaker and depends on JTC only for parsing evaluation
outcomes. JTC does not import, install, schedule, or run the watcher.

Install its optional runtime:

```bash
python -m pip install 'primebeaker[watcher]'
```

The watcher has a Python Fire CLI. The normal first-time or clean regeneration
command is:

```bash
primebeaker watcher rebuild \
  --database=watcher.sqlite3 \
  --repo_root=/weka/gfaria/jtc \
  --wandb_entity=graf
```

`rebuild` indexes into a temporary database in the destination directory,
runs SQLite integrity and foreign-key checks, snapshots the old database, and
only then atomically swaps in the new file. It refuses to replace a healthy
database when the scan has indexing failures, invalid TOMLs, or zero indexed
checkpoints. Output directories targeted by genuinely different resume TOMLs
are reported as `unresolved`; the watcher never guesses which configuration
created a checkpoint.

Use `update` for the quick day-to-day path. It skips checkpoints already in
SQLite by default:

```bash
primebeaker watcher update --wandb_entity=graf
primebeaker watcher status
```

The remaining commands index one checkpoint, inspect its stored graph, and
start the browser:

```bash
primebeaker watcher index \
  /weka/gfaria/prime_sft/outputs/my-rl-run/weights/step_400 \
  --wandb_entity=graf

primebeaker watcher show \
  /weka/gfaria/prime_sft/outputs/my-rl-run/weights/step_400

primebeaker watcher serve --port=8790
```

Run `primebeaker watcher COMMAND --help` for every option. Discovery-root
flags accept one path, a Fire list, or comma-separated paths. Use
`--details=True` on `rebuild` or `update` when the full indexed/skipped path
lists are useful. `--backup=False`, `--require_clean=False`, and
`--require_models=False` are explicit escape hatches for unusual recovery
workflows.

Discovery uses `output_dir` to find training TOMLs, the model field to find
evaluation YAMLs, task environment variables to find launch/W&B metadata, and
materialization manifest paths or UIDs to find source mixes. For old launches
that predate reserved W&B IDs, the watcher recovers IDs from local
`wandb/run-<timestamp>-<id>` directories. If neither source exists it retains
the project/name with a null `run_id`; use `--rl_wandb_run_id` or
`--sft_wandb_run_id` to fill that gap.

Historical MOSS outcomes under `/weka/gfaria/records/moss-evals` (plus the legacy
`/weka/gfaria/records/evalmoss` and `/weka/gfaria/records/eval-moss` spellings) are also imported when
their `launcher.log` names an exact indexed checkpoint. Archive rows are
deduplicated when a current evaluation YAML already points at the same result
directory.

`wandb_runs` is intentionally one-to-many per RL or SFT training stage. Failed
attempts and restarts with different W&B IDs are all retained. Duplicate local
directories using the same ID (for example the top-level and `run_default`
directories written by one distributed Prime-RL run) collapse to one logical
W&B run.

Default manifest discovery includes both the saved Weka data area
`/weka/gfaria/prime_sft/data` (derived from the repository parent) and
repository-local data directories. Source parquet paths in SQLite are absolute;
the original manifest entry is retained in `raw_source_json`.

Configured datasets without a surviving manifest are still recorded, but have
no source rows. `--manifests` can restrict discovery to one or more explicitly
chosen materialization manifests when legacy filenames are ambiguous.

The convenience views are `model_provenance`, `model_wandb_runs`,
`dataset_mix`, and `evaluation_results`.

## Local checkpoint browser

Start the watcher page against the populated database:

```bash
primebeaker watcher serve --database=watcher.sqlite3 --port=8790
```

Open `http://127.0.0.1:8790`. The page provides checkpoint search and shows
the complete training lineage, every W&B attempt, source parquet mixtures, and
evaluation outcomes.

The sidebar has one entry per output rather than one per checkpoint. It includes
RL outputs and every SFT output with an evaluation configuration. Runs with
evaluations sort to the top and receive a purple highlight; all saved steps
remain selectable inside the run detail view. An SFT evaluation is available
from its standalone SFT entry and is also shown, labeled `sft`, in the evaluation
table of every indexed RL run whose training lineage names that exact SFT
checkpoint.

## Live W&B charts

Selecting a run starts a separate request for live W&B history; metric values
are never precomputed or stored in SQLite. The browser plots SFT `val/loss` and
RL eval accuracy over the real training `step`. Every distinct W&B attempt in
the selected checkpoint lineage is queried. Overlapping attempts are averaged
at each step and their minimum-to-maximum range is drawn as a subtle band.

Authentication uses `WANDB_API_KEY` when it is already in the server
environment, otherwise Watcher reads the `gfaria_WANDB_API_KEY` Beaker secret
in-process without printing or persisting it. Use `--wandb-secret` on either
serve entry point to select another secret.

The accuracy loader accepts both the requested
`val/jtc-tool-label-terminal-eval/all/metrics/correct_final_label/mean` key and
the `eval/jtc-tool-label-terminal-eval/all/metrics/correct_final_label/mean`
spelling currently present in the indexed Prime-RL runs.
