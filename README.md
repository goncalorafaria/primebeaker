# PrimeBeaker

PrimeBeaker is a standalone package for running Prime-RL SFT and RL on Beaker.
It includes:

- lossless SFT/RL TOML loading and editing;
- single-node and multi-node Beaker preview/submission;
- heterogeneous RL placement (mixed trainer/inference node plus dedicated inference nodes);
- distributed multi-node SFT through torchrun/FSDP;
- all 16 runnable Verifiers environments, using LiteRegistry tool clients;
- package-local prompt templates, tool-call wire parsing, and tool schemas;
- an immutable Prime-RL image catalog and reproducible Dockerfiles.
- safe whole-topology checkpoint discovery and resume;
- a thin CLI adapter over LiteRegistry's native Beaker service deployment.

It does not import `datadev` or `jtc_data_commons`. Prime-RL owns model
rendering and the GPU training runtime. Dataset production and evaluation are
also outside this package. Verifier datasets are JSON/JSONL, saved Hugging Face
datasets, or Hub datasets; PrimeBeaker does not include a Parquet adapter.
Convert legacy `.jtasks.parquet` inputs to JSONL or a saved Hugging Face dataset before launch.

## Install

Client implementations are provided by `literegistry-tool-client`, a standalone
LiteRegistry companion. Existing `primebeaker.client` imports remain compatible;
new callers can use `from literegistry_tool_client import SearchClient`.
Normal installation resolves `literegistry-tool-client` from PyPI.

For TOML and launch tooling:

```bash
pip install -e .
```

For environments, clients, the LiteRegistry gateway, and multi-node RL:

```bash
pip install -e '.[runtime]'
```

The runtime extra installs LiteRegistry's Python launchers, but service stacks
also require their separate container images in Beaker. Complete the runtime
installation by building the official Dockerfiles from the exact installed
LiteRegistry companion versions and importing the images into your workspace:

```bash
beaker account whoami
docker version
primebeaker services images install --workspace=ai2/oe-agents
```

The command downloads the matching source distributions, builds both the base
services and Podman/mirror stacks, uploads every image, and prints the immutable
Beaker IDs as `launcher_args`. Use those values for `services launch` or
`services podman launch`. To install only the images required by the Podman
terminal verifier, pass `--stack=podman`. A local LiteRegistry checkout can be
used without downloading sources via `--source-root=/path/to/literegistry`.
If the base deployment will run local search, also pass
`--build-local-search --jtc-build-context=/path/to/jtc`; that image needs JTC's
Lucene build assets and is therefore not part of the default build.

This is an explicit post-install step because Python package installation must
not silently mutate Docker or a remote Beaker workspace. It requires a running
Docker daemon, an authenticated `beaker` CLI, network access to the configured
Python and container indexes, and write access to the selected workspace.

PrimeBeaker targets Python 3.12, matching the cataloged Prime-RL runtime. The
`runtime` extra pins `verifiers[harbor]==0.2.1`, whose released PyPI wheel
contains the client and environment APIs used by PrimeBeaker. No Git checkout is
required for the runtime extra.

The authenticated `beaker` executable, Docker, and a Prime-RL GPU image are
external requirements. See [`src/primebeaker/images/README.md`](src/primebeaker/images/README.md) for immutable
training-image locations and exact rebuild/publish instructions.

## Tiny runnable examples

[`examples/`](examples/README.md) contains two-step SFT and RL setups. Each
uses four complete training records and two complete validation records, preserving
the existing data format without bundling a corpus. Both are ready for the
Fire-based `preview` and `submit` commands documented there.

## Slurm through Rex

For the Delta Slurm setup, see [PrimeBeaker through Rex interception](deploy/delta/README.md).
It uses an opt-in `beaker` command shim; PrimeBeaker's normal Beaker backend remains unchanged.

## Environments

```bash
python -m primebeaker.environments
python -m primebeaker.environments tool-label
```

TOMLs use normal module paths such as:

```text
primebeaker.environments.jtc_tool_label_env:load_environment
primebeaker.environments.jtc_reward_model_env:load_environment
primebeaker.environments.jtc_search_agent_env:load_environment
primebeaker.environments.podman_terminal_env:load_environment
```

The Podman terminal solver expects each dataset row to contain a prompt and an
initial image under `original_image`, `image`, or `container_image`. It starts a
fresh container directly from that image; saved trace fields are neither required
nor read. The model solves the task in that live container using exactly two
model-facing tools, `bash` and `submit`, and must finish with:

```text
echo TERMINAL_COMPLETE
```

Only after that marker succeeds, the environment uses the same Podman client to
clear any stale or model-written reward file, execute the configured unit-test
command against the model's final filesystem, capture its structured
stdout/stderr/exit status, and read `/logs/verifier/reward.txt`. Its numeric
value is clamped to `[0, 1]`; a missing or invalid file contributes exactly zero.
Four independent rubric weights are configured directly in the Prime-RL TOML:

```toml
[orchestrator.train.source.legacy.args]
reward_file_path = "/logs/verifier/reward.txt"
test_command = "bash /tests/test.sh"
test_timeout = 600
reward_file_weight = 1.0
termination_reward_weight = 0.1
podman_failure_penalty_weight = 1.0
fake_tool_penalty_weight = 0.1
```

The raw rubric values are respectively the captured unit-test score, `1` for a
successful standalone completion command, `-1` for a Podman setup or execution
failure, and `-1` per attempted tool name that the environment does not provide. A rollout that merely fails to terminate gets no
termination credit rather than a separate missing-marker penalty. Failure phase, type,
message, and an OOM indicator are retained in rollout state for diagnosis.

The launcher rewrites copied legacy environment paths to the `primebeaker`
namespace in a temporary runtime TOML; it never edits the source TOML.

## Single-node launch

```bash
primebeaker sft preview --toml /weka/path/sft.toml

primebeaker rl preview --toml /weka/path/rl.toml \
  --env TERMINAL_SERVER_URL=http://service/terminal

primebeaker rl submit --toml /weka/path/rl.toml \
  --image beaker://IMMUTABLE_PRIMEBEAKER_IMAGE \
  --workspace ai2/oe-agents \
  --cluster ai2/holmes
```

## Multi-node RL

The same command automatically selects the multi-node backend when the TOML
contains `deployment.type = "multi_node"`:

```bash
primebeaker rl preview \
  --toml /weka/path/multinode-rl.toml \
  --image beaker://IMMUTABLE_PRIMEBEAKER_IMAGE \
  --registry redis://registry-host:6379 \
  --required-service terminal=8 \
  --required-service localsearch:corpus=8
```

The generated Beaker task uses replicas, leader selection, synchronized start,
host networking, and failure/preemption propagation. Replica 0 runs the
trainer, orchestrator, global vLLM router, and local tool gateway. It may also
run an inference GPU slice. Later replicas run the dedicated inference slices.
All replicas share rendezvous state through the mounted Weka dataset and use
one W&B run ID.

The retained heterogeneous layout matches the extracted production launcher:
exactly one trainer node and one Prime-RL inference replica, with any number of
configured inference nodes. The TOML's `inference.parallel.tp`, `.dp`, and
`api_server_count` must describe every exposed inference rank.

Tool workers and Redis remain external services. `--required-service NAME=N`
adds an exact startup barrier without bundling service deployment into the
training package.

## LiteRegistry services

PrimeBeaker does not maintain a second service-stack implementation. The
`services` command delegates directly to LiteRegistry's
`literegistry-base-deployment` package, which owns gateway, Redis, Python,
terminal, web search/fetch, cache, local-search, and vLLM deployment:

```bash
primebeaker services preview \
  --head-registry=/weka/gfaria/registries/example \
  --service-cluster=ai2/jupiter \
  --python-replicas=2 \
  --terminal-replicas=2 \
  --web-search-replicas=2

primebeaker services launch \
  --head-registry=/weka/gfaria/registries/example \
  --service-cluster=ai2/jupiter

REGISTRY=redis://YOUR_REDIS_HOST:6379 primebeaker services podman preview \
  --registry=redis://YOUR_REDIS_HOST:6379 \
  --service-cluster=ai2/jupiter \
  --podman-replicas=4 \
  --docker-mirror-replicas=2

REGISTRY=redis://YOUR_REDIS_HOST:6379 primebeaker services podman yaml preview \
  --config=examples/podman-terminal/services.yaml
```

All service flags are passed into LiteRegistry's native
`BaseDeploymentConfig`; PrimeBeaker adds no stack rendering or coordination
logic. The `podman` subgroup similarly delegates to the native
`literegistry-podman-beaker` package. Its YAML form is `services podman yaml`:
it uses the same strict `primebeaker.services/v1` schema as `services yaml` but
passes the `services` object to the Podman launcher. The same operations remain
available through those two upstream executables.

Both native launchers use `literegistry.coop.ports` for collision-safe
dynamic ports and child supervision, plus `literegistry.coop.endpoints` for
healthy endpoint publication and shutdown cleanup. The base launcher can publish through a `head+file://`, `head+sqlite://`, or
`head+redis://` registry URI. The native Podman launcher instead requires a
direct, persistent `redis://` or `rediss://` endpoint shared with training.
Pass the applicable URI to `primebeaker rl preview|submit|resume --registry=...`.

## Resume a multi-node RL experiment

Resume always creates a fresh complete topology. It first requires one step
to have every trainer rank shard, trainer `.metadata`, and the orchestrator's
`progress.pt`. It refuses to submit while any source job is active, preserves
the W&B run ID, moves stale broadcast handshakes aside, and writes an attempt
manifest under `<output_dir>/primebeaker_resume/`.

With no `--resume-step`, the latest complete common checkpoint is selected:

```bash
primebeaker rl resume \
  --from-experiment=01SOURCE \
  --dry-run

primebeaker rl resume \
  --from-experiment=01SOURCE \
  --resume-step=220 \
  --registry=head+file:///weka/gfaria/registries/example
```

The source Beaker experiment supplies the image, workspace, clusters, Weka
mount, output directory, gateway settings, and training TOML. Any corresponding
resume flag is an explicit override. A source without an external registry
must be given `--registry`; `primebeaker services launch` can create that stack.

## Multi-node SFT

PrimeBeaker also translates Prime-RL's SLURM-oriented multi-node SFT topology
to Beaker:

```toml
[deployment]
type = "multi_node"
num_nodes = 4
gpus_per_node = 8
```

```bash
primebeaker sft preview \
  --toml /weka/path/multinode-sft.toml \
  --image beaker://IMMUTABLE_PRIMEBEAKER_IMAGE
```

Each Beaker replica starts one torchrun node. Replica 0 publishes the rendezvous
address and materializes the validated Prime-RL trainer config; all nodes join
the same distributed job.

## Output and clients

Preview is non-mutating. `--write-spec` writes under
`<scratch-dir>/primebeaker/beaker_experiments/`; submit writes the same spec and
invokes `beaker experiment create`.

`primebeaker.client` is the canonical API and exposes terminal, Python, Podman, search, fetch, judge,
reward-model, web-terminal, and submit clients. Service URLs and credentials
remain explicit rather than being tied to JTC infrastructure.
