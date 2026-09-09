# PrimeBeaker

PrimeBeaker is a standalone package for running Prime-RL SFT and RL on Beaker.
It includes:

- lossless SFT/RL TOML loading and editing;
- single-node and multi-node Beaker preview/submission;
- heterogeneous RL placement (mixed trainer/inference node plus dedicated inference nodes);
- distributed multi-node SFT through torchrun/FSDP;
- all 15 runnable Verifiers environments and their HTTP tool clients;
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

For TOML and launch tooling:

```bash
pip install -e .
```

For environments, clients, the LiteRegistry gateway, and multi-node RL:

```bash
pip install -e '.[runtime]'
```

PrimeBeaker targets Python 3.12, matching the cataloged Prime-RL runtime. The
`runtime` extra pins the exact PrimeIntellect `verifiers` revision recorded by that
Prime-RL checkout, so a clean install cannot resolve to the incompatible v1-only
PyPI API. Git is therefore required when installing the runtime extra from source.

The authenticated `beaker` executable and a Prime-RL GPU image are external
requirements. See [`src/primebeaker/images/README.md`](src/primebeaker/images/README.md) for immutable image
locations and exact rebuild/publish instructions.

## Tiny runnable examples

[`examples/`](examples/README.md) contains two-step SFT and RL setups. Each
uses four complete training records and two complete validation records, preserving
the existing data format without bundling a corpus. Both are ready for the
Fire-based `preview` and `submit` commands documented there.

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
```

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

primebeaker services podman preview \
  --head-registry=/weka/gfaria/registries/example \
  --service-cluster=ai2/jupiter \
  --podman-replicas=4 \
  --docker-mirror-replicas=2
```

All service flags are passed into LiteRegistry's native
`BaseDeploymentConfig`; PrimeBeaker adds no stack rendering or coordination
logic. The `podman` subgroup similarly delegates to the native
`literegistry-podman-beaker` package. The same operations remain available
through those two upstream executables.

Both native launchers use `literegistry.coop.ports` for collision-safe
dynamic ports and child supervision, plus `literegistry.coop.endpoints` for
healthy endpoint publication and shutdown cleanup. Pass the resulting
`head+file://`, `head+sqlite://`, or `head+redis://` registry URI to
`primebeaker rl preview|submit|resume --registry=...`; training resolves the
current Redis endpoint through LiteRegistry rather than assuming a fixed port.

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
