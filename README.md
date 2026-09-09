# PrimeBeaker

PrimeBeaker is a standalone package for running Prime-RL SFT and RL on Beaker.
It includes:

- lossless SFT/RL TOML loading and editing;
- single-node and multi-node Beaker preview/submission;
- heterogeneous RL placement (mixed trainer/inference node plus dedicated inference nodes);
- distributed multi-node SFT through torchrun/FSDP;
- all 15 runnable Verifiers environments, using LiteRegistry tool clients;
- package-local prompt templates, tool-call wire parsing, and tool schemas;
- an immutable Prime-RL image catalog and reproducible Dockerfiles.

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
