# Tiny training examples

For a live Podman terminal-agent setup, see
[podman-terminal](podman-terminal). It includes single-node and multi-node RL
TOMLs, tiny task JSONL, and a launchable LiteRegistry Podman service stack.

For a full local-search WebTerminal topology, see
[search-agent-webterminal](search-agent-webterminal). It includes a four-node
RL TOML and a directly consumable LiteRegistry service-stack YAML.

These fixtures are literal, deliberately tiny slices of the existing Prime-RL
training data. They preserve the JSONL shapes without duplicating a corpus:

- `sft/data/train.jsonl`: four complete SFT conversations;
- `sft/data/validation.jsonl`: two complete SFT conversations;
- `rl/data/train.jsonl`: four complete label-RL records;
- `rl/data/validation.jsonl`: two complete label-RL records.

The rows are complete, unmodified records sampled at evenly spaced indexes from:

- SFT: `/weka/gfaria/prime_sft/data/sft_datadev_1c4e4ba436ab860f/{train,validation}.jsonl`;
- RL: `/weka/gfaria/prime_sft/data/rldata_datadev_4203b91b21b25100_{train,validation}.jsonl`.

The source files are not modified and are not required after these slices are copied.

Run commands from the PrimeBeaker repository root so the relative paths in the
TOMLs resolve inside the mounted Weka dataset.

## Install the required images

Installing the Python package does not install images into Beaker. Every
example needs a Prime-RL GPU runtime image, and service-backed examples also
need their LiteRegistry service images installed in the target workspace.

For the Podman example, install the Podman stack images:

```bash
beaker account whoami
docker version
primebeaker services images install \
  --workspace=ai2/oe-agents \
  --stack=podman
```

For the WebTerminal search example, install the base stack and local-search
image:

```bash
primebeaker services images install \
  --workspace=ai2/oe-agents \
  --stack=base \
  --build-local-search \
  --jtc-build-context=/path/to/jtc
```

Each command builds the official images matching the installed LiteRegistry
launcher version, uploads them to Beaker, and prints immutable IDs under
`launcher_args`. Put those IDs in the example's `services.yaml` before launch.
This installation is required once for each workspace and service-image
version. The topology-specific READMEs list the exact YAML fields.

## Select the training image

Use the immutable default image cataloged by PrimeBeaker:

```bash
primebeaker sft preview --toml examples/sft/config.toml
```

The CLI chooses that image when `--image` is omitted. The selected Beaker
workspace must be able to read it. To inspect, pull, rebuild, or publish the
runtime image into another workspace, follow
[`src/primebeaker/images/README.md`](../src/primebeaker/images/README.md).
You may instead pass an explicit immutable Beaker image:

```bash
--image=beaker://IMAGE_ID
```

The image must contain Prime-RL's `sft` and `rl` executables and PrimeBeaker
installed with its runtime dependencies.

## Preview and launch SFT

```bash
cd /weka/path/to/primebeaker

primebeaker sft preview \
  --toml=examples/sft/config.toml \
  --working-dir="$(pwd)" \
  --scratch-dir="$(pwd)" \
  --home-dir="$(pwd)"

primebeaker sft submit \
  --toml=examples/sft/config.toml \
  --working-dir="$(pwd)" \
  --scratch-dir="$(pwd)" \
  --home-dir="$(pwd)"
```

## Preview and launch RL

The RL fixture uses `primebeaker.environments.jtc_label_env`, a service-free
single-turn verifier. No terminal, search, judge, or JTC service is required.

```bash
cd /weka/path/to/primebeaker

primebeaker rl preview \
  --toml=examples/rl/config.toml \
  --working-dir="$(pwd)" \
  --scratch-dir="$(pwd)" \
  --home-dir="$(pwd)"

primebeaker rl submit \
  --toml=examples/rl/config.toml \
  --working-dir="$(pwd)" \
  --scratch-dir="$(pwd)" \
  --home-dir="$(pwd)"
```

`submit` uses the default `ai2/oe-agents` workspace, `ai2/holmes` cluster,
and `oe-adapt-default` Weka dataset. Override those Fire flags when your Beaker
setup differs. The examples run only two optimizer steps; increase `max_steps`
and replace the fixture paths for real training.

## Data shapes

SFT records contain `messages` and a JSON-encoded `tools` field, including
assistant tool calls and tool-result messages exactly as stored in the source.

RL records retain the source fields `prompt`, `answer`, `output`, `feedback`,
`record_id`, `rubric_index`, `source`, and `tools`. The label environment consumes
`prompt` and `answer`; the remaining fields preserve the richer interchange format.
