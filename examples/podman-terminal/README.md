# Live Podman terminal-agent RL

This directory is a complete example for training an agent to solve a task in a
fresh Podman container. It does not replay a saved trace. Every rollout starts
from the dataset row's image, exposes the environment's `bash` tool, and runs
the configured unit-test command only after the agent executes exactly:

```bash
echo TERMINAL_COMPLETE
```

The bundle contains:

- `rl.toml`: single-node, two-GPU smoke/training configuration;
- `multinode-rl.toml`: one 8-GPU trainer node and one 8-GPU inference node;
- `services.yaml`: 16 Podman workers, two Docker-mirror workers, and the native
  LiteRegistry gateway;
- `data/`: tiny live-task JSONL fixtures using the public Python 3.12 image.

The tiny task asks the model to write one file under `/tmp`. Its test command is
inline so the example needs no custom task image. For real tasks, put the private
tests in the task image, set `test_command = "bash /tests/test.sh"`, and have the
test write a score in `[0, 1]` to `reward_file_path`.

## Install the runtime and service images

Run from the repository root:

```bash
pip install -e '.[runtime]'
beaker account whoami
docker version
primebeaker services images install \
  --workspace=ai2/oe-agents \
  --stack=podman
```

This image installation is required before the first stack launch in each
workspace, and again when the installed LiteRegistry Podman launcher version
changes. Installing the Python runtime alone does not create remote Beaker
images.

The installer downloads the exact source distribution matching the installed
`literegistry-podman-beaker`, builds its official Redis, gateway, Podman, and
Docker-mirror images, uploads them to the workspace, and prints immutable
Beaker IDs under `launcher_args`. Add those four IDs to `services.yaml` as
`redis_image`, `gateway_image`, `podman_image`, and `docker_mirror_image` so the
stack uses the images installed in your workspace.

## Choose the shared registry

The native Podman stack accepts a direct `redis://` or `rediss://` registry.
Use one persistent endpoint reachable from both the Jupiter service tasks and
the Holmes training tasks:

```bash
export REGISTRY=redis://YOUR_REDIS_HOST:6379
```

Do not omit this value for a separately launched training experiment. When the
Podman launcher creates its own internal Redis, that dynamically allocated URL
is scoped to its stack and is not a stable input for a later training launch.

## Preview and launch the service stack

`services podman yaml` consumes `services.yaml` directly. Environment variables
inside YAML values are expanded and unresolved variables are rejected.

```bash
primebeaker services podman yaml preview \
  --config=examples/podman-terminal/services.yaml

primebeaker services podman yaml launch \
  --config=examples/podman-terminal/services.yaml

literegistry detail --registry "$REGISTRY"
```

Wait until the registry reports all 16 `podman` instances before training.

## Single-node run

The direct single-node launcher exports `REGISTRY` but does not own a gateway,
so the setup command below starts one at the URL used by `rl.toml`.

```bash
primebeaker rl preview \
  --toml=examples/podman-terminal/rl.toml \
  --registry="$REGISTRY" \
  --workspace=ai2/oe-agents \
  --cluster=ai2/holmes \
  --working-dir=/weka/gfaria/primebeaker \
  --scratch-dir=/weka/gfaria/primebeaker \
  --home-dir=/weka/gfaria/primebeaker \
  --setup-command='python3 -m primebeaker.gateway --registry "$REGISTRY" --port 1212 --workers 8 >/tmp/primebeaker-gateway.log 2>&1 & gateway_pid=$!; sleep 5; kill -0 "$gateway_pid"'
```

Change `preview` to `submit` to launch it.

## Multi-node run

The multi-node role script starts and probes a local gateway on the trainer
node. It also blocks on the requested number of registered Podman workers.

```bash
primebeaker rl preview \
  --toml=examples/podman-terminal/multinode-rl.toml \
  --registry="$REGISTRY" \
  --required-service=podman=16 \
  --workspace=ai2/oe-agents \
  --cluster=ai2/holmes \
  --priority=high \
  --min-runtime-hours=4 \
  --working-dir=/weka/gfaria/primebeaker \
  --scratch-dir=/weka/gfaria/primebeaker \
  --home-dir=/weka/gfaria/primebeaker
```

Change `preview` to `submit` to launch it.

## Reward rules

Both TOMLs expose the weights next to the environment arguments:

- unit-test reward file: `1.0`;
- valid terminal completion marker: `+0.1`;
- Podman exception, including an OOM: `-1.0`;
- each call to an unavailable/fake tool: `-0.1`.

A missing or invalid reward file scores zero. Tests never run before the exact
completion command, so a rollout that simply exhausts its turn budget gets no
termination credit.
