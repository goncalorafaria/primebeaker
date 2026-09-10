# WebTerminal search-agent RL on four nodes

This is a complete PrimeBeaker topology example for the local-corpus
WebTerminal search harness.

- services.yaml provisions 32 terminal workers and 32 workers for the
  localsearch:bc-rl-v1 BM25 corpus on Jupiter. It does not request a CPU
  resource shape.
- multinode-rl.toml launches one 8-GPU trainer node and three 8-GPU inference
  nodes on Holmes, for 8 trainer ranks and 24 tensor-parallel-1 inference
  ranks.
- run.yaml combines those service and RL settings for one-command scheduling
  and trainer-owned service cleanup.

The TOML uses primebeaker.environments.jtc_search_agent_webterminal_judge_env.
The final answer reward comes only from the registered judge model. The
environment also applies the configured terminal stderr and missing-final-answer
penalties; it does not use exact string matching as an answer reward.

## Prerequisites

Install the runtime extras, install the required LiteRegistry images into the
service workspace, and make the private dataset available to Beaker tasks
through the HF_TOKEN secret.

~~~bash
cd /weka/gfaria/primebeaker
pip install -e '.[runtime]'
beaker account whoami
docker version
primebeaker services images install \
  --workspace=ai2/oe-agents \
  --stack=base \
  --build-local-search \
  --jtc-build-context=/path/to/jtc
export HEAD_REGISTRY=sqlite:///weka/gfaria/primebeaker/registries/search-agent-webterminal.sqlite3
export REGISTRY=head+sqlite:///weka/gfaria/primebeaker/registries/search-agent-webterminal.sqlite3
~~~

Image installation is a required, explicit step: `pip install` installs the
Python launchers but cannot create images in a remote Beaker workspace. The
installer builds the Redis, base-services, terminal, vLLM, and local-search
images matching the installed launcher, uploads them, and prints immutable
Beaker IDs under `launcher_args`.

Copy those IDs into the selected configuration: either `services.yaml` or the
`service` object in `run.yaml`. Set `redis_image`, `services_image`,
`terminal_image`, `vllm_image`, and `local_search_image`. The judge-model
pool is launched separately from these YAMLs; use the installed vLLM image for
that pool. The GPU training job also needs the immutable PrimeBeaker runtime image
passed through `--image` later in this README.

The service stack owns the data-plane Redis. The SQLite HEAD_REGISTRY is its
stable control plane: Redis publishes its live, dynamically allocated URL
there, then the terminal and local-search workers connect through REGISTRY.
Training passes the same head+sqlite URI, so a Redis restart does not require
rewriting the GPU launch configuration.

This uses the head-aware LiteRegistry BaseDeployment launcher. If a local
installation predates its head_registry field, update that deployment companion
before launching the YAML.

The Weka paths in services.yaml are paths visible inside Beaker tasks:

~~~text
/weka/stevenc/data-dr/search_agents/bc_rl_v1/collection/part-00000.jsonl
/weka/stevenc/data-dr/search_agents/bc_rl_v1/bm25
~~~

## Launch both experiments with one command

`run.yaml` contains the complete service and RL launch configuration. Preview
or launch both Beaker experiments together:

~~~bash
primebeaker run preview \
  --config=examples/search-agent-webterminal/run.yaml

primebeaker run launch \
  --config=examples/search-agent-webterminal/run.yaml
~~~

The launcher submits the service experiment first and immediately submits RL.
It does not wait for Redis on the submitting machine: the stable SQLite head
lets the training replicas discover Redis when it appears, and their existing
`required_services` barrier waits for terminal, local-search, and judge
capacity.

Trainer replica 0 receives the service experiment ID and stops that experiment
when training succeeds, fails, or receives a handled termination signal. The
`BEAKER_TOKEN` secret named by `lifecycle.beaker_token_secret` must exist in
the RL workspace and be authorized to stop the service experiment in the
service workspace. If RL submission itself fails, the local launcher stops the
newly created service experiment immediately.

The judge pool remains separately managed because it may be shared across runs;
`run.yaml` waits for its registered `judge` endpoint but does not own or stop
it.

## Launch the service and RL experiments independently

The original commands remain available when services should outlive one run or
you want to inspect the registry manually. First launch the service stack:

~~~bash
primebeaker services yaml preview \
  --config=examples/search-agent-webterminal/services.yaml

primebeaker services yaml launch \
  --config=examples/search-agent-webterminal/services.yaml
~~~

The YAML passes head_registry to LiteRegistry BaseDeploymentConfig. Because it
does not supply registry, the native stack creates a Redis task, connects it
to the SQLite head, and supplies the derived head+sqlite registry URI to its
services. It keeps omit_service_resources true, so neither terminal nor
local-search workers declare a CPU allocation.

Optionally confirm the registry has enough services before submitting RL:

~~~bash
literegistry detail --registry "$REGISTRY"
~~~

The judge-scored verifier additionally requires a service registered as
model_path judge and the judge-model vLLM pool named in multinode-rl.toml. The
rollout calls the local gateway at port 1212, which routes its judge request to
that proxy. Both must use the same head+sqlite registry as this stack.

## Judge profile catalog

PrimeBeaker ships the judge-model catalog and all templates it references. The
TOML enables validate_judge_profile, so the rollout checks that its exact
judge_model_path is present in this catalog before a training rollout begins.

~~~bash
primebeaker judge path
primebeaker judge show \
  --model=/weka/gfaria/prime_sft/outputs/quokka-sft-qwen35-9b-rltracer-xmlv1-search25-baseformula-ba8d07-step400/weights/step_400
~~~

For the existing judge-server entrypoint, configure that catalog explicitly:

~~~bash
export JUDGE_MODEL_PROFILES_DIR="$(primebeaker judge path)"
~~~

Point the judge service model-profile-directory setting at the printed path.
This is the catalog the service must use to select the Quokka WebTerminal
profile; it contains the profile template, tool policy, and rollout limits.

Then preview or submit the multi-node run:

~~~bash
primebeaker rl preview \
  --toml=examples/search-agent-webterminal/multinode-rl.toml \
  --image=beaker://goncalof/prime-rl-v071dev83-qwen35-renderer-eval-inflight-holmes \
  --workspace=ai2/oe-agents-holmes \
  --cluster=ai2/holmes \
  --priority=high \
  --min-runtime-hours=4 \
  --registry="$REGISTRY" \
  --required-service=terminal=32 \
  --required-service=localsearch:bc-rl-v1=32 \
  --required-service=judge=1 \
  --working-dir=/weka/gfaria/primebeaker \
  --scratch-dir=/weka/gfaria/primebeaker \
  --home-dir=/weka/gfaria/primebeaker

primebeaker rl submit \
  --toml=examples/search-agent-webterminal/multinode-rl.toml \
  --image=beaker://goncalof/prime-rl-v071dev83-qwen35-renderer-eval-inflight-holmes \
  --workspace=ai2/oe-agents-holmes \
  --cluster=ai2/holmes \
  --priority=high \
  --min-runtime-hours=4 \
  --registry="$REGISTRY" \
  --required-service=terminal=32 \
  --required-service=localsearch:bc-rl-v1=32 \
  --required-service=judge=1 \
  --working-dir=/weka/gfaria/primebeaker \
  --scratch-dir=/weka/gfaria/primebeaker \
  --home-dir=/weka/gfaria/primebeaker
~~~

Each GPU task starts a local PrimeBeaker gateway on port 1212. The verifier
search and terminal URLs therefore stay local, while the gateway resolves
named workers through REGISTRY.

The proxy itself can resolve and call the judge-model vLLM pool through the
LiteRegistry API directly; it need not make another gateway hop.
