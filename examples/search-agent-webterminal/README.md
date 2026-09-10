# WebTerminal search-agent RL on four nodes

This is a complete PrimeBeaker topology example for the local-corpus
WebTerminal search harness.

- services.yaml provisions 32 terminal workers and 32 workers for the
  localsearch:bc-rl-v1 BM25 corpus on Jupiter. It does not request a CPU
  resource shape.
- multinode-rl.toml launches one 8-GPU trainer node and three 8-GPU inference
  nodes on Holmes, for 8 trainer ranks and 24 tensor-parallel-1 inference
  ranks.

The TOML uses primebeaker.environments.jtc_search_agent_webterminal_judge_env.
The final answer reward comes only from the registered judge model. The
environment also applies the configured terminal stderr and missing-final-answer
penalties; it does not use exact string matching as an answer reward.

## Prerequisites

Install the runtime extras and make the private dataset available to Beaker
tasks through the HF_TOKEN secret.

~~~bash
cd /weka/gfaria/primebeaker
pip install -e '.[runtime]'
export HEAD_REGISTRY=sqlite:///weka/gfaria/primebeaker/registries/search-agent-webterminal.sqlite3
export REGISTRY=head+sqlite:///weka/gfaria/primebeaker/registries/search-agent-webterminal.sqlite3
~~~

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

## Start and inspect the service stack

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

Confirm the registry has enough services before submitting the four-node run:

~~~bash
literegistry detail --registry "$REGISTRY"
~~~

The judge-scored verifier additionally requires a service registered as
model_path judge and the judge-model vLLM pool named in multinode-rl.toml. The
rollout calls the local gateway at port 1212, which routes its judge request to
that proxy. Both must use the same head+sqlite registry as this stack.

## Preview and submit the multi-node run

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
