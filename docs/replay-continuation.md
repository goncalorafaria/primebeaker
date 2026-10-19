# Replay continuation

Load `primebeaker.environments.podman_terminal_replay_env.load_environment` with a dataset path and `podman_gateway_url` pointing to a network-enabled LiteRegistry gateway. The service defaults to `podman`.

Rows need an original system/user `prompt` and metadata under `info`: `original_image` plus `replay` containing `source_success=true`, `source_dataset`, `source_id`, and ordered `steps`. Each step has `command`, integer `exit_code`, and optional `assistant_content`, `reasoning_content`, and `exit_code_semantics`. Use `success_boolean` only when source metadata records every nonzero result as 1; the default is exact exit codes.

The image must resolve to Docker Hub. A registry reference does not prove historical digest identity; use the dataset's validated image mapping. Runtime setup archives and extra service injection are rejected.

`replay_fraction` selects complete prefix commands and must be between 0 and 1. The default is 0.8; set it to 0.65 for a 65% handoff. At least one command is left for continuation. The prefix's commands and fresh outputs can enter the prompt; the reference suffix never does. The configured tokenizer checks the handoff token budget. This environment requires successful references and aborts on prefix exit divergence; the separate judge-dataset pipeline can retain reproducible failures using matching final test counts.

Each call starts fresh Bash in the persisted cwd, initially `/app` with `/workspace` fallback. Only cwd and exported variables persist. No forced HOME/PYTHONPATH/pipefail or service startup is added. Commands exceeding the Podman command-field size are transferred through stdin in the same container. HTTP requests with unknown execution outcomes are not automatically retried by the replay adapter.

Commands, setup, and final grading allow 300 seconds. Request timeout defaults to 360 seconds and handshake timeout to 900 seconds. The worker's startup timeout is a deployment setting. Solver completion uses the standalone `echo TERMINAL_COMPLETE` bash command, not rubric `submit`.

Install current runtime dependencies using uv's documented overrides. JTC 0.3.0 pins older registry dependencies; pip resolution of the combined judge/runtime extras may require a compatible JTC release. This PR does not update JTC or the separately running judge harness.

No live validation of this packaged integration has run. Existing results describe external replay bundles, not a validation run of this branch.
