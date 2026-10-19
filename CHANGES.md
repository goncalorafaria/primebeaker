# Changes

## Replay and rubric judging — October 2026

This records the replay workflow used for the latest dataset and judge runs. Several implementation changes still live in local scripts or run bundles rather than the published PrimeBeaker package. This document is not a claim that those changes have been integrated into the package.

### Active replay workflow

- Use original published Docker Hub task-image references matched to source tasks. Do not substitute reconstructed images or locally rebuilt archives. Catalog-tag agreement does not establish historical digest identity.
- Allocate a fresh task container through the LiteRegistry Podman client and shared `podman` service. Use the network-enabled deployment for outbound package downloads.
- Follow SWERL shell behavior: start in `/app` with `/workspace` fallback, run a fresh Bash per call, and persist cwd and exported variables. Do not force `HOME`, task-specific `PYTHONPATH`, `pipefail`, or additional service startup.
- Replay the task command sequence. Respect the source tool-call/response pairing; retain ordinary checks and task outputs. A final unresponded completion-marker action is not a task command to execute.
- The validation runtime uses 300-second command, shell-setup, and final-test deadlines, 600-second worker startup, and a 900-second handshake request deadline. These are run settings, not universal package defaults.
- Run the private final tests using the source's observed stop-on-first-failure policy, and compare the complete recorded outcome-count dictionary. Matching failures are eligible; a pass-only filter would discard reproducible negatives.
- Exclude infrastructure failures and final-count mismatches. Record command timeouts separately; the latest dataset retains them when the complete replay reaches a valid final grader with matching counts.
- Preserve source traces unchanged. Store fresh replay observations, grader output, container provenance, and exclusions separately.
- Normalize tool-output formatting and sort complete lines while preserving duplicate lines; compare exit status separately. Output agreement and final test-count agreement are distinct measurements.

### Judge workflow

- Match existing rubric tasks to source rollouts by normalized prompt and identical stripped unit-test source.
- Present the source-agent conversation as chat JSON with a `messages` array, supplied on stdin to `trace_inspection`. Each inspection response is capped at 2,000 characters.
- Give the judge a `bash` tool bound to the same container after fresh replay, allowing inspection and execution of the agent's work.
- Require one `submit` call per rubric with `rubric_id`, binary `label`, and concrete `feedback`; complete when all submissions are accepted.
- Exclude private final harness grader feedback, outcome counts, source success/reward labels, and private test source from judge context. Do not stage the private final grader in the judge container. Agent-authored or ordinary task test files may still exist and be inspected.
- Keep source traces, replay validation evidence, and judge conversations distinct in exports and visualization. A viewer split called `train` does not imply that a training job ran.

### Integration status

| Component | Current location/status |
| --- | --- |
| Podman solver environment | `src/primebeaker/environments/podman_terminal_env.py`; the shared local checkout still enforces a 60-second command/test limit. It is not aligned with the active 300-second replay settings. |
| Replay continuation environment | Local, untracked `src/primebeaker/environments/podman_terminal_replay_env.py`; contains both a legacy persistent shell and a Vanillux fresh-Bash wrapper. Its helper setup calls still use 10-second deadlines. Do not assume it is identical to the active run bundle. |
| Replay comparison/filter scripts | Local files under `examples/podman-terminal/`; older filters use successful final grading rather than the latest exact-count criterion. |
| Current validated dataset pipeline | External run bundle `records/tmax-rubric-matched-replay-20261019`, outside this repository. It owns the latest count matching, source preservation, and judge-input sanitization. |
| Active replay runtime | External dataset bundle's `runtime/glm_replay_core.py`, copied into run-specific workflows. Package integration remains outstanding. |
| Judge orchestration | JTC grouped binary-submit workflow, with a run-specific `bash` alias adapter; not implemented solely by PrimeBeaker. |
| Gateway/Redis/client fixes and network-enabled worker image | LiteRegistry PRs, listed below; these require the corresponding client versions and rebuilt deployment images. |

External bundle paths above are operational references, not files shipped with PrimeBeaker. Keep the bundles and their manifests when reproducing those runs.

### Related LiteRegistry PRs

- [#4](https://github.com/goncalorafaria/literegistry/pull/4), merged: indexed Redis worker roster.
- [#5](https://github.com/goncalorafaria/literegistry/pull/5) and [#6](https://github.com/goncalorafaria/literegistry/pull/6), merged: typed session-loss handling and bounded affinity-owner probes without command replay.
- [#11](https://github.com/goncalorafaria/literegistry/pull/11), merged: retain healthy Redis connections without routine head-registry polling.
- [#14](https://github.com/goncalorafaria/literegistry/pull/14), merged: cached gateway discovery, indexed endpoints, and obsolete head-refresh removal from #12.
- [#15](https://github.com/goncalorafaria/literegistry/pull/15), open at this writing: Redis operation exception logging before invalidation.
- [#16](https://github.com/goncalorafaria/literegistry/pull/16), open at this writing: deployment limits, affinity renewal, 300-second command support, and default `pasta` networking with rootless TUN preparation.
- Streaming remains separate in [#13](https://github.com/goncalorafaria/literegistry/pull/13).

### Validation evidence and limitations

The latest selected 5,000-row replay sample yielded 4,908 traces with matching final test counts: 4,783 Shashank and 125 Hamish 9B. Exclusions were 73 count mismatches and 19 infrastructure failures. This sample used prior eligibility evidence and is not an unbiased estimate for all source data.

The network-enabled successful-Shashank cohort had 10,307 exact final-count matches among 11,095 selected traces. GLM reconstructed-image results are quarantined as evidence of original-image fidelity; verified original image mapping is still required.

Remaining discrepancies include historical dependency-version drift, user-namespace permission behavior, background-service timing/port occupancy, and variable timestamps, addresses, certificate keys, and archive metadata. Matching final counts alone does not establish identical intermediate behavior or validate rubric labels.

This documentation update changes no runtime behavior and runs no tests. It does not resume RL training or redeploy the fleet.
