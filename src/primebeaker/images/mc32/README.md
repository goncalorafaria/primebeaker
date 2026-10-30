# MC32 source overlay

The standard score-centering build includes these patches against its published
PrimeRL, Renderers, Verifiers and vLLM pins. They promote the existing experimental
MC32 implementation into a reproducible image recipe; no local worktree is needed
to build it. `manifest.json` records every patch and source checksum. Applying a
patch to different source contents fails before modifying that component.

The patches cover the complete path: 32 independent categorical probes per
response position, processed sampler probabilities, duplicate-preserving NumPy
transport, trace alignment, batch packing, and the IcePop/IPO score-centering
surrogate. The generated response token is separate from the probes. A mode tag
prevents interpreting MC draws as a deduplicated top-k head.

`packed_heads.py` is shared by PrimeRL, Renderers and Verifiers. It is copied into
the editable base source during the build and installed into the final image's
Python site-packages. vLLM patches are applied to the installed Python source
after installing the pinned fork with its matching binary extensions. The
manifest remains in the image at `/opt/primebeaker-mc32/manifest.json`; installed
vLLM's `direct_url.json` identifies the base commit, and the manifest identifies
the additional changes.

## Enable MC32

Merge [rl-fragment.toml](rl-fragment.toml) into a complete RL config. The standard
image contains MC32 but leaves it disabled: the default estimator is `topk`, and
`VLLM_SCORE_CENTERING_MC32` defaults to `0`. Both the trainer estimator and rollout
sampling mode must be selected explicitly. Ordinary evaluation requests remain
supported on an MC32-enabled server.

Requirements:

- Exactly 32 IID categorical probes, with replacement; retain duplicate IDs.
- Positive temperature, `top_p = 1.0`, and no `top_k` or `score_centering_top_k`.
- Token stop IDs, no detokenization, no speculative decoding.
- V2 runner, `processed_logprobs`, and `max_logprobs >= 32`.
- End-to-end image stack from this recipe; a sampler-only overlay cannot supply
  the trainer/config/transport changes.

No model optimization/reduction dtype defaults or existing KL penalties change.
The Monte Carlo surrogate is an estimator conditional on the rollout
distribution; this does not imply that the whole RL algorithm is unbiased.

## Validation status

The source implementation was exercised in the separate experimental MC32 GPU
verification. That does not validate this newly assembled Docker image or
establish a performance improvement. This recipe change has not run a Docker
build, new tests, or a GPU training job. Validate and publish the resulting image
before using its immutable Beaker URI for a new experiment. Existing image
catalog entries and existing training jobs are unaffected.
