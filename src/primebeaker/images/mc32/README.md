# Native MC32 fork versions

The standard recipe builds native MC32 support from merged fork commits. It does
not apply source patches, copy Python modules into site-packages, or depend on
launch-local runtime overlays.

| Component | Merged commit | PR |
| --- | --- | --- |
| PrimeRL | `243cbb7d77702017b5f0e247afd7df35b07ee4e9` | [#5](https://github.com/goncalorafaria/prime-rl/pull/5) |
| Renderers | `a56bb9d046ada1c7a1f6fdb183549e0b0f5a7e1f` | [#2](https://github.com/goncalorafaria/renderers/pull/2) |
| Verifiers | `8a44b6145905b19a94ec777dd7d8b06d4e4c9b0f` | [#1](https://github.com/goncalorafaria/verifiers/pull/1) |
| vLLM | `409c9efd1aeb0fdd3f32d06554e33f520534e273` | [#2](https://github.com/goncalorafaria/vllm/pull/2) |

PrimeRL pins the merged Renderers and Verifiers commits as submodules. The shared
transport helper is packaged as `renderers.packed_heads`. The builder fetches
PrimeRL with its recursive submodules and installs the pinned vLLM fork with its
matching 0.29 / CUDA 13 binary extensions. Image labels record PrimeRL/vLLM commits;
vLLM's `direct_url.json` identifies its native implementation commit.

## Enable MC32

Merge [rl-fragment.toml](rl-fragment.toml) into a complete native RL config. MC32
is available in the image but remains disabled by default: the estimator is
`topk` and `VLLM_SCORE_CENTERING_MC32=0`. Select both the trainer estimator and
rollout sampling mode explicitly. Ordinary evaluation requests remain supported.

Requirements:

- Exactly 32 IID categorical probes with replacement; retain duplicate IDs.
- Positive temperature, `top_p=1.0`, and no `top_k` or `score_centering_top_k`.
- Token stop IDs, no detokenization, no speculative decoding.
- V2 runner, `processed_logprobs`, and `max_logprobs>=32`.
- The matching native trainer, renderer, verifier and sampler commits above.

The sampled response token is separate from the probes. A mode tag is preserved
through trace alignment, packing and tensorization, preventing interpretation of
MC draws as deduplicated top-k heads. Numerical dtype defaults and existing KL
penalties are unchanged. The conditional estimator guarantee does not imply
that the entire RL algorithm is unbiased.

## Validation status

The source implementation had separate experimental GPU correctness verification.
No new tests, model evaluation, Docker image build or GPU training were run for
this fork integration and recipe update. Build and validate the assembled image
before publishing a new immutable Beaker URI. Existing image catalog entries and
the Writing training run's older image are unaffected.
