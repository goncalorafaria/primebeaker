# PrimeBeaker image catalog

PrimeBeaker owns the immutable Prime-RL base-image catalog. The base image
provides the tested CUDA, Torch, vLLM, and Prime-RL binary stack.

See [current fork links and Podman RL setup](../../../docs/current-training-stack.md)
for published branch heads, exact submodule pins, and the additional local
packing/offload overlays used by current runs.

## Tested immutable base images

List the package-local catalog:

```bash
python -m primebeaker.images list
```

The default workspace currently uses
`beaker://01KZVDND2PYP538F5JSGJ469EC`; Holmes uses
`beaker://01M0E15WYCV7T0J1CMCFPBQ21P`.

Pull the chosen base into Docker:

```bash
python -m primebeaker.images pull \
  --workspace ai2/oe-agents \
  --tag prime-rl-base:01KZVDND2PYP538F5JSGJ469EC
```

JTC owns the combined evaluation application image at
`/weka/gfaria/jtc/docker/Dockerfile.eval`. That Dockerfile layers exact released
PrimeBeaker and JTC wheels onto one of these base images; PrimeBeaker does not
carry a duplicate evaluation Dockerfile.

## Standard score-centering and MC32 image build

Build native MC32 support from the merged forks:

```bash
bash src/primebeaker/images/build-score-centering.sh \
  prime-rl-score-centering:243cbb7-409c9ef-mc32
```

The builder pins PrimeRL to `243cbb7d77702017b5f0e247afd7df35b07ee4e9` and vLLM to
`409c9efd1aeb0fdd3f32d06554e33f520534e273`. PrimeRL's recursive submodules pin Renderers to
`a56bb9d046ada1c7a1f6fdb183549e0b0f5a7e1f` and Verifiers to
`8a44b6145905b19a94ec777dd7d8b06d4e4c9b0f`. All MC32 code is native to these commits: no
build-time source patches or copied runtime modules are used.

Docker with BuildKit, Git, and access to the source/package registries are
required. The build records source revisions in image labels. Commit hashes may
be overridden with `PRIME_RL_REVISION` and `VLLM_REVISION`; use a distinct image
tag and compatible native dependency commits. It builds locally and does not
publish to Beaker.

Use the [MC32 configuration fragment](mc32/rl-fragment.toml) and
[fork-version notes](mc32/README.md) to enable the estimator. The default remains
`topk`, with `VLLM_SCORE_CENTERING_MC32=0`; existing loss and sampling settings are
not changed implicitly. The newly assembled image has not been built or
GPU-validated.

## Compact logprob overlay

This optional recipe is for experiment images; the default image catalog is unchanged.

`Dockerfile.compact-logprobs` installs `goncalorafaria/vllm` at
`409c9efd1aeb0fdd3f32d06554e33f520534e273` with native compact and MC32 support. It requires a
PrimeRL 0.9 base with vLLM 0.29.0, Torch 2.13 and CUDA 13.0, and reuses matching
upstream binary extensions. For MC32 training, use the standard builder above:
this Dockerfile alone cannot supply missing trainer/config/transport changes.

```bash
docker build --build-arg BASE_IMAGE=prime-rl-cu130:local \
  -f src/primebeaker/images/Dockerfile.compact-logprobs \
  -t prime-rl-compact:409c9ef-mc32 src/primebeaker/images
```

Publish the resulting image to Beaker and pass its immutable URI as the launcher
`image`. The cataloged historical images do not contain this fork. Keep experiment images explicit in the launch configuration; do not replace the
default catalog entries.
For extraction, use `processed_logprobs` and request `logprobs=128` while retaining
normal sampling (`top_k=-1`, `top_p=1.0`). Set `VLLM_USE_COMPACT_LOGPROBS=0` for
the native comparison baseline. This flag does not request logprobs by itself.
