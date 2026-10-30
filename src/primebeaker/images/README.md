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

Build the merged PrimeRL source and vLLM kernel together:

```bash
bash src/primebeaker/images/build-score-centering.sh \
  prime-rl-score-centering:edac0a4-0ad34f5-mc32
```

The script pins PrimeRL to `edac0a49313100c7b502602d02467bf5e35f6c2c`
(PR #3, `main`) and vLLM to `0ad34f5b6bd55ec8a3647ab39b2c8bf322d65fa0`
(PR #1, `releases/v0.29.0`). It checks out PrimeRL and its pinned submodules,
applies the bundled MC32 and packed-transport patches to PrimeRL, Renderers and
Verifiers, builds its `Dockerfile.cuda`, then applies the vLLM overlay below.
MC32 support is included in every standard build; choosing the estimator remains
opt-in. Source revisions are recorded in image labels, and
[`mc32/manifest.json`](mc32/manifest.json) records patch and file checksums.
Docker with BuildKit, Git, Python 3, and access to
the source and package registries are required. Full commit hashes can be
overridden with `PRIME_RL_REVISION` and `VLLM_REVISION` only when recorded source
checksums still match. Different contents require updating patches and the
manifest, and a distinct image tag. The script builds locally; it does not publish
to Beaker.

Use the [MC32 configuration fragment](mc32/rl-fragment.toml) and
[implementation notes](mc32/README.md) to enable the estimator. The image keeps
`topk` as the default and `VLLM_SCORE_CENTERING_MC32=0`; existing loss and sampling
settings are not changed implicitly. This combined image recipe has not yet been
built or GPU-validated.

## Compact logprob overlay

This optional recipe is for experiment images; the default image catalog is unchanged.

`Dockerfile.compact-logprobs` installs `goncalorafaria/vllm` at
`0ad34f5b6bd55ec8a3647ab39b2c8bf322d65fa0`. It requires a Prime RL 0.9
base with vLLM 0.29.0, Torch 2.13, and CUDA 13.0. It reuses the matching
upstream binary extensions and adds the bundled MC32 kernel and storage modules.
For MC32 training, use the standard builder above: applying only this Dockerfile
to an unrelated base does not provide the trainer/config/transport patches.

```bash
docker build --build-arg BASE_IMAGE=prime-rl-cu130:local \
  -f src/primebeaker/images/Dockerfile.compact-logprobs \
  -t prime-rl-compact:0ad34f5-mc32 src/primebeaker/images
```

Publish the resulting image to Beaker and pass its immutable URI as the launcher
`image`. The cataloged historical images do not contain this fork. Keep experiment images explicit in the launch configuration; do not replace the
default catalog entries.
For extraction, use `processed_logprobs` and request `logprobs=128` while retaining
normal sampling (`top_k=-1`, `top_p=1.0`). Set `VLLM_USE_COMPACT_LOGPROBS=0` for
the native comparison baseline. This flag does not request logprobs by itself.
