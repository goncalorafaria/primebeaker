# Prime-RL images

PrimeBeaker ships an immutable image catalog plus two Dockerfiles. The base
images preserve a tested CUDA, Torch, vLLM, and Prime-RL binary stack. The
runtime layer adds PrimeBeaker's launchers, clients, environments,
and templates without re-resolving that GPU dependency stack.

## Tested immutable images

List the package-local catalog:

```bash
python -m primebeaker.images list
```

The default workspace uses:

```text
workspace: ai2/oe-agents
image:     beaker://01KZVDND2PYP538F5JSGJ469EC
```

Holmes uses:

```text
workspace: ai2/oe-agents-holmes
image:     beaker://01M0E15WYCV7T0J1CMCFPBQ21P
```

Inspect the selected image through Beaker:

```bash
python -m primebeaker.images inspect --workspace ai2/oe-agents
python -m primebeaker.images inspect --workspace ai2/oe-agents-holmes
```

The `primebeaker` launch CLI selects the cataloged image matching
`--workspace` when `--image` is omitted.

## Pull the image locally

Authenticate the Beaker CLI first, then run:

```bash
python -m primebeaker.images pull \
  --workspace ai2/oe-agents \
  --tag prime-rl-base:01KZVDND2PYP538F5JSGJ469EC
```

Equivalent direct command:

```bash
beaker image pull \
  01KZVDND2PYP538F5JSGJ469EC \
  prime-rl-base:01KZVDND2PYP538F5JSGJ469EC
```

## Build the self-contained PrimeBeaker runtime

From the repository root:

```bash
docker build \
  --build-arg BASE_IMAGE=prime-rl-base:01KZVDND2PYP538F5JSGJ469EC \
  --file src/primebeaker/images/Dockerfile.runtime \
  --tag primebeaker-runtime:0.3.0 \
  .
```

The build fails unless both `rl` and `sft` exist and every bundled environment
module imports successfully.

Publish the resulting local Docker image to Beaker:

```bash
beaker image create primebeaker-runtime:0.3.0 \
  --name primebeaker-runtime-0.3.0 \
  --workspace ai2/oe-agents
```

Use the immutable `beaker://...` URI returned by that command for training.

## Rebuild with the exact maintained Prime-RL fork

The tested fork is:

```text
repository: https://github.com/goncalorafaria/prime-rl.git
branch:     feat/datadev-training-extensions
revision:   171c669dac1c83b35559b4adbedf113569eb5579
upstream:   v0.7.1.dev83 / 2ffe374e0
```

Create an isolated build context:

```bash
mkdir primebeaker-image-context
git clone --recurse-submodules \
  --branch feat/datadev-training-extensions \
  https://github.com/goncalorafaria/prime-rl.git \
  primebeaker-image-context/prime-rl
git -C primebeaker-image-context/prime-rl checkout --detach \
  171c669dac1c83b35559b4adbedf113569eb5579
git -C primebeaker-image-context/prime-rl submodule update --init --recursive
cp -R primebeaker primebeaker-image-context/primebeaker
```

Then build the source overlay:

```bash
docker build \
  --build-arg BASE_IMAGE=prime-rl-base:01KZVDND2PYP538F5JSGJ469EC \
  --file primebeaker-image-context/primebeaker/src/primebeaker/images/Dockerfile.full \
  --tag primebeaker-full:0.3.0 \
  primebeaker-image-context
```

The historical immutable base image predates cleanup of the fork's Git
history. Its Beaker ID identifies the tested deployed bytes; the revision
above is the maintained rebuild source.

## Local smoke test

```bash
docker run --rm --gpus all --ipc=host \
  --volume /weka:/weka \
  --entrypoint /bin/bash \
  primebeaker-runtime:0.3.0 \
  -lc 'python -m primebeaker.environments >/tmp/environments && command -v rl && command -v sft'
```

## LiteRegistry tool-client dependency

Both Dockerfiles install `literegistry==1.0.48` and
`literegistry-tool-client==0.1.0` from PyPI explicitly, before overlaying
PrimeBeaker with `--no-deps`. PrimeBeaker environments import directly from
`literegistry_tool_client`; the old `primebeaker.client` imports re-export the
same classes. Build-time checks verify the client version, compatibility
imports, and all environment imports.

The image contains the installed clients and needs no LiteRegistry checkout.
Existing SIFs and cataloged images must be rebuilt to include this change.
