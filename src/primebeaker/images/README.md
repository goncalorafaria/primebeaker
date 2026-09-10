# PrimeBeaker + JTC runtime image

PrimeBeaker owns the supported training/runtime image. The immutable base image
provides the tested CUDA, Torch, vLLM, and Prime-RL binary stack. The
application layer is installed exclusively from version-pinned PyPI releases:

- `primebeaker[runtime]==0.3.0`
- `jtc[harness]==0.2.0`
- released PyPI dependencies including JTCFlow, LiteRegistry, the LiteRegistry
  tool client, and Verifiers

The Dockerfile contains no source-tree `COPY`, editable install, or VCS URL.
Datasets, TOML configs, checkpoints, and outputs remain external runtime
mounts; they are not baked into the image.

## Release order

Publish both wheels before building the image:

```bash
cd /weka/gfaria/primebeaker
python -m build
python -m twine upload dist/primebeaker-0.3.0*

cd /weka/gfaria/jtc
python -m build
python -m twine upload dist/jtc-0.2.0*
```

The image build intentionally fails if either exact version is unavailable
from public PyPI. There is no alternate-index or local-source fallback.

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

## Build from PyPI

From the PrimeBeaker repository root:

```bash
docker build \
  --build-arg BASE_IMAGE=prime-rl-base:01KZVDND2PYP538F5JSGJ469EC \
  --build-arg PRIMEBEAKER_VERSION=0.3.0 \
  --build-arg JTC_VERSION=0.2.0 \
  --file src/primebeaker/images/Dockerfile.runtime \
  --tag primebeaker-jtc-runtime:0.3.0-jtc0.2.0 \
  .
```

Build-time checks load every PrimeBeaker environment, the JTC verifier workflow,
and all required CLIs. This is where a missing or incompatible wheel is caught.

Publish the resulting image to Beaker:

```bash
beaker image create primebeaker-jtc-runtime:0.3.0-jtc0.2.0 \
  --name primebeaker-jtc-runtime-0.3.0-jtc0.2.0 \
  --workspace ai2/oe-agents
```

Use the returned immutable `beaker://...` URI as the workflow or training
`--image`; do not use the mutable local Docker tag in configs.

## Smoke test

Run the image with `/weka` mounted, then verify these commands inside it:

```bash
python -m primebeaker.environments
jtc-workflow list
command -v rl
command -v sft
```
