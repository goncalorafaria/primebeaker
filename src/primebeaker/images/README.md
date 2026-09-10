# PrimeBeaker image catalog

PrimeBeaker owns the immutable Prime-RL base-image catalog. The base image
provides the tested CUDA, Torch, vLLM, and Prime-RL binary stack.

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
