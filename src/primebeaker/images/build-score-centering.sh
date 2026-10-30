#!/usr/bin/env bash
set -euo pipefail

# Build native MC32 support from merged fork revisions and pinned submodules.
PRIME_RL_REVISION=${PRIME_RL_REVISION:-243cbb7d77702017b5f0e247afd7df35b07ee4e9}
VLLM_REVISION=${VLLM_REVISION:-409c9efd1aeb0fdd3f32d06554e33f520534e273}
IMAGE_TAG=${1:-prime-rl-score-centering:243cbb7-409c9ef-mc32}
BASE_TAG="${IMAGE_TAG}-base"
recipe_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
build_dir=$(mktemp -d "${TMPDIR:-/tmp}/primebeaker-score-build.XXXXXXXX")
trap 'rm -rf -- "$build_dir"' EXIT

git init "$build_dir/prime-rl"
git -C "$build_dir/prime-rl" remote add origin https://github.com/goncalorafaria/prime-rl.git
git -C "$build_dir/prime-rl" fetch --depth 1 origin "$PRIME_RL_REVISION"
git -C "$build_dir/prime-rl" checkout --detach FETCH_HEAD
test "$(git -C "$build_dir/prime-rl" rev-parse HEAD)" = "$PRIME_RL_REVISION"
git -C "$build_dir/prime-rl" -c 'url.https://github.com/.insteadOf=git@github.com:' \
    submodule update --init --recursive

docker build --file "$build_dir/prime-rl/Dockerfile.cuda" \
    --label "org.primebeaker.prime-rl.revision=$PRIME_RL_REVISION" \
    --tag "$BASE_TAG" "$build_dir/prime-rl"
docker build --file "$recipe_dir/Dockerfile.compact-logprobs" \
    --build-arg "BASE_IMAGE=$BASE_TAG" \
    --build-arg "VLLM_REVISION=$VLLM_REVISION" \
    --label "org.primebeaker.prime-rl.revision=$PRIME_RL_REVISION" \
    --label "org.primebeaker.mc32.available=true" \
    --tag "$IMAGE_TAG" "$recipe_dir"
printf 'Built %s\nPrimeRL: %s\nvLLM: %s\nMC32 available; estimator remains opt-in.\n' "$IMAGE_TAG" "$PRIME_RL_REVISION" "$VLLM_REVISION"
