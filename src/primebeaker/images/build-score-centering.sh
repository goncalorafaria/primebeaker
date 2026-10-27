#!/usr/bin/env bash
set -euo pipefail

# Build both repositories at the merged score-centering revisions.
PRIME_RL_REVISION=${PRIME_RL_REVISION:-edac0a49313100c7b502602d02467bf5e35f6c2c}
VLLM_REVISION=${VLLM_REVISION:-0ad34f5b6bd55ec8a3647ab39b2c8bf322d65fa0}
IMAGE_TAG=${1:-prime-rl-score-centering:edac0a4-0ad34f5}
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
    --tag "$IMAGE_TAG" "$recipe_dir"
printf 'Built %s\nPrimeRL: %s\nvLLM: %s\n' "$IMAGE_TAG" "$PRIME_RL_REVISION" "$VLLM_REVISION"
