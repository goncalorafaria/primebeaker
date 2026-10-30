#!/usr/bin/env bash
set -euo pipefail

# Build the pinned score-centering stack with bundled, opt-in MC32 support.
PRIME_RL_REVISION=${PRIME_RL_REVISION:-edac0a49313100c7b502602d02467bf5e35f6c2c}
VLLM_REVISION=${VLLM_REVISION:-0ad34f5b6bd55ec8a3647ab39b2c8bf322d65fa0}
IMAGE_TAG=${1:-prime-rl-score-centering:edac0a4-0ad34f5-mc32}
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

python3 "$recipe_dir/mc32/apply.py" prime-rl "$build_dir/prime-rl"
python3 "$recipe_dir/mc32/apply.py" renderers "$build_dir/prime-rl/deps/renderers"
python3 "$recipe_dir/mc32/apply.py" verifiers "$build_dir/prime-rl/deps/verifiers"
# Editable source installs need the helper while building the base image too.
cp "$recipe_dir/mc32/packed_heads.py" "$build_dir/prime-rl/src/packed_heads.py"

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
