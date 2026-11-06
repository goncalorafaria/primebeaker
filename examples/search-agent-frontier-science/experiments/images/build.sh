#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
context=$(mktemp -d /tmp/search-agent-image.XXXXXXXX)
trap 'rm -rf "$context"' EXIT
mkdir -p "$context/bundle"
cp -a "$root/." "$context/bundle/"
# No credentials, datasets, checkpoint weights or outputs.
beaker image pull 01M3WPHYKDAZ4533VCDZGNZGHZ search-judge-base:local
beaker image pull 01M27G8Y678BBF5MWXX6R12BSB search-services-base:local
for role in policy judge services; do
 docker build -f "$root/images/Dockerfile.$role" -t "search-agent-$role:local" "$context"
done
