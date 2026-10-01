#!/usr/bin/env bash
# Run the drone demo headless inside the drone6dof container.
#
# Usage: scripts/run_headless.sh [controller]
set -euo pipefail

controller="${1:-ds_guidance}"
root="$(cd "$(dirname "$0")/.." && pwd)"

mkdir -p "${root}/out"
exec docker run --rm \
    -v "${root}:/work" -w /work \
    -v "${root}/out:/data" \
    drone6dof:latest python -m drone6dof --vis none \
    --controller "${controller}" --out /data/telemetry.csv
