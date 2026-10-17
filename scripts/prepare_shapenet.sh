#!/usr/bin/env bash
# Build the ShapeNet geodesic dataset. Usage: scripts/prepare_shapenet.sh ShapeNetCore.v2.zip DATA_ROOT [WORKERS]
# ~190 CPU-seconds per shape (watertight + remesh + 500 exact MMP sources); ~1000 CPU-hours for 18.5k shapes.
# For clusters, run the process stage as k jobs with --shard i --num_shards k (finished shapes are skipped).
set -euo pipefail
ZIP=${1:?path to ShapeNetCore.v2.zip}
OUT=${2:?output dataset root}
WORKERS=${3:-$(nproc)}
python -m prism.preprocess select --zip "$ZIP" --out "$OUT"
python -m prism.preprocess process --out "$OUT" --workers "$WORKERS"
python -m prism.preprocess split --out "$OUT"
