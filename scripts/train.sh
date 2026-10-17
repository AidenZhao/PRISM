#!/usr/bin/env bash
# Train PRISM on one node. Usage: scripts/train.sh DATA_ROOT [S|B|L] [NGPU]
# Reproduces the released models: per-GPU batch 8 x 8 GPUs = 64, 1000 epochs, AdamW 3e-4 + cosine.
set -euo pipefail
DATA=${1:?dataset root from prism.preprocess}
SIZE=${2:-B}
NGPU=${3:-8}
RUN=runs/prism_${SIZE}
mkdir -p "$RUN"
torchrun --standalone --nproc_per_node "$NGPU" -m prism.train \
  --data "$DATA" --out "$RUN" --size "$SIZE" --bs 8 --epochs 1000 --workers 12 \
  "${@:4}" 2>&1 | tee -a "$RUN/train.log"
