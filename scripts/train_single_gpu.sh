#!/usr/bin/env bash
# Train on one GPU (or CPU). Usage: bash scripts/train_single_gpu.sh /path/to/dataset [model]
# model: cvt-13 (default) | cvt-21 | cvt-w24 | cvt-13-384 | ...
set -euo pipefail

DATA_DIR="${1:?usage: $0 /path/to/dataset [model_name]}"
MODEL="${2:-cvt-13}"

python train.py \
  --data_dir "$DATA_DIR" \
  --model "$MODEL" \
  --epochs 30 \
  --batch_size 32 \
  --class_weights
