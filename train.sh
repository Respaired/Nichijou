#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if [ "$#" -lt 1 ]; then
  printf '%s\n' "Usage: ./train.sh /path/to/datasets.json [additional train.py arguments]" >&2
  exit 2
fi
datasets="$1"
shift
accelerate launch train.py \
  --config configs/nichijou_60s.json \
  --datasets "$datasets" \
  --tokenizer assets/tokenizer \
  --exp_dir runs/nichijou_60s \
  --length_training_stage \
  "$@"
