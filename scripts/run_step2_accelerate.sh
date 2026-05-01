#!/usr/bin/env bash
set -euo pipefail

# Helper to run Step2 under `accelerate` with deepspeed offload (L40S recommended)
# Usage: ./scripts/run_step2_accelerate.sh [path/to/config.json]

CONFIG=${1:-configs/step2.full-train.hf.json}

echo "Using config: ${CONFIG}"

python -m accelerate launch \
  --config_file configs/accelerate/l40s_config.yaml \
  --deepspeed_config_file configs/deepspeed/offload_z3.json \
  scripts/train_step2_full.py --config "${CONFIG}"
