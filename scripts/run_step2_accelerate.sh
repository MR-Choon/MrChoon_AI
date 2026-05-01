#!/usr/bin/env bash
set -euo pipefail

# Helper to run Step2 under `accelerate` with deepspeed offload (L40S recommended)
# Usage: ./scripts/run_step2_accelerate.sh [path/to/config.json]

CONFIG=${1:-configs/step2.full-train.hf.json}

echo "Using config: ${CONFIG}"

# Check if accelerate is installed, if not install it
python -c "import accelerate" 2>/dev/null || pip install accelerate

python -m accelerate.commands.launch \
  --config_file configs/accelerate/l40s_config.yaml \
  --deepspeed_config_file configs/deepspeed/offload_z3.json \
  scripts/train_step2_full.py --config "${CONFIG}"
