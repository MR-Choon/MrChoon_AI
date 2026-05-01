#!/usr/bin/env bash
set -euo pipefail

# Helper to run Step2 under `accelerate` with deepspeed offload (L40S recommended)
# Usage: ./scripts/run_step2_accelerate.sh [path/to/config.json]

CONFIG=${1:-configs/step2.full-train.hf.json}

echo "Using config: ${CONFIG}"

# Ensure tokenizer and required dependencies are installed
echo "Installing required dependencies..."
python -m pip install --upgrade -q sentencepiece tiktoken accelerate 2>&1 | grep -v "already satisfied" || true

# Verify critical packages
echo "Verifying dependencies..."
python -c "import sentencepiece; import tiktoken; import accelerate" || {
  echo "ERROR: Failed to import required packages. Attempting reinstall..."
  python -m pip install --force-reinstall sentencepiece tiktoken accelerate
}

echo "Starting training..."

python -m accelerate.commands.launch \
  --config_file configs/accelerate/l40s_config.yaml \
  --deepspeed_config_file configs/deepspeed/offload_z3.json \
  scripts/train_step2_full.py --config "${CONFIG}"
