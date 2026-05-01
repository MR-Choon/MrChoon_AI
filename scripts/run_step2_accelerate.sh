#!/usr/bin/env bash
set -euo pipefail

# Helper to run Step2 under `accelerate` with deepspeed offload (L40S recommended)
# Usage: ./scripts/run_step2_accelerate.sh [path/to/config.json]

CONFIG=${1:-configs/step2.full-train.hf.json}

echo "Using config: ${CONFIG}"

# Fix all dependency issues at once
echo "Resolving all dependencies..."
python -m pip install --upgrade pip setuptools wheel -q
python -m pip install -e .[hf] --upgrade -q 2>&1 | tail -20 || true

# Force correct tokenizers version
python -m pip install "tokenizers>=0.22.0,<=0.23.0" -q

echo "Verifying dependencies..."
python -c "
import sys
try:
    import torch
    import transformers
    import peft
    import datasets
    import accelerate
    import sentencepiece
    import tokenizers
    print('All dependencies OK')
except ImportError as e:
    print(f'ERROR: {e}', file=sys.stderr)
    sys.exit(1)
"

echo "Starting training..."

python -m accelerate.commands.launch \
  --config_file configs/accelerate/l40s_config.yaml \
  --deepspeed_config_file configs/deepspeed/offload_z3.json \
  scripts/train_step2_full.py --config "${CONFIG}"
