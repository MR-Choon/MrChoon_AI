#!/usr/bin/env bash
set -euo pipefail

# Helper to run Step2 under `accelerate` with deepspeed offload (L40S recommended)
# Usage: ./scripts/run_step2_accelerate.sh [path/to/config.json]

CONFIG=${1:-configs/step2.full-train.hf.json}

echo "Using config: ${CONFIG}"

# Install ALL critical dependencies explicitly
echo "Installing all dependencies..."
python -m pip install --upgrade -q pip "setuptools<82" wheel
python -m pip install -q --no-cache-dir sentencepiece
python -m pip install -q --no-cache-dir tiktoken
python -m pip install -q "tokenizers>=0.22.0,<=0.23.0"
python -m pip install -q accelerate

# Verify installation
echo "Verifying sentencepiece installation..."
python -c "import sentencepiece; print(f'sentencepiece version: {sentencepiece.__version__}')"

echo "Verifying tiktoken installation..."
python -c "import tiktoken; print('tiktoken OK')"

echo "Starting training..."

python -m accelerate.commands.launch \
  --config_file configs/accelerate/l40s_config.yaml \
  --deepspeed_config_file configs/deepspeed/offload_z3.json \
  scripts/train_step2_full.py --config "${CONFIG}"
