# Minecraft VLA Project

This repository is organized for staged development:

- Step 1: Tokenizer compatibility and dry run
- Step 2: Build VLA model (Qwen3.5 text backbone + separate vision encoder)
- Step 3: Minecraft server integration test

## Project layout

- `src/minecraft_vla/`: core package
- `configs/`: json configs for each step
- `scripts/`: executable entrypoints
- `tests/`: lightweight tests
- `artifacts/`: generated outputs (ignored by git)
- `prompt.md`: original task definition

## Step 1 quick start (mock)

Run Step 1 tokenizer remapping and dry run:

python scripts/run_step1.py --config configs/step1.mock.json

## Step 1 hf backend (optional)

Install optional dependencies first:

pip install -e .[hf]

Then run:

python scripts/run_step1.py --config configs/step1.hf.json

## Step 2 quick start (mock)

Run Step 2 dry run without external model downloads:

python scripts/run_step2.py --config configs/step2.mock.json

## Step 2 hf backend (optional)

Install optional dependencies first:

pip install -e .[hf]

Then run:

python scripts/run_step2.py --config configs/step2.hf.json

## Step 2 full QLoRA training

Install training dependencies:

pip install -e .[hf]

Run full training pipeline:

python scripts/train_step2_full.py --config configs/step2.full-train.hf.json

Notes:

- Default config uses `dry_run=true` and `max_steps=20` for a safe first pass.
- Set `dry_run=false` and adjust runtime fields for a real long-run training job.
- In dry run, `model.dry_run_model_id` is used to avoid loading a full 9B backbone.
- Full training now uses a multimodal objective: text backbone + vision projector + action head.
- Step1 artifacts are required by default (`mapping_table_path` and local tokenizer directory).
- `strict_action_id_mapping=true` prevents silent fallback when Step1 action remapping is missing.
- Eval leakage is blocked by default via holdout split when train/eval split are identical.
- Model artifacts are written under `result/step2/full-train` by default.
- Run-level analysis is saved to `result/step2/full-train/model_analysis.json`.
- Tokenized train/eval datasets can be saved with `dataset.save_dataset=true` and reused with `dataset.use_saved_dataset=true`.

Reuse saved dataset (after the first run):

python scripts/train_step2_full.py --config configs/step2.full-train.hf.reuse.json
- Training graphs can be enabled with `runtime.report_to` in config (default: `tensorboard`).
- TensorBoard event logs are written to `runtime.logging_dir` (default: `result/step2/full-train/tb`).

### Monitor graphs on RunPod (phone-friendly)

1) Start training with log file:

python scripts/train_step2_full.py --config configs/step2.full-train.hf.json 2>&1 | tee train.log

2) TensorBoard graph view (recommended first):

pip install tensorboard
tensorboard --logdir result/step2/full-train/tb --host 0.0.0.0 --port 6006

- Open RunPod's exposed HTTP endpoint for port 6006 in your phone browser.
- You can keep CLI logs open in parallel with `tail -f train.log`.

3) Optional: Weights & Biases mobile dashboard:

pip install wandb
wandb login

- Change config `runtime.report_to` to `["wandb"]` or `["tensorboard", "wandb"]`.
- Then open the W&B run page on your phone.

## Step 3 quick start (mock)

Run Step 3 Minecraft integration dry run with Python mock client:

python scripts/run_step3.py --config configs/step3.mock.json

## Step 3 tcp dry run (no real network call)

Run Step 3 using tcp mode with `dry_run=true`:

python scripts/run_step3.py --config configs/step3.tcp-dryrun.json
