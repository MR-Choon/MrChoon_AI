#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from minecraft_vla.training.step2_full_train_pipeline import run_step2_full_train_pipeline_from_file



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step2 full QLoRA training pipeline")
    parser.add_argument("--config", required=True, help="Path to full-train config json")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    report = run_step2_full_train_pipeline_from_file(args.config)
    print("[STEP2-FULL-TRAIN] Completed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
