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

from minecraft_vla.training.step1_pipeline import run_step1_pipeline_from_file



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step1 tokenizer compatibility pipeline")
    parser.add_argument("--config", required=True, help="Path to Step1 config json")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    result = run_step1_pipeline_from_file(args.config)
    print("[STEP1] Completed")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
