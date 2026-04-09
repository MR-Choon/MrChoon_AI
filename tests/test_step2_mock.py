from __future__ import annotations

import json
from pathlib import Path

from minecraft_vla.config import load_step2_config
from minecraft_vla.training.step2_pipeline import run_step2_pipeline



def test_step2_mock_pipeline(tmp_path: Path) -> None:
    config = load_step2_config("configs/step2.mock.json")
    config.output_dir = str(tmp_path / "step2")

    report = run_step2_pipeline(config)

    assert report["backend"] == "mock"
    assert report["sample_count"] > 0
    assert report["action_token_count"] > 0
    assert report["forward"]["logits_size"] == config.action_head.num_actions

    report_path = Path(config.output_dir) / "step2_report.json"
    assert report_path.exists()

    loaded = json.loads(report_path.read_text(encoding="utf-8"))
    assert loaded["run_name"] == config.run_name
