from __future__ import annotations

import json
from pathlib import Path

from minecraft_vla.config import load_step3_config
from minecraft_vla.training.step3_pipeline import run_step3_pipeline



def test_step3_mock_pipeline(tmp_path: Path) -> None:
    fake_step2 = {
        "model": {
            "num_actions": 32,
        }
    }
    step2_report_path = tmp_path / "step2_report.json"
    step2_report_path.write_text(json.dumps(fake_step2), encoding="utf-8")

    config = load_step3_config("configs/step3.mock.json")
    config.output_dir = str(tmp_path / "step3")
    config.policy.step2_report_path = str(step2_report_path)

    report = run_step3_pipeline(config)

    assert report["server"]["mode"] == "mock"
    assert report["evaluation"]["total_steps"] == config.evaluation.episodes * config.evaluation.steps_per_episode
    assert report["policy"]["num_actions"] == 32

    out_dir = Path(config.output_dir)
    assert (out_dir / "step3_report.json").exists()
    assert (out_dir / "step3_trace.jsonl").exists()
