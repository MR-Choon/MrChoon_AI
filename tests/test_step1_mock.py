from __future__ import annotations

from pathlib import Path

from minecraft_vla.config import load_step1_config
from minecraft_vla.training.step1_pipeline import run_step1_pipeline



def test_step1_mock_pipeline(tmp_path: Path) -> None:
    config = load_step1_config("configs/step1.mock.json")
    config.output_dir = str(tmp_path / "step1")

    result = run_step1_pipeline(config)

    summary = result["summary"]
    dryrun = result["dryrun"]

    assert summary["backend"] == "mock"
    assert summary["sample_count"] > 0
    assert summary["mapping_count"] > 0
    assert summary["unmapped_count"] == 0
    assert dryrun["embedding_after"] >= dryrun["embedding_before"]

    assert Path(config.output_dir, "token_id_mapping.csv").exists()
    assert Path(config.output_dir, "dryrun_report.json").exists()
