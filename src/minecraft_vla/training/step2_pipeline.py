from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

from minecraft_vla.config import Step2Config, load_step2_config
from minecraft_vla.data.minecraft_sft import extract_action_token_ids, load_samples
from minecraft_vla.models.vla_model import MockVLABatch, build_vla_model
from minecraft_vla.training.qlora import apply_qlora
from minecraft_vla.utils.io import ensure_dir, write_json
from minecraft_vla.utils.seed import set_seed



def _build_mock_batch(config: Step2Config, action_ids: List[int]) -> MockVLABatch:
    input_ids = action_ids[:8] if action_ids else [0, 0, 0, 0]
    labels = [input_ids[0] if input_ids else 0]

    vision_features = []
    for i in range(config.vision_encoder.input_dim):
        value = float((i % 7) - 3) / 3.0
        vision_features.append(value)

    return MockVLABatch(
        input_ids=input_ids,
        vision_features=vision_features,
        labels=labels,
    )



def _run_mock_forward(model: Any, batch: MockVLABatch) -> Dict[str, Any]:
    out = model.forward(batch)
    return {
        "loss": float(out["loss"]),
        "logits_size": len(out["logits"]),
        "label": int(out["label"]),
    }



def _run_hf_forward(model: Any, config: Step2Config, action_ids: List[int]) -> Dict[str, Any]:  # pragma: no cover
    import torch  # type: ignore

    ids = action_ids[:8] if action_ids else [0, 0, 0, 0]
    labels = [ids[0] % config.action_head.num_actions]

    input_ids = torch.tensor([ids], dtype=torch.long)
    vision = torch.zeros((1, config.vision_encoder.input_dim), dtype=torch.float)
    label_tensor = torch.tensor(labels, dtype=torch.long)

    out = model.forward(input_ids=input_ids, vision_inputs=vision, labels=label_tensor)
    loss_value = out.get("loss")
    if loss_value is None:
        loss_float = 0.0
    else:
        loss_float = float(loss_value.detach().cpu().item())

    logits = out["logits"]
    return {
        "loss": loss_float,
        "logits_size": int(logits.shape[-1]),
        "label": int(labels[0]),
    }



def run_step2_pipeline(config: Step2Config) -> Dict[str, Any]:
    set_seed(config.seed)
    out_dir = ensure_dir(config.output_dir)

    samples = load_samples(
        backend=config.backend,
        dataset_id=config.dataset.dataset_id,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
    )
    action_ids, sample_count = extract_action_token_ids(samples)

    model, model_meta = build_vla_model(config)
    adapted_model, qlora_info = apply_qlora(model, config.qlora, config.backend)

    if config.backend == "mock":
        batch = _build_mock_batch(config, action_ids)
        forward_result = _run_mock_forward(adapted_model, batch)
    else:
        forward_result = _run_hf_forward(adapted_model, config, action_ids)

    adapter_dir = Path(out_dir) / "lora_adapter"
    if hasattr(adapted_model, "save_pretrained"):
        adapted_model.save_pretrained(adapter_dir)

    report = {
        "run_name": config.run_name,
        "backend": config.backend,
        "dataset_id": config.dataset.dataset_id,
        "sample_count": sample_count,
        "action_token_count": len(action_ids),
        "model": model_meta,
        "qlora": asdict(qlora_info),
        "forward": forward_result,
        "artifacts": {
            "output_dir": str(out_dir),
            "adapter_dir": str(adapter_dir),
        },
    }

    write_json(Path(out_dir) / "step2_report.json", report)
    write_json(Path(out_dir) / "step2_config_used.json", json.loads(json.dumps(asdict(config))))

    return report



def run_step2_pipeline_from_file(config_path: str | Path) -> Dict[str, Any]:
    config = load_step2_config(config_path)
    return run_step2_pipeline(config)
