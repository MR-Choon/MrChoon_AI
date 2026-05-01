from __future__ import annotations

import json
import random
import types
from pathlib import Path
from typing import Any, Dict, List, Sequence

from minecraft_vla.config import (
    Step2TrainConfig,
    Step2TrainDatasetConfig,
    Step2TrainLoraConfig,
    Step2TrainModelConfig,
    Step2TrainRuntimeConfig,
)
import minecraft_vla.training.step2_full_train_pipeline as full_train


class TinyDataset:
    def __init__(self, rows: Sequence[Dict[str, Any]]) -> None:
        self.rows = [dict(r) for r in rows]

    @property
    def column_names(self) -> List[str]:
        if not self.rows:
            return []
        return list(self.rows[0].keys())

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            return [row[item] for row in self.rows]
        return self.rows[item]

    def select(self, indices: Sequence[int]) -> "TinyDataset":
        return TinyDataset([self.rows[i] for i in indices if 0 <= i < len(self.rows)])

    def map(self, fn: Any, remove_columns: Sequence[str] | None = None, desc: str | None = None) -> "TinyDataset":
        mapped = [fn(dict(row)) for row in self.rows]
        return TinyDataset(mapped)

    def filter(self, fn: Any, desc: str | None = None) -> "TinyDataset":
        return TinyDataset([row for row in self.rows if fn(row)])

    def train_test_split(self, test_size: float, shuffle: bool, seed: int) -> Dict[str, "TinyDataset"]:
        rows = list(self.rows)
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(rows)

        n_test = max(1, int(len(rows) * float(test_size)))
        train_rows = rows[:-n_test] if len(rows) > n_test else rows[:1]
        test_rows = rows[-n_test:]
        return {"train": TinyDataset(train_rows), "test": TinyDataset(test_rows)}


class FakeTokenizer:
    def __init__(self) -> None:
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"

    def __call__(self, text: str, truncation: bool, max_length: int, padding: bool = False) -> Dict[str, List[int]]:
        ids = [((ord(ch) % 23) + 1) for ch in text][:max_length]
        if not ids:
            ids = [1]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}

    def pad(self, text_features: Sequence[Dict[str, Any]], padding: bool, return_tensors: str) -> Dict[str, Any]:
        import torch

        max_len = max(len(item["input_ids"]) for item in text_features)
        all_ids: List[List[int]] = []
        all_masks: List[List[int]] = []

        for item in text_features:
            ids = list(item["input_ids"])
            masks = list(item.get("attention_mask", [1] * len(ids)))
            pad_len = max_len - len(ids)
            all_ids.append(ids + [0] * pad_len)
            all_masks.append(masks + [0] * pad_len)

        return {
            "input_ids": torch.tensor(all_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_masks, dtype=torch.long),
        }

    def save_pretrained(self, save_dir: str | Path) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "tokenizer.json").write_text("{}", encoding="utf-8")


class FakeBackbone:  # small torch module with hidden_states output
    def __init__(self) -> None:
        import torch.nn as nn

        self.config = types.SimpleNamespace(hidden_size=8, use_cache=False)
        self.module = nn.Module()
        self.module.embedding = nn.Embedding(256, 8)

    def __call__(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ) -> Any:
        hidden = self.module.embedding(input_ids)
        return types.SimpleNamespace(hidden_states=(hidden,))

    def parameters(self):
        return self.module.parameters()

    def save_pretrained(self, save_dir: str | Path) -> None:
        import torch

        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.module.state_dict(), path / "fake_backbone.pt")


class FakeTrainingArguments:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = dict(kwargs)


class FakeTrainerState:
    def save_to_json(self, path: str) -> None:
        Path(path).write_text(json.dumps({"ok": True}), encoding="utf-8")


class FakeTrainer:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.state = FakeTrainerState()

    def train(self) -> Any:
        return types.SimpleNamespace(metrics={"train_loss": 0.42, "train_runtime": 0.1})

    def evaluate(self) -> Dict[str, float]:
        return {"eval_loss": 0.31, "eval_accuracy": 0.77}



def test_full_train_pipeline_integration(monkeypatch: Any, tmp_path: Path) -> None:
    rows = [
        {"instruction": "collect wood", "action_token_ids": [4, 5], "vision_features": [0.1, 0.2, 0.3]},
        {"instruction": "craft plank", "action_token_ids": [5, 4], "vision_features": [0.3, 0.2, 0.1]},
        {"instruction": "move forward", "action_token_ids": [4], "vision_features": [0.5, 0.6, 0.7]},
        {"instruction": "jump", "action_token_ids": [5], "vision_features": [0.9, 0.2, 0.8]},
    ]

    train_ds = TinyDataset(rows)
    eval_ds = TinyDataset(rows[:2])

    config = Step2TrainConfig(
        run_name="integration_test",
        seed=42,
        dry_run=True,
        dataset=Step2TrainDatasetConfig(
            dataset_id="dummy/ds",
            train_split="train",
            eval_split="train",
            max_train_samples=16,
            max_eval_samples=4,
            text_field_candidates=["instruction"],
            action_field_candidates=["action_token_ids"],
            vision_field_candidates=["vision_features"],
            vision_feature_dim=3,
            max_length=64,
            mapping_table_path="",
            eval_holdout_ratio=0.2,
            enforce_distinct_eval_split=True,
            require_step1_artifacts=False,
            strict_action_id_mapping=True,
        ),
        model=Step2TrainModelConfig(
            base_model_id="base",
            tokenizer_id="tok",
            trust_remote_code=True,
            use_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="bfloat16",
            bnb_4bit_use_double_quant=True,
            dry_run_model_id="tiny",
        ),
        lora=Step2TrainLoraConfig(
            enabled=False,
            r=8,
            alpha=16,
            dropout=0.0,
            target_modules=["q_proj"],
            bias="none",
        ),
        runtime=Step2TrainRuntimeConfig(
            output_dir=str(tmp_path / "result"),
            num_train_epochs=1.0,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            weight_decay=0.0,
            warmup_ratio=0.0,
            lr_scheduler_type="cosine",
            logging_steps=1,
            save_steps=5,
            eval_steps=5,
            max_grad_norm=1.0,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            max_steps=2,
            report_to=["tensorboard"],
            logging_dir=str(tmp_path / "result" / "tb"),
        ),
    )

    monkeypatch.setattr(full_train, "_require_hf_training_dependencies", lambda: None)
    monkeypatch.setattr(full_train, "_prepare_raw_splits", lambda _cfg: (train_ds, eval_ds))
    monkeypatch.setattr(full_train, "load_action_token_mapping", lambda _p: {4: "<ACT_MOVE>", 5: "<ACT_JUMP>"})
    monkeypatch.setattr(full_train, "load_action_id_mapping", lambda _p: {4: 40, 5: 50})
    monkeypatch.setattr(full_train, "_load_tokenizer", lambda _cfg: FakeTokenizer())
    monkeypatch.setattr(full_train, "_resolve_quantization_config_for_runtime", lambda _cfg: (None, False, ["q_off"]))
    monkeypatch.setattr(full_train, "_resolve_precision_flags", lambda _cfg: (False, False, ["bf16_off"]))
    monkeypatch.setattr(full_train, "_load_text_backbone", lambda _cfg, _q: FakeBackbone())
    monkeypatch.setattr(full_train, "_apply_lora_if_enabled", lambda model, _cfg: model)
    monkeypatch.setattr(full_train, "_get_trainer_components", lambda: (FakeTrainer, FakeTrainingArguments))

    report = full_train.run_step2_full_train_pipeline(config)

    out_dir = Path(config.runtime.output_dir)
    assert report["dataset"]["num_action_classes"] == 2
    assert "runtime_warnings" in report
    assert report["logging"]["report_to"] == ["tensorboard"]
    assert Path(report["logging"]["logging_dir"]).name == "tb"
    assert (out_dir / "step2_full_train_report.json").exists()
    assert (out_dir / "step2_full_train_config_used.json").exists()
    assert (out_dir / "model_analysis.json").exists()
    assert (out_dir / "model_bundle" / "pytorch_model.bin").exists()
    assert (out_dir / "action_label_map.json").exists()
    assert (out_dir / "vla_head_state.pt").exists()
