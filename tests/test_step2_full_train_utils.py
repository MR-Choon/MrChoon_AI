from __future__ import annotations

from minecraft_vla.config import (
    Step2TrainConfig,
    Step2TrainDatasetConfig,
    Step2TrainLoraConfig,
    Step2TrainModelConfig,
    Step2TrainRuntimeConfig,
)
from minecraft_vla.training.step2_full_train_pipeline import (
    build_model_analysis,
    build_training_text,
    collect_model_parameter_stats,
    load_action_id_mapping,
    load_action_token_mapping,
)



def test_build_training_text_with_action_mapping() -> None:
    row = {
        "instruction": "collect wood",
        "action_token_ids": [4, 5, 6],
    }
    mapping = {4: "<ACT_MOVE_FORWARD>", 5: "<ACT_JUMP>"}

    text = build_training_text(
        row=row,
        text_fields=["instruction"],
        action_fields=["action_token_ids"],
        id_to_token=mapping,
    )

    assert "collect wood" in text
    assert "<ACT_MOVE_FORWARD>" in text
    assert "<ACT_JUMP>" in text
    assert "<ACT_6>" in text



def test_load_action_token_mapping_missing_file() -> None:
    mapping = load_action_token_mapping("/tmp/not_existing_mapping_file.csv")
    assert mapping == {}


def test_load_action_id_mapping_missing_file() -> None:
    mapping = load_action_id_mapping("/tmp/not_existing_mapping_file.csv")
    assert mapping == {}


class _DummyParam:
    def __init__(self, n: int, requires_grad: bool) -> None:
        self._n = n
        self.requires_grad = requires_grad

    def numel(self) -> int:
        return self._n


class _DummyModel:
    def parameters(self):
        return [_DummyParam(10, True), _DummyParam(30, False)]


def test_collect_model_parameter_stats() -> None:
    stats = collect_model_parameter_stats(_DummyModel())
    assert stats["total_params"] == 40
    assert stats["trainable_params"] == 10
    assert abs(stats["trainable_ratio"] - 0.25) < 1e-9


def test_build_model_analysis() -> None:
    config = Step2TrainConfig(
        run_name="test_run",
        seed=42,
        dry_run=True,
        dataset=Step2TrainDatasetConfig(
            dataset_id="dummy/ds",
            train_split="train",
            eval_split="eval",
            max_train_samples=10,
            max_eval_samples=2,
            text_field_candidates=["text"],
            action_field_candidates=["action_token_ids"],
            vision_field_candidates=["vision_features"],
            vision_feature_dim=8,
            max_length=128,
            mapping_table_path="",
            eval_holdout_ratio=0.1,
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
        ),
        lora=Step2TrainLoraConfig(
            enabled=True,
            r=16,
            alpha=32,
            dropout=0.05,
            target_modules=["q_proj"],
            bias="none",
        ),
        runtime=Step2TrainRuntimeConfig(
            output_dir="result/step2/full-train",
            num_train_epochs=1.0,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            weight_decay=0.0,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            logging_steps=5,
            save_steps=10,
            eval_steps=10,
            max_grad_norm=1.0,
            fp16=False,
            bf16=True,
            gradient_checkpointing=True,
            max_steps=5,
            report_to=["tensorboard"],
            logging_dir="result/step2/full-train/tb",
        ),
    )

    analysis = build_model_analysis(
        config=config,
        train_metrics={"train_loss": 0.9},
        eval_metrics={"eval_loss": 1.1},
        model_stats={"total_params": 40, "trainable_params": 10, "trainable_ratio": 0.25},
    )

    assert analysis["run_name"] == "test_run"
    assert analysis["metrics"]["train_loss"] == 0.9
    assert analysis["metrics"]["eval_loss"] == 1.1
    assert analysis["parameter_stats"]["trainable_params"] == 10
