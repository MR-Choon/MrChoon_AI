from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class Step1DatasetConfig:
    dataset_id: str
    split: str
    max_samples: int


@dataclass
class Step1Config:
    run_name: str
    backend: str
    seed: int
    output_dir: str
    source_tokenizer_id: str
    target_tokenizer_id: str
    dataset: Step1DatasetConfig
    add_missing_tokens: bool

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step1Config":
        backend = str(data["backend"]).lower()
        if backend not in {"mock", "hf"}:
            raise ValueError(f"Unsupported backend: {backend}")

        return cls(
            run_name=str(data["run_name"]),
            backend=backend,
            seed=int(data.get("seed", 42)),
            output_dir=str(data["output_dir"]),
            source_tokenizer_id=str(data["source_tokenizer_id"]),
            target_tokenizer_id=str(data["target_tokenizer_id"]),
            dataset=Step1DatasetConfig(**data["dataset"]),
            add_missing_tokens=bool(data.get("add_missing_tokens", True)),
        )


@dataclass
class DatasetConfig:
    dataset_id: str
    split: str
    max_samples: int
    action_key_hints: List[str]


@dataclass
class TextBackboneConfig:
    model_id: str
    hidden_size: int
    vocab_size: int


@dataclass
class VisionEncoderConfig:
    name: str
    input_dim: int
    hidden_size: int


@dataclass
class ActionHeadConfig:
    num_actions: int


@dataclass
class QLoRAConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]


@dataclass
class Step2Config:
    run_name: str
    backend: str
    seed: int
    output_dir: str
    dataset: DatasetConfig
    text_backbone: TextBackboneConfig
    vision_encoder: VisionEncoderConfig
    action_head: ActionHeadConfig
    qlora: QLoRAConfig


    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step2Config":
        backend = str(data["backend"]).lower()
        if backend not in {"mock", "hf"}:
            raise ValueError(f"Unsupported backend: {backend}")

        return cls(
            run_name=str(data["run_name"]),
            backend=backend,
            seed=int(data.get("seed", 42)),
            output_dir=str(data["output_dir"]),
            dataset=DatasetConfig(**data["dataset"]),
            text_backbone=TextBackboneConfig(**data["text_backbone"]),
            vision_encoder=VisionEncoderConfig(**data["vision_encoder"]),
            action_head=ActionHeadConfig(**data["action_head"]),
            qlora=QLoRAConfig(**data["qlora"]),
        )


@dataclass
class Step2TrainDatasetConfig:
    dataset_id: str
    train_split: str
    eval_split: str
    max_train_samples: int
    max_eval_samples: int
    text_field_candidates: List[str]
    action_field_candidates: List[str]
    max_length: int
    mapping_table_path: str


@dataclass
class Step2TrainModelConfig:
    base_model_id: str
    tokenizer_id: str
    trust_remote_code: bool
    use_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_compute_dtype: str
    bnb_4bit_use_double_quant: bool


@dataclass
class Step2TrainLoraConfig:
    enabled: bool
    r: int
    alpha: int
    dropout: float
    target_modules: List[str]
    bias: str


@dataclass
class Step2TrainRuntimeConfig:
    output_dir: str
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    logging_steps: int
    save_steps: int
    eval_steps: int
    max_grad_norm: float
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    max_steps: int


@dataclass
class Step2TrainConfig:
    run_name: str
    seed: int
    dry_run: bool
    dataset: Step2TrainDatasetConfig
    model: Step2TrainModelConfig
    lora: Step2TrainLoraConfig
    runtime: Step2TrainRuntimeConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step2TrainConfig":
        return cls(
            run_name=str(data["run_name"]),
            seed=int(data.get("seed", 42)),
            dry_run=bool(data.get("dry_run", False)),
            dataset=Step2TrainDatasetConfig(**data["dataset"]),
            model=Step2TrainModelConfig(**data["model"]),
            lora=Step2TrainLoraConfig(**data["lora"]),
            runtime=Step2TrainRuntimeConfig(**data["runtime"]),
        )


def load_step2_config(path: str | Path) -> Step2Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Step2Config.from_dict(data)


def load_step2_train_config(path: str | Path) -> Step2TrainConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Step2TrainConfig.from_dict(data)


def load_step1_config(path: str | Path) -> Step1Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Step1Config.from_dict(data)


@dataclass
class Step3ServerConfig:
    mode: str
    host: str
    port: int
    connect_timeout_sec: float
    read_timeout_sec: float
    username: str


@dataclass
class Step3PolicyConfig:
    source: str
    step2_report_path: str
    default_action_id: int
    max_action_id: int


@dataclass
class Step3EvalConfig:
    episodes: int
    steps_per_episode: int


@dataclass
class Step3Config:
    run_name: str
    seed: int
    output_dir: str
    dry_run: bool
    server: Step3ServerConfig
    policy: Step3PolicyConfig
    evaluation: Step3EvalConfig

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Step3Config":
        mode = str(data["server"]["mode"]).lower()
        if mode not in {"mock", "tcp"}:
            raise ValueError(f"Unsupported server mode: {mode}")

        source = str(data["policy"]["source"]).lower()
        if source not in {"step2_report", "fixed"}:
            raise ValueError(f"Unsupported policy source: {source}")

        server = Step3ServerConfig(**data["server"])
        server.mode = mode

        policy = Step3PolicyConfig(**data["policy"])
        policy.source = source

        return cls(
            run_name=str(data["run_name"]),
            seed=int(data.get("seed", 42)),
            output_dir=str(data["output_dir"]),
            dry_run=bool(data.get("dry_run", True)),
            server=server,
            policy=policy,
            evaluation=Step3EvalConfig(**data["evaluation"]),
        )


def load_step3_config(path: str | Path) -> Step3Config:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return Step3Config.from_dict(data)
