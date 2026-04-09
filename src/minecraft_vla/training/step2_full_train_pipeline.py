from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from minecraft_vla.config import Step2TrainConfig, load_step2_train_config
from minecraft_vla.utils.io import ensure_dir, write_json
from minecraft_vla.utils.seed import set_seed


def _require_hf_training_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError(
            "Missing training dependencies. Install with: pip install -e .[hf]"
        ) from exc



def _resolve_dtype(name: str) -> Any:  # pragma: no cover - requires torch runtime
    import torch  # type: ignore

    table = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    lowered = str(name).strip().lower()
    return table.get(lowered, torch.bfloat16)



def _safe_get(row: Dict[str, Any], path: str) -> Any:
    node: Any = row
    for key in path.split("."):
        if isinstance(node, dict) and key in node:
            node = node[key]
        else:
            return None
    return node



def _pick_first_string(row: Dict[str, Any], candidates: Sequence[str]) -> str:
    for key in candidates:
        value = _safe_get(row, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    for value in row.values():
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""



def _pick_action_values(row: Dict[str, Any], candidates: Sequence[str]) -> List[Any]:
    for key in candidates:
        value = _safe_get(row, key)
        if isinstance(value, list):
            return value
    return []



def load_action_token_mapping(mapping_table_path: str) -> Dict[int, str]:
    path = Path(mapping_table_path)
    if not mapping_table_path or not path.exists():
        return {}

    result: Dict[int, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                old_id = int(row.get("old_id", ""))
                token_str = str(row.get("token_str", ""))
            except ValueError:
                continue
            if token_str:
                result[old_id] = token_str
    return result



def _stringify_action_values(action_values: Sequence[Any], id_to_token: Dict[int, str]) -> str:
    converted: List[str] = []
    for value in action_values:
        if isinstance(value, int):
            converted.append(id_to_token.get(value, f"<ACT_{value}>"))
        elif isinstance(value, str):
            converted.append(value)
        else:
            converted.append(str(value))
    return " ".join(converted)



def build_training_text(
    row: Dict[str, Any],
    text_fields: Sequence[str],
    action_fields: Sequence[str],
    id_to_token: Dict[int, str],
) -> str:
    text = _pick_first_string(row, text_fields)
    action_values = _pick_action_values(row, action_fields)

    if not text and not action_values:
        return json.dumps(row, ensure_ascii=False)

    action_text = _stringify_action_values(action_values, id_to_token)
    if action_text:
        return (
            "<observation>\n"
            f"{text}\n"
            "</observation>\n"
            "<action>\n"
            f"{action_text}\n"
            "</action>"
        )
    return text



def _truncate_dataset(ds: Any, max_samples: int) -> Any:
    if max_samples <= 0:
        return ds
    size = len(ds)
    take = min(size, int(max_samples))
    if take <= 0:
        return ds
    return ds.select(range(take))



def _build_quantization_config(config: Step2TrainConfig) -> Optional[Any]:  # pragma: no cover
    if not config.model.use_4bit:
        return None

    from transformers import BitsAndBytesConfig  # type: ignore

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.model.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=_resolve_dtype(config.model.bnb_4bit_compute_dtype),
        bnb_4bit_use_double_quant=config.model.bnb_4bit_use_double_quant,
    )



def _load_tokenizer(config: Step2TrainConfig) -> Any:  # pragma: no cover
    from transformers import AutoTokenizer  # type: ignore

    tokenizer_id = config.model.tokenizer_id or config.model.base_model_id
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=config.model.trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def _load_model(config: Step2TrainConfig, quantization_config: Optional[Any]) -> Any:  # pragma: no cover
    from transformers import AutoModelForCausalLM  # type: ignore

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "device_map": "auto",
    }

    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = _resolve_dtype(
            "bfloat16" if config.runtime.bf16 else "float16" if config.runtime.fp16 else "float32"
        )

    model = AutoModelForCausalLM.from_pretrained(config.model.base_model_id, **model_kwargs)
    if config.runtime.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model



def _apply_lora_if_enabled(model: Any, config: Step2TrainConfig) -> Any:  # pragma: no cover
    if not config.lora.enabled:
        return model

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore

    if config.model.use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
    )
    model = get_peft_model(model, lora_config)
    return model



def _tokenize_record(
    row: Dict[str, Any],
    tokenizer: Any,
    max_length: int,
    text_fields: Sequence[str],
    action_fields: Sequence[str],
    id_to_token: Dict[int, str],
) -> Dict[str, Any]:
    text = build_training_text(row, text_fields, action_fields, id_to_token)
    encoded = tokenizer(text, truncation=True, max_length=max_length, padding=False)
    encoded["labels"] = list(encoded["input_ids"])
    return encoded



def _prepare_datasets(config: Step2TrainConfig, tokenizer: Any, id_to_token: Dict[int, str]) -> Dict[str, Any]:  # pragma: no cover
    from datasets import load_dataset  # type: ignore

    train_ds = load_dataset(config.dataset.dataset_id, split=config.dataset.train_split)
    eval_ds = load_dataset(config.dataset.dataset_id, split=config.dataset.eval_split)

    max_train = config.dataset.max_train_samples
    max_eval = config.dataset.max_eval_samples
    if config.dry_run:
        max_train = min(max_train if max_train > 0 else 256, 256)
        max_eval = min(max_eval if max_eval > 0 else 64, 64)

    train_ds = _truncate_dataset(train_ds, max_train)
    eval_ds = _truncate_dataset(eval_ds, max_eval)

    remove_columns_train = list(train_ds.column_names)
    remove_columns_eval = list(eval_ds.column_names)

    train_ds = train_ds.map(
        lambda row: _tokenize_record(
            row=row,
            tokenizer=tokenizer,
            max_length=config.dataset.max_length,
            text_fields=config.dataset.text_field_candidates,
            action_fields=config.dataset.action_field_candidates,
            id_to_token=id_to_token,
        ),
        remove_columns=remove_columns_train,
        desc="Tokenizing train split",
    )

    eval_ds = eval_ds.map(
        lambda row: _tokenize_record(
            row=row,
            tokenizer=tokenizer,
            max_length=config.dataset.max_length,
            text_fields=config.dataset.text_field_candidates,
            action_fields=config.dataset.action_field_candidates,
            id_to_token=id_to_token,
        ),
        remove_columns=remove_columns_eval,
        desc="Tokenizing eval split",
    )

    return {"train": train_ds, "eval": eval_ds}


def collect_model_parameter_stats(model: Any) -> Dict[str, Any]:
    total_params = 0
    trainable_params = 0

    parameters_fn = getattr(model, "parameters", None)
    if callable(parameters_fn):
        for param in parameters_fn():
            numel_fn = getattr(param, "numel", None)
            if not callable(numel_fn):
                continue
            count = int(numel_fn())
            total_params += count
            if bool(getattr(param, "requires_grad", False)):
                trainable_params += count

    ratio = float(trainable_params / total_params) if total_params > 0 else 0.0
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "trainable_ratio": ratio,
    }


def build_model_analysis(
    config: Step2TrainConfig,
    train_metrics: Dict[str, Any],
    eval_metrics: Dict[str, Any],
    model_stats: Dict[str, Any],
) -> Dict[str, Any]:
    train_loss = train_metrics.get("train_loss")
    eval_loss = eval_metrics.get("eval_loss")

    if isinstance(eval_loss, (int, float)):
        perplexity = float(math.exp(float(eval_loss))) if float(eval_loss) < 20 else float("inf")
    else:
        perplexity = None

    quality_flags: List[str] = []
    if isinstance(train_loss, (int, float)) and float(train_loss) > 5.0:
        quality_flags.append("high_train_loss")
    if isinstance(eval_loss, (int, float)) and float(eval_loss) > 5.0:
        quality_flags.append("high_eval_loss")
    if model_stats.get("trainable_ratio", 0.0) <= 0.0:
        quality_flags.append("no_trainable_parameters")

    return {
        "run_name": config.run_name,
        "model": {
            "base_model_id": config.model.base_model_id,
            "use_4bit": config.model.use_4bit,
            "lora_enabled": config.lora.enabled,
            "lora_target_modules": config.lora.target_modules,
        },
        "dataset": {
            "dataset_id": config.dataset.dataset_id,
            "train_split": config.dataset.train_split,
            "eval_split": config.dataset.eval_split,
        },
        "metrics": {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_perplexity": perplexity,
        },
        "parameter_stats": model_stats,
        "quality_flags": quality_flags,
        "notes": [
            "This report is generated after training and is intended for quick run-level diagnostics.",
            "Use trainer_state.json and checkpoints for detailed step-by-step analysis.",
        ],
    }



def run_step2_full_train_pipeline(config: Step2TrainConfig) -> Dict[str, Any]:  # pragma: no cover
    _require_hf_training_dependencies()
    set_seed(config.seed)

    out_dir = ensure_dir(config.runtime.output_dir)
    id_to_token = load_action_token_mapping(config.dataset.mapping_table_path)

    tokenizer = _load_tokenizer(config)
    quantization_config = _build_quantization_config(config)
    model = _load_model(config, quantization_config)
    model = _apply_lora_if_enabled(model, config)

    datasets = _prepare_datasets(config, tokenizer, id_to_token)
    train_ds = datasets["train"]
    eval_ds = datasets["eval"]

    from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments  # type: ignore

    evaluation_strategy = "steps" if len(eval_ds) > 0 else "no"
    max_steps = config.runtime.max_steps
    if config.dry_run and max_steps <= 0:
        max_steps = 10

    args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=config.run_name,
        num_train_epochs=config.runtime.num_train_epochs,
        per_device_train_batch_size=config.runtime.per_device_train_batch_size,
        per_device_eval_batch_size=config.runtime.per_device_eval_batch_size,
        gradient_accumulation_steps=config.runtime.gradient_accumulation_steps,
        learning_rate=config.runtime.learning_rate,
        weight_decay=config.runtime.weight_decay,
        warmup_ratio=config.runtime.warmup_ratio,
        lr_scheduler_type=config.runtime.lr_scheduler_type,
        logging_steps=config.runtime.logging_steps,
        save_steps=config.runtime.save_steps,
        eval_steps=config.runtime.eval_steps,
        max_grad_norm=config.runtime.max_grad_norm,
        fp16=config.runtime.fp16,
        bf16=config.runtime.bf16,
        gradient_checkpointing=config.runtime.gradient_checkpointing,
        evaluation_strategy=evaluation_strategy,
        save_strategy="steps",
        max_steps=max_steps,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate() if evaluation_strategy == "steps" else {}
    model_stats = collect_model_parameter_stats(model)

    tokenizer_dir = Path(out_dir) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    if config.lora.enabled and hasattr(model, "save_pretrained"):
        adapter_dir = Path(out_dir) / "lora_adapter"
        model.save_pretrained(adapter_dir)
        model_artifact = str(adapter_dir)
    else:
        model_dir = Path(out_dir) / "full_model"
        trainer.save_model(model_dir)
        model_artifact = str(model_dir)

    trainer.state.save_to_json(str(Path(out_dir) / "trainer_state.json"))

    analysis = build_model_analysis(
        config=config,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
        model_stats=model_stats,
    )
    write_json(Path(out_dir) / "model_analysis.json", analysis)

    report = {
        "run_name": config.run_name,
        "dry_run": config.dry_run,
        "dataset": {
            "dataset_id": config.dataset.dataset_id,
            "train_split": config.dataset.train_split,
            "eval_split": config.dataset.eval_split,
            "train_size": len(train_ds),
            "eval_size": len(eval_ds),
        },
        "model": {
            "base_model_id": config.model.base_model_id,
            "tokenizer_id": config.model.tokenizer_id,
            "use_4bit": config.model.use_4bit,
            "lora_enabled": config.lora.enabled,
            "parameter_stats": model_stats,
        },
        "metrics": {
            "train": train_metrics,
            "eval": eval_metrics,
        },
        "artifacts": {
            "output_dir": str(out_dir),
            "model_artifact": model_artifact,
            "tokenizer_dir": str(tokenizer_dir),
            "trainer_state": str(Path(out_dir) / "trainer_state.json"),
            "model_analysis": str(Path(out_dir) / "model_analysis.json"),
        },
    }

    write_json(Path(out_dir) / "step2_full_train_report.json", report)
    write_json(Path(out_dir) / "step2_full_train_config_used.json", json.loads(json.dumps(asdict(config))))
    return report



def run_step2_full_train_pipeline_from_file(config_path: str | Path) -> Dict[str, Any]:
    config = load_step2_train_config(config_path)
    return run_step2_full_train_pipeline(config)
