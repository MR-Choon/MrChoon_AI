from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from minecraft_vla.config import Step2TrainConfig, load_step2_train_config
from minecraft_vla.utils.io import ensure_dir, write_json
from minecraft_vla.utils.seed import set_seed

ACTION_TOKEN_PATTERN = re.compile(r"<\|reserved_special_token_\d+\|>")


try:  # pragma: no cover - import fallback for environments without torch
    import torch.nn as _torch_nn  # type: ignore

    _BaseTorchModule = _torch_nn.Module
except Exception:  # pragma: no cover
    class _BaseTorchModule:  # type: ignore
        pass


def _require_hf_training_dependencies() -> None:
    try:
        import datasets  # noqa: F401
        import peft  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise RuntimeError("Missing training dependencies. Install with: pip install -e .[hf]") from exc



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
    return table.get(str(name).strip().lower(), torch.bfloat16)



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

    tokens: List[str] = []
    conv = row.get("conversations", [])
    if isinstance(conv, list):
        for msg in conv:
            if not isinstance(msg, dict):
                continue
            if str(msg.get("role", "")).lower() != "assistant":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for part in content:
                if not isinstance(part, dict):
                    continue
                if part.get("type") != "text":
                    continue
                text = part.get("text") or ""
                if text:
                    tokens.extend(ACTION_TOKEN_PATTERN.findall(text))
    if tokens:
        return tokens
    return []



def load_action_token_mapping(mapping_table_path: str) -> Dict[int, str]:
    if not mapping_table_path:
        return {}

    mapping: Dict[int, str] = {}

    def _iter_rows(text: str):
        reader = csv.DictReader(text.splitlines())
        for row in reader:
            yield row

    if isinstance(mapping_table_path, str) and mapping_table_path.startswith("http"):
        try:
            import requests

            resp = requests.get(mapping_table_path, timeout=30)
            resp.raise_for_status()
            text = resp.text
        except Exception:
            return {}
        rows_iter = _iter_rows(text)
    else:
        path = Path(mapping_table_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            rows_iter = csv.DictReader(f)

    for row in rows_iter:
        try:
            old_id = int(row.get("old_id", ""))
        except (ValueError, TypeError):
            continue
        if old_id < 0:
            continue
        token_str = str(row.get("token_str", ""))
        if token_str:
            mapping[old_id] = token_str
    return mapping



def load_action_id_mapping(mapping_table_path: str) -> Dict[int, int]:
    if not mapping_table_path:
        return {}

    mapping: Dict[int, int] = {}

    if isinstance(mapping_table_path, str) and mapping_table_path.startswith("http"):
        try:
            import requests

            resp = requests.get(mapping_table_path, timeout=30)
            resp.raise_for_status()
            text = resp.text
            reader = csv.DictReader(text.splitlines())
        except Exception:
            return {}
    else:
        path = Path(mapping_table_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

    for row in reader:
        try:
            old_id = int(row.get("old_id", ""))
            mapped_new_id = int(row.get("mapped_new_id", ""))
        except (ValueError, TypeError):
            continue
        if old_id < 0:
            continue
        mapping[old_id] = mapped_new_id
    return mapping


def load_token_str_to_new_id(mapping_table_path: str) -> Dict[str, int]:
    if not mapping_table_path:
        return {}

    mapping: Dict[str, int] = {}

    if isinstance(mapping_table_path, str) and mapping_table_path.startswith("http"):
        try:
            import requests

            resp = requests.get(mapping_table_path, timeout=30)
            resp.raise_for_status()
            text = resp.text
            reader = csv.DictReader(text.splitlines())
        except Exception:
            return {}
    else:
        path = Path(mapping_table_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

    for row in reader:
        token_str = str(row.get("token_str", "")).strip()
        if not token_str:
            continue
        try:
            mapped_new_id = int(row.get("mapped_new_id", ""))
        except (ValueError, TypeError):
            continue
        mapping[token_str] = mapped_new_id
    return mapping



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
    take = min(len(ds), int(max_samples))
    if take <= 0:
        return ds
    return ds.select(range(take))



def _collect_numeric_values(value: Any, output: List[float], max_values: int = 4096) -> None:
    if len(output) >= max_values:
        return

    if isinstance(value, (int, float)):
        output.append(float(value))
        return

    if isinstance(value, list):
        for item in value:
            if len(output) >= max_values:
                break
            _collect_numeric_values(item, output, max_values=max_values)
        return

    if isinstance(value, dict):
        for item in value.values():
            if len(output) >= max_values:
                break
            _collect_numeric_values(item, output, max_values=max_values)



def _fit_feature_vector(values: Sequence[float], feature_dim: int) -> List[float]:
    vec = list(values[:feature_dim])
    if len(vec) < feature_dim:
        vec.extend([0.0] * (feature_dim - len(vec)))
    return vec



def _extract_vision_features(row: Dict[str, Any], candidates: Sequence[str], feature_dim: int) -> List[float]:
    feature_dim = max(1, int(feature_dim))

    for key in candidates:
        value = _safe_get(row, key)
        if value is None:
            continue
        nums: List[float] = []
        _collect_numeric_values(value, nums)
        if nums:
            return _fit_feature_vector(nums, feature_dim)

    # Fallback: deterministic minimal signal from text length.
    fallback = [0.0] * feature_dim
    text_hint = _pick_first_string(row, ["instruction", "prompt", "text", "obs", "observation"])
    if text_hint:
        fallback[0] = float(min(len(text_hint), 4096)) / 4096.0
    return fallback



def _extract_action_label(
    row: Dict[str, Any],
    action_fields: Sequence[str],
    old_to_new_action_id: Dict[int, int],
    token_str_to_new_id: Dict[str, int],
    strict_action_id_mapping: bool,
) -> Optional[int]:
    action_values = _pick_action_values(row, action_fields)
    for value in action_values:
        if isinstance(value, int):
            if strict_action_id_mapping:
                if value not in old_to_new_action_id:
                    return None
                return int(old_to_new_action_id[value])
            return int(old_to_new_action_id.get(value, value))
        if isinstance(value, str) and value.strip().isdigit():
            parsed = int(value.strip())
            if strict_action_id_mapping:
                if parsed not in old_to_new_action_id:
                    return None
                return int(old_to_new_action_id[parsed])
            return int(old_to_new_action_id.get(parsed, parsed))
        if isinstance(value, str):
            token_str = value.strip()
            if token_str in token_str_to_new_id:
                return int(token_str_to_new_id[token_str])
    return None



def _validate_step1_artifacts(config: Step2TrainConfig) -> None:
    if not config.dataset.require_step1_artifacts:
        return

    if not config.dataset.use_saved_dataset:
        mapping_path = config.dataset.mapping_table_path
        # allow remote mapping URLs (http/https) when running remotely
        if not (isinstance(mapping_path, str) and mapping_path.startswith("http")):
            mp = Path(mapping_path)
            if not mp.exists():
                raise RuntimeError(
                    "Step1 mapping table is required but missing. "
                    f"Expected: {mp}"
                )

    # Allow tokenizer to be a remote HF repo with optional subfolder indicated by 'repo:subfolder'
    tok_id = config.model.tokenizer_id
    if isinstance(tok_id, str) and tok_id.startswith("http"):
        # remote URL — assume available
        return
    if isinstance(tok_id, str) and ":" in tok_id:
        repo, sub = tok_id.split(":", 1)
        # do not require local existence for repo-based tokenizer
        return
    tokenizer_path = Path(tok_id)
    if not tokenizer_path.exists():
        raise RuntimeError(
            "Step1 tokenizer artifact is required. "
            "Set model.tokenizer_id to a local tokenizer directory produced by Step1, "
            "or to an HF repo with subfolder using the format 'repo:subfolder'. "
            f"Current: {config.model.tokenizer_id}"
        )


def _resolve_quantization_config_for_runtime(config: Step2TrainConfig) -> Tuple[Optional[Any], bool, List[str]]:  # pragma: no cover
    warnings: List[str] = []
    if not config.model.use_4bit:
        return None, False, warnings

    import torch  # type: ignore

    if not torch.cuda.is_available():
        warnings.append("4bit_requested_but_cuda_unavailable_fallback_full_precision")
        return None, False, warnings

    return _build_quantization_config(config), True, warnings


def _resolve_precision_flags(config: Step2TrainConfig) -> Tuple[bool, bool, List[str]]:  # pragma: no cover
    warnings: List[str] = []
    import torch  # type: ignore

    fp16 = bool(config.runtime.fp16)
    bf16 = bool(config.runtime.bf16)

    if fp16 and bf16:
        warnings.append("both_fp16_and_bf16_requested_using_bf16")
        fp16 = False

    cuda_available = bool(torch.cuda.is_available())

    if bf16:
        bf16_supported = bool(cuda_available and hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
        if not bf16_supported:
            warnings.append("bf16_requested_but_not_supported")
            bf16 = False
            if cuda_available:
                fp16 = True

    if fp16 and not cuda_available:
        warnings.append("fp16_requested_but_cuda_unavailable")
        fp16 = False

    return fp16, bf16, warnings



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
    tok_id = config.model.tokenizer_id
    # support 'repo:subfolder' format
    if isinstance(tok_id, str) and ":" in tok_id and not tok_id.startswith("http"):
        repo, subfolder = tok_id.split(":", 1)
        tokenizer = AutoTokenizer.from_pretrained(
            repo,
            subfolder=subfolder,
            trust_remote_code=config.model.trust_remote_code,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_id,
            trust_remote_code=config.model.trust_remote_code,
        )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer



def _load_text_backbone(config: Step2TrainConfig, quantization_config: Optional[Any]) -> Any:  # pragma: no cover
    from transformers import AutoModelForCausalLM  # type: ignore

    kwargs: Dict[str, Any] = {
        "trust_remote_code": config.model.trust_remote_code,
        "device_map": "auto",
    }

    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    else:
        kwargs["torch_dtype"] = _resolve_dtype(
            "bfloat16" if config.runtime.bf16 else "float16" if config.runtime.fp16 else "float32"
        )

    model_id = config.model.base_model_id
    if config.dry_run and config.model.dry_run_model_id:
        model_id = config.model.dry_run_model_id

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if config.runtime.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    return model



def _apply_lora_if_enabled(text_backbone: Any, config: Step2TrainConfig) -> Any:  # pragma: no cover
    if not config.lora.enabled:
        return text_backbone

    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training  # type: ignore

    if config.model.use_4bit:
        text_backbone = prepare_model_for_kbit_training(text_backbone)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias=config.lora.bias,
    )
    return get_peft_model(text_backbone, lora_config)


class QwenVisionActionModel(_BaseTorchModule):  # pragma: no cover
    def __init__(self, text_backbone: Any, vision_feature_dim: int, num_actions: int) -> None:
        import torch.nn as nn  # type: ignore

        super().__init__()

        if num_actions <= 0:
            raise ValueError("num_actions must be > 0")

        self.text_backbone = text_backbone
        self.vision_feature_dim = max(1, int(vision_feature_dim))
        self.num_actions = int(num_actions)

        hidden_size = self._infer_hidden_size(text_backbone)
        self.vision_projector = nn.Linear(self.vision_feature_dim, hidden_size)
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.fusion_norm = nn.LayerNorm(hidden_size)
        self.action_head = nn.Linear(hidden_size, self.num_actions)

    @staticmethod
    def _infer_hidden_size(model: Any) -> int:
        cfg = getattr(model, "config", None)
        for attr in ("hidden_size", "n_embd", "d_model"):
            value = getattr(cfg, attr, None)
            if isinstance(value, int) and value > 0:
                return value
        raise RuntimeError("Unable to infer hidden size from text backbone config")

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        vision_inputs: Any = None,
        labels: Any = None,
    ) -> Dict[str, Any]:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        text_outputs = self.text_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(text_outputs, "hidden_states", None)
        if hidden_states is None:
            raise RuntimeError("Text backbone did not return hidden_states")

        text_state = hidden_states[-1][:, -1, :]
        if vision_inputs is None:
            vision_inputs = torch.zeros(
                (text_state.shape[0], self.vision_feature_dim),
                dtype=text_state.dtype,
                device=text_state.device,
            )
        vision_inputs = vision_inputs.to(dtype=text_state.dtype, device=text_state.device)

        vision_state = self.vision_projector(vision_inputs)
        fused = torch.tanh(self.fusion(torch.cat([text_state, vision_state], dim=-1)))
        fused = self.fusion_norm(fused)
        logits = self.action_head(fused)

        output: Dict[str, Any] = {"logits": logits}
        if labels is not None:
            output["loss"] = F.cross_entropy(logits, labels.long())
        return output

    def save_pretrained(self, output_dir: str | Path) -> None:
        import torch  # type: ignore

        target_dir = ensure_dir(output_dir)
        torch.save(self.state_dict(), Path(target_dir) / "pytorch_model.bin")
        write_json(
            Path(target_dir) / "model_config.json",
            {
                "vision_feature_dim": self.vision_feature_dim,
                "num_actions": self.num_actions,
                "text_backbone_class": self.text_backbone.__class__.__name__,
                "model_class": self.__class__.__name__,
            },
        )


class VLABatchCollator:  # pragma: no cover
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        import torch  # type: ignore

        text_features = [
            {
                "input_ids": item["input_ids"],
                "attention_mask": item.get("attention_mask", [1] * len(item["input_ids"])),
            }
            for item in features
        ]
        batch = self.tokenizer.pad(text_features, padding=True, return_tensors="pt")
        batch["vision_inputs"] = torch.tensor([item["vision_inputs"] for item in features], dtype=torch.float32)
        batch["labels"] = torch.tensor([int(item["labels"]) for item in features], dtype=torch.long)
        return batch


def _row_to_intermediate(
    row: Dict[str, Any],
    text_fields: Sequence[str],
    action_fields: Sequence[str],
    vision_fields: Sequence[str],
    vision_dim: int,
    token_mapping: Dict[int, str],
    action_id_mapping: Dict[int, int],
    token_str_to_new_id: Dict[str, int],
    strict_action_id_mapping: bool,
) -> Dict[str, Any]:
    action_id = _extract_action_label(
        row=row,
        action_fields=action_fields,
        old_to_new_action_id=action_id_mapping,
        token_str_to_new_id=token_str_to_new_id,
        strict_action_id_mapping=strict_action_id_mapping,
    )

    return {
        "text": build_training_text(row, text_fields, action_fields, token_mapping),
        "action_id": int(action_id) if action_id is not None else -1,
        "vision_inputs": _extract_vision_features(row, vision_fields, vision_dim),
    }


def _cap_sample_limits(config: Step2TrainConfig) -> Tuple[int, int]:
    max_train = int(config.dataset.max_train_samples)
    max_eval = int(config.dataset.max_eval_samples)

    if config.dry_run:
        max_train = min(max_train if max_train > 0 else 256, 256)
        max_eval = min(max_eval if max_eval > 0 else 64, 64)
    return max_train, max_eval


def _load_dataset_split_with_limit(dataset_id: str, split_name: str, max_samples: int) -> Any:  # pragma: no cover
    from datasets import load_dataset  # type: ignore

    if max_samples > 0:
        split_spec = f"{split_name}[:{int(max_samples)}]"
    else:
        split_spec = split_name
    return load_dataset(dataset_id, split=split_spec)


def _estimate_required_source_size(max_train: int, max_eval: int, holdout_ratio: float) -> int:
    ratio = max(1e-6, min(1.0 - 1e-6, float(holdout_ratio)))

    if max_train > 0 and max_eval > 0:
        return int(max_train + max_eval)
    if max_train > 0:
        return int(math.ceil(max_train / (1.0 - ratio)))
    if max_eval > 0:
        return int(math.ceil(max_eval / ratio))
    return 0


def _prepare_raw_splits(config: Step2TrainConfig) -> Tuple[Any, Any]:  # pragma: no cover
    max_train, max_eval = _cap_sample_limits(config)

    if config.dataset.eval_split != config.dataset.train_split:
        train_ds = _load_dataset_split_with_limit(
            dataset_id=config.dataset.dataset_id,
            split_name=config.dataset.train_split,
            max_samples=max_train,
        )
        eval_ds = _load_dataset_split_with_limit(
            dataset_id=config.dataset.dataset_id,
            split_name=config.dataset.eval_split,
            max_samples=max_eval,
        )
        return train_ds, eval_ds

    if config.dataset.enforce_distinct_eval_split:
        ratio = float(config.dataset.eval_holdout_ratio)
        if ratio <= 0.0 or ratio >= 1.0:
            raise ValueError("eval_holdout_ratio must be in (0, 1) when enforce_distinct_eval_split=true")

        source_limit = _estimate_required_source_size(max_train, max_eval, ratio)
        source_ds = _load_dataset_split_with_limit(
            dataset_id=config.dataset.dataset_id,
            split_name=config.dataset.train_split,
            max_samples=source_limit,
        )
        split_ds = source_ds.train_test_split(test_size=ratio, shuffle=True, seed=config.seed)
        train_ds = _truncate_dataset(split_ds["train"], max_train)
        eval_ds = _truncate_dataset(split_ds["test"], max_eval)
        return train_ds, eval_ds

    shared_limit = max(max_train, max_eval)
    shared_ds = _load_dataset_split_with_limit(
        dataset_id=config.dataset.dataset_id,
        split_name=config.dataset.train_split,
        max_samples=shared_limit,
    )
    train_ds = _truncate_dataset(shared_ds, max_train)
    eval_ds = _truncate_dataset(shared_ds, max_eval)
    return train_ds, eval_ds


def _resolve_saved_dataset_dir(config: Step2TrainConfig) -> Path:
    if config.dataset.saved_dataset_dir:
        return Path(config.dataset.saved_dataset_dir)
    return Path(config.runtime.output_dir) / "saved_dataset"


def _save_dataset_splits(
    config: Step2TrainConfig,
    train_ds: Any,
    eval_ds: Any,
    action_id_to_label: Dict[int, int],
    label_to_action_id: Dict[int, int],
) -> None:  # pragma: no cover
    if not config.dataset.save_dataset:
        return

    if not hasattr(train_ds, "save_to_disk") or not hasattr(eval_ds, "save_to_disk"):
        raise RuntimeError("Dataset saving requested but dataset object does not support save_to_disk")

    base_dir = _resolve_saved_dataset_dir(config)
    train_dir = base_dir / "train"
    eval_dir = base_dir / "eval"
    base_dir.mkdir(parents=True, exist_ok=True)

    train_ds.save_to_disk(str(train_dir))
    eval_ds.save_to_disk(str(eval_dir))

    write_json(
        base_dir / "label_map.json",
        {
            "action_id_to_label": {str(k): int(v) for k, v in action_id_to_label.items()},
            "label_to_action_id": {str(k): int(v) for k, v in label_to_action_id.items()},
        },
    )


def _load_saved_datasets(config: Step2TrainConfig) -> Tuple[Any, Any, Dict[int, int], Dict[int, int]]:  # pragma: no cover
    from datasets import load_from_disk  # type: ignore

    base_dir = _resolve_saved_dataset_dir(config)
    train_dir = base_dir / "train"
    eval_dir = base_dir / "eval"
    label_map_path = base_dir / "label_map.json"

    if not train_dir.exists() or not eval_dir.exists():
        raise RuntimeError(f"Saved dataset not found under: {base_dir}")
    if not label_map_path.exists():
        raise RuntimeError(f"Saved label map not found: {label_map_path}")

    train_ds = load_from_disk(str(train_dir))
    eval_ds = load_from_disk(str(eval_dir))

    with label_map_path.open("r", encoding="utf-8") as f:
        label_map = json.load(f)

    action_id_to_label = {int(k): int(v) for k, v in label_map.get("action_id_to_label", {}).items()}
    label_to_action_id = {int(k): int(v) for k, v in label_map.get("label_to_action_id", {}).items()}
    if not action_id_to_label or not label_to_action_id:
        raise RuntimeError(f"Saved label map is empty or invalid: {label_map_path}")

    return train_ds, eval_ds, action_id_to_label, label_to_action_id



def _build_intermediate_dataset(
    ds: Any,
    split_name: str,
    text_fields: Sequence[str],
    action_fields: Sequence[str],
    vision_fields: Sequence[str],
    vision_dim: int,
    token_mapping: Dict[int, str],
    action_id_mapping: Dict[int, int],
    token_str_to_new_id: Dict[str, int],
    strict_action_id_mapping: bool,
) -> Any:  # pragma: no cover
    original_columns = list(ds.column_names)

    mapped = ds.map(
        lambda row: _row_to_intermediate(
            row=row,
            text_fields=text_fields,
            action_fields=action_fields,
            vision_fields=vision_fields,
            vision_dim=vision_dim,
            token_mapping=token_mapping,
            action_id_mapping=action_id_mapping,
            token_str_to_new_id=token_str_to_new_id,
            strict_action_id_mapping=strict_action_id_mapping,
        ),
        remove_columns=original_columns,
        desc=f"Building {split_name} multimodal examples",
    )

    mapped = mapped.filter(lambda row: int(row["action_id"]) >= 0, desc=f"Filtering {split_name} invalid actions")
    return mapped



def _build_label_maps(train_ds: Any) -> Tuple[Dict[int, int], Dict[int, int]]:
    action_ids = [int(x) for x in train_ds["action_id"]]
    unique_ids = sorted(set(action_ids))
    if not unique_ids:
        raise RuntimeError("No valid action labels found in training split")

    action_id_to_label = {action_id: idx for idx, action_id in enumerate(unique_ids)}
    label_to_action_id = {idx: action_id for action_id, idx in action_id_to_label.items()}
    return action_id_to_label, label_to_action_id



def _finalize_dataset(ds: Any, tokenizer: Any, max_length: int, action_id_to_label: Dict[int, int], split_name: str) -> Any:  # pragma: no cover
    remove_columns = list(ds.column_names)

    finalized = ds.map(
        lambda row: {
            **tokenizer(row["text"], truncation=True, max_length=max_length, padding=False),
            "vision_inputs": row["vision_inputs"],
            "labels": int(action_id_to_label.get(int(row["action_id"]), -1)),
        },
        remove_columns=remove_columns,
        desc=f"Tokenizing {split_name} split",
    )

    finalized = finalized.filter(lambda row: int(row["labels"]) >= 0, desc=f"Filtering {split_name} unknown labels")
    return finalized



def _compute_classification_metrics(eval_pred: Any) -> Dict[str, float]:  # pragma: no cover
    try:
        import numpy as np  # type: ignore

        logits, labels = eval_pred
        if isinstance(logits, tuple):
            logits = logits[0]
        preds = np.argmax(logits, axis=-1)
        accuracy = float((preds == labels).mean()) if len(labels) > 0 else 0.0
        return {"accuracy": accuracy}
    except Exception:
        return {}


def _get_trainer_components() -> Tuple[Any, Any]:  # pragma: no cover
    from transformers import Trainer, TrainingArguments  # type: ignore

    return Trainer, TrainingArguments


def _resolve_report_targets(config: Step2TrainConfig) -> List[str]:
    targets: List[str] = []
    for item in config.runtime.report_to:
        name = str(item).strip().lower()
        if not name:
            continue
        if name in {"none", "off", "disable", "disabled"}:
            return []
        if name not in targets:
            targets.append(name)
    return targets



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
    eval_accuracy = eval_metrics.get("eval_accuracy")

    quality_flags: List[str] = []
    if isinstance(train_loss, (int, float)) and float(train_loss) > 5.0:
        quality_flags.append("high_train_loss")
    if isinstance(eval_loss, (int, float)) and float(eval_loss) > 5.0:
        quality_flags.append("high_eval_loss")
    if isinstance(eval_accuracy, (int, float)) and float(eval_accuracy) < 0.2:
        quality_flags.append("low_eval_accuracy")
    if model_stats.get("trainable_ratio", 0.0) <= 0.0:
        quality_flags.append("no_trainable_parameters")

    return {
        "run_name": config.run_name,
        "model": {
            "base_model_id": config.model.base_model_id,
            "use_4bit": config.model.use_4bit,
            "lora_enabled": config.lora.enabled,
            "lora_target_modules": config.lora.target_modules,
            "architecture": "text_backbone + vision_projector + fusion + action_head",
        },
        "dataset": {
            "dataset_id": config.dataset.dataset_id,
            "train_split": config.dataset.train_split,
            "eval_split": config.dataset.eval_split,
            "enforce_distinct_eval_split": config.dataset.enforce_distinct_eval_split,
        },
        "metrics": {
            "train_loss": train_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
            "legacy_eval_perplexity": float(math.exp(float(eval_loss))) if isinstance(eval_loss, (int, float)) and float(eval_loss) < 20 else None,
        },
        "parameter_stats": model_stats,
        "quality_flags": quality_flags,
        "notes": [
            "This report is generated after multimodal action training.",
            "See action_label_map.json for label-to-action-id mapping.",
        ],
    }



def run_step2_full_train_pipeline(config: Step2TrainConfig) -> Dict[str, Any]:  # pragma: no cover
    _require_hf_training_dependencies()
    _validate_step1_artifacts(config)
    set_seed(config.seed)

    out_dir = ensure_dir(config.runtime.output_dir)

    token_mapping = load_action_token_mapping(config.dataset.mapping_table_path)
    action_id_mapping = load_action_id_mapping(config.dataset.mapping_table_path)
    token_str_to_new_id = load_token_str_to_new_id(config.dataset.mapping_table_path)

    if (
        config.dataset.require_step1_artifacts
        and not action_id_mapping
        and not token_str_to_new_id
        and not config.dataset.use_saved_dataset
    ):
        raise RuntimeError("Step1 mapping table is present but has no usable action mappings")

    runtime_warnings: List[str] = []

    tokenizer = _load_tokenizer(config)
    quantization_config, quantization_enabled, quant_warnings = _resolve_quantization_config_for_runtime(config)
    runtime_warnings.extend(quant_warnings)

    resolved_fp16, resolved_bf16, precision_warnings = _resolve_precision_flags(config)
    runtime_warnings.extend(precision_warnings)

    text_backbone = _load_text_backbone(config, quantization_config)
    text_backbone = _apply_lora_if_enabled(text_backbone, config)

    if config.dataset.use_saved_dataset:
        train_ds, eval_ds, action_id_to_label, label_to_action_id = _load_saved_datasets(config)
    else:
        raw_train_ds, raw_eval_ds = _prepare_raw_splits(config)
        train_intermediate = _build_intermediate_dataset(
            ds=raw_train_ds,
            split_name="train",
            text_fields=config.dataset.text_field_candidates,
            action_fields=config.dataset.action_field_candidates,
            vision_fields=config.dataset.vision_field_candidates,
            vision_dim=config.dataset.vision_feature_dim,
            token_mapping=token_mapping,
            action_id_mapping=action_id_mapping,
            token_str_to_new_id=token_str_to_new_id,
            strict_action_id_mapping=config.dataset.strict_action_id_mapping,
        )
        eval_intermediate = _build_intermediate_dataset(
            ds=raw_eval_ds,
            split_name="eval",
            text_fields=config.dataset.text_field_candidates,
            action_fields=config.dataset.action_field_candidates,
            vision_fields=config.dataset.vision_field_candidates,
            vision_dim=config.dataset.vision_feature_dim,
            token_mapping=token_mapping,
            action_id_mapping=action_id_mapping,
            token_str_to_new_id=token_str_to_new_id,
            strict_action_id_mapping=config.dataset.strict_action_id_mapping,
        )

        action_id_to_label, label_to_action_id = _build_label_maps(train_intermediate)

        train_ds = _finalize_dataset(
            ds=train_intermediate,
            tokenizer=tokenizer,
            max_length=config.dataset.max_length,
            action_id_to_label=action_id_to_label,
            split_name="train",
        )
        eval_ds = _finalize_dataset(
            ds=eval_intermediate,
            tokenizer=tokenizer,
            max_length=config.dataset.max_length,
            action_id_to_label=action_id_to_label,
            split_name="eval",
        )

        _save_dataset_splits(
            config=config,
            train_ds=train_ds,
            eval_ds=eval_ds,
            action_id_to_label=action_id_to_label,
            label_to_action_id=label_to_action_id,
        )

    model = QwenVisionActionModel(
        text_backbone=text_backbone,
        vision_feature_dim=config.dataset.vision_feature_dim,
        num_actions=len(action_id_to_label),
    )

    Trainer, TrainingArguments = _get_trainer_components()

    evaluation_strategy = "steps" if len(eval_ds) > 0 else "no"
    max_steps = config.runtime.max_steps
    if config.dry_run and max_steps <= 0:
        max_steps = 10

    report_to = _resolve_report_targets(config)
    logging_dir = config.runtime.logging_dir.strip() if config.runtime.logging_dir else ""
    if not logging_dir:
        logging_dir = str(Path(out_dir) / "tb")

    training_args_kwargs: Dict[str, Any] = {
        "output_dir": str(out_dir),
        "run_name": config.run_name,
        "num_train_epochs": config.runtime.num_train_epochs,
        "per_device_train_batch_size": config.runtime.per_device_train_batch_size,
        "per_device_eval_batch_size": config.runtime.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.runtime.gradient_accumulation_steps,
        "learning_rate": config.runtime.learning_rate,
        "weight_decay": config.runtime.weight_decay,
        "warmup_ratio": config.runtime.warmup_ratio,
        "lr_scheduler_type": config.runtime.lr_scheduler_type,
        "logging_steps": config.runtime.logging_steps,
        "save_steps": config.runtime.save_steps,
        "eval_steps": config.runtime.eval_steps,
        "max_grad_norm": config.runtime.max_grad_norm,
        "fp16": resolved_fp16,
        "bf16": resolved_bf16,
        "gradient_checkpointing": config.runtime.gradient_checkpointing,
        "evaluation_strategy": evaluation_strategy,
        "save_strategy": "steps",
        "max_steps": max_steps,
        "report_to": report_to,
        "logging_dir": logging_dir,
        "remove_unused_columns": False,
        "dataloader_pin_memory": False,
    }

    try:
        args = TrainingArguments(**training_args_kwargs)
    except TypeError as exc:
        if "evaluation_strategy" in str(exc):
            training_args_kwargs.pop("evaluation_strategy", None)
            training_args_kwargs["eval_strategy"] = evaluation_strategy
            args = TrainingArguments(**training_args_kwargs)
        else:
            raise

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=VLABatchCollator(tokenizer),
        compute_metrics=_compute_classification_metrics if evaluation_strategy != "no" else None,
    )

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate() if evaluation_strategy == "steps" else {}
    model_stats = collect_model_parameter_stats(model)

    model_bundle_dir = Path(out_dir) / "model_bundle"
    model.save_pretrained(model_bundle_dir)

    tokenizer_dir = Path(out_dir) / "tokenizer"
    tokenizer.save_pretrained(tokenizer_dir)

    if config.lora.enabled and hasattr(model.text_backbone, "save_pretrained"):
        text_backbone_dir = Path(out_dir) / "text_backbone_lora"
        model.text_backbone.save_pretrained(text_backbone_dir)
        text_model_artifact = str(text_backbone_dir)
    elif hasattr(model.text_backbone, "save_pretrained"):
        text_backbone_dir = Path(out_dir) / "text_backbone_full"
        model.text_backbone.save_pretrained(text_backbone_dir)
        text_model_artifact = str(text_backbone_dir)
    else:
        import torch  # type: ignore

        state_path = Path(out_dir) / "text_backbone_state.pt"
        torch.save(model.text_backbone.state_dict(), state_path)
        text_model_artifact = str(state_path)

    import torch  # type: ignore

    vla_head_path = Path(out_dir) / "vla_head_state.pt"
    torch.save(
        {
            "vision_projector": model.vision_projector.state_dict(),
            "fusion": model.fusion.state_dict(),
            "fusion_norm": model.fusion_norm.state_dict(),
            "action_head": model.action_head.state_dict(),
            "num_actions": model.num_actions,
            "vision_feature_dim": model.vision_feature_dim,
        },
        vla_head_path,
    )

    action_label_map_path = Path(out_dir) / "action_label_map.json"
    write_json(
        action_label_map_path,
        {
            "action_id_to_label": {str(k): int(v) for k, v in action_id_to_label.items()},
            "label_to_action_id": {str(k): int(v) for k, v in label_to_action_id.items()},
        },
    )

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
            "num_action_classes": len(action_id_to_label),
            "save_dataset": config.dataset.save_dataset,
            "use_saved_dataset": config.dataset.use_saved_dataset,
            "saved_dataset_dir": str(_resolve_saved_dataset_dir(config)),
        },
        "model": {
            "base_model_id": config.model.base_model_id,
            "effective_base_model_id": config.model.dry_run_model_id if config.dry_run and config.model.dry_run_model_id else config.model.base_model_id,
            "tokenizer_id": config.model.tokenizer_id,
            "use_4bit": config.model.use_4bit,
            "effective_4bit": quantization_enabled,
            "lora_enabled": config.lora.enabled,
            "parameter_stats": model_stats,
            "architecture": "qwen_text_backbone + vision_projector + action_head",
        },
        "runtime_warnings": runtime_warnings,
        "logging": {
            "report_to": report_to,
            "logging_dir": logging_dir,
        },
        "metrics": {
            "train": train_metrics,
            "eval": eval_metrics,
        },
        "artifacts": {
            "output_dir": str(out_dir),
            "model_bundle": str(model_bundle_dir),
            "text_model_artifact": text_model_artifact,
            "vla_head_state": str(vla_head_path),
            "action_label_map": str(action_label_map_path),
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
