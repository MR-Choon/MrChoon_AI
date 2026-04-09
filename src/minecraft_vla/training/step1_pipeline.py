from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from minecraft_vla.config import Step1Config, load_step1_config
from minecraft_vla.data.minecraft_sft import load_samples
from minecraft_vla.utils.io import ensure_dir, write_json
from minecraft_vla.utils.seed import set_seed


@dataclass
class MappingRow:
    old_id: int
    token_str: str
    mapped_new_id: int
    match_type: str
    confidence: float
    notes: str


class MockTokenizer:
    def __init__(self, vocab: Dict[str, int], unk_token: str = "<unk>") -> None:
        self.vocab = dict(vocab)
        self.id_to_token = {v: k for k, v in self.vocab.items()}
        self.unk_token = unk_token
        if self.unk_token not in self.vocab:
            next_id = max(self.vocab.values(), default=-1) + 1
            self.vocab[self.unk_token] = next_id
            self.id_to_token[next_id] = self.unk_token

    @property
    def unk_token_id(self) -> int:
        return self.vocab[self.unk_token]

    def get_vocab(self) -> Dict[str, int]:
        return dict(self.vocab)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.id_to_token.get(token_id, self.unk_token)

    def convert_tokens_to_ids(self, token_str: str) -> int:
        return self.vocab.get(token_str, self.unk_token_id)

    def add_tokens(self, token_list: Sequence[str]) -> int:
        next_id = max(self.vocab.values(), default=-1) + 1
        added = 0
        for token in token_list:
            if token in self.vocab:
                continue
            self.vocab[token] = next_id
            self.id_to_token[next_id] = token
            next_id += 1
            added += 1
        return added

    def save_pretrained(self, save_dir: str | Path) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "vocab.json").open("w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False, indent=2)
        with (path / "tokenizer_config.json").open("w", encoding="utf-8") as f:
            json.dump({"unk_token": self.unk_token}, f, ensure_ascii=False, indent=2)

    def __len__(self) -> int:
        return len(self.vocab)


class MockCausalLM:
    def __init__(self, vocab_size: int, hidden_size: int = 16) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embeddings = [[0.0] * hidden_size for _ in range(vocab_size)]

    def resize_token_embeddings(self, new_size: int) -> None:
        if new_size < self.vocab_size:
            self.embeddings = self.embeddings[:new_size]
        else:
            for _ in range(new_size - self.vocab_size):
                self.embeddings.append([0.0] * self.hidden_size)
        self.vocab_size = new_size

    def forward(self, input_ids: Sequence[int]) -> Dict[str, float]:
        if not input_ids:
            raise ValueError("input_ids must not be empty")
        for token_id in input_ids:
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token id out of range: {token_id} >= {self.vocab_size}")
        return {"loss": float(sum(input_ids) / len(input_ids))}


class MockLoraModel:
    def __init__(self, base_model: MockCausalLM, target_modules: Sequence[str]) -> None:
        self.base_model = base_model
        self.target_modules = list(target_modules)

    def forward(self, input_ids: Sequence[int]) -> Dict[str, float]:
        return self.base_model.forward(input_ids)

    def save_pretrained(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with (path / "adapter_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "peft_type": "LORA",
                    "target_modules": self.target_modules,
                    "mock": True,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        (path / "adapter_model.safetensors").write_bytes(b"MOCK_ADAPTER")



def _normalize_token(token: str) -> str:
    normalized = token.replace("\u2581", " ").replace("\u0120", " ")
    normalized = " ".join(normalized.split())
    return normalized.strip()



def _build_normalized_index(vocab: Dict[str, int]) -> Dict[str, int]:
    index: Dict[str, int] = {}
    for token, token_id in vocab.items():
        key = _normalize_token(token)
        if key and key not in index:
            index[key] = token_id
    return index



def _iter_action_candidate_lists(obj: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], List[int]]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_action_candidate_lists(value, path + (str(key),))
        return

    if isinstance(obj, list):
        if obj and all(isinstance(x, int) for x in obj):
            path_text = ".".join(path).lower()
            if "action" in path_text:
                yield path, [int(x) for x in obj]
        for idx, value in enumerate(obj):
            yield from _iter_action_candidate_lists(value, path + (str(idx),))



def _collect_source_ids(samples: Sequence[Dict[str, Any]]) -> Tuple[List[int], Dict[str, int]]:
    ids_counter: Dict[int, int] = {}
    path_counter: Dict[str, int] = {}

    for sample in samples:
        for path, values in _iter_action_candidate_lists(sample):
            key = ".".join(path)
            path_counter[key] = path_counter.get(key, 0) + 1
            for token_id in values:
                ids_counter[token_id] = ids_counter.get(token_id, 0) + 1

    return sorted(ids_counter), path_counter



def _build_mapping_rows(
    source_tokenizer: Any,
    target_tokenizer: Any,
    source_ids: Sequence[int],
    add_missing_tokens: bool,
) -> Tuple[List[MappingRow], int, int]:
    target_vocab = target_tokenizer.get_vocab()
    normalized_target_vocab = _build_normalized_index(target_vocab)

    rows: List[MappingRow] = []
    missing_tokens: List[str] = []

    for old_id in source_ids:
        token_str = str(source_tokenizer.convert_ids_to_tokens(old_id))

        if token_str in target_vocab:
            rows.append(
                MappingRow(
                    old_id=old_id,
                    token_str=token_str,
                    mapped_new_id=int(target_vocab[token_str]),
                    match_type="exact",
                    confidence=1.0,
                    notes="",
                )
            )
            continue

        normalized = _normalize_token(token_str)
        if normalized in normalized_target_vocab:
            rows.append(
                MappingRow(
                    old_id=old_id,
                    token_str=token_str,
                    mapped_new_id=int(normalized_target_vocab[normalized]),
                    match_type="normalized",
                    confidence=0.7,
                    notes=f"normalized={normalized}",
                )
            )
            continue

        missing_tokens.append(token_str)
        rows.append(
            MappingRow(
                old_id=old_id,
                token_str=token_str,
                mapped_new_id=int(target_tokenizer.unk_token_id),
                match_type="fallback_unk",
                confidence=0.2,
                notes="token not found in target vocab",
            )
        )

    added_count = 0
    if add_missing_tokens:
        unique_missing = sorted(set(missing_tokens))
        added_count = int(target_tokenizer.add_tokens(unique_missing))
        if added_count:
            for row in rows:
                if row.match_type == "fallback_unk" and row.token_str in unique_missing:
                    row.mapped_new_id = int(target_tokenizer.convert_tokens_to_ids(row.token_str))
                    row.match_type = "added_token"
                    row.confidence = 0.85
                    row.notes = "added to target tokenizer"

    unmapped_count = sum(1 for row in rows if row.match_type == "fallback_unk")
    return rows, added_count, unmapped_count



def _save_mapping_artifacts(output_dir: Path, rows: Sequence[MappingRow], metadata: Dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "token_id_mapping.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["old_id", "token_str", "mapped_new_id", "match_type", "confidence", "notes"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))

    with (output_dir / "token_id_mapping.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    write_json(output_dir / "mapping_summary.json", metadata)



def _load_mock_tokenizers() -> Tuple[MockTokenizer, MockTokenizer]:
    source_vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "mine": 2,
        "craft": 3,
        "<ACT_MOVE_FORWARD>": 4,
        "<ACT_JUMP>": 5,
        "stone": 6,
    }
    target_vocab = {
        "<unk>": 0,
        "<pad>": 1,
        "mine": 2,
        "craft": 3,
        "<ACT_MOVE_FORWARD>": 4,
        "stone": 5,
    }
    return MockTokenizer(source_vocab), MockTokenizer(target_vocab)



def _load_hf_tokenizers(source_id: str, target_id: str) -> Tuple[Any, Any]:
    from transformers import AutoTokenizer  # type: ignore

    source = AutoTokenizer.from_pretrained(source_id, trust_remote_code=True)
    target = AutoTokenizer.from_pretrained(target_id, trust_remote_code=True)

    if source.unk_token_id is None:
        source.add_special_tokens({"unk_token": "<unk>"})
    if target.unk_token_id is None:
        target.add_special_tokens({"unk_token": "<unk>"})

    return source, target



def _run_mock_dryrun(vocab_size: int, output_dir: Path, mapped_ids: Sequence[int]) -> Dict[str, Any]:
    model = MockCausalLM(vocab_size=vocab_size)
    before = model.vocab_size
    model.resize_token_embeddings(vocab_size)
    after = model.vocab_size

    lora = MockLoraModel(model, target_modules=["mock_attn_q", "mock_attn_v"])
    input_ids = list(mapped_ids[:8]) if mapped_ids else [0, 0, 0, 0]
    out = lora.forward(input_ids)

    adapter_dir = output_dir / "lora_adapter"
    lora.save_pretrained(adapter_dir)

    return {
        "backend": "mock",
        "embedding_before": before,
        "embedding_after": after,
        "forward_tokens": len(input_ids),
        "forward_loss": float(out["loss"]),
        "adapter_dir": str(adapter_dir),
    }



def _run_hf_dryrun(vocab_size: int, output_dir: Path, mapped_ids: Sequence[int]) -> Dict[str, Any]:  # pragma: no cover
    import torch  # type: ignore
    from peft import LoraConfig, TaskType, get_peft_model  # type: ignore
    from transformers import GPT2Config, GPT2LMHeadModel  # type: ignore

    model_cfg = GPT2Config(
        vocab_size=vocab_size,
        n_embd=64,
        n_layer=2,
        n_head=2,
        n_positions=128,
    )
    model = GPT2LMHeadModel(model_cfg)

    before = int(model.get_input_embeddings().weight.shape[0])
    model.resize_token_embeddings(vocab_size)
    after = int(model.get_input_embeddings().weight.shape[0])

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["c_attn"],
    )
    lora_model = get_peft_model(model, lora_cfg)

    ids = list(mapped_ids[:8]) if mapped_ids else [0, 0, 0, 0]
    input_ids = torch.tensor([ids], dtype=torch.long)
    outputs = lora_model(input_ids=input_ids, labels=input_ids)

    adapter_dir = output_dir / "lora_adapter"
    lora_model.save_pretrained(adapter_dir)

    return {
        "backend": "hf",
        "embedding_before": before,
        "embedding_after": after,
        "forward_tokens": int(input_ids.shape[1]),
        "forward_loss": float(outputs.loss.detach().cpu().item()),
        "adapter_dir": str(adapter_dir),
    }



def run_step1_pipeline(config: Step1Config) -> Dict[str, Any]:
    set_seed(config.seed)
    output_dir = ensure_dir(config.output_dir)

    if config.backend == "mock":
        source_tokenizer, target_tokenizer = _load_mock_tokenizers()
    else:
        source_tokenizer, target_tokenizer = _load_hf_tokenizers(
            config.source_tokenizer_id,
            config.target_tokenizer_id,
        )

    samples = load_samples(
        backend=config.backend,
        dataset_id=config.dataset.dataset_id,
        split=config.dataset.split,
        max_samples=config.dataset.max_samples,
    )

    source_ids, action_paths = _collect_source_ids(samples)
    rows, added_count, unmapped_count = _build_mapping_rows(
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        source_ids=source_ids,
        add_missing_tokens=config.add_missing_tokens,
    )

    mapped_ids = [row.mapped_new_id for row in rows]
    summary = {
        "run_name": config.run_name,
        "backend": config.backend,
        "source_tokenizer_id": config.source_tokenizer_id,
        "target_tokenizer_id": config.target_tokenizer_id,
        "dataset_id": config.dataset.dataset_id,
        "dataset_split": config.dataset.split,
        "sample_count": len(samples),
        "action_paths": action_paths,
        "source_action_token_id_count": len(source_ids),
        "mapping_count": len(rows),
        "added_token_count": int(added_count),
        "unmapped_count": int(unmapped_count),
        "low_confidence_count": sum(1 for r in rows if r.confidence < 0.5),
        "target_vocab_size_after": len(target_tokenizer),
    }

    _save_mapping_artifacts(output_dir, rows, summary)
    target_tokenizer.save_pretrained(output_dir / "target_tokenizer_after")

    if config.backend == "mock":
        dryrun = _run_mock_dryrun(len(target_tokenizer), output_dir, mapped_ids)
    else:
        dryrun = _run_hf_dryrun(len(target_tokenizer), output_dir, mapped_ids)

    write_json(output_dir / "dryrun_report.json", dryrun)
    write_json(output_dir / "step1_config_used.json", json.loads(json.dumps(asdict(config))))

    return {
        "summary": summary,
        "dryrun": dryrun,
        "artifacts": {
            "output_dir": str(output_dir),
            "mapping_csv": str(output_dir / "token_id_mapping.csv"),
            "mapping_jsonl": str(output_dir / "token_id_mapping.jsonl"),
            "summary_json": str(output_dir / "mapping_summary.json"),
            "dryrun_report": str(output_dir / "dryrun_report.json"),
        },
    }



def run_step1_pipeline_from_file(config_path: str | Path) -> Dict[str, Any]:
    config = load_step1_config(config_path)
    return run_step1_pipeline(config)
