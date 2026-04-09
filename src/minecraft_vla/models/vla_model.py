from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

from minecraft_vla.config import Step2Config


@dataclass
class MockVLABatch:
    input_ids: List[int]
    vision_features: List[float]
    labels: List[int]


class MockVisionEncoder:
    def __init__(self, input_dim: int, hidden_size: int) -> None:
        self.input_dim = input_dim
        self.hidden_size = hidden_size

    def encode(self, features: Sequence[float]) -> List[float]:
        values = list(features)
        if len(values) < self.input_dim:
            values.extend([0.0] * (self.input_dim - len(values)))
        values = values[: self.input_dim]
        encoded: List[float] = []
        for i in range(self.hidden_size):
            source = values[i % self.input_dim]
            encoded.append(math.tanh(source * 0.1 + i * 0.001))
        return encoded


class MockTextBackbone:
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

    def encode(self, input_ids: Sequence[int]) -> List[float]:
        if not input_ids:
            input_ids = [0]
        state = [0.0] * self.hidden_size
        for token in input_ids:
            norm = float(token % max(1, self.vocab_size)) / max(1.0, float(self.vocab_size))
            for i in range(self.hidden_size):
                state[i] += norm * (i + 1) * 0.001
        scale = 1.0 / len(input_ids)
        return [s * scale for s in state]


class MockActionHead:
    def __init__(self, hidden_size: int, num_actions: int) -> None:
        self.hidden_size = hidden_size
        self.num_actions = num_actions

    def logits(self, fused_state: Sequence[float]) -> List[float]:
        total = sum(fused_state)
        return [total * (i + 1) * 0.01 for i in range(self.num_actions)]


class MockVLAModel:
    def __init__(self, text_backbone: MockTextBackbone, vision_encoder: MockVisionEncoder, action_head: MockActionHead) -> None:
        self.text_backbone = text_backbone
        self.vision_encoder = vision_encoder
        self.action_head = action_head

    def forward(self, batch: MockVLABatch) -> Dict[str, Any]:
        text_state = self.text_backbone.encode(batch.input_ids)
        vision_state = self.vision_encoder.encode(batch.vision_features)
        fused = [(t + v) * 0.5 for t, v in zip(text_state, vision_state)]
        logits = self.action_head.logits(fused)

        if not batch.labels:
            label = 0
        else:
            label = int(batch.labels[0] % self.action_head.num_actions)

        target_logit = logits[label]
        max_logit = max(logits) if logits else 0.0
        loss = float((max_logit - target_logit) ** 2)
        return {"loss": loss, "logits": logits, "label": label}


class TinyTorchVLAModel:  # pragma: no cover - optional torch backend
    def __init__(self, vocab_size: int, hidden_size: int, vision_input_dim: int, num_actions: int) -> None:
        import torch.nn as nn  # type: ignore

        self._nn = nn
        self.module = nn.Module()
        self.module.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.module.vision_proj = nn.Linear(vision_input_dim, hidden_size)
        self.module.fusion = nn.Linear(hidden_size * 2, hidden_size)
        self.module.classifier = nn.Linear(hidden_size, num_actions)

    def forward(self, input_ids: Any, vision_inputs: Any, labels: Any = None) -> Dict[str, Any]:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore

        text_state = self.module.token_embedding(input_ids).mean(dim=1)
        vision_state = self.module.vision_proj(vision_inputs)
        fused = torch.tanh(self.module.fusion(torch.cat([text_state, vision_state], dim=-1)))
        logits = self.module.classifier(fused)
        out: Dict[str, Any] = {"logits": logits}

        if labels is not None:
            out["loss"] = F.cross_entropy(logits, labels)
        return out


class TorchModelAdapter:  # pragma: no cover - optional torch backend
    def __init__(self, tiny_torch_model: TinyTorchVLAModel) -> None:
        self.tiny_torch_model = tiny_torch_model
        self.module = tiny_torch_model.module

    def forward(self, input_ids: Any, vision_inputs: Any, labels: Any = None) -> Dict[str, Any]:
        return self.tiny_torch_model.forward(input_ids=input_ids, vision_inputs=vision_inputs, labels=labels)



def build_vla_model(config: Step2Config) -> Tuple[Any, Dict[str, Any]]:
    if config.backend == "mock":
        text = MockTextBackbone(
            vocab_size=config.text_backbone.vocab_size,
            hidden_size=config.text_backbone.hidden_size,
        )
        vision = MockVisionEncoder(
            input_dim=config.vision_encoder.input_dim,
            hidden_size=config.vision_encoder.hidden_size,
        )
        head = MockActionHead(
            hidden_size=config.text_backbone.hidden_size,
            num_actions=config.action_head.num_actions,
        )
        model = MockVLAModel(text_backbone=text, vision_encoder=vision, action_head=head)
        meta = {
            "backend": "mock",
            "text_backbone_model_id": config.text_backbone.model_id,
            "text_hidden_size": config.text_backbone.hidden_size,
            "vision_encoder": config.vision_encoder.name,
            "num_actions": config.action_head.num_actions,
        }
        return model, meta

    torch_model = TinyTorchVLAModel(
        vocab_size=config.text_backbone.vocab_size,
        hidden_size=config.text_backbone.hidden_size,
        vision_input_dim=config.vision_encoder.input_dim,
        num_actions=config.action_head.num_actions,
    )
    model = TorchModelAdapter(torch_model)
    meta = {
        "backend": "hf",
        "text_backbone_model_id": config.text_backbone.model_id,
        "text_hidden_size": config.text_backbone.hidden_size,
        "vision_encoder": config.vision_encoder.name,
        "num_actions": config.action_head.num_actions,
    }
    return model, meta
