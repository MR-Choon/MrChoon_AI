from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from minecraft_vla.config import QLoRAConfig


@dataclass
class QLoRAResult:
    applied: bool
    backend: str
    mode: str
    target_modules: List[str]
    details: Dict[str, Any]


class SimulatedLoraWrapper:
    def __init__(self, model: Any, target_modules: List[str], backend: str, details: Dict[str, Any]) -> None:
        self.model = model
        self.target_modules = target_modules
        self.backend = backend
        self.details = details

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model.forward(*args, **kwargs)

    def save_pretrained(self, output_dir: str | Path) -> None:
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        with (target / "adapter_config.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "peft_type": "LORA",
                    "backend": self.backend,
                    "mode": "simulated",
                    "target_modules": self.target_modules,
                    "details": self.details,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        (target / "adapter_model.safetensors").write_bytes(b"SIMULATED_LORA")



def apply_qlora(model: Any, config: QLoRAConfig, backend: str) -> Tuple[Any, QLoRAResult]:
    if not config.enabled:
        return (
            model,
            QLoRAResult(
                applied=False,
                backend=backend,
                mode="disabled",
                target_modules=config.target_modules,
                details={},
            ),
        )

    if backend == "mock":
        wrapped = SimulatedLoraWrapper(
            model=model,
            target_modules=config.target_modules,
            backend=backend,
            details={
                "r": config.r,
                "alpha": config.alpha,
                "dropout": config.dropout,
            },
        )
        return (
            wrapped,
            QLoRAResult(
                applied=True,
                backend=backend,
                mode="simulated",
                target_modules=config.target_modules,
                details={
                    "r": config.r,
                    "alpha": config.alpha,
                    "dropout": config.dropout,
                },
            ),
        )

    # HF backend: try PEFT first; if unavailable, fallback to simulated wrapper.
    try:  # pragma: no cover - optional path
        from peft import LoraConfig, get_peft_model  # type: ignore

        lora_config = LoraConfig(
            r=config.r,
            lora_alpha=config.alpha,
            lora_dropout=config.dropout,
            target_modules=config.target_modules,
        )
        peft_model = get_peft_model(model.module, lora_config)

        class PeftAdapter:
            def __init__(self, wrapped: Any) -> None:
                self.wrapped = wrapped

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                return self.wrapped(*args, **kwargs)

            def save_pretrained(self, output_dir: str | Path) -> None:
                self.wrapped.save_pretrained(output_dir)

        return (
            PeftAdapter(peft_model),
            QLoRAResult(
                applied=True,
                backend=backend,
                mode="peft",
                target_modules=config.target_modules,
                details={
                    "r": config.r,
                    "alpha": config.alpha,
                    "dropout": config.dropout,
                },
            ),
        )
    except Exception as exc:  # pragma: no cover - optional path
        wrapped = SimulatedLoraWrapper(
            model=model,
            target_modules=config.target_modules,
            backend=backend,
            details={"fallback_reason": str(exc)},
        )
        return (
            wrapped,
            QLoRAResult(
                applied=True,
                backend=backend,
                mode="simulated_fallback",
                target_modules=config.target_modules,
                details={"fallback_reason": str(exc)},
            ),
        )
