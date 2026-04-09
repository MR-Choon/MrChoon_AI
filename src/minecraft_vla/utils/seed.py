from __future__ import annotations

import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional in mock mode.
        return
