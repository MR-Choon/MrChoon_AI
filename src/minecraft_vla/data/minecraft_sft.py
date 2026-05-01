from __future__ import annotations

import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _iter_action_candidate_lists(obj: Any, path: Tuple[str, ...] = ()) -> Iterable[List[int]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield from _iter_action_candidate_lists(value, path + (str(key),))
        return

    if isinstance(obj, list):
        if obj and all(isinstance(x, int) for x in obj):
            path_text = ".".join(path).lower()
            if "action" in path_text:
                yield [int(x) for x in obj]
        for idx, value in enumerate(obj):
            yield from _iter_action_candidate_lists(value, path + (str(idx),))


def extract_action_token_ids(samples: Iterable[Dict[str, Any]]) -> tuple[List[int], int]:
    ids: List[int] = []
    sample_count = 0
    for row in samples:
        sample_count += 1
        for action_list in _iter_action_candidate_lists(row):
            ids.extend(action_list)
    return ids, sample_count


def _load_mock_samples(max_samples: int) -> List[Dict[str, Any]]:
    base_rows = [
        {
            "obs": "mock_frame_0",
            "action_token_ids": [1, 2, 3, 2],
        },
        {
            "obs": "mock_frame_1",
            "nested": {
                "action_ids": [2, 4, 2, 1],
            },
        },
    ]

    rows: List[Dict[str, Any]] = []
    count = max(1, min(max_samples, 1000))
    for i in range(count):
        rows.append(base_rows[i % len(base_rows)])
    return rows


def _load_hf_samples(dataset_id: str, split: str, max_samples: int) -> Iterable[Dict[str, Any]]:
    from datasets import load_dataset  # type: ignore

    stream = load_dataset(dataset_id, split=split, streaming=True)
    print(f"[DATASET] Starting to stream {dataset_id}/{split}...", file=sys.stderr, flush=True)
    
    progress_interval = 10000  # Print every 10k samples
    idx = 0
    for item in stream:
        yield dict(item)
        idx += 1
        if idx % progress_interval == 0:
            if max_samples > 0:
                print(f"[DATASET] Streamed {idx}/{max_samples} samples", file=sys.stderr, flush=True)
            else:
                print(f"[DATASET] Streamed {idx} samples...", file=sys.stderr, flush=True)
        if max_samples > 0 and idx >= max_samples:
            break


def load_samples(backend: str, dataset_id: str, split: str, max_samples: int) -> Iterable[Dict[str, Any]]:
    if backend == "mock":
        return _load_mock_samples(max_samples)
    return _load_hf_samples(dataset_id, split, max_samples)
