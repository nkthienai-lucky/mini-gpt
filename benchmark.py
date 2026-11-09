"""Generation timing helpers for comparing KV-cache modes."""

from __future__ import annotations

import statistics
import time
from typing import Any

import torch

from model import MiniGPT


def _synchronize(device: torch.device) -> bool:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        return True
    return False


def benchmark_generation(
    model: MiniGPT,
    prompt_ids: torch.Tensor,
    new_tokens: int,
    repeats: int = 3,
    warmup: int = 1,
    device: torch.device | None = None,
) -> dict[str, dict[str, Any]]:
    if new_tokens < 0:
        raise ValueError("new_tokens must be non-negative")
    if repeats < 1 or warmup < 0:
        raise ValueError("repeats must be positive and warmup cannot be negative")
    device = device or next(model.parameters()).device
    prompt_ids = prompt_ids.to(device)
    if prompt_ids.ndim != 2:
        raise ValueError("prompt_ids must have shape (batch, sequence)")
    if prompt_ids.size(0) != 1:
        raise ValueError("benchmark expects a single prompt batch")

    results: dict[str, dict[str, Any]] = {}
    for mode, use_cache in (("uncached", False), ("cached", True)):
        for _ in range(warmup):
            model.generate(prompt_ids, new_tokens, temperature=0, use_kv_cache=use_cache)
        _synchronize(device)
        durations = []
        for _ in range(repeats):
            start = time.perf_counter()
            model.generate(prompt_ids, new_tokens, temperature=0, use_kv_cache=use_cache)
            _synchronize(device)
            durations.append(time.perf_counter() - start)
        total_seconds = statistics.median(durations)
        results[mode] = {
            "device": str(device),
            "total_seconds": total_seconds,
            "tokens_per_second": (
                new_tokens / total_seconds if total_seconds > 0 and new_tokens else 0.0
            ),
            "milliseconds_per_token": (
                total_seconds * 1000 / new_tokens if new_tokens else 0.0
            ),
            "repeats": repeats,
            "warmup": warmup,
            "prompt_tokens": prompt_ids.size(1),
            "generated_tokens": new_tokens,
            "cuda_synchronized": device.type == "cuda",
        }
    return results
