"""Portable checkpoint serialization for Mini-GPT."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from config import Config
from data import CharTokenizer
from model import MiniGPT


FORMAT_VERSION = 1


def _config_dict(config: Config) -> dict[str, Any]:
    return dict(vars(config))


def save_checkpoint(
    path: Path,
    model: MiniGPT,
    optimizer: torch.optim.Optimizer,
    config: Config,
    tokenizer: CharTokenizer,
    step: int,
    metrics: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "config": _config_dict(config),
        "vocab_size": tokenizer.vocab_size,
        "tokenizer": {"stoi": tokenizer.stoi, "itos": tokenizer.itos},
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "metrics": metrics,
    }
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    device: torch.device,
) -> tuple[MiniGPT, CharTokenizer, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format_version')!r}"
        )
    config = Config(**payload["config"])
    tokenizer = CharTokenizer(payload["tokenizer"]["stoi"])
    model = MiniGPT(payload["vocab_size"], config).to(device)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    metadata = {
        "step": payload["step"],
        "metrics": payload["metrics"],
        "config": config,
    }
    return model, tokenizer, metadata
