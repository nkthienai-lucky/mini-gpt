"""Character-level dataset and batch utilities for Mini-GPT."""

from dataclasses import dataclass
from pathlib import Path

import torch


class CharTokenizer:
    """A deterministic character-to-integer tokenizer."""

    def __init__(self, stoi: dict[str, int]) -> None:
        self.stoi = dict(stoi)
        self.itos = {index: char for char, index in self.stoi.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        if not text:
            raise ValueError("text must contain at least two characters")
        return cls({char: index for index, char in enumerate(sorted(set(text)))})

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> torch.Tensor:
        unknown = sorted(set(text) - self.stoi.keys())
        if unknown:
            raise ValueError(f"text contains unknown characters: {unknown!r}")
        return torch.tensor([self.stoi[char] for char in text], dtype=torch.long)

    def decode(self, ids: torch.Tensor | list[int]) -> str:
        values = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        try:
            return "".join(self.itos[int(index)] for index in values)
        except KeyError as exc:
            raise ValueError(f"unknown token id: {exc.args[0]}") from exc


@dataclass(frozen=True)
class TextDataset:
    tokenizer: CharTokenizer
    train: torch.Tensor
    val: torch.Tensor


def load_text_dataset(path: Path, train_fraction: float = 0.9) -> TextDataset:
    text = path.read_text(encoding="utf-8")
    if len(text) < 2:
        raise ValueError("text must contain at least two characters")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")

    tokenizer = CharTokenizer.from_text(text)
    encoded = tokenizer.encode(text)
    split_index = int(len(encoded) * train_fraction)
    if split_index < 1 or len(encoded) - split_index < 1:
        raise ValueError("text must provide at least one token for train and validation")
    return TextDataset(
        tokenizer=tokenizer,
        train=encoded[:split_index],
        val=encoded[split_index:],
    )


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if data.ndim != 1:
        raise ValueError("data must be a one-dimensional token tensor")
    if block_size < 1 or data.numel() < block_size + 1:
        raise ValueError("data must contain at least block_size + 1 tokens")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    starts = torch.randint(
        data.numel() - block_size,
        (batch_size,),
        generator=generator,
        device=data.device,
    )
    offsets = torch.arange(block_size, device=data.device)
    indices = starts.unsqueeze(1) + offsets
    x = data[indices]
    y = data[indices + 1]
    return x.to(device), y.to(device)
