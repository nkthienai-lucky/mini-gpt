"""Hand-written byte-level BPE tokenizers (Karpathy minbpe style)."""

from __future__ import annotations

from collections import Counter


def _merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every non-overlapping occurrence of ``pair`` in ``ids``."""
    merged: list[int] = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BasicTokenizer:
    """Byte-level BPE without any pre-splitting (minbpe's BasicTokenizer)."""

    def __init__(self) -> None:
        self.merges: dict[tuple[int, int], int] = {}
        self.vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        if vocab_size < 256:
            raise ValueError(f"vocab_size must be at least 256, got {vocab_size}")
        ids: list[int] = list(text.encode("utf-8"))
        for i in range(vocab_size - 256):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break
            # max() breaks ties by first-seen pair, matching minbpe.
            best = max(counts, key=counts.get)
            new_id = 256 + i
            ids = _merge(ids, best, new_id)
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            if verbose:
                print(f"merge {i + 1}: {best} -> {new_id}")

    def encode(self, text: str) -> list[int]:
        ids: list[int] = list(text.encode("utf-8"))
        while len(ids) >= 2:
            counts = Counter(zip(ids, ids[1:]))
            pair = min(
                (p for p in counts if p in self.merges),
                key=lambda p: self.merges[p],
                default=None,
            )
            if pair is None:
                break
            ids = _merge(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: list[int]) -> str:
        pieces: list[bytes] = []
        for token_id in ids:
            if token_id not in self.vocab:
                raise ValueError(f"unknown token id: {token_id}")
            pieces.append(self.vocab[token_id])
        return b"".join(pieces).decode("utf-8", errors="replace")
