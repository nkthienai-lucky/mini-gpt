"""Central configuration for the educational Mini-GPT baseline."""

from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 1337
    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 5_000
    eval_interval: int = 250
    eval_iters: int = 50
    warmup_steps: int = 100
    device: str = "auto"
    temperature: float = 0.8
    top_k: int | None = 50
