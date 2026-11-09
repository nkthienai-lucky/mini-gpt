"""A small decoder-only Transformer implemented for learning."""

from __future__ import annotations

import math
from typing import TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from config import Config


KVPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
KVCache: TypeAlias = list[KVPair]


class CausalSelfAttention(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        self.n_head = config.n_head
        self.head_dim = config.n_embd // config.n_head
        self.q_proj = nn.Linear(config.n_embd, config.n_embd)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, channels = x.shape
        return x.view(batch, seq_len, self.n_head, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(batch, seq_len, -1)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVPair | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVPair | None]:
        if x.ndim != 3:
            raise ValueError("attention input must have shape (batch, sequence, embedding)")

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        past_len = 0
        if kv_cache is not None:
            past_k, past_v = kv_cache
            if past_k.shape[:2] != k.shape[:2] or past_v.shape[:2] != v.shape[:2]:
                raise ValueError("KV cache batch or head dimensions do not match input")
            if past_k.device != k.device or past_v.device != v.device:
                raise ValueError("KV cache and input must be on the same device")
            past_len = past_k.size(2)
            k = torch.cat((past_k, k), dim=2)
            v = torch.cat((past_v, v), dim=2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        query_positions = past_len + torch.arange(x.size(1), device=x.device)
        key_positions = torch.arange(k.size(2), device=x.device)
        causal_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
        weights = F.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)
        output = self._merge_heads(weights @ v)
        output = self.resid_dropout(self.out_proj(output))
        new_cache = (k, v) if use_cache else None
        return output, new_cache


class MLP(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.fc_in = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.activation = nn.GELU()
        self.fc_out = nn.Linear(4 * config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(self.activation(self.fc_in(x))))


class Block(nn.Module):
    def __init__(self, config: Config) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: KVPair | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, KVPair | None]:
        attention_output, new_cache = self.attention(
            self.ln_1(x), kv_cache=kv_cache, use_cache=use_cache
        )
        x = x + attention_output
        x = x + self.mlp(self.ln_2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, vocab_size: int, config: Config) -> None:
        super().__init__()
        if config.n_embd % config.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if vocab_size < 2:
            raise ValueError("vocab_size must be at least 2")
        self.config = config
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.final_norm = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _past_length(self, kv_cache: KVCache | None) -> int:
        return 0 if kv_cache is None else kv_cache[0][0].size(2)

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        kv_cache: KVCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, KVCache | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        if idx.size(1) == 0:
            raise ValueError("idx must contain at least one token")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        if kv_cache is not None and len(kv_cache) != len(self.blocks):
            raise ValueError("KV cache must contain one entry per Transformer block")

        past_len = self._past_length(kv_cache)
        if past_len + idx.size(1) > self.config.block_size:
            raise ValueError("sequence plus KV cache exceeds model block_size")

        positions = torch.arange(
            past_len, past_len + idx.size(1), device=idx.device
        )
        x = self.token_embedding(idx) + self.position_embedding(positions)
        new_cache: KVCache | None = [] if use_cache else None
        for layer_index, block in enumerate(self.blocks):
            layer_cache = None if kv_cache is None else kv_cache[layer_index]
            x, layer_new_cache = block(x, kv_cache=layer_cache, use_cache=use_cache)
            if new_cache is not None:
                assert layer_new_cache is not None
                new_cache.append(layer_new_cache)

        logits = self.lm_head(self.final_norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss, new_cache

    @staticmethod
    def _sample_next_token(
        logits: torch.Tensor,
        temperature: float,
        top_k: int | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor:
        if temperature <= 0:
            return torch.argmax(logits, dim=-1, keepdim=True)
        logits = logits / temperature
        if top_k is not None:
            if top_k < 1:
                raise ValueError("top_k must be positive when provided")
            top_k = min(top_k, logits.size(-1))
            values, _ = torch.topk(logits, top_k, dim=-1)
            threshold = values[..., -1, None]
            logits = logits.masked_fill(logits < threshold, float("-inf"))
        probabilities = F.softmax(logits, dim=-1)
        return torch.multinomial(probabilities, num_samples=1, generator=generator)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        use_kv_cache: bool = True,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if idx.ndim != 2:
            raise ValueError("idx must have shape (batch, sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if idx.size(1) + max_new_tokens > self.config.block_size:
            raise ValueError("generation would exceed model block_size")
        output = idx
        if max_new_tokens == 0:
            return output

        if use_kv_cache:
            logits, _, cache = self(output, use_cache=True)
            for step in range(max_new_tokens):
                next_token = self._sample_next_token(
                    logits[:, -1], temperature, top_k, generator
                )
                output = torch.cat((output, next_token), dim=1)
                if step + 1 < max_new_tokens:
                    logits, _, cache = self(
                        next_token, kv_cache=cache, use_cache=True
                    )
            return output

        for _ in range(max_new_tokens):
            logits, _, _ = self(output, use_cache=False)
            next_token = self._sample_next_token(
                logits[:, -1], temperature, top_k, generator
            )
            output = torch.cat((output, next_token), dim=1)
        return output
