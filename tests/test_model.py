from pathlib import Path

import pytest
import torch

from config import Config
from model import MiniGPT


def small_config(**overrides) -> Config:
    values = {
        "block_size": 8,
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
    }
    values.update(overrides)
    return Config(**values)


def test_forward_returns_logits_and_scalar_cross_entropy():
    model = MiniGPT(vocab_size=11, config=small_config())
    idx = torch.randint(11, (2, 6))
    targets = torch.randint(11, (2, 6))

    logits, loss, cache = model(idx, targets=targets)

    assert logits.shape == (2, 6, 11)
    assert loss is not None and loss.ndim == 0
    assert cache is None


def test_forward_backward_produces_finite_gradients():
    model = MiniGPT(vocab_size=11, config=small_config())
    idx = torch.randint(11, (2, 6))
    targets = torch.randint(11, (2, 6))

    _, loss, _ = model(idx, targets=targets)
    assert loss is not None
    loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_future_tokens_do_not_change_earlier_logits():
    model = MiniGPT(vocab_size=11, config=small_config())
    model.eval()
    prefix = torch.tensor([[1, 2, 3, 4]])
    changed_future = torch.tensor([[1, 2, 3, 9]])

    with torch.no_grad():
        first_logits, _, _ = model(prefix)
        second_logits, _, _ = model(changed_future)

    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6)


def test_non_divisible_embedding_dimension_is_rejected():
    with pytest.raises(ValueError, match="n_embd must be divisible by n_head"):
        MiniGPT(vocab_size=11, config=small_config(n_embd=15))


def test_cached_next_token_logits_match_uncached_logits():
    model = MiniGPT(vocab_size=11, config=small_config())
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4]])
    next_token = torch.tensor([[5]])

    with torch.no_grad():
        _, _, cache = model(prompt, use_cache=True)
        cached_logits, _, _ = model(next_token, kv_cache=cache, use_cache=True)
        uncached_logits, _, _ = model(torch.cat([prompt, next_token], dim=1))

    assert torch.allclose(cached_logits[:, -1], uncached_logits[:, -1], atol=1e-5)


def test_model_source_does_not_use_forbidden_attention_helpers():
    source = Path("model.py").read_text(encoding="utf-8")

    for forbidden in (
        "nn.Transformer",
        "nn.TransformerEncoder",
        "nn.TransformerDecoder",
        "nn.MultiheadAttention",
        "scaled_dot_product_attention",
    ):
        assert forbidden not in source
