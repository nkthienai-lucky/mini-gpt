import torch
import pytest

from config import Config
from data import CharTokenizer
from model import MiniGPT
from sample import generate_text
from benchmark import benchmark_generation


def small_config(**overrides) -> Config:
    values = {
        "seed": 11,
        "block_size": 12,
        "n_layer": 1,
        "n_head": 2,
        "n_embd": 16,
        "dropout": 0.0,
    }
    values.update(overrides)
    return Config(**values)


def make_model_and_tokenizer():
    tokenizer = CharTokenizer.from_text("abcd")
    model = MiniGPT(tokenizer.vocab_size, small_config())
    model.eval()
    return model, tokenizer


def test_generate_text_returns_prompt_and_continuation():
    model, tokenizer = make_model_and_tokenizer()

    result = generate_text(
        model,
        tokenizer,
        prompt="ab",
        max_new_tokens=3,
        temperature=0,
        top_k=None,
        use_kv_cache=True,
        device=torch.device("cpu"),
        seed=3,
    )

    assert result.startswith("ab")
    assert len(result) == 5


def test_fixed_seed_and_settings_reproduce_sampled_text():
    model, tokenizer = make_model_and_tokenizer()
    kwargs = {
        "model": model,
        "tokenizer": tokenizer,
        "prompt": "ab",
        "max_new_tokens": 4,
        "temperature": 0.8,
        "top_k": 2,
        "use_kv_cache": True,
        "device": torch.device("cpu"),
        "seed": 19,
    }

    assert generate_text(**kwargs) == generate_text(**kwargs)


def test_cached_and_uncached_greedy_generation_match():
    model, tokenizer = make_model_and_tokenizer()
    prompt = tokenizer.encode("ab").unsqueeze(0)

    cached = model.generate(prompt, 5, temperature=0, use_kv_cache=True)
    uncached = model.generate(prompt, 5, temperature=0, use_kv_cache=False)

    assert torch.equal(cached, uncached)


def test_default_context_supports_documented_64_token_sample():
    tokenizer = CharTokenizer.from_text("abcd")
    model = MiniGPT(tokenizer.vocab_size, Config())
    prompt = tokenizer.encode("ab").unsqueeze(0)

    result = model.generate(prompt, 64, temperature=0)

    assert result.shape == (1, 66)


def test_generation_rejects_context_overflow():
    model, tokenizer = make_model_and_tokenizer()
    prompt = tokenizer.encode("abcdabcdabcd").unsqueeze(0)

    with pytest.raises(ValueError, match="exceed model block_size"):
        model.generate(prompt, 1, temperature=0)

    shorter = tokenizer.encode("abcd").unsqueeze(0)
    result = model.generate(shorter, 1, temperature=0)
    assert result.shape == (1, 5)


def test_benchmark_reports_both_modes_and_token_counts():
    model, tokenizer = make_model_and_tokenizer()
    prompt_ids = tokenizer.encode("ab").unsqueeze(0)

    result = benchmark_generation(model, prompt_ids, new_tokens=3, repeats=2, warmup=1)

    assert set(result) == {"uncached", "cached"}
    for record in result.values():
        assert record["generated_tokens"] == 3
        assert record["prompt_tokens"] == 2
        assert record["total_seconds"] >= 0
        assert record["milliseconds_per_token"] >= 0
        assert record["tokens_per_second"] >= 0
