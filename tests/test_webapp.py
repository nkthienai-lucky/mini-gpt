"""Tests for the web demo logic (UI-framework free)."""

import pytest
import torch

from config import Config
from data import CharTokenizer
from model import MiniGPT
from webapp import (
    DEFAULT_PROMPT,
    available_new_tokens,
    fit_to_context,
    generate_demo,
    sanitize_prompt,
)


def make_tokenizer() -> CharTokenizer:
    return CharTokenizer.from_text("ROMEO:\nabcdefghijklmnopqrstuvwxyz ,.")


def make_model(tokenizer: CharTokenizer) -> MiniGPT:
    config = Config(block_size=128, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = MiniGPT(tokenizer.vocab_size, config)
    model.eval()
    return model


def test_sanitize_prompt_removes_unknown_characters():
    tokenizer = make_tokenizer()

    cleaned = sanitize_prompt(tokenizer, "Xin chào, ROMEO:")

    assert set(cleaned) <= set(tokenizer.stoi)
    assert cleaned == "in cho, ROMEO:"


def test_sanitize_prompt_keeps_fully_known_text_unchanged():
    tokenizer = make_tokenizer()

    assert sanitize_prompt(tokenizer, "ROMEO: the king, first citizen.") == (
        "ROMEO: the king, first citizen."
    )


def test_sanitize_prompt_can_return_empty_string():
    tokenizer = make_tokenizer()

    assert sanitize_prompt(tokenizer, "日本語") == ""


def test_fit_to_context_keeps_prompt_within_limit():
    assert fit_to_context("ROMEO:", 128) == "ROMEO:"
    assert fit_to_context("a" * 96, 128, reserved=32) == "a" * 96


def test_fit_to_context_truncates_long_prompts_to_the_tail():
    prompt = "abcdefghij" * 30

    fitted = fit_to_context(prompt, 128, reserved=32)

    assert len(fitted) == 96
    assert fitted == prompt[-96:]


def test_available_new_tokens_is_context_minus_prompt():
    assert available_new_tokens(6, 128) == 122
    assert available_new_tokens(1, 128) == 127


def test_generate_demo_returns_prompt_with_continuation():
    tokenizer = make_tokenizer()
    model = make_model(tokenizer)

    output = generate_demo(
        model, tokenizer, "ROMEO:", 10, 0.0, None, use_kv_cache=True, seed=1
    )

    assert output.startswith("ROMEO:")
    assert len(output) > len("ROMEO:")


def test_generate_demo_is_deterministic_for_same_seed():
    tokenizer = make_tokenizer()
    model = make_model(tokenizer)

    first = generate_demo(
        model, tokenizer, "the king", 20, 0.8, 5, use_kv_cache=True, seed=7
    )
    second = generate_demo(
        model, tokenizer, "the king", 20, 0.8, 5, use_kv_cache=True, seed=7
    )

    assert first == second


def test_generate_demo_falls_back_to_default_prompt_when_unknown():
    tokenizer = make_tokenizer()
    model = make_model(tokenizer)

    output = generate_demo(
        model, tokenizer, "日本語", 5, 0.0, None, use_kv_cache=True, seed=1
    )

    assert output.startswith(DEFAULT_PROMPT)


def test_generate_demo_caps_new_tokens_to_context():
    tokenizer = make_tokenizer()
    model = make_model(tokenizer)
    prompt = "a" * 120

    output = generate_demo(
        model, tokenizer, prompt, 50, 0.0, None, use_kv_cache=True, seed=3
    )

    assert len(output) == 128


def test_generate_demo_accepts_unicode_without_raising():
    tokenizer = make_tokenizer()
    model = make_model(tokenizer)

    output = generate_demo(
        model, tokenizer, "🤖 Xin chào, ROMEO:", 8, 0.8, 10, use_kv_cache=True, seed=9
    )

    assert isinstance(output, str)
    assert output


def test_resolve_checkpoint_prefers_environment_override(tmp_path, monkeypatch):
    from webapp import resolve_checkpoint

    override = tmp_path / "custom.pt"
    override.write_bytes(b"placeholder")
    monkeypatch.setenv("MINIGPT_CHECKPOINT", str(override))

    assert resolve_checkpoint() == override


def test_resolve_checkpoint_falls_back_to_repo_checkpoint(monkeypatch):
    from webapp import resolve_checkpoint

    monkeypatch.delenv("MINIGPT_CHECKPOINT", raising=False)

    resolved = resolve_checkpoint()

    assert resolved.name == "shakespeare.pt"
    assert resolved.is_file()
