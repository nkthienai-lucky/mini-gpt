from pathlib import Path

import torch

from checkpoint import load_checkpoint, save_checkpoint
from config import Config
from data import CharTokenizer, TextDataset, get_batch
from model import MiniGPT
from train import estimate_loss, train_model


def tiny_config(**overrides) -> Config:
    values = {
        "seed": 7,
        "block_size": 4,
        "n_layer": 1,
        "n_head": 1,
        "n_embd": 8,
        "dropout": 0.0,
        "batch_size": 4,
        "learning_rate": 0.05,
        "max_steps": 10,
        "eval_interval": 5,
        "eval_iters": 2,
        "warmup_steps": 0,
        "device": "cpu",
    }
    values.update(overrides)
    return Config(**values)


def test_tiny_training_run_reduces_loss(tmp_path: Path):
    data_path = tmp_path / "input.txt"
    data_path.write_text("abcd" * 40, encoding="utf-8")
    checkpoint_path = tmp_path / "checkpoint.pt"

    metrics = train_model(tiny_config(max_steps=20), data_path, checkpoint_path)

    assert checkpoint_path.exists()
    assert metrics["train_loss"][-1] < metrics["train_loss"][0]


def test_estimate_loss_returns_finite_train_and_validation_values():
    tokenizer = CharTokenizer.from_text("abcd" * 20)
    tokens = tokenizer.encode("abcd" * 20)
    dataset = TextDataset(tokenizer, tokens[:60], tokens[60:])
    config = tiny_config(eval_iters=2)
    model = MiniGPT(tokenizer.vocab_size, config)

    losses = estimate_loss(model, dataset, config)

    assert set(losses) == {"train", "val"}
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())


def test_checkpoint_round_trip_restores_logits_and_metadata(tmp_path: Path):
    config = tiny_config()
    tokenizer = CharTokenizer.from_text("abcd" * 20)
    model = MiniGPT(tokenizer.vocab_size, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    path = tmp_path / "checkpoint.pt"
    idx = torch.tensor([[0, 1, 2, 3]])

    with torch.no_grad():
        expected_logits, _, _ = model(idx)
    save_checkpoint(path, model, optimizer, config, tokenizer, step=12, metrics={"train_loss": [1.0]})

    restored, restored_tokenizer, metadata = load_checkpoint(path, torch.device("cpu"))
    with torch.no_grad():
        actual_logits, _, _ = restored(idx)

    assert torch.equal(actual_logits, expected_logits)
    assert restored.config == config
    assert restored_tokenizer.stoi == tokenizer.stoi
    assert metadata["step"] == 12


def test_checkpoint_loads_with_explicit_cpu_map_location(tmp_path: Path):
    config = tiny_config()
    tokenizer = CharTokenizer.from_text("abcd" * 20)
    model = MiniGPT(tokenizer.vocab_size, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    path = tmp_path / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, config, tokenizer, step=0, metrics={})
    restored, _, _ = load_checkpoint(path, torch.device("cpu"))

    assert next(restored.parameters()).device == torch.device("cpu")
