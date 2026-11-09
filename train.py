"""Training loop and command-line entrypoint for Mini-GPT."""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch

from checkpoint import save_checkpoint
from config import Config
from data import TextDataset, get_batch, load_text_dataset
from model import MiniGPT


def resolve_device(config: Config) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(config.device)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def estimate_loss(
    model: MiniGPT,
    dataset: TextDataset,
    config: Config,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device
    losses: dict[str, float] = {}
    with torch.no_grad():
        for name, data in (("train", dataset.train), ("val", dataset.val)):
            values = []
            for _ in range(config.eval_iters):
                x, y = get_batch(
                    data,
                    config.block_size,
                    config.batch_size,
                    device,
                )
                _, loss, _ = model(x, targets=y)
                assert loss is not None
                values.append(loss.item())
            losses[name] = sum(values) / len(values)
    model.train()
    return losses


def learning_rate_at(config: Config, step: int) -> float:
    if config.max_steps <= config.warmup_steps:
        return config.learning_rate
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return config.learning_rate * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


def train_model(
    config: Config,
    data_path: Path,
    checkpoint_path: Path,
) -> dict[str, Any]:
    set_seed(config.seed)
    device = resolve_device(config)
    dataset = load_text_dataset(data_path)
    if min(dataset.train.numel(), dataset.val.numel()) < config.block_size + 1:
        raise ValueError("both train and validation splits must fit block_size + 1")

    model = MiniGPT(dataset.tokenizer.vocab_size, config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    metrics: dict[str, list[float] | float] = {
        "steps": [],
        "train_loss": [],
        "val_loss": [],
        "best_val_loss": float("inf"),
        "best_step": -1.0,
    }
    latest_path = checkpoint_path.parent / "latest.pt"
    best_path = checkpoint_path.parent / "best.pt"
    generator = torch.Generator(device=dataset.train.device).manual_seed(config.seed)

    for step in range(config.max_steps):
        current_lr = learning_rate_at(config, step)
        for group in optimizer.param_groups:
            group["lr"] = current_lr
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, dataset, config)
            metrics["steps"].append(float(step))
            metrics["train_loss"].append(losses["train"])
            metrics["val_loss"].append(losses["val"])
            if losses["val"] < metrics["best_val_loss"]:
                metrics["best_val_loss"] = losses["val"]
                metrics["best_step"] = float(step)
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    config,
                    dataset.tokenizer,
                    step=step,
                    metrics=metrics,
                )
            save_checkpoint(
                latest_path,
                model,
                optimizer,
                config,
                dataset.tokenizer,
                step=step,
                metrics=metrics,
            )
            print(
                f"step {step:>6} | train loss {losses['train']:.4f} "
                f"| val loss {losses['val']:.4f} | lr {current_lr:.2e}"
            )
        x, y = get_batch(
            dataset.train,
            config.block_size,
            config.batch_size,
            device,
            generator=generator,
        )
        _, loss, _ = model(x, targets=y)
        assert loss is not None
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        config,
        dataset.tokenizer,
        step=config.max_steps,
        metrics=metrics,
    )
    save_checkpoint(
        latest_path,
        model,
        optimizer,
        config,
        dataset.tokenizer,
        step=config.max_steps,
        metrics=metrics,
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Mini-GPT character model")
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config(
        device=args.device,
        max_steps=args.max_steps if args.max_steps is not None else Config.max_steps,
        eval_interval=(
            args.eval_interval if args.eval_interval is not None else Config.eval_interval
        ),
        batch_size=args.batch_size if args.batch_size is not None else Config.batch_size,
        seed=args.seed if args.seed is not None else Config.seed,
    )
    print(f"device={resolve_device(config)} config={config}")
    train_model(config, args.data, args.checkpoint)


if __name__ == "__main__":
    main()
