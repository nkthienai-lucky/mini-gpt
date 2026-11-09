"""Text generation CLI and user-facing sampling helper."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from benchmark import benchmark_generation
from checkpoint import load_checkpoint
from data import CharTokenizer
from model import MiniGPT


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(name)


def generate_text(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    use_kv_cache: bool,
    device: torch.device,
    seed: int,
) -> str:
    prompt_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        use_kv_cache=use_kv_cache,
        generator=generator,
    )
    return tokenizer.decode(generated[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a Mini-GPT checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--no-kv-cache", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--benchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = _resolve_device(args.device)
    model, tokenizer, metadata = load_checkpoint(args.checkpoint, device)
    output = generate_text(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_k,
        not args.no_kv_cache,
        device,
        args.seed,
    )
    print(output)
    if args.benchmark:
        prompt_ids = tokenizer.encode(args.prompt).unsqueeze(0).to(device)
        results = benchmark_generation(
            model, prompt_ids, args.max_new_tokens, device=device
        )
        print(json.dumps({"checkpoint_step": metadata["step"], **results}, indent=2))


if __name__ == "__main__":
    main()
