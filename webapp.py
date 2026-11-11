"""Framework-free generation logic for the web demo."""

from __future__ import annotations

import os
from pathlib import Path

import torch

from data import CharTokenizer
from model import MiniGPT

DEFAULT_PROMPT = "ROMEO:"
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "shakespeare.pt"
DEFAULT_RESERVED = 32
HF_REPO_ID = "Thienpkae/mini-gpt-shakespeare"
HF_FILENAME = "shakespeare.pt"


def hf_hub_download(repo_id: str, filename: str, local_dir: Path) -> Path:  # pragma: no cover - thin wrapper
    from huggingface_hub import hf_hub_download as _download

    return Path(_download(repo_id=repo_id, filename=filename, local_dir=local_dir))


def sanitize_prompt(tokenizer: CharTokenizer, prompt: str) -> str:
    """Drop every character the character-level vocabulary cannot encode."""
    return "".join(char for char in prompt if char in tokenizer.stoi)


def fit_to_context(prompt: str, block_size: int, reserved: int = DEFAULT_RESERVED) -> str:
    """Keep the tail of the prompt so prompt + reserved tokens fit the context."""
    budget = max(block_size - reserved, 0)
    return prompt[-budget:] if len(prompt) > budget else prompt


def available_new_tokens(prompt_length: int, block_size: int) -> int:
    return max(block_size - prompt_length, 0)


def generate_demo(
    model: MiniGPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    use_kv_cache: bool,
    seed: int,
) -> str:
    """Generate text from a user prompt, never raising on unknown characters."""
    block_size = model.config.block_size
    cleaned = sanitize_prompt(tokenizer, prompt)
    if not cleaned:
        cleaned = DEFAULT_PROMPT
    cleaned = fit_to_context(cleaned, block_size)
    budget = min(max_new_tokens, available_new_tokens(len(cleaned), block_size))
    prompt_ids = tokenizer.encode(cleaned).unsqueeze(0)
    generator = torch.Generator(device=prompt_ids.device).manual_seed(seed)
    generated = model.generate(
        prompt_ids,
        max_new_tokens=budget,
        temperature=temperature,
        top_k=top_k,
        use_kv_cache=use_kv_cache,
        generator=generator,
    )
    return tokenizer.decode(generated[0])


def resolve_checkpoint() -> Path:
    """Resolve the demo checkpoint: env override first, then the repo default."""
    override = os.environ.get("MINIGPT_CHECKPOINT")
    if override:
        return Path(override)
    return DEFAULT_CHECKPOINT


def ensure_checkpoint(path: Path | None = None) -> Path:
    """Return a readable checkpoint path, downloading from the HF Hub if missing."""
    path = path or resolve_checkpoint()
    if path.is_file():
        return path
    downloaded = hf_hub_download(
        HF_REPO_ID,
        HF_FILENAME,
        path.parent,
    )
    if not downloaded.is_file():
        raise FileNotFoundError(f"checkpoint not found locally or on the Hub: {path}")
    return downloaded
