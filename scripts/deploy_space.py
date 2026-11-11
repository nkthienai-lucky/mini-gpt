"""Deploy the web demo to a Hugging Face Space.

Usage (from repo root, with an HF token logged in):
    python scripts/deploy_space.py

Creates the Space if missing, then uploads the runtime files and the trained
checkpoint. Idempotent: re-running updates the Space in place.
"""

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
SPACE_ID = "Thienpkae/mini-gpt"
SDK_VERSION = "6.31.0"

README = f"""---
title: Mini-GPT Shakespeare
emoji: 🎭
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: {SDK_VERSION}
app_file: app.py
pinned: false
license: mit
short_description: Character-level Transformer trained from scratch on tiny Shakespeare
---

# Mini-GPT — Shakespeare

A decoder-only Transformer **trained from scratch** on tiny Shakespeare (char-level,
65-token vocabulary). The attention core is hand-written — no `nn.Transformer`,
no `nn.MultiheadAttention`, no `scaled_dot_product_attention`:

- 4 layers, 4 heads, 128-dim embeddings, context 128
- trained 5,000 steps on an RTX 5060: validation loss 4.24 → **1.61**
- generation supports temperature, top-k, seeding, and a KV-cache toggle

Source and training evidence: [github.com/nkthienai-lucky/mini-gpt](https://github.com/nkthienai-lucky/mini-gpt)

The prompt is sanitized to the Shakespeare vocabulary; unknown characters
(Vietnamese diacritics, emoji) are dropped before generation.
"""

RUNTIME_FILES = [
    "app.py",
    "webapp.py",
    "model.py",
    "data.py",
    "config.py",
    "checkpoint.py",
]

REQUIREMENTS = "torch>=2.11\ngradio>=6.0\n"


def main() -> int:
    api = HfApi()
    who = api.whoami()
    owner = who["name"]
    space_id = f"{owner}/mini-gpt"
    print(f"deploying as {owner} -> {space_id}")

    api.create_repo(
        repo_id=space_id,
        repo_type="space",
        space_sdk="gradio",
        private=False,
        exist_ok=True,
    )

    api.upload_file(
        path_or_fileobj=README.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=space_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=REQUIREMENTS.encode("utf-8"),
        path_in_repo="requirements.txt",
        repo_id=space_id,
        repo_type="space",
    )

    for name in RUNTIME_FILES:
        path = REPO_ROOT / name
        if not path.is_file():
            print(f"missing runtime file: {path}", file=sys.stderr)
            return 1
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=space_id,
            repo_type="space",
        )
        print(f"uploaded {name}")

    checkpoint = REPO_ROOT / "checkpoints" / "shakespeare.pt"
    if not checkpoint.is_file():
        print(f"missing checkpoint: {checkpoint}", file=sys.stderr)
        return 1
    api.upload_file(
        path_or_fileobj=checkpoint,
        path_in_repo="checkpoints/shakespeare.pt",
        repo_id=space_id,
        repo_type="space",
    )
    print("uploaded checkpoints/shakespeare.pt")

    print(f"Space: https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
