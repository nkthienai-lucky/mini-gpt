"""One-off: verify HF model repo creation still works on the free tier,
then upload the trained checkpoint with a model card."""

import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_ID = "mini-gpt-shakespeare"

MODEL_CARD = """---
language: en
license: mit
tags:
- text-generation
- transformer
- from-scratch
- educational
widget:
- text: "ROMEO:"
---

# mini-gpt-shakespeare

A decoder-only Transformer **trained from scratch** (no `nn.Transformer`,
no `scaled_dot_product_attention` — attention is hand-written Q/K/V, scaling,
causal mask, softmax, weighted values).

| | |
|---|---|
| Corpus | tiny Shakespeare (char-level, 65-token vocab) |
| Architecture | 4 layers, 4 heads, 128-dim, context 128, pre-norm |
| Training | 5,000 steps, AdamW, RTX 5060 |
| Validation loss | 4.24 → **1.61** |

Checkpoint format: `torch.save` dict with `format_version`, `config`,
`tokenizer`, `model_state_dict`, `step`, `metrics` — loads with
`checkpoint.load_checkpoint(path, device)` from
[nkthienai-lucky/mini-gpt](https://github.com/nkthienai-lucky/mini-gpt).

Demo prompt (greedy): `"ROMEO:"` → Shakespeare-like speaker headings and verse.
"""


def main() -> int:
    api = HfApi()
    owner = api.whoami()["name"]
    repo_id = f"{owner}/{MODEL_ID}"
    print(f"creating model repo {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)

    checkpoint = REPO_ROOT / "checkpoints" / "shakespeare.pt"
    if not checkpoint.is_file():
        print(f"missing checkpoint: {checkpoint}", file=sys.stderr)
        return 1
    api.upload_file(
        path_or_fileobj=checkpoint,
        path_in_repo="shakespeare.pt",
        repo_id=repo_id,
        repo_type="model",
    )
    print("uploaded shakespeare.pt")
    api.upload_file(
        path_or_fileobj=MODEL_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
    )
    print("uploaded model card")
    print(f"https://huggingface.co/{repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
