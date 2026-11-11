"""One-off inspection: compare checkpoint metadata (not part of the suite)."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from checkpoint import load_checkpoint  # noqa: E402

for name in ("shakespeare.pt", "latest.pt", "cuda-final.pt"):
    _, _, meta = load_checkpoint(Path("checkpoints") / name, torch.device("cpu"))
    metrics = meta["metrics"]
    best = metrics.get("best_val_loss")
    best_step = metrics.get("best_step")
    print(
        f"{name}: step={meta['step']:.0f} "
        f"best_val={best:.4f} at step {best_step:.0f}"
    )
