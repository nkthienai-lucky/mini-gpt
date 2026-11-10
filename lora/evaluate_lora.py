"""Before/after JSON-compliance evaluation for the LoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from lora.dataset import load_pairs
from lora.format_metric import evaluate_model
from lora.qlora import load_base_model, load_tokenizer


def run_eval(
    adapter_dir: Path | None,
    questions: list[str],
    data_path: Path | None = None,
) -> dict[str, Any]:
    """Evaluate base vs adapter on the same questions, greedy decoding."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    base = load_base_model().to(device)
    before = evaluate_model(base, tokenizer, questions, device=device)

    if adapter_dir is None:
        return {"before": before, "after": before}

    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(base, adapter_dir).to(device)
    after = evaluate_model(peft_model, tokenizer, questions, device=device)
    return {"before": before, "after": after}


def main(
    adapter_dir: Path = Path("lora/adapter"),
    data_path: Path = Path("lora/data/eval.jsonl"),
    out_path: Path = Path("lora/results.json"),
) -> None:
    questions = [p["messages"][0]["content"] for p in load_pairs(data_path)]
    result = run_eval(adapter_dir=adapter_dir, questions=questions)

    print("| Phase | n | parse_ok | schema_ok | schema_rate |")
    print("|---|---:|---:|---:|---:|")
    for phase in ("before", "after"):
        m = result[phase]
        print(
            f"| {phase} | {m.get('n')} | {m.get('parse_ok', '-')} "
            f"| {m.get('schema_ok', '-')} | {m['schema_rate']:.2%} |"
        )
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"results written to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Before/after JSON-format evaluation")
    parser.add_argument("--adapter-dir", type=Path, default=Path("lora/adapter"))
    parser.add_argument("--data", type=Path, default=Path("lora/data/eval.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("lora/results.json"))
    args = parser.parse_args()
    main(adapter_dir=args.adapter_dir, data_path=args.data, out_path=args.out)
