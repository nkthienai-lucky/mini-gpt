"""QLoRA training loop with assistant-only loss masking."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from lora.dataset import load_pairs
from lora.qlora import attach_lora, load_base_model, load_tokenizer

MAX_LEN = 512
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
GRAD_ACCUM = 2
EPOCHS = 3
LOG_EVERY = 10


def build_example(tokenizer, pair: dict, max_len: int = MAX_LEN) -> tuple[list[int], list[int]]:
    """Tokenize one pair; label only the assistant answer (+terminator).

    Applies the chat template as STRINGS and encodes explicitly:
    ``apply_chat_template(tokenize=True)`` on fast tokenizers can return a
    list of Encoding objects rather than a plain id list, which silently
    breaks prefix-based masking.
    """
    user_msg, assistant_msg = pair["messages"]
    prompt_text = tokenizer.apply_chat_template(
        [user_msg], tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        [user_msg, assistant_msg], tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError("chat template prompt is not a prefix of the full render")
    prompt_len = len(prompt_ids)
    # keep the newest tokens: if over budget, drop oldest prompt tokens
    if len(full_ids) > max_len:
        overflow = len(full_ids) - max_len
        full_ids = full_ids[overflow:]
        prompt_len = max(prompt_len - overflow, 0)
    labels = [-100] * len(full_ids)
    for i in range(prompt_len, len(full_ids)):
        labels[i] = full_ids[i]
    return full_ids, labels


def train(
    out_dir: Path = Path("lora/adapter"),
    data_path: Path = Path("lora/data/train.jsonl"),
    max_steps: int | None = None,
) -> dict[str, Any]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = load_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_base_model().to(device)
    peft_model = attach_lora(model)

    pairs = load_pairs(Path(data_path))
    examples = [build_example(tokenizer, p) for p in pairs]

    optimizer = torch.optim.AdamW(
        (p for p in peft_model.parameters() if p.requires_grad), lr=LEARNING_RATE
    )
    peft_model.train()

    steps: list[int] = []
    losses: list[float] = []
    generator = torch.Generator().manual_seed(1337)
    order = torch.randperm(len(examples), generator=generator).tolist()
    accum_batches: list[list[tuple[list[int], list[int]]]] = []
    for start in range(0, len(order), BATCH_SIZE * GRAD_ACCUM):
        chunk = [examples[i] for i in order[start : start + BATCH_SIZE * GRAD_ACCUM]]
        for b in range(0, len(chunk), BATCH_SIZE):
            accum_batches.append(chunk[b : b + BATCH_SIZE])

    step = 0
    for epoch in range(EPOCHS):
        if max_steps is not None and step >= max_steps:
            break
        g = torch.Generator().manual_seed(1337 + epoch)
        epoch_order = torch.randperm(len(accum_batches), generator=g).tolist()
        for batch_idx in epoch_order:
            batch = accum_batches[batch_idx]
            maxlen = max(len(ids) for ids, _ in batch)
            input_ids = torch.full((len(batch), maxlen), tokenizer.pad_token_id, dtype=torch.long)
            labels = torch.full((len(batch), maxlen), -100, dtype=torch.long)
            attn = torch.zeros((len(batch), maxlen), dtype=torch.long)
            for row, (ids, lab) in enumerate(batch):
                input_ids[row, : len(ids)] = torch.tensor(ids)
                labels[row, : len(lab)] = torch.tensor(lab)
                attn[row, : len(ids)] = 1
            out = peft_model(
                input_ids=input_ids.to(device),
                attention_mask=attn.to(device),
                labels=labels.to(device),
            )
            loss = out.loss / GRAD_ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(
                    (p for p in peft_model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if step % LOG_EVERY == 0:
                print(f"step {step:>5} | epoch {epoch} | loss {loss.item() * GRAD_ACCUM:.4f}")
                steps.append(step)
                losses.append(loss.item() * GRAD_ACCUM)
            step += 1
            if max_steps is not None and step >= max_steps:
                break

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(out_dir)
    print(f"adapter saved to {out_dir}")
    return {"steps": steps, "loss": losses}


def main() -> None:
    parser = argparse.ArgumentParser(description="QLoRA-train the JSON-format adapter")
    parser.add_argument("--out-dir", type=Path, default=Path("lora/adapter"))
    parser.add_argument("--data", type=Path, default=Path("lora/data/train.jsonl"))
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()
    train(out_dir=args.out_dir, data_path=args.data, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
