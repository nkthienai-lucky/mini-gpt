"""JSON extraction, schema validation and response scoring."""

from __future__ import annotations

import json
from typing import Any

from lora.dataset import PAIR_SCHEMA_KEYS


def extract_json(text: str) -> str | None:
    """Return the first balanced {...} substring, or None.

    Tracks string-literal state so braces inside JSON strings (e.g. '}') do
    not close the object early.
    """
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def validate_answer(text: str) -> bool:
    """True iff text contains exactly one schema-conformant JSON object."""
    extracted = extract_json(text)
    if extracted is None:
        return False
    try:
        obj = json.loads(extracted)
    except json.JSONDecodeError:
        return False
    if not isinstance(obj, dict):
        return False
    if set(obj.keys()) != set(PAIR_SCHEMA_KEYS):
        return False
    if not all(isinstance(v, str) for v in obj.values()):
        return False
    return obj["lang"] == "vi"


def score_responses(responses: list[str]) -> dict[str, Any]:
    """Aggregate parse/schema statistics over raw model responses."""
    parse_ok = 0
    schema_ok = 0
    for response in responses:
        extracted = extract_json(response)
        if extracted is not None:
            try:
                json.loads(extracted)
                parse_ok += 1
            except json.JSONDecodeError:
                pass
        if validate_answer(response):
            schema_ok += 1
    n = len(responses)
    return {
        "n": n,
        "parse_ok": parse_ok,
        "schema_ok": schema_ok,
        "schema_rate": schema_ok / n if n else 0.0,
    }


def evaluate_model(
    model,
    tokenizer,
    questions: list[str],
    device: str = "cuda",
    max_new_tokens: int = 200,
) -> dict[str, Any]:
    """Generate greedy answers for each question and score JSON compliance."""
    model.eval()
    responses: list[str] = []
    for question in questions:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        with __import__("torch").inference_mode():
            out = model.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        responses.append(tokenizer.decode(out[0][prompt.shape[1]:], skip_special_tokens=True))
    return {**score_responses(responses), "responses": responses}
