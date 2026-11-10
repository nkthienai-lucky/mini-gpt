"""Dataset schema validation and JSONL IO for the LoRA exercise."""

from __future__ import annotations

import json
from pathlib import Path

PAIR_SCHEMA_KEYS = ("answer", "reason", "lang")


def validate_pair(pair: dict) -> None:
    """Validate one training record; raise ValueError describing the violation."""
    messages = pair.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise ValueError("record must contain exactly two messages")
    user_msg, assistant_msg = messages
    if user_msg.get("role") != "user" or assistant_msg.get("role") != "assistant":
        raise ValueError("messages must be ordered user then assistant")
    question = user_msg.get("content")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("user content must be a non-empty string")
    try:
        answer = json.loads(assistant_msg.get("content", ""))
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError(f"assistant content is not valid JSON: {exc}") from exc
    if not isinstance(answer, dict):
        raise ValueError("assistant content must be a JSON object")
    if set(answer.keys()) != set(PAIR_SCHEMA_KEYS):
        raise ValueError(
            f"answer keys must be exactly {PAIR_SCHEMA_KEYS}, got {sorted(answer.keys())}"
        )
    for key, value in answer.items():
        if not isinstance(value, str):
            raise ValueError(f"value of '{key}' must be a string")
    if answer["lang"] != "vi":
        raise ValueError(f"lang must be 'vi', got {answer['lang']!r}")


def load_pairs(path: Path) -> list[dict]:
    """Load and validate a JSONL file of records; blank lines are skipped."""
    pairs: list[dict] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: line {line_no}: not valid JSON: {exc}") from exc
            try:
                validate_pair(record)
            except ValueError as exc:
                raise ValueError(f"{path}: line {line_no}: {exc}") from exc
            pairs.append(record)
    return pairs
