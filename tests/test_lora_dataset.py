import json
from pathlib import Path

import pytest

from lora.dataset import load_pairs, validate_pair


def make_pair(question="Transformer là gì?", answer="Kiến trúc neural.", reason="Vì nó dùng attention."):
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": json.dumps(
            {"answer": answer, "reason": reason, "lang": "vi"}, ensure_ascii=False)},
    ]}


def write_jsonl(path: Path, records: list[dict], blank_lines: tuple[int, ...] = ()) -> None:
    lines = [json.dumps(r, ensure_ascii=False) for r in records]
    for idx in sorted(blank_lines, reverse=True):
        lines.insert(idx, "")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_validate_pair_accepts_valid_record():
    validate_pair(make_pair())  # no raise


def test_validate_pair_rejects_missing_key():
    pair = make_pair()
    content = json.loads(pair["messages"][1]["content"])
    del content["reason"]
    pair["messages"][1]["content"] = json.dumps(content, ensure_ascii=False)
    with pytest.raises(ValueError, match="reason"):
        validate_pair(pair)


def test_validate_pair_rejects_extra_key():
    pair = make_pair()
    content = json.loads(pair["messages"][1]["content"])
    content["extra"] = "x"
    pair["messages"][1]["content"] = json.dumps(content, ensure_ascii=False)
    with pytest.raises(ValueError, match="extra"):
        validate_pair(pair)


def test_validate_pair_rejects_wrong_lang():
    pair = make_pair()
    content = json.loads(pair["messages"][1]["content"])
    content["lang"] = "en"
    pair["messages"][1]["content"] = json.dumps(content, ensure_ascii=False)
    with pytest.raises(ValueError, match="lang"):
        validate_pair(pair)


def test_validate_pair_rejects_non_json_assistant_content():
    pair = make_pair()
    pair["messages"][1]["content"] = "Câu trả lời thường thì là vậy."
    with pytest.raises(ValueError):
        validate_pair(pair)


def test_validate_pair_rejects_empty_question():
    pair = make_pair(question="  ")
    with pytest.raises(ValueError):
        validate_pair(pair)


def test_load_pairs_reads_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "pairs.jsonl"
    write_jsonl(path, [make_pair(), make_pair(), make_pair()], blank_lines=(1, 2))
    pairs = load_pairs(path)
    assert len(pairs) == 3


def test_load_pairs_reports_line_number_of_bad_record(tmp_path):
    path = tmp_path / "pairs.jsonl"
    records = [make_pair(), make_pair(), make_pair()]
    records[1]["messages"][1]["content"] = "not json"
    write_jsonl(path, records)
    with pytest.raises(ValueError, match="line 2"):
        load_pairs(path)
