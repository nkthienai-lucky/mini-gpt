import json
from pathlib import Path

from lora.build_dataset import build_all
from lora.dataset import load_pairs


def test_build_is_deterministic(tmp_path):
    build_all(out_dir=tmp_path / "a")
    build_all(out_dir=tmp_path / "b")
    for name in ("train.jsonl", "eval.jsonl"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_build_produces_expected_counts(tmp_path):
    counts = build_all(out_dir=tmp_path)
    assert counts == {"train": 350, "eval": 30}
    assert len(load_pairs(tmp_path / "train.jsonl")) == 350
    assert len(load_pairs(tmp_path / "eval.jsonl")) == 30


def test_eval_questions_disjoint_from_train(tmp_path):
    build_all(out_dir=tmp_path)
    train_q = {p["messages"][0]["content"] for p in load_pairs(tmp_path / "train.jsonl")}
    eval_q = {p["messages"][0]["content"] for p in load_pairs(tmp_path / "eval.jsonl")}
    assert len(train_q & eval_q) == 0


def test_eval_answers_match_their_question_topics(tmp_path):
    import json as jsonlib

    build_all(out_dir=tmp_path)
    pairs = load_pairs(tmp_path / "eval.jsonl")
    by_question = {p["messages"][0]["content"]: p for p in pairs}
    # spot-check known question→topic alignments (question fragment → answer must mention)
    checks = {
        "Cache các tensor nào": "cache",
        "Adapter trong LoRA": "LoRA",
        "Nén model xuống 4-bit": "4-bit",
        "sinh JSON từ LLM nên đặt temperature": "Temperature",
        "Hàm mất mát của model sinh ngôn ngữ": "Cross-entropy",
        "Chuẩn hóa lớp trước hay sau block": "LayerNorm",
        "Chia nhỏ tài liệu ảnh hưởng": "Chunking",
        "Gradient accumulation giúp gì": "Batch size",  # topic's answer covers batching/VRAM; accumulation is in its reason
        "Residual connection đặt ở đâu": "Residual",
    }
    for fragment, keyword in checks.items():
        question = next(q for q in by_question if fragment in q)
        answer = jsonlib.loads(by_question[question]["messages"][1]["content"])["answer"]
        assert keyword.lower() in answer.lower(), (question, answer)
