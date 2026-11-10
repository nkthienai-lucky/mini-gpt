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
