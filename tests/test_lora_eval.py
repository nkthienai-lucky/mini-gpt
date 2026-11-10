import json
from pathlib import Path

from lora.evaluate_lora import run_eval


def test_run_eval_returns_both_phases(monkeypatch, tmp_path):
    calls = []

    def fake_evaluate_model(model, tokenizer, qs, device="cuda", max_new_tokens=200):
        calls.append(qs)
        return {"n": len(qs), "parse_ok": 0, "schema_ok": 0, "schema_rate": 0.0,
                "responses": ["x"] * len(qs)}

    monkeypatch.setattr("lora.evaluate_lora.evaluate_model", fake_evaluate_model)

    class FakeModel:
        def to(self, device):
            return self

    monkeypatch.setattr(
        "lora.evaluate_lora.load_base_model", lambda device="cuda": FakeModel()
    )
    monkeypatch.setattr("lora.evaluate_lora.load_tokenizer", lambda: object())

    result = run_eval(adapter_dir=None, questions=["Q1", "Q2", "Q3"])

    assert set(result.keys()) == {"before", "after"}
    assert result["before"]["n"] == 3 and result["after"]["n"] == 3
    assert result["before"] is result["after"]  # no adapter → single evaluation reused
    assert len(calls) == 1


def test_main_writes_results_json(monkeypatch, tmp_path):
    import lora.evaluate_lora as ev

    def fake_run_eval(adapter_dir, questions):
        return {
            "before": {"n": 2, "schema_rate": 0.0},
            "after": {"n": 2, "schema_rate": 1.0},
        }

    monkeypatch.setattr(ev, "run_eval", fake_run_eval)
    monkeypatch.setattr(
        ev, "load_pairs", lambda path: [
            {"messages": [{"role": "user", "content": "A"}, {"role": "assistant", "content": "{}"}]},
            {"messages": [{"role": "user", "content": "B"}, {"role": "assistant", "content": "{}"}]},
        ],
    )
    out = tmp_path / "results.json"
    ev.main(adapter_dir=tmp_path, data_path=Path("fake.jsonl"), out_path=out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["before"]["schema_rate"] == 0.0
    assert data["after"]["schema_rate"] == 1.0
