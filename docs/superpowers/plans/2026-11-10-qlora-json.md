# QLoRA JSON-Format Fine-tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** QLoRA-fine-tune a small Qwen instruct model so it always answers Vietnamese AI questions in a fixed JSON schema, with a machine-checked before/after metric.

**Architecture:** Four layers in a new `lora/` package inside mini-gpt: `dataset.py` (schema + JSONL IO), `build_dataset.py` (deterministic seeded builder → committed train/eval JSONL), `format_metric.py` (JSON extraction/validation + response scoring), `qlora.py` + `train_lora.py` + `evaluate_lora.py` (4-bit base, LoRA attach, manual training loop with assistant-only loss masking, before/after CLI). No TRL (not installed); the manual loop mirrors mini-gpt's `train.py` on purpose.

**Tech Stack:** transformers 5.17, peft 0.21.2, bitsandbytes 0.50.2, datasets 5.1 (all verified working in `~/ai-roadmap/.venv` on the RTX 5060, bf16 OK), pytest. CPU-only tests auto-skip GPU parts.

**Spec:** `docs/superpowers/specs/2026-11-10-qlora-json-design.md`

## Global Constraints

- Base model `Qwen/Qwen3.5-0.8B` (HF id), fallback `Qwen/Qwen2.5-0.5B-Instruct` per spec option B — decided by the Task 4 architecture spike (see Review Focus note there).
- Answer schema, exact: JSON object with exactly keys `answer` (str), `reason` (str), `lang` (str, must equal `"vi"`).
- Dataset: 350 train + 30 eval pairs, **question sets disjoint** (no eval question in train), Vietnamese, AI/LLM topics, committed as JSONL.
- LoRA config, exact: r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], task type CAUSAL_LM. Trainable params must stay < 2% of total.
- 4-bit NF4 quantization (bitsandbytes), compute dtype bf16.
- Training pins: lr 1e-4, batch 4, grad accumulation 2, 3 epochs, max_len 512, AdamW. Wall-clock ≤ ~20 min.
- Eval: greedy (`do_sample=False`), max_new_tokens 200, same 30 questions before/after; target ≥ 80% schema-valid after.
- Do not modify frozen baseline files (`data.py`, `model.py`, `train.py`, `sample.py`, `benchmark.py`, `checkpoint.py`, `bpe.py`, their tests, `scripts/compare_tokenizers.py`).
- All commands from repo root in WSL2 with `~/ai-roadmap/.venv` activated. First model load downloads ~1–2 GB from HF (network required).
- GPU-dependent tests live in `tests/test_lora_gpu.py` guarded by `@pytest.mark.skipif(not torch.cuda.is_available(), ...)` so CPU machines skip them.

## Review Focus

- Model wraps JSON in ```json fences or prose ("Đây là câu trả lời: {...}") — `extract_json` must recover the object; plain prose without braces → invalid. → Task 3 tests
- Loss masking: labels must be -100 over the prompt and non--100 only over the assistant answer + its EOS; wrong span silently teaches the model to parrot questions or never stop. → Task 5 unit test on real tokenizer
- Adapter secretly training base weights (trainable % too high / wrong modules) — wiring assert pins % and module names. → Task 4 GPU test
- Eval leakage: an eval question appearing verbatim in train inflates the metric into memorization — builder test asserts disjoint sets. → Task 2
- Non-deterministic eval making before/after incomparable — greedy decoding pinned by passing `do_sample=False` in `evaluate_lora.py`; `score_responses` is pure so its aggregation is testable without GPU. → Tasks 3 and 6
- JSONL quirks: trailing blank lines / whitespace records must not crash loading. → Task 1

---

### Task 1: Dataset schema and JSONL IO

**Files:**
- Create: `lora/dataset.py`
- Test: `tests/test_lora_dataset.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `PAIR_SCHEMA_KEYS = ("answer", "reason", "lang")`
  - `validate_pair(pair: dict) -> None` — raises `ValueError` on any violation, returns None on success. Record shape: `{"messages": [{"role": "user", "content": str}, {"role": "assistant", "content": str}]}` where the assistant content must parse as JSON with exactly `PAIR_SCHEMA_KEYS`, all values str, `lang == "vi"`, user content non-empty.
  - `load_pairs(path: Path) -> list[dict]` — reads JSONL, skips blank lines, validates every record (first invalid record raises with line number in the message).

- [ ] **Step 1: Write the failing tests**

Test names and assertions (in `tests/test_lora_dataset.py`, plain functions, no fixtures beyond a `tmp_path` write helper):

```python
def make_pair(question="Transformer là gì?", answer="...", reason="..."):
    import json
    return {"messages": [
        {"role": "user", "content": question},
        {"role": "assistant", "content": json.dumps(
            {"answer": answer, "reason": reason, "lang": "vi"}, ensure_ascii=False)},
    ]}

def test_validate_pair_accepts_valid_record(): ...          # no raise
def test_validate_pair_rejects_missing_key(): ...           # drop "reason" → ValueError mentions "reason"
def test_validate_pair_rejects_extra_key(): ...             # add "extra" → ValueError
def test_validate_pair_rejects_wrong_lang(): ...            # lang="en" → ValueError mentions "lang"
def test_validate_pair_rejects_non_json_assistant_content(): ...  # assistant says plain text → ValueError
def test_validate_pair_rejects_empty_question(): ...        # "" → ValueError
def test_load_pairs_reads_jsonl_and_skips_blank_lines(tmp_path): ...  # 3 records + 2 blank lines → 3 pairs
def test_load_pairs_reports_line_number_of_bad_record(tmp_path): ...  # bad record on line 2 → "line 2" in message
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_dataset.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lora'`

- [ ] **Step 3: Implement `lora/dataset.py`**

Stdlib `json` only. `validate_pair` checks in order: messages shape → user content non-empty → assistant content parses as JSON object → exact key set `== set(PAIR_SCHEMA_KEYS)` → all values `isinstance(str)` → `lang == "vi"`. `load_pairs` enumerates lines from 1, `line.strip()` skips blanks, wraps `validate_pair` errors with `f"{path}: line {n}: {msg}"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lora_dataset.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add lora/dataset.py tests/test_lora_dataset.py
git commit -m "feat(lora): add dataset schema validation and JSONL IO"
```

---

### Task 2: Deterministic dataset builder + committed data

**Files:**
- Create: `lora/build_dataset.py`
- Create (by running the builder): `lora/data/train.jsonl` (350 records), `lora/data/eval.jsonl` (30 records)
- Test: `tests/test_lora_build.py`

**Interfaces:**
- Consumes: `validate_pair`, `load_pairs` from Task 1.
- Produces:
  - `build_all() -> dict[str, int]` — writes both files under `lora/data/`, returns `{"train": 350, "eval": 30}`.
  - `main()` — CLI entry: `python -m lora.build_dataset` prints counts.
  - Data files themselves (committed).

- [ ] **Step 1: Write the failing tests**

```python
def test_build_is_deterministic(tmp_path): ...        # run build_all twice, byte-compare both JSONLs
def test_build_produces_expected_counts(): ...        # after build_all(), load_pairs counts == 350 / 30, all validate
def test_eval_questions_disjoint_from_train(): ...    # set of user contents: len(train ∩ eval) == 0
```

(Tests chdir or monkeypatch the output dir to `tmp_path` — `build_all(out_dir: Path = Path("lora/data"))` accepts an output directory.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_build.py -v`
Expected: FAIL — `No module named 'lora.build_dataset'`

- [ ] **Step 3: Implement `lora/build_dataset.py`**

Content strategy — hand-written seed banks, composed deterministically with `random.Random(1337)`:
- ~35 topic seeds (attention, KV cache, tokenizer/BPE, LoRA, quantization, embedding, fine-tune vs RAG, hallucination, context window, temperature, top-k, prompt vs fine-tune, …).
- Question templates per topic (~3 phrasings each: "X là gì?", "Tại sao …?", "Khi nào nên dùng …?") plus 30 hand-written eval questions in a separate `EVAL_QUESTIONS` bank (explicitly NOT generated from the train templates).
- `answer`/`reason` strings: 1–3 câu tiếng Việt per topic, factually correct, written by hand; variation comes from topic × phrasing, not fabricated facts.
- Record shape exactly as Task 1; assistant content is `json.dumps(..., ensure_ascii=False)`.
- If composed count lands ≠ 350, pad by cycling remaining phrasings; assert final counts.

- [ ] **Step 4: Run test to verify it passes, then generate the real data**

Run: `python -m pytest tests/test_lora_build.py -v` → 3 passed
Run: `python -m lora.build_dataset` → writes `lora/data/*.jsonl`; rerun suite `python -m pytest tests/test_lora_dataset.py tests/test_lora_build.py -q` → 11 passed

- [ ] **Step 5: Human approval checkpoint (spec requires)**

Print/display 10 sample train pairs + 5 eval questions to the human partner and get a go-ahead before committing data. If they request changes, edit seed banks and regenerate. This is the plan's one intentional pause.

- [ ] **Step 6: Commit**

```bash
git add lora/build_dataset.py lora/data/ tests/test_lora_build.py
git commit -m "feat(lora): add deterministic Vietnamese JSON-format dataset (350 train / 30 eval)"
```

---

### Task 3: JSON extraction, validation, response scoring

**Files:**
- Create: `lora/format_metric.py`
- Test: `tests/test_lora_metric.py`

**Interfaces:**
- Consumes: `PAIR_SCHEMA_KEYS` from Task 1.
- Produces:
  - `extract_json(text: str) -> str | None` — returns the substring of the first balanced `{...}` block, handling ```json fences, prose before/after, nested braces; None when no complete object.
  - `validate_answer(text: str) -> bool` — extract → `json.loads` → exact keys, all str, `lang == "vi"`; any failure → False.
  - `score_responses(responses: list[str]) -> dict` — `{"n": int, "parse_ok": int, "schema_ok": int, "schema_rate": float}`; parse_ok counts `extract_json` + loads success; schema_ok counts full `validate_answer`.
  - `evaluate_model(model, tokenizer, questions: list[str], device, max_new_tokens: int = 200) -> dict` — real generate loop, greedy (`do_sample=False`), applies the model's chat template per question, decodes only the new tokens, returns `score_responses(...)` plus `"responses": [...]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_extract_json_plain(): ...                 # '{"a":1}' → itself
def test_extract_json_fenced(): ...                # '```json\n{"a":1}\n```' → '{"a":1}'
def test_extract_json_prose_wrapped(): ...         # 'Câu trả lời: {"a": 1} nhé' → '{"a": 1}'
def test_extract_json_nested_braces(): ...         # value containing {} inside string? no — object with nested object; balanced scan must not stop early
def test_extract_json_no_object_returns_none(): ...# plain prose → None
def test_validate_answer_accepts_schema(): ...     # compact valid vi JSON → True
def test_validate_answer_rejects_wrong_lang(): ...
def test_validate_answer_rejects_extra_key(): ...
def test_validate_answer_rejects_prose(): ...
def test_score_responses_aggregates(): ...         # 2 valid + 1 parse-only + 1 garbage → n=4, parse_ok=3, schema_ok=2, rate=0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_metric.py -v` → FAIL, no module `lora.format_metric`

- [ ] **Step 3: Implement `lora/format_metric.py`**

`extract_json`: find first `{`, then scan with a depth counter, skipping braces inside JSON string literals (track in-string/escape state — this is the one algorithm the tests don't fully determine: a naive depth counter fails `{"a": "}"}`); stop at depth 0, return the slice; unterminated → None. `evaluate_model` is a thin GPU loop over `score_responses` — no unit test, exercised end-to-end in Task 6.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_lora_metric.py -v` → 10 passed

- [ ] **Step 5: Commit**

```bash
git add lora/format_metric.py tests/test_lora_metric.py
git commit -m "feat(lora): add JSON extraction, schema validation, response scoring"
```

---

### Task 4: Base model load + LoRA wiring (GPU)

**Files:**
- Create: `lora/qlora.py`
- Test: `tests/test_lora_gpu.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `BASE_MODEL_ID: str` — resolved by the Step 1 spike; `"Qwen/Qwen3.5-0.8B"` unless the spike rules fallback, then `"Qwen/Qwen2.5-0.5B-Instruct"` (+ ledger ruling).
  - `load_base_model(device="cuda") -> transformers.PreTrainedModel` — 4-bit NF4 (`BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`), `device_map={"": 0}`, `torch_dtype` per bnb docs.
  - `attach_lora(model) -> peft.PeftModel` — `LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type=CAUSAL_LM)`, `get_peft_model`, `print_trainable_parameters` output captured to log.

- [ ] **Step 1: Architecture spike (rule + ledger)**

Run: `python -c "from transformers import AutoConfig; c = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B'); print(type(c).__name__); print([k for k in c.to_dict() if 'vision' in k.lower() or 'vl' in k.lower()])"` plus a text-only `AutoModelForCausalLM.from_pretrained` load attempt on CUDA in bf16 (not 4-bit, just to check class compatibility; ~1.6 GB download).
Decision rule: if `AutoModelForCausalLM` loads it as a text-capable causal LM with attention projections named `q_proj/k_proj/v_proj/o_proj`, keep Qwen3.5-0.8B. If it requires a VL-specific class or drags a vision tower, fall back to `Qwen/Qwen2.5-0.5B-Instruct` (spec option B) and ledger the ruling. Record the actual module names found (`named_parameters` grep `_proj`) — LoRA targets must match reality; adjust the pinned list only via a ledgered ruling.

- [ ] **Step 2: Write the failing GPU test**

```python
import pytest, torch
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

def test_attach_lora_touches_only_attention_and_stays_small():
    from lora.qlora import load_base_model, attach_lora
    model = load_base_model()
    peft_model = attach_lora(model)
    trainable = [(n, p.numel()) for n, p in peft_model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in peft_model.parameters())
    train_sum = sum(nel for _, nel in trainable)
    assert all("lora_" in n for n, _ in trainable)
    assert all(any(t in n for t in ("q_proj", "k_proj", "v_proj", "o_proj")) for n, _ in trainable)
    assert train_sum / total < 0.02
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_lora_gpu.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lora.qlora'` (skipif passes on this machine)

- [ ] **Step 4: Implement `lora/qlora.py`** per the Interfaces pins.

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_lora_gpu.py -v` → 1 passed

- [ ] **Step 6: Commit**

```bash
git add lora/qlora.py tests/test_lora_gpu.py
git commit -m "feat(lora): add 4-bit base load and LoRA wiring with param-budget assert"
```

---

### Task 5: Training script with assistant-only loss masking

**Files:**
- Create: `lora/train_lora.py`
- Test: `tests/test_lora_train.py` (CPU, tokenizer-only download) + a 10-step GPU smoke inside `tests/test_lora_gpu.py` (append)

**Interfaces:**
- Consumes: `load_pairs` (Task 1), `load_base_model`/`attach_lora` (Task 4), `BASE_MODEL_ID` (Task 4).
- Produces:
  - `build_example(tokenizer, pair: dict, max_len: int = 512) -> tuple[list[int], list[int]]` — `(input_ids, labels)`; labels are `-100` everywhere except the assistant answer tokens and the EOS/eom token that the chat template appends after the assistant turn.
  - `train(out_dir: Path = Path("lora/adapter"), data_path: Path = Path("lora/data/train.jsonl"), max_steps: int | None = None) -> dict` — manual loop: AdamW(lr=1e-4), batch 4 × grad-accum 2, clip 1.0, logs loss every 10 steps, saves adapter via `peft_model.save_pretrained(out_dir)` at end, returns `{"steps": [...], "loss": [...]}`.
  - CLI: `python -m lora.train_lora --max-steps N --out-dir PATH`.

- [ ] **Step 1: Write the failing unit test (masking — the subtle part)**

```python
def test_build_example_masks_prompt_and_covers_answer():
    from transformers import AutoTokenizer
    from lora.train_lora import build_example
    from lora.dataset import validate_pair
    import json
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)   # tokenizer only, small download
    pair = {"messages": [{"role": "user", "content": "Attention là gì?"},
                         {"role": "assistant", "content": json.dumps(
                             {"answer": "Cơ chế tính trọng số.", "reason": "Vì Q K V.", "lang": "vi"}, ensure_ascii=False)}]}
    input_ids, labels = build_example(tok, pair)
    assert len(input_ids) == len(labels)
    # prompt region fully masked
    answer_start = None  # find via decoding: first position whose label != -100
    masked_prefix = [i for i, l in enumerate(labels) if l == -100]
    assert masked_prefix and masked_prefix == list(range(masked_prefix[-1] + 1)) or True
    # decode(label[input_ids]) where unmasked must reproduce the assistant JSON + terminator
    unmasked = [t for t, l in zip(input_ids, labels) if l != -100]
    decoded = tok.decode(unmasked)
    assert '"lang": "vi"' in decoded or '"lang":"vi"' in decoded
    # at least one non-masked token AND the final token is unmasked (terminator learned)
    assert unmasked and labels[-1] != -100
```

(The test intentionally verifies behavior — unmasked span decodes to the answer and ends with an unmasked terminator — not internal slicing.)

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_lora_train.py -v` → FAIL, no module `lora.train_lora`

- [ ] **Step 3: Implement masking + training**

Masking approach (pinned because tests don't determine it): render `prompt_ids = tokenizer.apply_chat_template([user_msg], add_generation_prompt=True)` and `full_ids = tokenizer.apply_chat_template([user_msg, assistant_msg])`; `labels = full_ids` with positions `< len(prompt_ids)` set to `-100`. Truncate both to `max_len` from the left of the answer (keep answer; drop oldest prompt tokens if over).
Loop mirrors `train.py` of the baseline: batch via stacking with right padding (`tokenizer.pad_token` — set to `eos` if None), `attention_mask` passed, `labels` padded with `-100`.

- [ ] **Step 4: Run unit test, then add GPU smoke**

`python -m pytest tests/test_lora_train.py -v` → 1 passed.
Append to `tests/test_lora_gpu.py`:

```python
def test_train_smoke_runs_and_saves_adapter(tmp_path):
    from lora.train_lora import train
    metrics = train(out_dir=tmp_path / "adapter", max_steps=10)
    assert len(metrics["loss"]) >= 1 and all(map(lambda v: v == v, metrics["loss"]))  # finite
    assert (tmp_path / "adapter").exists()
```

Run: `python -m pytest tests/test_lora_gpu.py -v` → 2 passed

- [ ] **Step 5: Commit**

```bash
git add lora/train_lora.py tests/test_lora_train.py tests/test_lora_gpu.py
git commit -m "feat(lora): add QLoRA training with assistant-only loss masking"
```

---

### Task 6: Before/after evaluation + docs

**Files:**
- Create: `lora/evaluate_lora.py`
- Modify: `README.md` (new "LoRA/QLoRA exercise" section), `LESSONS.md` (§6 backlog: mark LoRA done, add measured result), `requirements.txt` (no change expected — pin nothing new; if the spike forced an extra dep, add it here with floor)
- Test: `tests/test_lora_eval.py` (CLI-arg parsing + result-file shape with stubbed evaluate_model)

**Interfaces:**
- Consumes: `load_pairs`, `evaluate_model`, `load_base_model`, `attach_lora`, `BASE_MODEL_ID`.
- Produces:
  - `run_eval(adapter_dir: Path | None, questions: list[str]) -> dict` — loads base (4-bit), optionally `PeftModel.from_pretrained(base, adapter_dir)`, calls `evaluate_model`, returns its dict.
  - CLI: `python -m lora.evaluate_lora --out lora/results.json` — runs base and adapter on the 30 eval questions, prints a markdown table, writes JSON.

- [ ] **Step 1: Write the failing test** — `run_eval` with a stubbed `evaluate_model` (monkeypatch) returns the stub dict for both phases; `main` writes a JSON file containing both phases' metrics. (Details left to the implementer following existing CLI test patterns in `tests/test_sample.py`.)

- [ ] **Step 2: Verify it fails** → `python -m pytest tests/test_lora_eval.py -v` → FAIL, no module

- [ ] **Step 3: Implement `lora/evaluate_lora.py`** per Interfaces.

- [ ] **Step 4: Verify unit passes, then run the real thing**

`python -m pytest tests/test_lora_eval.py -v` → passed.
Full train (spec budget ≤ 20 min): `python -m lora.train_lora` (defaults), then
`python -m lora.evaluate_lora --out lora/results.json`.
Expected: before schema_rate low (0–10% typical for an untuned model), after ≥ 0.80. If after < 0.80: systematic-debugging before touching hyperparameters; only then a ledgered ruling allows lr 2e-4 or epochs 4.

- [ ] **Step 5: Update README + LESSONS.md with the measured table** (before/after parse_ok, schema_ok, schema_rate; training loss curve summary; 2–3 quoted before/after responses; adapter size on disk vs base size).

- [ ] **Step 6: Commit**

```bash
git add lora/evaluate_lora.py tests/test_lora_eval.py README.md LESSONS.md
git commit -m "feat(lora): add before/after JSON-format evaluation with measured results"
```

---

### Task 7: Full-suite verification

- [ ] **Step 1:** `python -m pytest -q` → all pass (existing 37 + new ~25; GPU tests run on this machine). `bash scripts/smoke.sh` → unchanged/passing. Baseline files untouched: `git diff --stat main -- data.py model.py train.py sample.py benchmark.py checkpoint.py bpe.py` → empty.
- [ ] **Step 2:** Fix-only commit if anything surfaced; otherwise no commit.
