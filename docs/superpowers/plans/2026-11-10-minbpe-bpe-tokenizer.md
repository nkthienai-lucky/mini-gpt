# minbpe-style BPE Tokenizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hand-written byte-level BPE tokenizer (Karpathy `minbpe` style) plus a char-vs-BPE comparison experiment to the mini-gpt repo, without touching the frozen baseline.

**Architecture:** One new module `bpe.py` with two classes — `BasicTokenizer` (pure merge loop on raw bytes) and `RegexTokenizer` (GPT-4-style pattern pre-splitting, subclass of `BasicTokenizer`) — following the repo's plain-module layout. A standalone script `scripts/compare_tokenizers.py` consumes both plus the existing `CharTokenizer` to measure compression on `data/input.txt`. No changes to `data.py`, `model.py`, or any reviewed baseline file.

**Tech Stack:** Python 3.12, stdlib `re` for BasicTokenizer, `regex` (new dependency, for `\p{L}` in the GPT-4 split pattern), pytest. CPU only.

**Spec:** `docs/superpowers/specs/2026-11-10-minbpe-bpe-tokenizer-design.md`

## Global Constraints

- Do not modify `data.py`, `model.py`, `train.py`, `sample.py`, `benchmark.py`, `checkpoint.py`, or their tests (spec anti-goal).
- No `tiktoken`/`tokenizers` library usage — BPE algorithm must be hand-written (spec goal).
- Byte-level encoding: text → UTF-8 bytes → merges on byte/byte-pair ids; vocab 0–255 are the raw bytes, learned merges get ids 256+.
- All commands run from repo root in WSL2 with `~/ai-roadmap/.venv` activated.
- `regex` is the only new allowed dependency; add it to `requirements.txt` with a floor (`regex>=2024.4.16`).

## Review Focus

- Empty text / no merges trained: `encode("")` → `[]`, `decode([])` → `""`; training with `vocab_size=256` (no merges) must not crash. → Task 1 tests
- Decode of unknown id ≥ vocab size: should raise a clear `ValueError`, not index-error silently. → Task 1 tests
- Encoding text with characters never seen in training: byte-level fallback must still roundtrip (no `KeyError`). → Task 1 tests
- Multi-byte UTF-8 (Vietnamese diacritics) through `RegexTokenizer`: merges must never cross the regex part boundaries; roundtrip must hold exactly. → Task 2 tests
- Lone surrogate / invalid UTF-8 edge (e.g. `"\ud83d"` style string): decode of arbitrary ids must map bytes back through `errors="replace"` rather than crashing. → Task 2 tests

---

### Task 1: BasicTokenizer — train, encode, decode

**Files:**
- Create: `bpe.py`
- Modify: `requirements.txt` (add `regex>=2024.4.16` — installed in Task 2, line added now so Task 1 stays one commit)
- Test: `tests/test_bpe.py`

**Interfaces:**
- Consumes: nothing new.
- Produces (used by Task 2 and the comparison script):
  - `class BasicTokenizer` with `__init__(self)`, `train(self, text: str, vocab_size: int, verbose: bool = False) -> None`, `encode(self, text: str) -> list[int]`, `decode(self, ids: list[int]) -> str`, and attributes `merges: dict[tuple[int, int], int]`, `vocab: dict[int, bytes]`.

- [ ] **Step 1: Write the failing tests**

```python
import pytest

from bpe import BasicTokenizer


def test_basic_tokenizer_round_trips_training_text():
    text = "aaabdaaabac"
    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=259)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_basic_tokenizer_learns_expected_merges_on_karpathy_example():
    text = "aaabdaaabac"
    tokenizer = BasicTokenizer()
    tokenizer.train(text, vocab_size=259)
    # ids: 97='a' 98='b' 99='c' 100='d'; merges rank-ordered. Derived by hand:
    # merge1 (97,97)="aa"; then "aaabdaaabac" -> [256,97,98,100,256,98,97,99]
    # where every pair count is 1, so first-inserted (256,97)="aaa" wins the
    # tie; then [257,98,100,256,98,97,99] ties again -> (257,98)="aaab".
    assert tokenizer.merges == {(97, 97): 256, (256, 97): 257, (257, 98): 258}


def test_basic_tokenizer_vocab_cannot_shrink_below_256():
    with pytest.raises(ValueError, match="vocab_size"):
        BasicTokenizer().train("abc", vocab_size=255)


def test_basic_tokenizer_round_trips_unseen_text_and_empty():
    text = "the quick brown fox 🦊 jumps"
    tokenizer = BasicTokenizer()
    tokenizer.train("small training corpus zz", vocab_size=258)
    assert tokenizer.decode(tokenizer.encode("")) == ""
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_basic_tokenizer_decode_rejects_unknown_ids():
    tokenizer = BasicTokenizer()
    tokenizer.train("abc", vocab_size=256)
    with pytest.raises(ValueError, match="id"):
        tokenizer.decode([999])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bpe'`

- [ ] **Step 3: Implement `bpe.py` (BasicTokenizer only)**

Follows the `minbpe` algorithm: `train` maps text to `list(text.encode("utf-8"))`, then loops `vocab_size - 256` times — count adjacent pairs, merge the most frequent (ties broken by first-seen pair in the count dict, i.e. `max(count, key=count.get)`), record `merges[pair] = 256 + i`, build `vocab` (0–255 single bytes, then concatenated merge bytes). `encode` starts from UTF-8 bytes and applies learned merges in rank order (lowest merge id first) until none apply. `decode` joins `vocab[id]` bytes and returns `bytes(...).decode("utf-8", errors="replace")`, raising `ValueError(f"unknown token id: {id}")` for ids not in vocab. `train` raises `ValueError` if `vocab_size < 256`. NOTE on Step 1's expected merges: after `(97,97)`, remaining pair counts all tie at 1, so dict insertion order decides — verify the asserted `{(97,97):256, (256,97):257, (257,98):258}` by actually running it; if Python's dict ordering gives a different tie winner, the test's expected dict is wrong, not the algorithm — recompute by hand and fix the test before proceeding.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add bpe.py tests/test_bpe.py requirements.txt
git commit -m "feat: add hand-written byte-level BPE BasicTokenizer"
```

---

### Task 2: RegexTokenizer — GPT-4-style pre-splitting

**Files:**
- Modify: `bpe.py` (add `RegexTokenizer` and the pattern constant)
- Modify: `requirements.txt` (already has `regex` floor from Task 1)
- Test: `tests/test_bpe.py` (append)

**Interfaces:**
- Consumes: `BasicTokenizer` from Task 1.
- Produces (used by comparison script):
  - `GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""` (raw string, used with `regex.compile(...)`).
  - `class RegexTokenizer(BasicTokenizer)` overriding `train` and `encode` (inherits `decode`, `merges`, `vocab`).

- [ ] **Step 1: Write the failing tests (append)**

```python
from bpe import GPT4_SPLIT_PATTERN, RegexTokenizer


def test_gpt4_split_pattern_parts_are_expected():
    import regex
    parts = regex.findall(GPT4_SPLIT_PATTERN, "Hello world, hasn't örökké 12345!")
    assert parts == ["Hello", " world", ",", " hasn", "'t", " örökké", " 123", "45", "!"]
def test_regex_tokenizer_round_trips_english_and_vietnamese():
    text = "Xin chào, tôi là kỹ sư AI. Nice to meet you!"
    tokenizer = RegexTokenizer()
    tokenizer.train(text * 20, vocab_size=320)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_regex_tokenizer_merges_never_cross_part_boundaries():
    text = "ab cd ab cd ab cd"
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=258)
    # "a" and "b" live in different parts; (97, 98) must not be a learned merge
    assert (97, 98) not in tokenizer.merges


def test_regex_tokenizer_decode_handles_invalid_utf8_bytes():
    tokenizer = RegexTokenizer()
    tokenizer.train("abc", vocab_size=256)
    # 0xFF is invalid standalone UTF-8; decode must not crash
    assert tokenizer.decode([255]) == "�"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: the 4 new tests FAIL with `ImportError: cannot import name 'RegexTokenizer'`

- [ ] **Step 3: Implement `RegexTokenizer` in `bpe.py`**

Install `regex` first: `python -m pip install 'regex>=2024.4.16'`. `train` splits text with the compiled pattern, concatenates the parts' UTF-8 bytes into one id sequence **for merge counting only**, and — key difference from minbpe — after picking each merge, verifies it never merges ids across part boundaries by tracking part boundary positions in the id sequence; simplest correct approach: run the merge loop per-part-id-list (train merges on the concatenated stream but apply merges per part, counting pairs only within parts). `encode` splits with the pattern, encodes each part with the merge loop, concatenates. `decode` inherited.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add bpe.py tests/test_bpe.py
git commit -m "feat: add RegexTokenizer with GPT-4-style pattern pre-splitting"
```

---

### Task 3: Char-vs-BPE comparison script + README evidence

**Files:**
- Create: `scripts/compare_tokenizers.py`
- Modify: `README.md` (new "Tokenizers: char vs BPE" section; move BPE line out of the backlog list)
- Test: `tests/test_bpe.py` (append — script correctness is covered by unit tests on its pure helpers)

**Interfaces:**
- Consumes: `BasicTokenizer`, `RegexTokenizer` (`bpe.py`), `CharTokenizer` (`data.py`).
- Produces:
  - `compare_on_text(text: str, vocab_size: int) -> dict` — pure function, keys: `"chars"`, `"bytes"`, `"basic_ids"`, `"regex_ids"`, `"basic_ratio"`, `"regex_ratio"` where ratio = ids/chars (lower is better).
  - `render_boundaries(tokenizer: BasicTokenizer, text: str) -> str` — renders each token as `[bytes]` concatenated.
  - CLI: `python scripts/compare_tokenizers.py --data data/input.txt --vocab-size 500` prints the comparison table and boundary renderings for a fixed English sample and a fixed Vietnamese sample, both embedded in the script.

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_compare_on_text_reports_consistent_counts():
    from scripts.compare_tokenizers import compare_on_text
    text = "aaabdaaabac aaabdaaabac aaabdaaabac"
    result = compare_on_text(text, vocab_size=300)
    assert result["chars"] == len(text)
    assert result["bytes"] == len(text.encode("utf-8"))
    assert result["basic_ids"] >= 1 and result["regex_ids"] >= 1
    assert result["basic_ratio"] == pytest.approx(result["basic_ids"] / len(text))


def test_render_boundaries_brackets_each_token():
    from scripts.compare_tokenizers import compare_on_text, render_boundaries
    tokenizer = BasicTokenizer()
    tokenizer.train("aaabdaaabac", vocab_size=259)
    rendered = render_boundaries(tokenizer, "aaab")
    assert rendered.count("[") == len(tokenizer.encode("aaab"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_bpe.py -v`
Expected: 2 new tests FAIL with `ModuleNotFoundError: No module named 'scripts'`

- [ ] **Step 3: Implement `scripts/compare_tokenizers.py`**

Pure helpers at top (importable because `scripts/` gets a `sys.path` shim or the tests import via `scripts.compare_tokenizers` with `scripts/__init__.py` absent — use `importlib` or make tests insert `scripts/` on `sys.path`; simplest: `sys.path.insert(0, str(Path(__file__).parent.parent))` inside the script and tests import the module through a tiny `conftest.py` path hook — implementer picks the conventional one). CLI main trains both tokenizers on the corpus (default `data/input.txt`, vocab 500), prints a table: corpus characters, corpus bytes, char-level ids, Basic BPE ids + ratio, Regex BPE ids + ratio, then boundary renderings for the two embedded samples.

- [ ] **Step 4: Run test to verify it passes, then run the CLI on the real corpus**

Run: `python -m pytest tests/test_bpe.py -v` → 11 passed
Run: `python scripts/compare_tokenizers.py --data data/input.txt --vocab-size 500`
Expected: table prints with regex ratio well below 1.0 (compression vs char-level); Vietnamese sample renders multi-byte diacritics as single bracketed tokens where merged.

- [ ] **Step 5: Update README**

Add "Tokenizers: char vs BPE" section after "Generate and benchmark": the exact CLI command, the measured table (fill in real numbers from Step 4), the two boundary renderings, and one line: why byte-level BPE handles Vietnamese without a vocab problem while char-level would need codepoint-level vocab. Remove the "BPE/token-length comparison … follow-up" line from the backlog list (keep "Vietnamese data" if still relevant — it now is: the experiment shows boundaries but we still don't *train* on Vietnamese).

- [ ] **Step 6: Commit**

```bash
git add scripts/compare_tokenizers.py tests/test_bpe.py README.md
git commit -m "feat: add char-vs-BPE tokenizer comparison with measured results"
```

---

### Task 4: Full-suite verification

**Files:**
- Modify: none expected (fix-only if Review Focus failures surface).

- [ ] **Step 1: Run the full suite and the baseline smoke test**

Run: `python -m pytest -q` → all pass (26 existing + 11 new = 37)
Run: `bash scripts/smoke.sh` → passes unchanged (baseline untouched)

- [ ] **Step 2: Commit any fixes (if none, nothing to commit)**

Only if fixes were needed; otherwise this task is verification-only, no commit.
