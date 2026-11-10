# minbpe-style BPE tokenizer — Design (short)

**Date:** 2026-11-10 · **Status:** approved (option A, chosen in conversation)

## Intent & why

Stage 2, week 4 of the AI Engineer Roadmap calls for writing a simple BPE
tokenizer by hand (Karpathy `minbpe` exercise). The mini-gpt baseline uses a
character-level tokenizer (vocab 65), which is educational but inefficient for
real text — especially Vietnamese. This adds the missing learning artifact and
ticks the README backlog item "BPE/token-length comparison".

## Goals

- Hand-written BPE: `BasicTokenizer` (byte-level, merge loop) and
  `RegexTokenizer` (GPT-4-style pattern pre-splitting), no `tiktoken`/
  `tokenizers` libraries.
- Encode/decode roundtrip on English and Vietnamese (multi-byte UTF-8) text.
- Comparison experiment: same corpus, char-level vs BPE token counts,
  compression ratio, plus token boundary renderings for English and Vietnamese
  samples.
- pytest suite following the repo's existing patterns.

## Non-goals

- Retraining mini-gpt on BPE tokens (baseline stays frozen).
- Special tokens, save/load file formats, training on Vietnamese corpus.
- LoRA/QLoRA (separate later exercise).

## Anti-goals

- Replacing or modifying `data.py`'s `CharTokenizer` or any reviewed baseline code.

## Constraints

- Same environment as the repo: WSL2 Ubuntu, `~/ai-roadmap/.venv` (Python 3.12),
  CPU-only is fine. New dependency allowed: `regex` (for `\p{L}` classes in the
  GPT-4 split pattern).

## Approach (chosen: A — standalone module + experiment)

- **A (chosen):** new `bpe.py` + `tests/test_bpe.py` + `scripts/compare_tokenizers.py`
  inside mini-gpt. Baseline untouched; learning value is in the merge loop and
  regex pre-split; comparison answers "why not char-level".
- **B (rejected):** integrate BPE into the training pipeline and retrain —
  changes the frozen, reviewed baseline and shifts focus from the algorithm to
  plumbing.
