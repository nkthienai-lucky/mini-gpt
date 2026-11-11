# Mini-GPT — Stage 2

Educational decoder-only GPT trained from scratch on the tiny Shakespeare corpus. This is the Stage 2 project from the AI Engineer Roadmap: the goal is to understand Transformer internals, not to replace the Stage 1 API chatbot.

## Scope

The baseline includes:

- character-level tokenization;
- a contiguous 90/10 train/validation split;
- manually implemented causal multi-head self-attention;
- pre-norm Transformer blocks with residual connections, LayerNorm, and GELU MLP;
- next-token cross-entropy training;
- temperature/top-k sampling; and
- a reproducible uncached versus KV-cached generation benchmark.

It intentionally does not include BPE-model serving, Vietnamese data, RoPE, LoRA/QLoRA serving, RAG, chat instruction tuning, or the QLoRA adapter in the web demo.

## Environment setup

The supported workflow is WSL2 Ubuntu with the roadmap environment:

```bash
cd /mnt/d/Thien/Code/mini-gpt
source ~/ai-roadmap/.venv/bin/activate
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The same commands work on CPU when CUDA is unavailable. Dependencies are listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The baseline corpus is `data/input.txt`, downloaded from Karpathy's tiny Shakespeare source:

`https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`

SHA-256: `86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`

## Project structure

```text
mini-gpt/
├── config.py       # one Config dataclass for model/training/sampling defaults
├── data.py         # character tokenizer, contiguous split, random windows
├── model.py        # manual attention, Transformer block, MiniGPT, generation
├── train.py        # training loop, evaluation, CLI
├── checkpoint.py   # portable model/tokenizer/optimizer metadata
├── benchmark.py    # CPU/CUDA-safe generation timing
├── sample.py       # generation and benchmark CLI
├── webapp.py       # framework-free web demo logic (sanitization, hub fetch)
├── app.py          # Gradio interface served on Render
├── render.yaml     # Render free deploy config
├── data/input.txt  # tiny Shakespeare baseline corpus
├── tests/           # data, model, training, checkpoint, sampling, web tests
└── scripts/         # smoke, tokenizer comparison, hub publish, e2e check
```

## Architecture and tensor shapes

For batch `B`, sequence length `T`, embedding size `C`, heads `H`, and head size `D=C/H`:

```text
idx                         (B, T)
  └─ token + learned position embeddings  (B, T, C)
       └─ LayerNorm → manual causal attention
            Q, K, V         (B, H, T, D)
            QKᵀ / √D        (B, H, T, T)
            causal mask → softmax → weights @ V
       └─ residual addition  (B, T, C)
       └─ LayerNorm → Linear(C, 4C) → GELU → Linear(4C, C)
       └─ residual addition  (B, T, C)
  └─ final LayerNorm → language-model head  (B, T, vocab_size)
```

The core does not use `nn.Transformer`, `nn.MultiheadAttention`, or `scaled_dot_product_attention`; the Q/K/V, scaling, mask, softmax, and weighted-value operations are visible in `model.py`.

## Train

A short CPU run is enough to verify the pipeline:

```bash
python train.py \
  --data data/input.txt \
  --device cpu \
  --max-steps 20 \
  --eval-interval 10 \
  --checkpoint checkpoints/smoke.pt
```

For a longer run, use CUDA when available and keep the checkpoint outside Git:

```bash
python train.py --device cuda --checkpoint checkpoints/shakespeare.pt
```

The baseline config uses a 128-character context, 4 layers, 4 heads, 128-dimensional embeddings, AdamW, and a learning rate of `3e-4`. Training evaluates both splits periodically and writes the requested checkpoint plus `latest.pt` and validation-best `best.pt`. In the verified 20-step CPU smoke run, train loss decreased from `4.2373` to `3.2459`.

Verified CPU evaluation log for that run:

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 4.2373 | 4.2389 |
| 10 | 3.5020 | 3.5188 |
| 19 | 3.2459 | 3.2572 |

A short continuation generated from `checkpoints/smoke.pt` with prompt `To `, greedy decoding, and 32 new tokens was:

```text
To the the the t the the the the t
```

### Full training run

The complete run is 5,000 steps on the RTX 5060 (same baseline config, `--eval-interval 250`, checkpoint `checkpoints/shakespeare.pt`):

```bash
python train.py --device cuda --max-steps 5000 --eval-interval 250 --checkpoint checkpoints/shakespeare.pt
```

Selected milestones (train/validation loss):

| Step | Train loss | Validation loss |
|---:|---:|---:|
| 0 | 4.2373 | 4.2389 |
| 500 | 2.2283 | 2.2533 |
| 1000 | 1.9049 | 1.9846 |
| 2000 | 1.6111 | 1.7735 |
| 3000 | 1.4863 | 1.6802 |
| 4000 | 1.4208 | 1.6275 |
| 4999 | 1.3981 | 1.6114 |

Best validation loss `1.6075` at step 4,500 (saved as `checkpoints/best.pt`). Cosine decay is fully exhausted by the end of the schedule.

Sample from `checkpoints/shakespeare.pt` (prompt `ROMEO:`, temperature 0.8, top-k 50, seed 1337):

```text
ROMEO:
You thank'd cowing to love the did of Banna I am
grave the gate
he art this unquit to before that end.

HENRY BOLINGBRO
```

Greedy continuation (prompt `To be, or not to be:`, temperature 0, from `checkpoints/best.pt`):

```text
To be, or not to be:
The world the shall be the world the shall be thee,
That what I was the should shall be the words,
And t
```

The output is recognizably Shakespeare-like: speaker-name headings, verse line breaks, and archaic word forms — while remaining locally incoherent at the sentence level, which is expected for a 4-layer character-level model. Note that the prompt plus generated tokens must fit within `block_size=128` (122 new tokens for a 6-character prompt).

## Generate and benchmark

```bash
python sample.py \
  --checkpoint checkpoints/smoke.pt \
  --prompt "ROMEO:" \
  --max-new-tokens 32 \
  --temperature 0 \
  --benchmark \
  --device cpu
```

`--temperature 0` selects greedy decoding, which makes cached and uncached output directly comparable. Sampling mode accepts `--temperature 0.8 --top-k 50`. Add `--no-kv-cache` to display only the uncached path.

The benchmark excludes checkpoint loading and tokenization. It runs one warmup and three timed repetitions, reports the median duration, and synchronizes CUDA immediately before/after timed work. The CPU and RTX 5060 results from verification are:

| Mode | Device | Prompt tokens | Generated tokens | Median ms/token | Tokens/s |
|---|---|---:|---:|---:|---:|
| Uncached | CPU | 6 | 32 | 1.754 | 570 |
| Cached | CPU | 6 | 32 | 1.740 | 575 |
| Uncached | RTX 5060 CUDA | 6 | 64 | 3.821 | 262 |
| Cached | RTX 5060 CUDA | 6 | 64 | 3.346 | 299 |

## Web demo

`app.py` serves the trained model as a Gradio interface (prompt, new tokens, temperature, top-k, seed, KV-cache toggle); `webapp.py` holds the framework-free logic covered by `tests/test_webapp.py`:

- unknown characters (Vietnamese diacritics, emoji) are dropped from the prompt instead of crashing the 65-character vocabulary;
- prompts longer than the context keep their tail (reserved 32 slots);
- an empty sanitized prompt falls back to `ROMEO:`;
- `ensure_checkpoint()` downloads the checkpoint from the Hub when missing.

Run locally:

```bash
python app.py    # http://127.0.0.1:7860
```

End-to-end server check (start, poll, exercise the `/generate` API, stop):

```bash
bash scripts/e2e_check.sh
```

### Deploy

The demo deploys on Render free (same pattern as the stage 1 project) via `render.yaml`. The build bakes the checkpoint from the Hugging Face Hub — the checkpoint lives at [Thienpkae/mini-gpt-shakespeare](https://huggingface.co/Thienpkae/mini-gpt-shakespeare) (model repos stay free; Gradio *Spaces* would require HF Pro), so `checkpoints/` stays gitignored.

1. Push to GitHub (Render deploys from the Git repo).
2. Create a **Web Service** on [render.com](https://render.com) → connect the repo → Render reads `render.yaml`.
3. Wait for the build (installs torch, downloads the checkpoint once) → take the public URL.

HF Spaces note: creating a Gradio Space returned `402 Payment Required` on the free tier (2026) — free accounts get **static** Spaces only, which cannot run this Python app; Render free is the working alternative.

## Tokenizers: char vs BPE

The baseline model uses a character-level tokenizer (vocab 65). `bpe.py` adds a hand-written byte-level BPE (Karpathy `minbpe` style, no tokenizer libraries): `BasicTokenizer` (pure merge loop) and `RegexTokenizer` (GPT-4-style pattern pre-splitting, so merges never cross part boundaries). Compare them on the corpus:

```bash
python scripts/compare_tokenizers.py --data data/input.txt --vocab-size 500
```

Measured on the tiny Shakespeare corpus (1,115,394 chars ≡ bytes, all ASCII):

| Tokenizer | ids for full corpus | ratio vs char-level |
|---|---:|---:|
| char-level (the baseline) | 1,115,394 | 1.000 |
| Basic BPE (vocab 500) | 574,057 | 0.515 |
| Regex BPE (vocab 500) | 553,176 | **0.496** |

English sample boundaries (Regex BPE): `[F][ir][st][ C][it][i][z][en][:] …` — frequent words become multi-character tokens while rare letters stay solo.

Vietnamese sample boundaries, after training only on English: `[X][in][ c][h][�][�][o][,] …` — diacritics shatter into raw bytes (`�`), because the ASCII corpus never offered Vietnamese byte pairs to merge. This is the point of the exercise: **byte-level BPE never breaks on unseen text** (roundtrip still holds exactly, every byte has a token), but compression quality for a language depends entirely on that language appearing in the training data. A production multilingual tokenizer (Qwen, GPT-4) trains merges on many languages so `à`, `ễ` etc. become single tokens; a char-level Vietnamese tokenizer would instead need a codepoint vocabulary with no sub-word structure at all.

The exact values vary by machine; the command records the conditions in its JSON output. CPU timing is sufficient when CUDA is unavailable. When CUDA is available, run the same command with `--device cuda` and record the GPU result.

## LoRA/QLoRA exercise

`lora/` adds the stage-2 bonus: QLoRA fine-tuning of `Qwen/Qwen3.5-0.8B` (HF weights, 4-bit NF4) to always answer Vietnamese AI questions in the fixed JSON schema `{"answer", "reason", "lang": "vi"}` — behavior shaping, not knowledge injection (knowledge is RAG's job, stage 3).

```bash
python -m lora.train_lora            # ~263 steps, RTX 5060, loss 3.05 → 0.54
python -m lora.evaluate_lora         # before/after on 30 held-out questions
```

Measured (greedy decoding, same 30 questions, answers never seen in training):

| Phase | n | parse_ok | schema_ok | schema_rate |
|---|---:|---:|---:|---:|
| Base 4-bit (before) | 30 | 0 | 0 | 0.00% |
| + LoRA adapter (after) | 30 | 28 | 28 | **93.33%** |

- Adapter: r=16, α=32, dropout 0.05, targets q/k/v/o_proj — **1,081,344 trainable of 753,474,368 params (0.14%)**, ~13 MB on disk vs ~600 MB base. The base weights are untouched: the adapter is the whole behavioral delta.
- Before: model answers in free-form markdown prose (typical instruct style). After: compact Vietnamese JSON with exactly the three keys.
- The 2 remaining failures are instructive: one degenerates into a repetition loop inside `reason`; one switches to English — format learned ≠ fluency guaranteed on every phrasing.

Quoted before/after (held-out question "Số tầng và số head ảnh hưởng gì đến model?"):

```text
before: "Chào bạn, đây là một câu hỏi rất quan trọng trong lĩnh vực **Deep Learning**..."
after:  {"answer": "Self-attention là cơ chế quan trọng trong RNN và transformer tự mình
         làm attention.", "reason": "Tự mình làm attention nên model hiểu ngữ nghĩa chính
         xác hơn...", "lang": "vi"}
```

Training notes: assistant-only loss masking (prompt tokens labeled −100); 3 epochs, batch 4 × grad-accum 2, AdamW lr 1e-4, ~6 min wall-clock. Two `apply_chat_template` pitfalls were hit and fixed with regression tests: with fast tokenizers `tokenize=True` returns Encoding lists / `BatchEncoding`, not plain id tensors — always render the template as a string and `.encode()` it, or take `.input_ids` explicitly (`lora/train_lora.py`, `lora/format_metric.py`).

## Why attention is O(n²)

For `n` tokens, `QKᵀ` computes a score for every query/key pair, producing an `n × n` attention matrix. The attention score computation therefore costs `O(n² · d)` and the score matrix uses `O(n²)` memory. Doubling context length roughly quadruples this part of the work.

## What does KV cache save?

During autoregressive decoding, the historical keys and values do not change. Without a cache, every new token recomputes K and V for the entire prefix. With the explicit per-layer KV cache in this project, the prompt is processed once; each subsequent step computes Q/K/V only for the new token, appends its K/V to the cache, and attends against the stored history. This saves repeated K/V projection work, at the cost of memory proportional to layers, heads, head dimension, and cached tokens.

## Verification

Run the full CPU workflow:

```bash
bash scripts/smoke.sh
```

Or run only tests:

```bash
pytest -q
```

The final verified suite has 82 tests covering tokenizer/data validation, shapes and gradients, causal masking, cache equivalence, overfit/loss reduction, checkpoint portability, deterministic generation, context overflow, benchmark metadata, and web-demo sanitization/hub-fetch behavior.

## Limitations and follow-up backlog

- A char-level vocabulary is educational but inefficient for real text, especially Vietnamese.
- The baseline is not instruction-tuned; the 5,000-step model produces recognizable Shakespeare structure but sentence-level incoherence.
- Learned positional embeddings are used first; RoPE is a follow-up comparison.
- Training the BPE tokenizers on Vietnamese data (instead of ASCII-only Shakespeare) is a follow-up experiment; the comparison script already shows the failure mode.
- LoRA/QLoRA fine-tuning: done for `Qwen3.5-0.8B` (see "LoRA/QLoRA exercise"); merging the adapter into a GGUF for Ollama serving is a possible follow-up.
- Integration with `chatbot-cli` is intentionally deferred until the internals project is complete.
