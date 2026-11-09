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

It intentionally does not include BPE, Vietnamese data, RoPE, LoRA/QLoRA, RAG, chat instruction tuning, or web deployment.

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
├── data/input.txt  # tiny Shakespeare baseline corpus
├── tests/           # data, model, training, checkpoint, sampling tests
└── scripts/smoke.sh # full CPU smoke workflow
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

The baseline config uses a 128-character context, 4 layers, 4 heads, 128-dimensional embeddings, AdamW, and a learning rate of `3e-4`. Training evaluates both splits periodically. In the verified 20-step CPU smoke run, train loss decreased from `4.2373` to `3.2459`.

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

The exact values vary by machine; the command records the conditions in its JSON output. CPU timing is sufficient when CUDA is unavailable. When CUDA is available, run the same command with `--device cuda` and record the GPU result.

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

The final verified suite has 21 tests covering tokenizer/data validation, shapes and gradients, causal masking, cache equivalence, overfit/loss reduction, checkpoint portability, deterministic generation, context overflow, and benchmark metadata.

## Limitations and follow-up backlog

- A char-level vocabulary is educational but inefficient for real text, especially Vietnamese.
- The small baseline is not instruction-tuned and will produce low-quality Shakespeare-like continuations after short training.
- Learned positional embeddings are used first; RoPE is a follow-up comparison.
- BPE/token-length comparison and Vietnamese data are follow-up experiments.
- LoRA/QLoRA fine-tuning of `Qwen2.5-0.5B` is a later GPU exercise.
- Integration with `chatbot-cli` is intentionally deferred until the internals project is complete.
