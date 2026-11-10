# QLoRA Qwen3.5-0.8B — JSON-format fine-tune (Design: short)

**Date:** 2026-11-10 · **Status:** approved (option A chosen in conversation; base model Qwen3.5-0.8B per recon)

## Intent & why

Stage 2 bonus exercise: understand LoRA/QLoRA hands-on — the "bend behavior, don't inject knowledge" counterpart to the from-scratch pretraining done in mini-gpt. A small instruct model is fine-tuned to always answer in a fixed JSON schema, with a machine-checkable before/after metric.

## Goals

- QLoRA fine-tune `Qwen/Qwen3.5-0.8B` locally: 4-bit NF4 base (bitsandbytes), LoRA adapters (r=16, alpha=32, dropout 0.05, targets q/k/v/o_proj), a few hundred steps on RTX 5060 8GB.
- Dataset: ~300–400 synthetic Vietnamese Q&A pairs about AI/LLM topics, each answer exactly `{"answer": str, "reason": str, "lang": "vi"}`. Generated locally (by the assistant's own tooling), human-approved sample before training.
- Metric: on a held-out 30-question eval set, `% of responses that parse as JSON AND validate against the schema`, greedy decoding, before vs after fine-tune. Before ≈ 0%, after target ≥ 80%.
- Artifacts: script(s), dataset + eval set committed (text), adapter checkpoint (outside git), README section with measured table, LESSONS.md update.
- pytest suite: dataset schema validation, adapter wiring (trainable % < 2, correct modules touched), save/load roundtrip, eval-metric function on hand-built responses.

## Non-goals

- Answer *quality* (only format compliance is measured).
- Multi-epoch tuning, hyperparameter search, merging into GGUF/Ollama, serving.
- English data, other behaviors (persona, domain knowledge).

## Anti-goals

- Touching the frozen mini-gpt baseline files.
- Fine-tuning to "inject knowledge" (that's RAG, stage 3).

## Constraints

- Same environment: WSL2, `~/ai-roadmap/.venv` (transformers 5.17 / peft 0.21 / bitsandbytes 0.50 / datasets 5.1 all verified working on the RTX 5060, bf16 OK).
- VRAM budget 8GB: 0.8B at 4-bit ≈ 0.6GB weights + LoRA + activations — ample.
- Training run ≤ ~20 min wall-clock.
- New code lives in a new `lora/` directory inside mini-gpt (or a sibling dir — builder's call, stated in plan).

## Approach (chosen: QLoRA on Qwen3.5-0.8B, local)

- **A (chosen):** QLoRA `Qwen/Qwen3.5-0.8B` (HF weights) with peft — matches the user's current qwen3.5 stack; 0.8B fits 8GB trivially; instruct-tuned so the JSON task is a behavior change, not capability.
- **B (rejected):** `Qwen2.5-0.5B-Instruct` exactly as the roadmap names — equally valid technically, but older line than the user's daily stack, and no advantage at this size.
- **C (rejected):** persona/tone or domain-knowledge datasets — weaker or fake metrics, and C blurs the line with RAG (stage 3).

## Verification approach

- TDD for the testable layers (data format, adapter wiring, metric function, save/load).
- Training verified by: loss curve decreasing; before/after table with parse% and schema% on the same 30 held-out prompts, same decoding (greedy); a few side-by-side sample outputs quoted in README.
- Human approval checkpoint: dataset sample (≥10 pairs) shown to the user before training starts.
