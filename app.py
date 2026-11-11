"""Gradio web demo for the trained Mini-GPT Shakespeare model.

Run locally:
    python app.py

Deployed on a Hugging Face Space (CPU basic, free): the Space runtime executes
this file directly and serves the Gradio interface on port 7860.
"""

from __future__ import annotations

import gradio as gr
import torch

from checkpoint import load_checkpoint
from webapp import DEFAULT_PROMPT, ensure_checkpoint, generate_demo


def build_demo() -> gr.Blocks:
    model, tokenizer, metadata = load_checkpoint(
        ensure_checkpoint(), torch.device("cpu")
    )
    config = metadata["config"]
    metrics = metadata.get("metrics", {})
    best_val = metrics.get("best_val_loss") or metrics.get("val_loss")

    def predict(prompt: str, max_new_tokens: int, temperature: float, top_k: int,
                seed: int, use_kv_cache: bool) -> str:
        return generate_demo(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            temperature,
            top_k if top_k > 0 else None,
            use_kv_cache,
            seed,
        )

    title = "Mini-GPT — Shakespeare character-level Transformer"
    model_summary = ", ".join(
        f"{key}={getattr(config, key)}"
        for key in ("n_layer", "n_head", "n_embd", "block_size")
    )
    checkpoint_line = f"checkpoint step {metadata['step']} · {model_summary}"
    if best_val is not None:
        checkpoint_line += f" · best val loss {best_val:.4f}"

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}")
        gr.Markdown(
            "A decoder-only Transformer trained from scratch (no `nn.Transformer`, "
            "no `scaled_dot_product_attention`) on tiny Shakespeare for the "
            "[AI Engineer Roadmap](https://github.com/nkthienai-lucky/mini-gpt) "
            f"stage 2. **{checkpoint_line}.**"
        )
        gr.Markdown(
            "ℹ️ The vocabulary is the 65 Shakespeare characters. Characters outside "
            "it (e.g. Vietnamese diacritics, emoji) are dropped from the prompt "
            "before generation; an empty result falls back to `ROMEO:`."
        )
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label="Prompt",
                    value=DEFAULT_PROMPT,
                    placeholder="ROMEO: (or any English Shakespeare-ish text)",
                )
                max_new_tokens = gr.Slider(
                    minimum=8, maximum=128, value=96, step=8, label="New tokens"
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=1.5, value=0.8, step=0.05, label="Temperature"
                )
                top_k = gr.Slider(
                    minimum=0, maximum=65, value=50, step=1,
                    label="Top-k (0 = off)",
                )
                seed = gr.Number(value=1337, precision=0, label="Seed")
                use_kv_cache = gr.Checkbox(value=True, label="Use KV cache")
                button = gr.Button("Generate", variant="primary")
            with gr.Column():
                output = gr.Textbox(label="Output", lines=12)
        button.click(
            predict,
            inputs=[prompt, max_new_tokens, temperature, top_k, seed, use_kv_cache],
            outputs=output,
            api_name="generate",
        )
    return demo


demo = build_demo()

if __name__ == "__main__":
    demo.launch()
