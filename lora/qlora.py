"""4-bit base model load and LoRA adapter wiring."""

from __future__ import annotations

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL_ID = "Qwen/Qwen3.5-0.8B"


def load_tokenizer():
    return AutoTokenizer.from_pretrained(BASE_MODEL_ID)


def load_base_model(device: str = "cuda"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    return AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        quantization_config=bnb,
        device_map={"": 0},
    )


def attach_lora(model):
    config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model
