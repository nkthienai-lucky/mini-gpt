import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_attach_lora_touches_only_attention_and_stays_small():
    from lora.qlora import attach_lora, load_base_model

    model = load_base_model()
    peft_model = attach_lora(model)
    trainable = [(n, p.numel()) for n, p in peft_model.named_parameters() if p.requires_grad]
    total = sum(p.numel() for p in peft_model.parameters())
    train_sum = sum(nel for _, nel in trainable)
    assert all("lora_" in n for n, _ in trainable)
    assert all(
        any(t in n for t in ("q_proj", "k_proj", "v_proj", "o_proj")) for n, _ in trainable
    )
    assert train_sum / total < 0.02
