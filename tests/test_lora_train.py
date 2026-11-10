import json

from lora.qlora import BASE_MODEL_ID


def test_build_example_masks_prompt_and_covers_answer():
    from transformers import AutoTokenizer

    from lora.train_lora import build_example

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    pair = {"messages": [
        {"role": "user", "content": "Attention là gì?"},
        {"role": "assistant", "content": json.dumps(
            {"answer": "Cơ chế tính trọng số.", "reason": "Vì Q K V.", "lang": "vi"},
            ensure_ascii=False)},
    ]}
    input_ids, labels = build_example(tok, pair)
    assert len(input_ids) == len(labels)
    unmasked = [t for t, l in zip(input_ids, labels) if l != -100]
    decoded = tok.decode(unmasked)
    assert '"lang": "vi"' in decoded or '"lang":"vi"' in decoded
    assert unmasked and labels[-1] != -100
    # prompt region is fully masked: the question text must NOT appear unmasked
    assert "Attention" not in decoded
