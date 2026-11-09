from pathlib import Path

import pytest
import torch

from data import CharTokenizer, get_batch, load_text_dataset


def test_char_tokenizer_round_trips_every_character():
    text = "To be, or not to be.\n"
    tokenizer = CharTokenizer.from_text(text)

    encoded = tokenizer.encode(text)

    assert encoded.dtype == torch.long
    assert tokenizer.decode(encoded) == text


def test_char_tokenizer_uses_deterministic_sorted_vocabulary():
    tokenizer = CharTokenizer.from_text("zabza")

    assert list(tokenizer.itos.values()) == ["a", "b", "z"]
    assert tokenizer.stoi == {"a": 0, "b": 1, "z": 2}


def test_char_tokenizer_rejects_text_shorter_than_two_characters():
    with pytest.raises(ValueError, match="at least two characters"):
        CharTokenizer.from_text("a")


def test_load_text_dataset_splits_character_stream_contiguously(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("abcdefghij", encoding="utf-8")

    dataset = load_text_dataset(path)

    assert dataset.train.tolist() == list(range(9))
    assert dataset.val.tolist() == [9]
    assert dataset.tokenizer.decode(dataset.train) == "abcdefghi"
    assert dataset.tokenizer.decode(dataset.val) == "j"


def test_get_batch_returns_shifted_windows():
    data = torch.arange(12, dtype=torch.long)

    x, y = get_batch(
        data,
        block_size=4,
        batch_size=3,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(7),
    )

    assert x.shape == (3, 4)
    assert y.shape == (3, 4)
    assert torch.equal(y, x + 1)


def test_short_corpus_is_rejected_with_clear_error(tmp_path: Path):
    path = tmp_path / "input.txt"
    path.write_text("a", encoding="utf-8")

    with pytest.raises(ValueError, match="at least two characters"):
        load_text_dataset(path)


def test_get_batch_rejects_data_shorter_than_block_plus_target():
    with pytest.raises(ValueError, match=r"block_size \+ 1"):
        get_batch(
            torch.arange(4, dtype=torch.long),
            block_size=4,
            batch_size=1,
            device=torch.device("cpu"),
        )
