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
