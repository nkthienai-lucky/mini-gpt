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


def test_gpt4_split_pattern_parts_are_expected():
    import regex

    from bpe import GPT4_SPLIT_PATTERN

    parts = regex.findall(GPT4_SPLIT_PATTERN, "Hello world, hasn't örökké 12345!")
    # Note: " 123" splits into " " + "123" — the possessive optional prefix
    # only joins letters, and \s+(?!\S) fails before a digit.
    assert parts == ["Hello", " world", ",", " hasn", "'t", " örökké", " ", "123", "45", "!"]


def test_regex_tokenizer_round_trips_english_and_vietnamese():
    from bpe import RegexTokenizer

    text = "Xin chào, tôi là kỹ sư AI. Nice to meet you!"
    tokenizer = RegexTokenizer()
    tokenizer.train(text * 20, vocab_size=320)
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_regex_tokenizer_merges_never_cross_part_boundaries():
    from bpe import RegexTokenizer

    # The pattern splits "a,b a,b a,b" into ['a', ',b', ' a', ',b', ' a', ',b']
    # (punctuation attaches to the following word). Per-part pair counting
    # therefore sees ",b" x3 and " a" x2 and learns exactly those merges.
    # A broken implementation counting on the concatenated stream would
    # instead learn cross-part merges starting with (97, 44) "a," — a
    # different dict — so pinning the exact merges discriminates the two.
    text = "a,b a,b a,b"
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=258)
    assert tokenizer.merges == {(44, 98): 256, (32, 97): 257}
    # Round trip must still hold.
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_regex_tokenizer_decode_handles_invalid_utf8_bytes():
    from bpe import RegexTokenizer

    tokenizer = RegexTokenizer()
    tokenizer.train("abc", vocab_size=256)
    # 0xFF is invalid standalone UTF-8; decode must not crash
    assert tokenizer.decode([255]) == "�"


def test_compare_on_text_reports_consistent_counts():
    from scripts.compare_tokenizers import compare_on_text

    text = "aaabdaaabac aaabdaaabac aaabdaaabac"
    result = compare_on_text(text, vocab_size=300)
    assert result["chars"] == len(text)
    assert result["bytes"] == len(text.encode("utf-8"))
    assert result["basic_ids"] >= 1 and result["regex_ids"] >= 1
    assert result["basic_ratio"] == pytest.approx(result["basic_ids"] / len(text))


def test_render_boundaries_brackets_each_token():
    from scripts.compare_tokenizers import render_boundaries

    tokenizer = BasicTokenizer()
    tokenizer.train("aaabdaaabac", vocab_size=259)
    rendered = render_boundaries(tokenizer, "aaab")
    assert rendered.count("[") == len(tokenizer.encode("aaab"))
