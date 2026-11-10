"""Compare char-level vs byte-level BPE tokenization on a corpus."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bpe import BasicTokenizer, RegexTokenizer  # noqa: E402

ENGLISH_SAMPLE = (
    "First Citizen: We are accounted poor citizens, the patricians good."
)
VIETNAMESE_SAMPLE = (
    "Xin chào, tôi là kỹ sư AI tại Hà Nội, Việt Nam."
)


def compare_on_text(text: str, vocab_size: int) -> dict[str, float]:
    """Token-count comparison of Basic/Regex BPE against char level."""
    basic = BasicTokenizer()
    basic.train(text, vocab_size=vocab_size)
    regex_tok = RegexTokenizer()
    regex_tok.train(text, vocab_size=vocab_size)
    chars = len(text)
    basic_ids = len(basic.encode(text))
    regex_ids = len(regex_tok.encode(text))
    return {
        "chars": chars,
        "bytes": len(text.encode("utf-8")),
        "basic_ids": basic_ids,
        "regex_ids": regex_ids,
        "basic_ratio": basic_ids / chars,
        "regex_ratio": regex_ids / chars,
    }


def render_boundaries(tokenizer: BasicTokenizer, text: str) -> str:
    """Render each token as [bytes] concatenated, minbpe-style."""
    ids = tokenizer.encode(text)
    return "".join(f"[{tokenizer.vocab[i].decode('utf-8', errors='replace')}]" for i in ids)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/input.txt"))
    parser.add_argument("--vocab-size", type=int, default=500)
    args = parser.parse_args()

    text = args.data.read_text(encoding="utf-8")
    print(f"corpus: {args.data} ({len(text):,} chars)")
    result = compare_on_text(text, args.vocab_size)

    print()
    print(f"corpus characters:            {result['chars']:>12,}")
    print(f"corpus UTF-8 bytes:           {result['bytes']:>12,}")
    print(f"char-level ids (= chars):     {result['chars']:>12,}")
    print(
        f"Basic BPE ids (vocab {args.vocab_size}): {result['basic_ids']:>8,}"
        f"  ratio vs chars: {result['basic_ratio']:.3f}"
    )
    print(
        f"Regex BPE ids (vocab {args.vocab_size}): {result['regex_ids']:>8,}"
        f"  ratio vs chars: {result['regex_ratio']:.3f}"
    )

    regex_tok = RegexTokenizer()
    regex_tok.train(text, vocab_size=args.vocab_size)
    basic = BasicTokenizer()
    basic.train(text, vocab_size=args.vocab_size)
    print()
    print(f"English boundaries:   {render_boundaries(regex_tok, ENGLISH_SAMPLE)}")
    print()
    print(f"Vietnamese boundaries: {render_boundaries(regex_tok, VIETNAMESE_SAMPLE)}")
    print()
    print(f"Vietnamese, Basic BPE (no split): {render_boundaries(basic, VIETNAMESE_SAMPLE)}")


if __name__ == "__main__":
    main()
