"""Train, persist, and sample a BPE tokenizer from text files."""

import argparse
from pathlib import Path

from tokenizers.model.bpe import BPE, BPE_Meta
from tokenizers.io import store_bpe, load_bpe

def main():
    """Run the command-line tokenizer preparation workflow."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="data")
    parser.add_argument("--output", type=str, default="out")
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--max-token-bytes", type=int, default=64)
    args = parser.parse_args()

    bpe = BPE(
        meta=BPE_Meta(
            vocab_size=args.vocab_size,
            max_token_bytes=args.max_token_bytes,
        )
    )

    file_paths = list(Path(args.data).glob("*.txt"))
    print(f"Found {len(file_paths)} text files...")

    print(f"Reading texts...")
    texts = [Path(path).read_text(encoding="utf-8") for path in file_paths]
    print(f"Reading finished...")

    print(f"Training BPE...")
    bpe.train(texts)
    print(f"Training finished...")

    print(f"Storing BPE...")
    store_bpe(bpe, Path(args.output) / "bpe.json")
    print(f"Storing finished...")

    print(f"Sampling for string 'Hello, world!'...")
    new_bpe = load_bpe(Path(args.output) / "bpe.json")
    print(new_bpe.encode("Hello, world!"))
    print(new_bpe.decode(new_bpe.encode("Hello, world!")))
    print(f"Sampling finished...")

    print(f"Done...")

if __name__ == "__main__":
    main()
