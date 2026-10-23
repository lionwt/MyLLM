"""Persist trained BPE tokenizers as JSON."""

import json
from pathlib import Path
from tokenizers.model.bpe import BPE

def store_bpe(bpe: BPE, path: str):
    """Write a BPE tokenizer and its metadata to a JSON file."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        merges = bpe.merges
        vocab = {str(k): v.decode("utf-8", errors="replace") for k, v in bpe.vocab.items()}
        meta = {
            "version": bpe.meta.version,
            "vocab_size": bpe.meta.vocab_size,
            "max_token_bytes": bpe.meta.max_token_bytes,
            "pattern": bpe.meta.pattern,
        }
        json.dump(
            {
                "meta": meta,
                "merges": merges,
                "vocab": vocab,
            },
            file
        )
