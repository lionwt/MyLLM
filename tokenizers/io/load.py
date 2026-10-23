"""Load persisted BPE tokenizers."""

import json
from tokenizers.model.bpe import BPE, BPE_Meta

def load_bpe(path: str) -> BPE:
    """Reconstruct a BPE tokenizer from a JSON file."""
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        meta = data["meta"]
        merges = data["merges"]

        # The stored vocab is a lossy, human-readable view (partial UTF-8 bytes
        # become U+FFFD), so the byte payloads are rebuilt from merges instead.
        return BPE(
            meta=BPE_Meta(
                version=meta["version"],
                vocab_size=meta["vocab_size"],
                max_token_bytes=meta["max_token_bytes"],
                pattern=meta["pattern"],
            ),
            merges=merges,
        )
