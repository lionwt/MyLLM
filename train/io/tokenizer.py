"""Load the tokenizer expected by training and checkpoint restoration."""

from pathlib import Path

from tokenizers.io import load_bpe


def load_tokenizer(path, vocab_size=None):
    """Load a BPE tokenizer and optionally enforce checkpoint vocabulary size."""
    token_path = Path(path)
    if not token_path.exists():
        raise FileNotFoundError(
            f"Missing tokenizer {token_path}; build it with "
            "`python -m tokenizers.prepare` or pass --token"
        )
    tok = load_bpe(token_path)
    if vocab_size is not None and len(tok.vocab) != vocab_size:
        raise ValueError(
            f"Tokenizer {token_path} has {len(tok.vocab)} tokens "
            f"but the checkpoint expects {vocab_size}"
        )
    return tok
