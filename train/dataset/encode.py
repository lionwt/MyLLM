"""Tokenizer validation and document encoding helpers."""


def encode_texts(tok, texts, paths):
    """Encode documents after verifying lossless tokenizer round trips."""
    encoded = []
    total_bytes = 0
    total_tokens = 0
    for text, path in zip(texts, paths):
        ids = tok.encode(text)
        if tok.decode(ids) != text:
            raise ValueError(f"Tokenizer round-trip failed for {path}")
        encoded.append(ids)
        total_bytes += len(text.encode("utf-8"))
        total_tokens += max(len(ids), 1)
    print(f"tokenizer round-trip ok, bytes/token={total_bytes / total_tokens:.3f}", flush=True)
    return encoded
