"""Tokenizer validation and document encoding helpers."""

from pathlib import Path


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


class LazyEncodedDocuments:
    """Load and tokenize documents on first access, then keep them in memory."""

    def __init__(self, paths, tok, name=""):
        self.paths = [Path(path) for path in paths]
        self.tok = tok
        self.name = name
        self._cache = [None] * len(self.paths)
        self.loaded = 0
        self.total_tokens = 0
        self.total_bytes = 0

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        cached = self._cache[index]
        if cached is not None:
            return cached
        path = self.paths[index]
        text = path.read_text(encoding="utf-8")
        ids = self.tok.encode(text)
        if self.tok.decode(ids) != text:
            raise ValueError(f"Tokenizer round-trip failed for {path}")
        self._cache[index] = ids
        self.loaded += 1
        self.total_tokens += max(len(ids), 1)
        self.total_bytes += len(text.encode("utf-8"))
        if self.loaded == 1 or self.loaded % 100 == 0 or self.loaded == len(self.paths):
            label = f"{self.name} " if self.name else ""
            print(
                f"{label}encoded {self.loaded}/{len(self.paths)} files, "
                f"{self.total_tokens} tokens, "
                f"bytes/token={self.total_bytes / self.total_tokens:.3f}",
                flush=True,
            )
        return ids
