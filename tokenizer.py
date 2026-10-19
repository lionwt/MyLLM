import json
import re
from collections import Counter
from pathlib import Path

TOKENIZER_PATH = "tokenizer.json"
TOKEN_PATH = "tokens.json"
PATTERN = r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+"


class ByteBPE:
    def __init__(self, pattern=PATTERN):
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.pattern = pattern
        self.pair_rank = {}
        self._cache = {}

    def _split(self, text):
        if not self.pattern:
            return [text] if text else []
        chunks = []
        pos = 0
        for match in re.finditer(self.pattern, text):
            if match.start() > pos:
                chunks.append(text[pos:match.start()])
            chunks.append(match.group())
            pos = match.end()
        if pos < len(text):
            chunks.append(text[pos:])
        return chunks

    def _rebuild(self):
        self.vocab = {i: bytes([i]) for i in range(256)}
        for i, (a, b) in enumerate(self.merges):
            self.vocab[256 + i] = self.vocab[a] + self.vocab[b]
        self.pair_rank = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self._cache = {}

    def train(self, texts, vocab_size=1024, max_token_bytes=16):
        counts = Counter()
        for text in texts:
            if not text:
                continue
            counts.update(self._split(text))
        sequences = [[list(chunk.encode("utf-8")), freq] for chunk, freq in counts.items() if chunk]
        for new_id in range(256, vocab_size):
            pair_counts = Counter()
            for seq, freq in sequences:
                for pair in zip(seq, seq[1:]):
                    pair_counts[pair] += freq
            if not pair_counts:
                break
            candidates = [
                (pair, frequency)
                for pair, frequency in pair_counts.items()
                if frequency >= 2
                and len(self.vocab[pair[0]]) + len(self.vocab[pair[1]]) <= max_token_bytes
            ]
            if not candidates:
                break
            pair, frequency = min(candidates, key=lambda item: (-item[1], item[0]))
            a, b = pair
            self.merges.append((a, b))
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
            sequences = [[self._merge(seq, pair, new_id), freq] for seq, freq in sequences]
            if (new_id - 255) % 100 == 0:
                print(f"BPE merges: {new_id - 255}, pair count: {frequency}", flush=True)
        self._rebuild()
        return self

    @staticmethod
    def _merge(ids, pair, new_id):
        out = []
        i = 0
        while i < len(ids):
            if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    def _encode_chunk(self, chunk):
        cached = self._cache.get(chunk)
        if cached is not None:
            return cached
        ids = list(chunk.encode("utf-8"))
        if self.pattern is None:
            for i, pair in enumerate(self.merges):
                ids = self._merge(ids, pair, 256 + i)
        else:
            while len(ids) >= 2:
                best_rank = None
                best_pair = None
                for pair in zip(ids, ids[1:]):
                    rank = self.pair_rank.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_pair = pair
                if best_pair is None:
                    break
                ids = self._merge(ids, best_pair, 256 + best_rank)
        self._cache[chunk] = ids
        return ids

    def encode(self, text):
        if self.pattern is None:
            return self._encode_chunk(text)
        ids = []
        for chunk in self._split(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    def _paths(self, path):
        dest = Path(path)
        if dest.suffix.lower() == ".json":
            directory = dest.parent
            tokenizer_path = dest
        else:
            directory = dest
            tokenizer_path = directory / TOKENIZER_PATH
        directory.mkdir(parents=True, exist_ok=True)
        return tokenizer_path, directory / TOKEN_PATH

    def save(self, path):
        payload = {"version": 2, "pattern": self.pattern, "merges": self.merges}
        tokenizer_path, tokens_path = self._paths(path)
        tokenizer_path.write_text(json.dumps(payload), encoding="utf-8")
        serializable = {
            str(i): token.decode("utf-8", errors="replace") for i, token in self.vocab.items()
        }
        tokens_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        print(f"Saved BPE model to {tokenizer_path} and {tokens_path}")

    def store_tokens(self, path=TOKEN_PATH):
        serializable = {
            i: token.decode("utf-8", errors="replace") for i, token in self.vocab.items()
        }
        Path(path).write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        dest = Path(path)
        if dest.is_dir() or dest.suffix.lower() != ".json":
            dest = dest / TOKENIZER_PATH
        data = json.loads(dest.read_text(encoding="utf-8"))
        pattern = data.get("pattern")
        tok = cls(pattern=pattern)
        tok.merges = [tuple(x) for x in data["merges"]]
        tok._rebuild()
        return tok
