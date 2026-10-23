"""Byte-level byte-pair encoding model and training primitives."""

import re
from collections import Counter, defaultdict


class BPE_Meta:
    """Hyperparameters for byte-level BPE.

    vocab_size: hard cap on ids (256 byte tokens + merges). Training may stop
        earlier if no remaining pair has frequency >= 2 or fits max_token_bytes.
    max_token_bytes: refuse a merge whose concatenated payloads would exceed this.
    pattern: GPT-2-style regex used to split text so merges cannot cross words.
    """

    __slots__ = ("version", "vocab_size", "max_token_bytes", "pattern")

    def __init__(
        self,
        version: int = 1,
        vocab_size: int = 2048,
        max_token_bytes: int = 64,
        pattern: str = r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d+| ?[^\s\w]+|\s+(?!\S)|\s+",
    ):
        """Initialize BPE training and segmentation settings."""
        self.version = version
        self.vocab_size = vocab_size
        self.max_token_bytes = max_token_bytes
        self.pattern = pattern


class BPE_Node:
    """One token in the doubly linked list for a unique regex chunk.

    Unique chunks are stored once. Every node in that list shares the same freq
    (how many times the chunk appeared). That weight is never replaced by the
    global pair total (top_freq).

    value: token id (0-255 for UTF-8 bytes, 256+ after merges).
    freq: chunk count; pair_counts change by ±freq, not ±1.
    alive: False after this node is swallowed by a merge (the right half).
    prev / next: neighboring tokens in this chunk.
    pos / word_id: original order, used to merge overlapping pairs left-to-right.
    """

    __slots__ = ("value", "freq", "alive", "prev", "next", "pos", "word_id")

    def __init__(self, value: int, freq: int, pos: int, word_id: int):
        """Initialize a linked-list token occurrence within a regex chunk."""
        self.value = value
        self.freq = freq
        self.alive = True
        self.prev = None
        self.next = None
        self.pos = pos
        self.word_id = word_id


class BPE:
    """Byte-level BPE model.

    merges: list of (left_id, right_id) in training order; rank is the index.
    vocab: id -> payload bytes. Keys are ints; values are bytes, never used
        as pair keys. New ids start at 256.
    pair_rank: (left_id, right_id) -> merge rank for greedy encode.
    """
    def __init__(
        self,
        meta: BPE_Meta | None = None,
        merges: list[tuple[int, int]] | None = None,
        vocab: dict[int, bytes] | None = None,
    ):
        """Byte-level BPE model.

        merges: list of (left_id, right_id) in training order; rank is the index.
        vocab: id -> payload bytes. Keys are ints; values are bytes, never used
            as pair keys. New ids start at 256.
        pair_rank: (left_id, right_id) -> merge rank for greedy encode.
        """
        self.meta = meta if meta is not None else BPE_Meta()
        self.merges = [tuple(pair) for pair in merges] if merges is not None else []
        if vocab is None:
            vocab = {i: bytes([i]) for i in range(256)}
            for i, (a, b) in enumerate(self.merges):
                vocab[256 + i] = vocab[a] + vocab[b]
        self.vocab = dict(vocab)
        self.pair_rank = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self._cache = {}

    def train(self, texts: list[str]) -> "BPE":
        """Learn merge rules from texts.

        1. Regex-split and count unique chunks (a chunk seen 10k times is one list
           with freq=10000).
        2. Build a linked list of byte ids per chunk and fill pair_counts.
        3. Repeatedly merge the most frequent pair that also satisfies
           freq >= 2 and max_token_bytes. Ties pick the smaller (left, right) ids.
        4. Apply that pair left-to-right in each word so overlapping aaa matches
           encode's i += 2 merge. Stop early if no legal pair remains.
        vocab_size is a cap, not a guarantee.
        """
        chunk_set = Counter()
        for text in texts:
            if not text:
                continue
            chunk_set.update(self._split(text))

        pair_index = defaultdict(set)
        pair_counts = Counter()
        word_id = 0
        for chunk, freq in chunk_set.items():
            if not chunk:
                continue
            tokens = list(chunk.encode("utf-8"))
            if not tokens:
                continue
            nodes = [
                BPE_Node(value, freq, pos, word_id) for pos, value in enumerate(tokens)
            ]
            for i, node in enumerate(nodes):
                if not i:
                    continue
                node.prev = nodes[i - 1]
                nodes[i - 1].next = node
                self._add_pair(pair_index, pair_counts, nodes[i - 1])
            word_id += 1

        vocab_size = self.meta.vocab_size
        max_token_bytes = self.meta.max_token_bytes
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        for new_id in range(256, vocab_size):
            candidates = [
                (pair, frequency)
                for pair, frequency in pair_counts.items()
                if frequency >= 2
                and len(self.vocab[pair[0]]) + len(self.vocab[pair[1]]) <= max_token_bytes
            ]
            if not candidates:
                break
            pair, frequency = min(candidates, key=lambda item: (-item[1], item[0]))
            occurrences = sorted(
                (node for node in pair_index.get(pair, ()) if node.alive),
                key=lambda node: (node.word_id, node.pos),
            )
            self.merges.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            for node in occurrences:
                nxt = node.next
                if (
                    not node.alive
                    or nxt is None
                    or not nxt.alive
                    or (node.value, nxt.value) != pair
                ):
                    continue
                self._merge_nodes(pair_index, pair_counts, node, new_id)
            if (new_id - 255) % 100 == 0:
                print(f"BPE merges: {new_id - 255}, pair count: {frequency}", flush=True)
        self.pair_rank = {tuple(pair): i for i, pair in enumerate(self.merges)}
        self._cache = {}
        return self

    def encode(self, text: str) -> list[int]:
        """Split text, BPE-encode each chunk, concatenate ids.

        Merges never cross chunk boundaries (e.g. letters will not glue to a comma).
        """
        ids = []
        for chunk in self._split(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Map ids to bytes, join, then one UTF-8 decode.

        A token may be only part of a multi-byte character, so decoding per id
        would fail. Invalid sequences become replacement characters.
        """
        return b"".join(self.vocab[int(i)] for i in ids).decode("utf-8", errors="replace")

    @staticmethod
    def _add_pair(pair_index, pair_counts, node):
        """Index (node.value, node.next.value) and add node.freq to pair_counts.

        pair_index maps the pair to the *left* node so later merges can jump to
        occurrences instead of scanning every chunk. No-op at end of list, if
        either node is dead, or if this left node is already recorded.
        """
        nxt = node.next
        if nxt is None or not node.alive or not nxt.alive:
            return
        pair = (node.value, nxt.value)
        bucket = pair_index[pair]
        if node in bucket:
            return
        bucket.add(node)
        pair_counts[pair] += node.freq

    @staticmethod
    def _remove_pair(pair_index, pair_counts, node):
        """Drop (node, node.next) from the index and subtract node.freq.

        Safe if the pair was already removed (stale pointer after a nearby merge).
        Deletes the Counter key when the remaining count hits 0.
        """
        nxt = node.next
        if nxt is None:
            return
        pair = (node.value, nxt.value)
        bucket = pair_index.get(pair)
        if not bucket or node not in bucket:
            return
        bucket.remove(node)
        if not bucket:
            del pair_index[pair]
        pair_counts[pair] -= node.freq
        if pair_counts[pair] <= 0:
            del pair_counts[pair]

    def _merge_nodes(self, pair_index, pair_counts, left, new_id):
        """Replace adjacent tokens (a, b) with new_id and patch only local counts.

        Before: prev ⇄ a ⇄ b ⇄ next
        After:  prev ⇄ new_id ⇄ next  (b marked dead)

        Count updates, each ± this chunk's freq (left.freq):
            -(prev, a), -(a, b), -(b, next)
            +(prev, new_id), +(new_id, next)
        left.value becomes the new id; left.freq is unchanged.
        """
        right = left.next
        if right is None or not left.alive or not right.alive:
            return
        prev_node = left.prev
        next_node = right.next
        if prev_node is not None:
            self._remove_pair(pair_index, pair_counts, prev_node)
        self._remove_pair(pair_index, pair_counts, left)
        self._remove_pair(pair_index, pair_counts, right)
        left.value = new_id
        left.next = next_node
        if next_node is not None:
            next_node.prev = left
        right.alive = False
        right.prev = None
        right.next = None
        if prev_node is not None:
            self._add_pair(pair_index, pair_counts, prev_node)
        self._add_pair(pair_index, pair_counts, left)

    @staticmethod
    def _merge(ids, pair, new_id):
        """Replace non-overlapping occurrences of pair with new_id, left to right.

        Both branches append integer ids. The payload bytes live in vocab, not here.
        """
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

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one regex chunk: UTF-8 bytes, then greedy lowest-rank merges.

        Rank 0 is the first trained merge. Replay must pick the earliest remaining
        pair at each step so encode matches training order on overlapping pairs.
        """
        ids = list(chunk.encode("utf-8"))
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
        return ids

    def _split(self, text: str) -> list[str]:
        """Split with meta.pattern; keep unmatched gaps and a trailing tail.

        Gaps (e.g. an emoji the regex skipped) are still encoded as UTF-8 bytes.
        Dropping them would make decode lose characters.
        """
        if not self.meta.pattern:
            return [text] if text else []
        pos = 0
        chunks = []
        for match in re.finditer(self.meta.pattern, text):
            st, ed = match.start(), match.end()
            if pos < st:
                chunks.append(text[pos:st])
            chunks.append(match.group())
            pos = ed
        if pos < len(text):
            chunks.append(text[pos:])
        return chunks
