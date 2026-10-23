"""Build shuffled next-token training windows from tokenized documents."""

import torch


def window_starts(documents, context):
    """Enumerate every valid ``(document, offset)`` training-window start."""
    starts = []
    for doc_i, seq in enumerate(documents):
        limit = len(seq) - context
        if limit > 0:
            starts.extend((doc_i, start) for start in range(limit))
    if not starts:
        raise ValueError("Need documents longer than context length")
    return starts


def slice_window(documents, context, start):
    """Extract ``context + 1`` tokens for shifted input and target sequences."""
    doc_i, offset = start
    return documents[doc_i][offset : offset + context + 1]


def xy_tensors(windows, device):
    """Convert token windows into input and one-token-shifted target tensors."""
    return (
        torch.tensor([w[:-1] for w in windows], dtype=torch.long, device=device),
        torch.tensor([w[1:] for w in windows], dtype=torch.long, device=device),
    )


def sample_windows(documents, n_windows, context, rng):
    """Sample valid windows with replacement using the supplied RNG."""
    starts = window_starts(documents, context)
    return [slice_window(documents, context, start) for start in rng.choices(starts, k=n_windows)]


class ShuffledWindowEpochs:
    """Yield each valid window once per shuffled epoch."""

    def __init__(self, documents, context, batch_size, rng):
        """Initialize deterministic shuffled traversal over all valid starts."""
        self.documents = documents
        self.context = context
        self.batch_size = batch_size
        self.rng = rng
        self.starts = window_starts(documents, context)
        self.order = []
        self.pos = 0
        self.epoch = 0
        self._reshuffle()

    def _reshuffle(self):
        """Start a new epoch with a freshly shuffled window order."""
        self.order = list(self.starts)
        self.rng.shuffle(self.order)
        self.pos = 0
        self.epoch += 1

    def next_start(self):
        """Return the next start, automatically advancing to a new epoch."""
        if self.pos >= len(self.order):
            self._reshuffle()
        item = self.order[self.pos]
        self.pos += 1
        return item

    def skip_batches(self, n_batches):
        """Advance traversal to match batches consumed before a resumed run."""
        for _ in range(n_batches * self.batch_size):
            self.next_start()

    def next_batch(self, device):
        """Materialize the next batch as input and target tensors."""
        windows = [
            slice_window(self.documents, self.context, self.next_start())
            for _ in range(self.batch_size)
        ]
        return xy_tensors(windows, device)
