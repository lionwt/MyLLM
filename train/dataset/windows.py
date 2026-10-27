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
    """Convert token windows into input and target tensors."""
    return (
        torch.tensor([w[:-1] for w in windows], dtype=torch.long, device=device),
        torch.tensor([w[1:] for w in windows], dtype=torch.long, device=device),
    )


def _sample_from_document(documents, doc_i, context, rng):
    """Sample one valid window from a document, or None if it is too short."""
    seq = documents[doc_i]
    limit = len(seq) - context
    if limit <= 0:
        return None
    offset = rng.randrange(limit)
    return seq[offset : offset + context + 1]


def sample_windows(documents, n_windows, context, rng):
    """Sample valid windows with replacement using the supplied RNG."""
    n_docs = len(documents)
    if n_docs == 0:
        raise ValueError("Need documents longer than context length")
    paths = getattr(documents, "paths", None)
    if paths is not None:
        weights = [max(path.stat().st_size, 1) for path in paths]
        windows = []
        attempts = 0
        limit = max(n_windows * 100, n_docs * 2)
        while len(windows) < n_windows:
            attempts += 1
            if attempts > limit:
                raise ValueError("Need documents longer than context length")
            doc_i = rng.choices(range(n_docs), weights=weights, k=1)[0]
            window = _sample_from_document(documents, doc_i, context, rng)
            if window is not None:
                windows.append(window)
        return windows
    starts = window_starts(documents, context)
    return [slice_window(documents, context, start) for start in rng.choices(starts, k=n_windows)]


class ShuffledWindowEpochs:
    """Yield each valid window once per shuffled epoch, loading documents lazily."""

    def __init__(self, documents, context, batch_size, rng):
        """Initialize deterministic shuffled traversal over documents then windows."""
        self.documents = documents
        self.context = context
        self.batch_size = batch_size
        self.rng = rng
        self.n_docs = len(documents)
        self.doc_order = []
        self.doc_cursor = 0
        self.current_starts = []
        self.pos = 0
        self.epoch = 0
        self._fill_current_doc()

    def _begin_epoch(self):
        """Start a new epoch with a freshly shuffled document order."""
        self.doc_order = list(range(self.n_docs))
        self.rng.shuffle(self.doc_order)
        self.doc_cursor = 0
        self.epoch += 1

    def _fill_current_doc(self):
        """Load the next document that has at least one valid training window."""
        attempts = 0
        while attempts < max(self.n_docs, 1):
            if self.doc_cursor >= len(self.doc_order):
                self._begin_epoch()
            doc_i = self.doc_order[self.doc_cursor]
            self.doc_cursor += 1
            attempts += 1
            seq = self.documents[doc_i]
            limit = len(seq) - self.context
            if limit > 0:
                starts = [(doc_i, start) for start in range(limit)]
                self.rng.shuffle(starts)
                self.current_starts = starts
                self.pos = 0
                return
        raise ValueError("Need documents longer than context length")

    def next_start(self):
        """Return the next start, automatically advancing to a new document or epoch."""
        if self.pos >= len(self.current_starts):
            self._fill_current_doc()
        item = self.current_starts[self.pos]
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
