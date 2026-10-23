"""Dataset splitting, encoding, and window-sampling utilities."""

from train.dataset.encode import encode_texts
from train.dataset.split import load_split, read_texts, save_split, split_documents
from train.dataset.windows import (
    ShuffledWindowEpochs,
    sample_windows,
    slice_window,
    window_starts,
    xy_tensors,
)

__all__ = [
    "encode_texts",
    "load_split",
    "read_texts",
    "save_split",
    "split_documents",
    "ShuffledWindowEpochs",
    "sample_windows",
    "slice_window",
    "window_starts",
    "xy_tensors",
]
