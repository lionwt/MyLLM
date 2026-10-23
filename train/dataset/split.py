"""Read documents and persist reproducible train/validation splits."""

import json
from pathlib import Path


def read_texts(paths):
    """Read UTF-8 text from each path in order."""
    return [Path(path).read_text(encoding="utf-8") for path in paths]


def split_documents(paths, rng, val_frac=0.1):
    """Shuffle documents into nonempty train and validation partitions."""
    if len(paths) < 2:
        raise ValueError("Need at least two documents for train/validation split")
    shuffled = list(paths)
    rng.shuffle(shuffled)
    n_val = min(max(1, round(val_frac * len(shuffled))), len(shuffled) - 1)
    return shuffled[n_val:], shuffled[:n_val]


def save_split(path, train_paths, val_paths):
    """Store split membership by filename as formatted JSON."""
    Path(path).write_text(
        json.dumps(
            {
                "train": [p.name for p in train_paths],
                "val": [p.name for p in val_paths],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_split(path, data_dir):
    """Restore a split and verify that every referenced document still exists."""
    split_path = Path(path)
    if not split_path.exists():
        raise FileNotFoundError(f"Cannot resume: missing {split_path}")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    data = Path(data_dir)
    train_paths = [data / name for name in split["train"]]
    val_paths = [data / name for name in split["val"]]
    missing = [str(path) for path in train_paths + val_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Cannot resume: missing documents: " + ", ".join(missing))
    return train_paths, val_paths
