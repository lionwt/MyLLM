"""Tests for dataset splitting, sampling, epochs, and devices."""

import random
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.dataset import (
    LazyEncodedDocuments,
    ShuffledWindowEpochs,
    sample_windows,
    split_documents,
    window_starts,
)
from train.device import resolve_device


class SplitTests(unittest.TestCase):
    """Verify document split coverage, bounds, and reproducibility."""

    def test_split_uses_every_document_once(self):
        paths = [Path(f"doc_{i}.txt") for i in range(20)]
        train_paths, val_paths = split_documents(paths, random.Random(0))
        self.assertEqual(len(val_paths), 2)
        self.assertEqual(sorted(train_paths + val_paths), sorted(paths))
        self.assertNotEqual(train_paths + val_paths, paths)

    def test_split_is_deterministic_for_a_seed(self):
        paths = [Path(f"doc_{i}.txt") for i in range(20)]
        self.assertEqual(
            split_documents(paths, random.Random(5)),
            split_documents(paths, random.Random(5)),
        )

    def test_split_keeps_at_least_one_train_and_val_document(self):
        paths = [Path("a.txt"), Path("b.txt")]
        train_paths, val_paths = split_documents(paths, random.Random(0), val_frac=0.9)
        self.assertEqual((len(train_paths), len(val_paths)), (1, 1))
        with self.assertRaises(ValueError):
            split_documents(paths[:1], random.Random(0))


class SamplingTests(unittest.TestCase):
    """Verify random training-window sampling behavior."""

    def test_longer_document_is_chosen_more_often(self):
        rng = random.Random(0)
        docs = [[0] * 20, [1] * 200]
        counts = {0: 0, 1: 0}
        for window in sample_windows(docs, 4000, 8, rng):
            counts[window[0]] += 1
        self.assertGreater(counts[1], counts[0] * 5)

    def test_sample_windows_uses_valid_range(self):
        rng = random.Random(2)
        docs = [[3] * 30, [4] * 80]
        windows = sample_windows(docs, 10, 8, rng)
        self.assertEqual(len(windows), 10)
        for window in windows:
            self.assertEqual(len(window), 9)
            self.assertTrue(set(window) <= {3, 4})


class EpochSamplingTests(unittest.TestCase):
    """Verify shuffled epoch traversal and resumable progress."""

    def test_window_starts_cover_every_valid_offset(self):
        docs = [[0] * 5, [1] * 3, [2] * 10]
        context = 4
        self.assertEqual(
            window_starts(docs, context),
            [(0, 0), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5)],
        )

    def test_window_starts_rejects_short_documents(self):
        with self.assertRaises(ValueError):
            window_starts([[0] * 4, [1] * 8], context=8)

    def test_one_epoch_visits_each_start_once(self):
        docs = [[0] * 12, [1] * 20]
        context = 8
        starts = window_starts(docs, context)
        loader = ShuffledWindowEpochs(docs, context, batch_size=5, rng=random.Random(0))
        seen = [loader.next_start() for _ in range(len(starts))]
        self.assertEqual(sorted(seen), starts)
        self.assertNotEqual(seen, starts)

    def test_second_epoch_is_a_new_shuffle(self):
        docs = [[0] * 30]
        context = 8
        starts = window_starts(docs, context)
        loader = ShuffledWindowEpochs(docs, context, batch_size=4, rng=random.Random(1))
        first = [loader.next_start() for _ in range(len(starts))]
        second = [loader.next_start() for _ in range(len(starts))]
        self.assertEqual(sorted(second), starts)
        self.assertNotEqual(first, second)

    def test_skip_batches_matches_consuming_batches(self):
        docs = [[0] * 40, [1] * 25]
        context = 8
        skipped = ShuffledWindowEpochs(docs, context, batch_size=4, rng=random.Random(3))
        consumed = ShuffledWindowEpochs(docs, context, batch_size=4, rng=random.Random(3))
        for _ in range(7 * 4):
            consumed.next_start()
        skipped.skip_batches(7)
        self.assertEqual(skipped.next_start(), consumed.next_start())
        self.assertEqual(skipped.epoch, consumed.epoch)
        self.assertEqual(skipped.pos, consumed.pos)


class _StubTokenizer:
    """Map characters to ids so lazy encoding can be tested without BPE."""

    def encode(self, text):
        return [ord(ch) for ch in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class LazyEncodeTests(unittest.TestCase):
    """Verify documents are tokenized on first access and then cached."""

    def test_loads_each_file_once(self):
        tok = _StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i, text in enumerate(["aaaaaaa", "bbbbbbbbbb"]):
                path = Path(tmp) / f"{i}.txt"
                path.write_text(text, encoding="utf-8")
                paths.append(path)
            docs = LazyEncodedDocuments(paths, tok, name="train")
            self.assertEqual(docs.loaded, 0)
            first = docs[0]
            self.assertEqual(first, [ord("a")] * 7)
            self.assertEqual(docs.loaded, 1)
            self.assertIs(docs[0], first)
            self.assertEqual(docs.loaded, 1)
            second = docs[1]
            self.assertEqual(second, [ord("b")] * 10)
            self.assertEqual(docs.loaded, 2)

    def test_epoch_loader_does_not_encode_every_file_up_front(self):
        tok = _StubTokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            paths = []
            for i in range(8):
                path = Path(tmp) / f"{i}.txt"
                path.write_text("x" * 20, encoding="utf-8")
                paths.append(path)
            docs = LazyEncodedDocuments(paths, tok)
            loader = ShuffledWindowEpochs(docs, context=8, batch_size=2, rng=random.Random(0))
            self.assertLess(docs.loaded, len(paths))
            loader.next_start()
            self.assertGreaterEqual(docs.loaded, 1)
            self.assertLess(docs.loaded, len(paths))


class DeviceTests(unittest.TestCase):
    """Verify automatic device selection remains usable."""

    def test_auto_never_raises(self):
        device = resolve_device("auto")
        self.assertIn(device.type, ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()
