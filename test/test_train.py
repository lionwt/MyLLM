import random
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from device import resolve_device
from train import _sentences, _weighted_choice, sample_windows, split_documents


class SplitTests(unittest.TestCase):
    def test_exact_duplicates_and_near_duplicates_stay_together(self):
        random.seed(0)
        paths = [Path(f"doc_{i}.txt") for i in range(4)]
        texts = [
            "The cat sat on the mat. The dog ran away.",
            "The cat sat on the mat. The dog ran away.",
            "A unique village story about a baker. She opened the shop at dawn.",
            "The cat sat on the mat. The dog ran home later.",
        ]
        train_paths, val_paths, train_texts, val_texts = split_documents(
            paths, texts, val_frac=0.5, threshold=0.4
        )
        names = {p.name for p in train_paths + val_paths}
        self.assertEqual(len(names), 3)
        cat_group = {"doc_0.txt", "doc_3.txt"}
        side = "train" if any(p.name in cat_group for p in train_paths) else "val"
        other = val_paths if side == "train" else train_paths
        self.assertTrue(all(p.name not in cat_group for p in other))

    def test_validation_sentences_are_not_in_train(self):
        random.seed(1)
        paths = [Path("a.txt"), Path("b.txt"), Path("c.txt")]
        texts = [
            "Shared filler appears here. Alpha story sentence one. Alpha story sentence two.",
            "Shared filler appears here. Beta story sentence one. Beta story sentence two.",
            "Shared filler appears here. Gamma story sentence one. Gamma story sentence two.",
        ]
        _, _, train_texts, val_texts = split_documents(paths, texts, val_frac=0.34, threshold=0.99)
        train_sents = set(_sentences(train_texts))
        for sentence in _sentences(val_texts):
            self.assertNotIn(sentence, train_sents)


class SamplingTests(unittest.TestCase):
    def test_longer_document_is_chosen_more_often(self):
        rng = random.Random(0)
        short = [0] * 20
        long = [1] * 200
        context = 8
        counts = {0: 0, 1: 0}
        for _ in range(4000):
            seq = _weighted_choice([short, long], context, rng)
            counts[seq[0]] += 1
        self.assertGreater(counts[1], counts[0] * 5)

    def test_sample_windows_uses_valid_range(self):
        rng = random.Random(2)
        docs = [[3] * 30, [4] * 80]
        windows = sample_windows(docs, 10, 8, rng)
        self.assertEqual(len(windows), 10)
        for window in windows:
            self.assertEqual(len(window), 9)
            self.assertTrue(set(window) <= {3, 4})


class DeviceTests(unittest.TestCase):
    def test_auto_never_raises(self):
        device = resolve_device("auto")
        self.assertIn(device.type, ("cpu", "cuda"))


if __name__ == "__main__":
    unittest.main()
