import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluate import repetition_stats
from generate import generate
from model import MiniGPT
from tokenizer import ByteBPE


class DummyTok:
    def encode(self, text):
        return [1, 2, 3]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


class DummyModel:
    context = 8

    def __init__(self, vocab=16):
        self.vocab = vocab
        self._param = torch.zeros(1)

    def parameters(self):
        yield self._param

    def eval(self):
        return self

    def __call__(self, x):
        b, t = x.shape
        logits = torch.zeros(b, t, self.vocab)
        logits[..., 1] = 100.0
        return logits


class GenerateTests(unittest.TestCase):
    def test_greedy_is_deterministic(self):
        tok = ByteBPE().train(["Once upon a time a cat sat on the mat."], vocab_size=300)
        torch.manual_seed(0)
        model = MiniGPT(vocab_size=len(tok.vocab), context=16, dim=16, heads=4, layers=1, dropout=0.0)
        model.eval()
        first, first_ids = generate(model, tok, "Once upon", new_tokens=20, temperature=0)
        second, second_ids = generate(model, tok, "Once upon", new_tokens=20, temperature=0)
        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first, second)

    def test_no_repeat_ngram(self):
        text, ids = generate(
            DummyModel(),
            DummyTok(),
            "prompt",
            new_tokens=24,
            temperature=0,
            no_repeat_ngram=3,
        )
        trigrams = [tuple(ids[i : i + 3]) for i in range(len(ids) - 2)]
        self.assertEqual(len(trigrams), len(set(trigrams)))
        self.assertTrue(text)

    def test_repetition_stats_flags_loops(self):
        looping = repetition_stats([7] * 20)
        self.assertTrue(looping["is_degenerate"])
        self.assertGreaterEqual(looping["longest_run"], 8)
        normal = repetition_stats(list(range(40)))
        self.assertFalse(normal["is_degenerate"])
        self.assertEqual(normal["longest_run"], 1)


if __name__ == "__main__":
    unittest.main()
