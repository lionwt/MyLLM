"""Tests for Mini-GPT model behavior and checkpoints."""

import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from train.device import resolve_device
from train.io import load_checkpoint, save_checkpoint
from train.model import MiniGPT


CFG = dict(vocab_size=32, context=8, dim=16, heads=4, layers=1, dropout=0.0)


class ModelTests(unittest.TestCase):
    """Verify model outputs, weight tying, and checkpoint compatibility."""

    def test_no_separate_head_after_device_move(self):
        device = resolve_device("cpu" if not torch.cuda.is_available() else "cuda")
        model = MiniGPT(**CFG).to(device)
        self.assertFalse(any(name.startswith("head") for name in model.state_dict()))
        x = torch.randint(0, CFG["vocab_size"], (2, CFG["context"]), device=device)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        self.assertEqual(tuple(logits.shape), (2, CFG["context"], CFG["vocab_size"]))
        self.assertTrue(torch.isfinite(logits).all())
        before = model.token.weight[0, 0].detach().item()
        with torch.no_grad():
            model.token.weight[0, 0] += 1.0
            changed = model(x)
        self.assertFalse(torch.equal(logits, changed))
        with torch.no_grad():
            model.token.weight[0, 0] = before

    def test_save_load_logits_match(self):
        model = MiniGPT(**CFG)
        model.eval()
        x = torch.randint(0, CFG["vocab_size"], (2, CFG["context"]))
        with torch.no_grad():
            expected = model(x)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "best.pt"
            save_checkpoint(path, model, CFG, step=1, val_loss=1.23)
            checkpoint = load_checkpoint(path)
        self.assertEqual(checkpoint["step"], 1)
        self.assertAlmostEqual(checkpoint["val_loss"], 1.23)
        restored = MiniGPT(**checkpoint["config"])
        restored.load_state_dict(checkpoint["model"])
        restored.eval()
        with torch.no_grad():
            got = restored(x)
        self.assertTrue(torch.allclose(expected, got))

    def test_legacy_checkpoint_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "old.pt"
            torch.save(
                {
                    "model": {
                        "head.weight": torch.zeros(2, 2),
                        "token.weight": torch.zeros(2, 2),
                    },
                    "config": {},
                },
                path,
            )
            with self.assertRaises(ValueError) as ctx:
                load_checkpoint(path)
        self.assertIn("old Mini-GPT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
