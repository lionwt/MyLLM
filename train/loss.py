"""Cross-entropy loss helpers for training and evaluation."""

import torch
import torch.nn.functional as F

from train.dataset.windows import xy_tensors


def batch_loss(model, xs, ys, vocab_size):
    """Compute next-token cross-entropy for one batch."""
    logits = model(xs)
    return F.cross_entropy(logits.reshape(-1, vocab_size), ys.reshape(-1))


def estimate_loss(model, windows, vocab_size, device, batch_size):
    """Compute mean loss over fixed windows without recording gradients."""
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            chunk = windows[i : i + batch_size]
            xs, ys = xy_tensors(chunk, device)
            total += batch_loss(model, xs, ys, vocab_size).item() * len(chunk)
            count += len(chunk)
    return total / count
