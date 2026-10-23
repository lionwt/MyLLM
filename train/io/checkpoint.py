"""Save, load, validate, and locate MiniGPT training checkpoints."""

import pickle
from pathlib import Path

import torch


def save_checkpoint(path, model, config, optimizer=None, **meta):
    """Serialize model state on CPU with configuration and optional metadata."""
    payload = {
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    payload.update(meta)
    torch.save(payload, path)


def _checkpoint_to_cpu(checkpoint):
    """Move model tensors to CPU so restored checkpoints are device-neutral."""
    state = checkpoint.get("model")
    if isinstance(state, dict):
        checkpoint["model"] = {
            name: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
            for name, tensor in state.items()
        }
    return checkpoint


def _reject_legacy_checkpoint(checkpoint):
    """Reject checkpoints whose separate output head is no longer supported."""
    state = checkpoint.get("model")
    if isinstance(state, dict) and "head.weight" in state:
        raise ValueError(
            "This checkpoint is from the old Mini-GPT version (tied head.weight). "
            "Please retrain with the current code."
        )


def load_checkpoint(path):
    """Load a checkpoint safely when possible and validate its model format."""
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = _checkpoint_to_cpu(checkpoint)
    _reject_legacy_checkpoint(checkpoint)
    return checkpoint


def resume_checkpoint_path(output):
    """Choose the latest resumable checkpoint, preferring ``last.pt``."""
    output = Path(output)
    for path in (output / "last.pt", output / "best.pt"):
        if path.exists():
            return path
    raise FileNotFoundError(f"Cannot resume: no last.pt or best.pt in {output}")
