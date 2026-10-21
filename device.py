import pickle

import torch


def _cuda_status():
    return (
        f"torch={torch.__version__}, "
        f"cuda_built={torch.version.cuda!r}, "
        f"cuda_available={torch.cuda.is_available()}"
    )


def _require_cuda():
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise RuntimeError(
        "CUDA is not available to this PyTorch install. "
        f"{_cuda_status()}. "
        "Install a CUDA wheel, then rerun without --device cpu:\n"
        "  pip uninstall -y torch\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
        "  python -c \"import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))\""
    )


def resolve_device(preference="cuda"):
    preference = (preference or "cuda").lower()
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if preference == "cuda":
        return _require_cuda()
    raise ValueError(f"Unknown device preference: {preference}")


def describe_device(device=None):
    device = torch.device(device) if device is not None else resolve_device()
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)


def save_checkpoint(path, model, config, optimizer=None, **meta):
    payload = {
        "model": {name: tensor.detach().cpu() for name, tensor in model.state_dict().items()},
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    payload.update(meta)
    torch.save(payload, path)


def _checkpoint_to_cpu(checkpoint):
    state = checkpoint.get("model")
    if isinstance(state, dict):
        checkpoint["model"] = {
            name: tensor.detach().cpu() if torch.is_tensor(tensor) else tensor
            for name, tensor in state.items()
        }
    return checkpoint


def _reject_legacy_checkpoint(checkpoint):
    state = checkpoint.get("model")
    if isinstance(state, dict) and "head.weight" in state:
        raise ValueError(
            "This checkpoint is from the old Mini-GPT version (tied head.weight). "
            "Please retrain with the current code."
        )


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint = _checkpoint_to_cpu(checkpoint)
    _reject_legacy_checkpoint(checkpoint)
    return checkpoint
