"""PyTorch compute-device selection and diagnostics."""

import torch


def _cuda_status():
    """Summarize the CUDA capabilities visible to the current PyTorch build."""
    return (
        f"torch={torch.__version__}, "
        f"cuda_built={torch.version.cuda!r}, "
        f"cuda_available={torch.cuda.is_available()}"
    )


def _require_cuda():
    """Return the CUDA device or raise an actionable installation error."""
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
    """Resolve ``cpu``, ``cuda``, or ``auto`` into a PyTorch device."""
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
    """Return a human-readable device description, including the GPU name."""
    device = torch.device(device) if device is not None else resolve_device()
    if device.type == "cuda":
        return f"{device} ({torch.cuda.get_device_name(device)})"
    return str(device)
