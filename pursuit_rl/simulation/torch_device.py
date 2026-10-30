"""Canonical CUDA device selection for policy, simulation and rollout.

``torch.device('cuda')`` has no index, whereas tensor.device is ``cuda:0``.
Comparing these directly incorrectly rejects zero-copy CUDA tensors.
"""

import torch


def resolve_torch_device(device):
    """Return a fully qualified device, e.g. ``cuda:0`` rather than ``cuda``.

    Raises immediately if CUDA was explicitly requested but is unavailable.
    Does not change the CUDA device or transfer any tensor.
    """
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device {selected} requested but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver, "
                "or run with --device cpu."
            )
        if selected.index is None:
            selected = torch.device("cuda", torch.cuda.current_device())
    return selected
