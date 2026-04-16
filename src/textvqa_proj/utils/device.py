from __future__ import annotations

from collections.abc import Iterable


def detect_available_devices(preferred: Iterable[str]) -> list[str]:
    try:
        import torch
    except ImportError:
        return ["cpu"]

    available: list[str] = []
    for device in preferred:
        if device == "cuda" and torch.cuda.is_available():
            available.append(device)
            continue
        if device == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if mps_backend and mps_backend.is_available():
                available.append(device)
                continue
        if device == "cpu":
            available.append(device)
    return available or ["cpu"]


def pick_device(preferred: Iterable[str]) -> str:
    return detect_available_devices(preferred)[0]
