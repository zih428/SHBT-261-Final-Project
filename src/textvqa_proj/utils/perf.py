from __future__ import annotations

import gc
from contextlib import suppress


def is_oom_error(exc: BaseException) -> bool:
    message = str(exc).casefold()
    return any(
        needle in message
        for needle in (
            "out of memory",
            "mps backend out of memory",
            "cuda out of memory",
            "not enough memory",
        )
    )


def release_torch_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        with suppress(RuntimeError):
            torch.mps.empty_cache()
