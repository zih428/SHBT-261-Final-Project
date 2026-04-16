from __future__ import annotations

import os

from textvqa_proj.config import Settings


def local_files_only(settings: Settings) -> bool:
    if settings.model.local_files_only:
        return True
    return any(
        os.getenv(name) == "1"
        for name in ("TEXTVQA_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
