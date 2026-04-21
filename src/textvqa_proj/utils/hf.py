from __future__ import annotations

import os
from pathlib import Path

from textvqa_proj.config import Settings


def local_files_only(settings: Settings) -> bool:
    if settings.model.local_files_only:
        return True
    return any(
        os.getenv(name) == "1"
        for name in ("TEXTVQA_OFFLINE", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def resolve_pretrained_source(
    repo_or_path: str,
    *,
    revision: str = "main",
    local_only: bool = False,
) -> str:
    path = Path(repo_or_path)
    if path.exists():
        return str(path)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        return repo_or_path

    return snapshot_download(
        repo_id=repo_or_path,
        revision=revision,
        local_files_only=local_only,
    )
