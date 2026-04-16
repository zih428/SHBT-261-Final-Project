from __future__ import annotations

import hashlib
from pathlib import Path


def fingerprint_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_key_path(root: Path, namespace: str, key: str, suffix: str) -> Path:
    digest = fingerprint_text(key)
    return root / namespace / f"{digest}{suffix}"
