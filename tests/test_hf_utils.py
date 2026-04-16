from __future__ import annotations

from textvqa_proj.config import Settings
from textvqa_proj.utils.hf import local_files_only


def test_local_files_only_respects_setting() -> None:
    settings = Settings()
    settings.model.local_files_only = True

    assert local_files_only(settings) is True


def test_local_files_only_respects_offline_env(monkeypatch) -> None:
    settings = Settings()
    monkeypatch.setenv("TEXTVQA_OFFLINE", "1")

    assert local_files_only(settings) is True
