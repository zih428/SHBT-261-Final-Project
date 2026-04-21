from __future__ import annotations

from pathlib import Path

from textvqa_proj.config import Settings
from textvqa_proj.utils.hf import local_files_only, resolve_pretrained_source


def test_local_files_only_respects_setting() -> None:
    settings = Settings()
    settings.model.local_files_only = True

    assert local_files_only(settings) is True


def test_local_files_only_respects_offline_env(monkeypatch) -> None:
    settings = Settings()
    monkeypatch.setenv("TEXTVQA_OFFLINE", "1")

    assert local_files_only(settings) is True


def test_resolve_pretrained_source_keeps_existing_path(tmp_path: Path) -> None:
    source = tmp_path / "model"
    source.mkdir()

    assert resolve_pretrained_source(str(source)) == str(source)


def test_resolve_pretrained_source_downloads_snapshot(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return "/tmp/fake-snapshot"

    import textvqa_proj.utils.hf as hf_utils

    monkeypatch.setattr(hf_utils, "Path", Path)
    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", type("HF", (), {"snapshot_download": staticmethod(fake_snapshot_download)}))

    assert resolve_pretrained_source("Qwen/Qwen2.5-VL-3B-Instruct", revision="main") == "/tmp/fake-snapshot"
    assert calls == [
        {
            "repo_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "revision": "main",
            "local_files_only": False,
        }
    ]
