import sys
import types
from pathlib import Path

from PIL import Image

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample, write_manifest
from textvqa_proj.data.prepare import materialize_external_ocr_manifest


def test_materialize_external_ocr_uses_internal_dev_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    manifest_path = tmp_path / "internal_dev.jsonl"
    output_path = tmp_path / "external_ocr.jsonl"
    write_manifest(
        manifest_path,
        [
            TextVQASample(
                sample_id="1",
                question="What word is shown?",
                image="dummy.jpg",
                answers=("open",),
                ocr_tokens=("OPEN",),
            )
        ],
    )

    settings = Settings()
    settings.data.internal_dev_manifest_path = str(manifest_path)

    class FakeRapidOCR:
        def __call__(self, array):
            del array
            return [((0, 0, 1, 1), "OPEN", 0.99)], None

    fake_module = types.SimpleNamespace(RapidOCR=lambda: FakeRapidOCR())
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)
    monkeypatch.setattr(
        "textvqa_proj.data.prepare.load_huggingface_split",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("HF fallback should not run")),
    )
    monkeypatch.setattr(
        "textvqa_proj.data.prepare.load_image",
        lambda source: Image.new("RGB", (2, 2), color="white"),
    )

    summary = materialize_external_ocr_manifest(
        settings,
        split="internal_dev",
        output_path=output_path,
    )

    assert summary["count"] == 1
    saved = output_path.read_text(encoding="utf-8")
    assert "OPEN" in saved
