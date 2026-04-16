import sys
import types
from pathlib import Path

from PIL import Image

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample, load_huggingface_split, write_manifest
from textvqa_proj.data.prepare import materialize_external_ocr_manifest
from textvqa_proj.utils.io import write_jsonl


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


def test_hf_loader_materializes_pil_images(tmp_path: Path, monkeypatch) -> None:
    fake_image = Image.new("RGB", (3, 3), color="white")

    def fake_load_dataset(name: str, *, split: str, cache_dir: str | None = None):
        del name, split, cache_dir
        return [
            {
                "question_id": 123,
                "question": "What word is shown?",
                "image": fake_image,
                "answers": ["open"],
                "ocr_tokens": ["OPEN"],
                "set_name": "validation",
                "image_id": "abc",
                "image_width": 3,
                "image_height": 3,
            }
        ]

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=fake_load_dataset),
    )

    samples = load_huggingface_split(
        "lmms-lab/textvqa",
        "validation",
        cache_dir=str(tmp_path / "hf"),
    )

    image_path = Path(samples[0].image)
    assert image_path.exists()
    assert image_path.suffix == ".jpg"


def test_materialize_external_ocr_manifest_resumes_existing_rows(
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
                image="one.jpg",
                answers=("open",),
                ocr_tokens=("OPEN",),
            ),
            TextVQASample(
                sample_id="2",
                question="What word is shown?",
                image="two.jpg",
                answers=("closed",),
                ocr_tokens=("CLOSED",),
            ),
        ],
    )
    write_jsonl(
        output_path,
        [
            {
                "sample_id": "1",
                "question_id": 0,
                "external_ocr_tokens": ["OPEN"],
                "metadata": {"engine": "rapidocr_onnxruntime", "split": "internal_dev"},
            }
        ],
    )

    settings = Settings()
    settings.data.internal_dev_manifest_path = str(manifest_path)
    seen_sources: list[str] = []

    class FakeRapidOCR:
        def __call__(self, array):
            del array
            return [((0, 0, 1, 1), "CLOSED", 0.99)], None

    fake_module = types.SimpleNamespace(RapidOCR=lambda: FakeRapidOCR())
    monkeypatch.setitem(sys.modules, "rapidocr_onnxruntime", fake_module)
    monkeypatch.setattr(
        "textvqa_proj.data.prepare.load_image",
        lambda source: seen_sources.append(str(source)) or Image.new("RGB", (2, 2), color="white"),
    )

    summary = materialize_external_ocr_manifest(
        settings,
        split="internal_dev",
        output_path=output_path,
    )

    assert summary["count"] == 2
    assert summary["resumed_count"] == 1
    assert seen_sources == ["two.jpg"]
    saved = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(saved) == 2
