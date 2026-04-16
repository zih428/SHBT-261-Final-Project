from pathlib import Path

from textvqa_proj.data.dataset import TextVQASample, load_manifest, write_manifest
from textvqa_proj.utils.io import write_jsonl


def test_load_manifest_merges_external_ocr_tokens(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.jsonl"
    external_ocr_path = tmp_path / "external_ocr.jsonl"
    write_manifest(
        manifest_path,
        [
            TextVQASample(
                sample_id="1",
                question="What word is shown?",
                image="dummy.jpg",
                answers=("open",),
                ocr_tokens=("closed",),
            )
        ],
    )
    write_jsonl(
        external_ocr_path,
        [{"sample_id": "1", "external_ocr_tokens": ["OPEN", "TODAY"]}],
    )

    samples = load_manifest(manifest_path, external_ocr_path=external_ocr_path)

    assert samples[0].external_ocr_tokens == ("OPEN", "TODAY")
