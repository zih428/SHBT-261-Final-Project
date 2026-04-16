from __future__ import annotations

from pathlib import Path
from typing import Any

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import load_huggingface_split, write_manifest
from textvqa_proj.data.splits import stratified_subset
from textvqa_proj.utils.io import ensure_dir, load_image, write_jsonl


def materialize_internal_dev_split(
    settings: Settings,
    *,
    subset_size: int,
    dev_output_path: Path,
    train_output_path: Path,
) -> dict[str, Any]:
    train_samples = load_huggingface_split(
        settings.data.hf_dataset_name,
        "train",
        cache_dir=settings.data.hf_cache_dir,
    )
    dev_samples = stratified_subset(
        train_samples,
        subset_size=subset_size,
        seed=settings.runtime.seed,
    )
    dev_ids = {sample.sample_id for sample in dev_samples}
    train_remainder = [
        sample for sample in train_samples if sample.sample_id not in dev_ids
    ]
    write_manifest(dev_output_path, dev_samples)
    write_manifest(train_output_path, train_remainder)
    return {
        "status": "completed",
        "subset_size": len(dev_samples),
        "train_remainder_size": len(train_remainder),
        "dev_output_path": str(dev_output_path),
        "train_output_path": str(train_output_path),
    }


def materialize_external_ocr_manifest(
    settings: Settings,
    *,
    split: str,
    output_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    try:
        import numpy as np
        from rapidocr_onnxruntime import RapidOCR
    except ImportError as exc:
        raise RuntimeError(
            "rapidocr-onnxruntime is required for external OCR materialization. "
            "Install it with uv pip install -e '.[ocr]'."
        ) from exc

    samples = load_huggingface_split(
        settings.data.hf_dataset_name,
        split,
        cache_dir=settings.data.hf_cache_dir,
        limit=limit,
    )
    ensure_dir(output_path.parent)
    engine = RapidOCR()
    records: list[dict[str, Any]] = []
    for sample in samples:
        image = load_image(sample.image)
        result, _ = engine(np.asarray(image))
        tokens = [
            entry[1].strip()
            for entry in (result or [])
            if len(entry) >= 2 and str(entry[1]).strip()
        ]
        records.append(
            {
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "external_ocr_tokens": tokens,
                "metadata": {"engine": "rapidocr_onnxruntime", "split": split},
            }
        )
    write_jsonl(output_path, records)
    return {
        "status": "completed",
        "split": split,
        "count": len(records),
        "output_path": str(output_path),
    }
