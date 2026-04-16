from __future__ import annotations

from pathlib import Path
from typing import Any

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import load_huggingface_split, load_manifest, write_manifest
from textvqa_proj.data.splits import stratified_subset
from textvqa_proj.utils.io import append_jsonl, ensure_dir, iter_jsonl, load_image


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

    normalized = split.replace("-", "_")
    manifest_map = {
        "train": settings.data.train_manifest_path,
        "internal_dev": settings.data.internal_dev_manifest_path,
        "train_remainder": settings.data.train_remainder_manifest_path,
        "train_rest": settings.data.train_remainder_manifest_path,
        "validation": settings.data.validation_manifest_path or settings.data.manifest_path,
        "test": settings.data.test_manifest_path,
    }
    manifest_path = manifest_map.get(normalized)
    if manifest_path and Path(manifest_path).exists():
        samples = load_manifest(Path(manifest_path), limit=limit)
    else:
        samples = load_huggingface_split(
            settings.data.hf_dataset_name,
            split,
            cache_dir=settings.data.hf_cache_dir,
            limit=limit,
        )
    ensure_dir(output_path.parent)
    completed_ids = {record["sample_id"] for record in iter_jsonl(output_path)}
    engine = RapidOCR()
    records_written = len(completed_ids)
    for sample in samples:
        if sample.sample_id in completed_ids:
            continue
        image = load_image(sample.image)
        result, _ = engine(np.asarray(image))
        tokens = [
            entry[1].strip()
            for entry in (result or [])
            if len(entry) >= 2 and str(entry[1]).strip()
        ]
        append_jsonl(
            output_path,
            {
                "sample_id": sample.sample_id,
                "question_id": sample.question_id,
                "external_ocr_tokens": tokens,
                "metadata": {"engine": "rapidocr_onnxruntime", "split": split},
            },
        )
        records_written += 1
    return {
        "status": "completed",
        "split": split,
        "count": records_written,
        "resumed_count": len(completed_ids),
        "output_path": str(output_path),
    }
