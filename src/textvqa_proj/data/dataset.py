from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from textvqa_proj.utils.io import iter_jsonl, write_jsonl


@dataclass(slots=True)
class TextVQASample:
    sample_id: str
    question: str
    image: str
    answers: tuple[str, ...]
    ocr_tokens: tuple[str, ...] = ()
    external_ocr_tokens: tuple[str, ...] = ()
    split: str | None = None
    question_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_record(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "question": self.question,
            "image": self.image,
            "answers": list(self.answers),
            "ocr_tokens": list(self.ocr_tokens),
            "external_ocr_tokens": list(self.external_ocr_tokens),
            "split": self.split,
            "question_id": self.question_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> TextVQASample:
        sample_id = record.get("sample_id") or str(record.get("question_id"))
        return cls(
            sample_id=sample_id,
            question=record["question"],
            image=record["image"],
            answers=tuple(record.get("answers", [])),
            ocr_tokens=tuple(record.get("ocr_tokens", [])),
            external_ocr_tokens=tuple(record.get("external_ocr_tokens", [])),
            split=record.get("split"),
            question_id=str(record.get("question_id"))
            if record.get("question_id") is not None
            else None,
            metadata=dict(record.get("metadata", {})),
        )


def _coerce_tokens(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    return tuple(str(token) for token in value if str(token).strip())


def load_external_ocr_map(path: Path) -> dict[str, tuple[str, ...]]:
    records: dict[str, tuple[str, ...]] = {}
    for record in iter_jsonl(path):
        sample_id = str(record.get("sample_id") or record.get("question_id") or "").strip()
        if not sample_id:
            continue
        tokens = _coerce_tokens(
            record.get("external_ocr_tokens") or record.get("ocr_tokens") or record.get("tokens")
        )
        records[sample_id] = tokens
    return records


def _attach_external_ocr(
    samples: list[TextVQASample],
    external_ocr: dict[str, tuple[str, ...]],
) -> None:
    if not external_ocr:
        return
    for sample in samples:
        keys = [sample.sample_id]
        if sample.question_id:
            keys.append(sample.question_id)
        for key in keys:
            tokens = external_ocr.get(key)
            if tokens:
                sample.external_ocr_tokens = tokens
                break


def load_manifest(
    path: Path,
    limit: int | None = None,
    *,
    external_ocr_path: str | Path | None = None,
) -> list[TextVQASample]:
    samples: list[TextVQASample] = []
    for index, record in enumerate(iter_jsonl(path)):
        if limit is not None and index >= limit:
            break
        samples.append(TextVQASample.from_record(record))
    if external_ocr_path:
        _attach_external_ocr(samples, load_external_ocr_map(Path(external_ocr_path)))
    return samples


def write_manifest(path: Path, samples: Iterable[TextVQASample]) -> None:
    write_jsonl(path, (sample.to_record() for sample in samples))


def load_huggingface_split(
    dataset_name: str,
    split: str,
    cache_dir: str | None = None,
    limit: int | None = None,
    *,
    external_ocr_path: str | Path | None = None,
) -> list[TextVQASample]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required to load the Hugging Face dataset") from exc

    dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
    samples: list[TextVQASample] = []
    for index, record in enumerate(dataset):
        if limit is not None and index >= limit:
            break
        image = record["image"]
        image_path = None
        if isinstance(image, dict) and "path" in image:
            image_path = image["path"]
        if image_path is None:
            image_path = record.get("flickr_300k_url") or record.get("flickr_original_url") or ""
        samples.append(
            TextVQASample(
                sample_id=str(record.get("question_id", index)),
                question=record["question"],
                image=str(image_path),
                answers=tuple(record.get("answers", [])),
                ocr_tokens=tuple(record.get("ocr_tokens", [])),
                external_ocr_tokens=(),
                split=record.get("set_name") or split,
                question_id=str(record.get("question_id"))
                if record.get("question_id") is not None
                else None,
                metadata={
                    "image_id": record.get("image_id"),
                    "image_width": record.get("image_width"),
                    "image_height": record.get("image_height"),
                },
            )
        )
    if external_ocr_path:
        _attach_external_ocr(samples, load_external_ocr_map(Path(external_ocr_path)))
    return samples
