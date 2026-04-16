from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textvqa_proj.config import PromptSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.data.normalization import normalize_tokens
from textvqa_proj.data.ocr_features import fuse_ocr_tokens
from textvqa_proj.data.splits import question_prefix
from textvqa_proj.prompting.templates import DEFAULT_SYSTEM_MESSAGE, PROMPT_TEMPLATES


@dataclass(slots=True)
class PromptBundle:
    system_message: str | None
    user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)


def select_ocr_tokens(sample: TextVQASample, settings: PromptSettings) -> list[str]:
    if not settings.include_ocr:
        return []

    source = settings.ocr_source.casefold()
    if source == "dataset":
        tokens = list(sample.ocr_tokens)
    elif source == "external":
        tokens = list(sample.external_ocr_tokens)
    elif source == "fused":
        tokens = fuse_ocr_tokens(sample.ocr_tokens, sample.external_ocr_tokens)
    else:
        raise ValueError(f"Unsupported OCR source {settings.ocr_source!r}")

    if settings.normalize_ocr:
        tokens = normalize_tokens(tokens)
    if settings.max_ocr_tokens is not None:
        tokens = tokens[: settings.max_ocr_tokens]
    return [token for token in tokens if token]


def build_prompt(sample: TextVQASample, settings: PromptSettings) -> PromptBundle:
    template = PROMPT_TEMPLATES[settings.template]
    ocr_tokens = select_ocr_tokens(sample, settings)
    ocr_text = ", ".join(token for token in ocr_tokens if token)
    user_message = template.format(
        question=sample.question,
        ocr_tokens=ocr_text or "N/A",
        question_type=question_prefix(sample.question),
    )
    return PromptBundle(
        system_message=settings.system_message or DEFAULT_SYSTEM_MESSAGE,
        user_message=user_message,
        metadata={
            "template": settings.template,
            "ocr_token_count": len(sample.ocr_tokens),
            "selected_ocr_token_count": len(ocr_tokens),
            "ocr_included": settings.include_ocr,
            "ocr_source": settings.ocr_source,
            "normalize_ocr": settings.normalize_ocr,
            "question_type": question_prefix(sample.question),
        },
    )
