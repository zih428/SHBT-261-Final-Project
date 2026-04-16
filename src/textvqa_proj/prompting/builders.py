from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from textvqa_proj.config import PromptSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.data.normalization import normalize_tokens
from textvqa_proj.prompting.templates import DEFAULT_SYSTEM_MESSAGE, PROMPT_TEMPLATES


@dataclass(slots=True)
class PromptBundle:
    system_message: str | None
    user_message: str
    metadata: dict[str, Any] = field(default_factory=dict)


def build_prompt(sample: TextVQASample, settings: PromptSettings) -> PromptBundle:
    template = PROMPT_TEMPLATES[settings.template]
    ocr_tokens = list(sample.ocr_tokens)
    if settings.normalize_ocr:
        ocr_tokens = normalize_tokens(ocr_tokens)
    ocr_text = ", ".join(token for token in ocr_tokens if token)
    user_message = template.format(question=sample.question, ocr_tokens=ocr_text or "N/A")
    return PromptBundle(
        system_message=settings.system_message or DEFAULT_SYSTEM_MESSAGE,
        user_message=user_message,
        metadata={
            "template": settings.template,
            "ocr_token_count": len(sample.ocr_tokens),
            "ocr_included": settings.include_ocr,
        },
    )
