from __future__ import annotations

import re

from textvqa_proj.data.normalization import normalize_answer

_ANSWER_PREFIX = re.compile(r"^\s*answer\s*[:\-]\s*", flags=re.IGNORECASE)


def clean_prediction(text: str) -> str:
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = _ANSWER_PREFIX.sub("", text)
    return text.strip()


def clean_and_normalize_prediction(text: str) -> tuple[str, str]:
    cleaned = clean_prediction(text)
    return cleaned, normalize_answer(cleaned)
