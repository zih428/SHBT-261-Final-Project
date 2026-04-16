from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.strip().casefold()
    text = _WHITESPACE.sub(" ", text)
    return text


def normalize_answer(text: str) -> str:
    normalized = normalize_text(text)
    return normalized.strip(" \t\r\n")


def normalize_tokens(tokens: list[str] | tuple[str, ...]) -> list[str]:
    return [normalize_text(token) for token in tokens if normalize_text(token)]
