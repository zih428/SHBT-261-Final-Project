from __future__ import annotations

import re
from dataclasses import dataclass

from textvqa_proj.config import GenerationSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.data.normalization import normalize_answer
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import PromptBundle, select_ocr_tokens

COLOR_WORDS = {
    "black",
    "blue",
    "brown",
    "gold",
    "gray",
    "green",
    "orange",
    "pink",
    "purple",
    "red",
    "silver",
    "white",
    "yellow",
}
QUESTION_STOPWORDS = {
    "a",
    "an",
    "answer",
    "appears",
    "color",
    "does",
    "how",
    "image",
    "in",
    "is",
    "it",
    "many",
    "number",
    "of",
    "on",
    "question",
    "say",
    "sign",
    "shown",
    "the",
    "this",
    "what",
    "which",
    "word",
}


@dataclass(frozen=True, slots=True)
class CandidatePhrase:
    text: str
    normalized: str
    token_count: int


def _candidate_phrases(tokens: list[str], *, max_phrase_tokens: int = 3) -> list[CandidatePhrase]:
    candidates: list[CandidatePhrase] = []
    seen: set[str] = set()
    for width in range(1, min(max_phrase_tokens, len(tokens)) + 1):
        for start in range(0, len(tokens) - width + 1):
            phrase = " ".join(tokens[start : start + width]).strip()
            normalized = normalize_answer(phrase)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                CandidatePhrase(text=phrase, normalized=normalized, token_count=width)
            )
    return candidates


def _looks_numeric(text: str) -> bool:
    return bool(re.fullmatch(r"[\d:./-]+", text))


def _question_words(question: str) -> set[str]:
    return {
        token
        for token in normalize_answer(question).split()
        if token and token not in QUESTION_STOPWORDS
    }


def _choose_numeric(candidates: list[CandidatePhrase], question: str) -> CandidatePhrase | None:
    numeric_candidates = [candidate for candidate in candidates if _looks_numeric(candidate.text)]
    if not numeric_candidates:
        return None
    question_lower = question.casefold()
    if "year" in question_lower:
        for candidate in numeric_candidates:
            if re.fullmatch(r"\d{4}", candidate.normalized):
                return candidate
    if "time" in question_lower:
        for candidate in numeric_candidates:
            if ":" in candidate.text:
                return candidate
    return min(
        numeric_candidates,
        key=lambda candidate: (candidate.token_count, len(candidate.text)),
    )


def _choose_color(candidates: list[CandidatePhrase]) -> CandidatePhrase | None:
    for candidate in candidates:
        if candidate.normalized in COLOR_WORDS:
            return candidate
    return None


class OCRLexicalAdapter(BaseModelAdapter):
    adapter_name = "ocr_lexical"

    def generate_one(
        self,
        sample: TextVQASample,
        prompt: PromptBundle,
        generation: GenerationSettings,
    ) -> str:
        del prompt, generation
        selected_tokens = select_ocr_tokens(sample, self.settings.prompt)
        if not selected_tokens:
            return ""

        candidates = _candidate_phrases(selected_tokens)
        if not candidates:
            return ""

        question_lower = sample.question.casefold()
        if question_lower.startswith(("is ", "are ", "do ", "does ", "did ", "was ", "were ")):
            return ""

        if any(
            cue in question_lower
            for cue in ("how many", "what number", "what year", "what time")
        ):
            numeric = _choose_numeric(candidates, sample.question)
            if numeric is not None:
                return numeric.text

        if "what color" in question_lower:
            color = _choose_color(candidates)
            if color is not None:
                return color.text

        question_words = _question_words(sample.question)

        def candidate_score(candidate: CandidatePhrase) -> tuple[float, int, int]:
            overlap = len(question_words & set(candidate.normalized.split()))
            informative_words = [
                token for token in candidate.normalized.split() if token not in QUESTION_STOPWORDS
            ]
            score = 0.0
            score += len(informative_words) * 0.8
            score -= overlap * 0.9
            score += min(len(candidate.normalized), 16) * 0.02
            return (score, -candidate.token_count, -len(candidate.text))

        best = max(candidates, key=candidate_score)
        return best.text
