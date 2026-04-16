from __future__ import annotations

from textvqa_proj.data.normalization import normalize_answer, normalize_tokens


def ocr_count_bucket(tokens: list[str] | tuple[str, ...]) -> str:
    count = len(tokens)
    if count == 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 15:
        return "6-15"
    if count <= 30:
        return "16-30"
    return "31+"


def answer_present_in_ocr(answer: str, tokens: list[str] | tuple[str, ...]) -> bool:
    normalized_answer = normalize_answer(answer)
    if not normalized_answer:
        return False
    return normalized_answer in set(normalize_tokens(tokens))


def fuse_ocr_tokens(*token_lists: list[str] | tuple[str, ...]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for token_list in token_lists:
        for token in normalize_tokens(token_list):
            if token not in seen:
                seen.add(token)
                merged.append(token)
    return merged
