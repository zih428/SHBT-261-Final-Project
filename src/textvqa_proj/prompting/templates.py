from __future__ import annotations

DEFAULT_SYSTEM_MESSAGE = (
    "You are answering TextVQA questions. Return a short final answer only, with no explanation."
)

PROMPT_TEMPLATES = {
    "plain": "Question: {question}\nAnswer:",
    "short_answer": (
        "Question: {question}\n"
        "Reply with the shortest exact answer string that best matches the image.\n"
        "Answer:"
    ),
    "ocr_copy_first": (
        "Question: {question}\n"
        "If the answer appears as visible text in the image or OCR tokens, copy it exactly.\n"
        "Answer:"
    ),
    "ocr_injected": "Question: {question}\nOCR tokens: {ocr_tokens}\nAnswer:",
}
