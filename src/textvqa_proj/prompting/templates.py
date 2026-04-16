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
        "If the answer appears as visible text in the image, copy it exactly.\n"
        "Answer:"
    ),
    "ocr_injected": "Question: {question}\nOCR tokens: {ocr_tokens}\nAnswer:",
    "ocr_injected_normalized": (
        "Question: {question}\n"
        "OCR tokens: {ocr_tokens}\n"
        "If OCR text nearly matches the answer, normalize only casing or punctuation when needed.\n"
        "Answer:"
    ),
    "ocr_fused": "Question: {question}\nFused OCR tokens: {ocr_tokens}\nAnswer:",
    "answer_format_constrained": (
        "Question: {question}\n"
        "OCR tokens: {ocr_tokens}\n"
        "Return only the final answer string. "
        "Use digits for numbers and preserve visible text when exact copying is appropriate.\n"
        "Answer:"
    ),
    "question_routed": (
        "Question type: {question_type}\n"
        "Question: {question}\n"
        "If this is a yes/no question, answer only yes or no. "
        "If it is numeric, answer with digits only. "
        "Otherwise return the shortest exact answer string.\n"
        "OCR tokens: {ocr_tokens}\n"
        "Answer:"
    ),
}
