from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.models.ocr_lexical import OCRLexicalAdapter
from textvqa_proj.prompting.builders import PromptBundle


def test_ocr_lexical_prefers_numeric_candidates() -> None:
    settings = Settings()
    settings.prompt.include_ocr = True
    sample = TextVQASample(
        sample_id="1",
        question="What year is shown on the poster?",
        image="dummy.jpg",
        answers=("2024",),
        ocr_tokens=("sale", "2024"),
    )
    prediction = OCRLexicalAdapter(settings).generate_one(
        sample,
        PromptBundle(system_message=None, user_message=""),
        settings.generation,
    )
    assert prediction == "2024"


def test_ocr_lexical_can_use_external_tokens() -> None:
    settings = Settings()
    settings.prompt.include_ocr = True
    settings.prompt.ocr_source = "external"
    settings.prompt.normalize_ocr = True
    sample = TextVQASample(
        sample_id="2",
        question="What word is on the sign?",
        image="dummy.jpg",
        answers=("open",),
        ocr_tokens=("closed",),
        external_ocr_tokens=("OPEN",),
    )
    prediction = OCRLexicalAdapter(settings).generate_one(
        sample,
        PromptBundle(system_message=None, user_message=""),
        settings.generation,
    )
    assert prediction == "open"
