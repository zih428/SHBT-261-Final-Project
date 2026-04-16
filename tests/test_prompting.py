from textvqa_proj.config import PromptSettings
from textvqa_proj.data.dataset import TextVQASample
from textvqa_proj.prompting.builders import build_prompt, select_ocr_tokens


def test_select_ocr_tokens_can_use_fused_external_tokens() -> None:
    sample = TextVQASample(
        sample_id="1",
        question="What word is shown?",
        image="dummy.jpg",
        answers=("open",),
        ocr_tokens=("OPEN", "sale"),
        external_ocr_tokens=("open", "hours"),
    )
    settings = PromptSettings(include_ocr=True, ocr_source="fused", normalize_ocr=True)
    assert select_ocr_tokens(sample, settings) == ["open", "sale", "hours"]


def test_build_prompt_respects_max_ocr_tokens() -> None:
    sample = TextVQASample(
        sample_id="2",
        question="What text appears?",
        image="dummy.jpg",
        answers=("open",),
        ocr_tokens=("OPEN", "NOW", "TODAY"),
    )
    settings = PromptSettings(
        template="ocr_injected",
        include_ocr=True,
        normalize_ocr=True,
        max_ocr_tokens=2,
    )
    prompt = build_prompt(sample, settings)
    assert "open, now" in prompt.user_message
    assert "today" not in prompt.user_message
