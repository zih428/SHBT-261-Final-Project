from textvqa_proj.models.registry import MODEL_REGISTRY


def test_real_backbones_are_registered() -> None:
    assert "qwen2_5_vl" in MODEL_REGISTRY
    assert "blip2" in MODEL_REGISTRY
    assert "llava_hf" in MODEL_REGISTRY
    assert "internvl2_5" in MODEL_REGISTRY
