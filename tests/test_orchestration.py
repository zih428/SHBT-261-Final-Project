from pathlib import Path

from textvqa_proj.orchestration import (
    EvaluationResult,
    select_finalist_specs,
    select_top_backbones,
    select_top_settings_for_backbone,
    select_winner_backbone,
)


def _result(model: str, setting: str, accuracy: float) -> EvaluationResult:
    return EvaluationResult(
        model_config=Path(f"configs/models/{model}.toml"),
        experiment_config=Path(f"configs/experiments/screening/{setting}.toml"),
        model_slug=model,
        run_name=setting,
        accuracy=accuracy,
    )


def test_select_finalists_uses_top_two_backbones_and_top_four_settings() -> None:
    results = [
        _result("qwen25_vl_3b", "plain", 0.66),
        _result("qwen25_vl_3b", "short_answer", 0.64),
        _result("qwen25_vl_3b", "ocr_copy_first", 0.63),
        _result("qwen25_vl_3b", "ocr_injected", 0.62),
        _result("qwen25_vl_3b", "ocr_injected_normalized", 0.61),
        _result("qwen25_vl_3b", "ocr_fused", 0.60),
        _result("internvl2_5_4b", "plain", 0.59),
        _result("internvl2_5_4b", "short_answer", 0.58),
        _result("internvl2_5_4b", "ocr_copy_first", 0.57),
        _result("internvl2_5_4b", "ocr_injected", 0.56),
        _result("internvl2_5_4b", "ocr_injected_normalized", 0.55),
        _result("internvl2_5_4b", "ocr_fused", 0.54),
        _result("llava_phi3_mini", "plain", 0.50),
        _result("llava_phi3_mini", "short_answer", 0.49),
        _result("llava_phi3_mini", "ocr_copy_first", 0.48),
        _result("llava_phi3_mini", "ocr_injected", 0.47),
    ]

    finalists = select_finalist_specs(results, Path("configs/experiments/finalists"))

    assert len(finalists) == 8
    assert finalists[0][0] == Path("configs/models/qwen25_vl_3b.toml")
    assert finalists[0][1] == Path("configs/experiments/finalists/plain.toml")
    assert finalists[-1][0] == Path("configs/models/internvl2_5_4b.toml")
    assert finalists[-1][1] == Path("configs/experiments/finalists/ocr_injected.toml")


def test_select_top_backbones_prefers_best_exact_match_then_mean() -> None:
    ranked = select_top_backbones(
        [
            _result("a", "plain", 0.70),
            _result("a", "short_answer", 0.10),
            _result("b", "plain", 0.69),
            _result("b", "short_answer", 0.68),
        ],
        limit=2,
    )

    assert [score.model_config.stem for score in ranked] == ["a", "b"]


def test_select_top_settings_for_backbone_orders_by_accuracy() -> None:
    selected = select_top_settings_for_backbone(
        [
            _result("qwen", "plain", 0.51),
            _result("qwen", "short_answer", 0.59),
            _result("qwen", "ocr_copy_first", 0.55),
            _result("other", "plain", 0.99),
        ],
        Path("configs/models/qwen.toml"),
        limit=2,
    )

    assert [result.setting_key for result in selected] == ["short_answer", "ocr_copy_first"]


def test_select_winner_backbone_uses_finalist_scores() -> None:
    winner = select_winner_backbone(
        [
            _result("qwen25_vl_3b", "plain", 0.61),
            _result("qwen25_vl_3b", "short_answer", 0.63),
            _result("internvl2_5_4b", "plain", 0.62),
            _result("internvl2_5_4b", "short_answer", 0.60),
        ]
    )

    assert winner.model_config == Path("configs/models/qwen25_vl_3b.toml")
