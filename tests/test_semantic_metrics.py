from __future__ import annotations

import sys
import types

from textvqa_proj.eval.semantic_metrics import try_optional_semantic_metrics


def test_try_optional_semantic_metrics_skips_meteor_when_wordnet_is_missing(
    monkeypatch,
) -> None:
    nltk_module = types.ModuleType("nltk")
    translate_module = types.ModuleType("nltk.translate")
    bleu_module = types.ModuleType("nltk.translate.bleu_score")
    meteor_module = types.ModuleType("nltk.translate.meteor_score")

    class FakeSmoothingFunction:
        def __init__(self) -> None:
            self.method1 = lambda *args, **kwargs: None

    bleu_module.SmoothingFunction = FakeSmoothingFunction
    bleu_module.corpus_bleu = lambda references, hypotheses, smoothing_function=None: 0.5

    def failing_meteor_score(*args, **kwargs):
        raise LookupError("wordnet not installed")

    meteor_module.meteor_score = failing_meteor_score
    translate_module.bleu_score = bleu_module
    translate_module.meteor_score = meteor_module
    nltk_module.translate = translate_module

    monkeypatch.setitem(sys.modules, "nltk", nltk_module)
    monkeypatch.setitem(sys.modules, "nltk.translate", translate_module)
    monkeypatch.setitem(sys.modules, "nltk.translate.bleu_score", bleu_module)
    monkeypatch.setitem(sys.modules, "nltk.translate.meteor_score", meteor_module)

    metrics = try_optional_semantic_metrics([("open", "open")])

    assert metrics["bleu"] == 0.5
    assert metrics["meteor"] is None
