from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from statistics import mean

from textvqa_proj.config import Settings
from textvqa_proj.data.dataset import TextVQASample, load_huggingface_split, load_manifest
from textvqa_proj.eval.bootstrap_ci import bootstrap_mean_ci
from textvqa_proj.eval.breakdowns import compute_prediction_breakdowns
from textvqa_proj.eval.semantic_metrics import (
    aggregate_token_overlap,
    try_optional_semantic_metrics,
)
from textvqa_proj.eval.textvqa_accuracy import aggregate_accuracy, score_prediction
from textvqa_proj.inference.postprocess import clean_and_normalize_prediction
from textvqa_proj.inference.run_store import PredictionRecord, RunStore
from textvqa_proj.models.base import BaseModelAdapter
from textvqa_proj.prompting.builders import build_prompt
from textvqa_proj.utils.io import ensure_dir
from textvqa_proj.utils.perf import is_oom_error, release_torch_cache
from textvqa_proj.utils.profiling import time_block

LOGGER = logging.getLogger(__name__)


def _select_reference_answer(answers: list[str]) -> str:
    if not answers:
        return ""
    counts = Counter(answers)
    return counts.most_common(1)[0][0]


def _resolve_manifest(settings: Settings, split: str) -> Path | None:
    normalized = split.replace("-", "_")
    if normalized == "train" and settings.data.train_manifest_path:
        return Path(settings.data.train_manifest_path)
    if normalized == "internal_dev" and settings.data.internal_dev_manifest_path:
        return Path(settings.data.internal_dev_manifest_path)
    if (
        normalized in {"train_remainder", "train_rest"}
        and settings.data.train_remainder_manifest_path
    ):
        return Path(settings.data.train_remainder_manifest_path)
    if normalized == "validation" and settings.data.validation_manifest_path:
        return Path(settings.data.validation_manifest_path)
    if normalized == "test" and settings.data.test_manifest_path:
        return Path(settings.data.test_manifest_path)
    if settings.data.manifest_path:
        return Path(settings.data.manifest_path)
    return None


def _validate_external_ocr_requirements(settings: Settings) -> None:
    if settings.prompt.ocr_source not in {"external", "fused"}:
        return
    external_path = settings.data.external_ocr_manifest_path
    if not external_path:
        raise RuntimeError(
            "This config requests external/fused OCR, but "
            "data.external_ocr_manifest_path is not set."
        )
    if not Path(external_path).exists():
        raise RuntimeError(
            f"External OCR manifest {external_path} does not exist. "
            "Run materialize-external-ocr first."
        )


def load_samples_for_settings(settings: Settings) -> list[TextVQASample]:
    _validate_external_ocr_requirements(settings)
    manifest_path = _resolve_manifest(settings, settings.experiment.split)
    if manifest_path and manifest_path.exists():
        return load_manifest(
            manifest_path,
            limit=settings.experiment.limit,
            external_ocr_path=settings.data.external_ocr_manifest_path,
        )
    return load_huggingface_split(
        settings.data.hf_dataset_name,
        settings.experiment.split,
        cache_dir=settings.data.hf_cache_dir,
        limit=settings.experiment.limit,
        external_ocr_path=settings.data.external_ocr_manifest_path,
    )


class ExperimentRunner:
    def __init__(self, settings: Settings, adapter: BaseModelAdapter) -> None:
        self.settings = settings
        self.adapter = adapter
        run_root = (
            Path(settings.runtime.output_root)
            / settings.experiment.name
            / settings.run_dir_name
        )
        self.run_store = RunStore(
            ensure_dir(run_root),
            settings,
        )

    def _generate_prediction_records(
        self,
        batch: list[TextVQASample],
        prompts: list,
    ) -> list[tuple[TextVQASample, object, str, float]]:
        try:
            with time_block() as timing:
                raw_predictions = self.adapter.generate_batch(
                    batch,
                    prompts,
                    self.settings.generation,
                )
        except RuntimeError as exc:
            if len(batch) == 1 or not is_oom_error(exc):
                raise
            midpoint = len(batch) // 2
            LOGGER.warning(
                "Batch size %s hit OOM for %s; retrying with %s and %s",
                len(batch),
                self.settings.model.model_name,
                midpoint,
                len(batch) - midpoint,
            )
            release_torch_cache()
            return [
                *self._generate_prediction_records(batch[:midpoint], prompts[:midpoint]),
                *self._generate_prediction_records(batch[midpoint:], prompts[midpoint:]),
            ]
        latency = timing["elapsed_seconds"] / max(len(batch), 1)
        return [
            (sample, prompt, raw_prediction, latency)
            for sample, prompt, raw_prediction in zip(
                batch,
                prompts,
                raw_predictions,
                strict=True,
            )
        ]

    def run(self) -> dict[str, object]:
        samples = load_samples_for_settings(self.settings)
        samples_by_id = {sample.sample_id: sample for sample in samples}
        completed_ids = (
            self.run_store.load_completed_ids() if self.settings.experiment.resume else set()
        )
        pending_samples = [sample for sample in samples if sample.sample_id not in completed_ids]
        LOGGER.info(
            "Loaded %s samples (%s pending, %s completed)",
            len(samples),
            len(pending_samples),
            len(completed_ids),
        )

        if pending_samples:
            self.adapter.load()
            try:
                batch_size = self.settings.eval_batch_size
                for start in range(0, len(pending_samples), batch_size):
                    batch = pending_samples[start : start + batch_size]
                    prompts = [build_prompt(sample, self.settings.prompt) for sample in batch]
                    prediction_records = self._generate_prediction_records(batch, prompts)
                    for sample, prompt, raw_prediction, latency in prediction_records:
                        cleaned_prediction, normalized_prediction = clean_and_normalize_prediction(
                            raw_prediction
                        )
                        score = score_prediction(cleaned_prediction, sample.answers)
                        record = PredictionRecord(
                            sample_id=sample.sample_id,
                            prediction=cleaned_prediction,
                            normalized_prediction=normalized_prediction,
                            answers=list(sample.answers),
                            normalized_answers=score.normalized_answers,
                            reference_answer=_select_reference_answer(score.normalized_answers),
                            any_match=score.any_match,
                            consensus_match=score.consensus_match,
                            question=sample.question,
                            prompt_template=prompt.metadata["template"],
                            latency_seconds=latency,
                            metadata={
                                "ocr_token_count": len(sample.ocr_tokens),
                                "selected_ocr_token_count": prompt.metadata.get(
                                    "selected_ocr_token_count", 0
                                ),
                                "ocr_source": prompt.metadata.get("ocr_source"),
                                "split": sample.split,
                                "question_id": sample.question_id,
                            },
                        )
                        self.run_store.append_prediction(record)
            finally:
                self.adapter.unload()

        predictions = self.run_store.load_predictions()
        match_results = [
            score_prediction(prediction.prediction, prediction.answers)
            for prediction in predictions
        ]
        primary_accuracy = aggregate_accuracy(match_results, self.settings.experiment.match_type)
        ci = bootstrap_mean_ci(
            [
                result.any_match
                if self.settings.experiment.match_type == "any"
                else result.consensus_match
                for result in match_results
            ],
            seed=self.settings.runtime.seed,
        )
        token_metrics = aggregate_token_overlap(
            (prediction.normalized_prediction, prediction.reference_answer)
            for prediction in predictions
        )
        metrics: dict[str, object] = {
            "match_type": self.settings.experiment.match_type,
            "accuracy": primary_accuracy,
            "consensus_accuracy": mean(record.consensus_match for record in predictions)
            if predictions
            else 0.0,
            "count": len(predictions),
            "latency_seconds_mean": mean(record.latency_seconds for record in predictions)
            if predictions
            else 0.0,
            "accuracy_ci95": {"lower": ci[0], "upper": ci[1]},
            **token_metrics,
        }
        if self.settings.experiment.include_semantic_metrics:
            metrics.update(
                try_optional_semantic_metrics(
                    (prediction.normalized_prediction, prediction.reference_answer)
                    for prediction in predictions
                )
            )
        self.run_store.write_breakdowns(compute_prediction_breakdowns(predictions, samples_by_id))
        self.run_store.finalize(metrics)
        return metrics
