from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from textvqa_proj.config import Settings, load_settings


def resolve_repo_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    model_config: Path
    experiment_config: Path
    model_slug: str
    run_name: str
    accuracy: float

    @property
    def model_key(self) -> str:
        return self.model_config.stem

    @property
    def setting_key(self) -> str:
        return self.experiment_config.stem


@dataclass(frozen=True, slots=True)
class BackboneScore:
    model_config: Path
    mean_accuracy: float
    best_accuracy: float


def evaluation_run_root(repo_root: Path, settings: Settings) -> Path:
    return (
        resolve_repo_path(repo_root, settings.runtime.output_root)
        / settings.experiment.name
        / settings.run_dir_name
    )


def training_run_root(repo_root: Path, settings: Settings) -> Path:
    return (
        resolve_repo_path(repo_root, settings.training.output_root)
        / settings.experiment.name
        / settings.run_dir_name
    )


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def load_evaluation_result(repo_root: Path, config_paths: list[Path]) -> EvaluationResult | None:
    settings = load_settings(config_paths)
    run_root = evaluation_run_root(repo_root, settings)
    progress = load_json(run_root / "progress.json")
    metrics = load_json(run_root / "metrics.json")
    if progress is None or metrics is None or progress.get("status") != "completed":
        return None
    return EvaluationResult(
        model_config=config_paths[-2],
        experiment_config=config_paths[-1],
        model_slug=settings.model_slug,
        run_name=settings.run_name,
        accuracy=float(metrics["accuracy"]),
    )


def evaluation_completed(repo_root: Path, config_paths: list[Path]) -> bool:
    return load_evaluation_result(repo_root, config_paths) is not None


def training_completed(repo_root: Path, config_paths: list[Path]) -> bool:
    settings = load_settings(config_paths)
    run_root = training_run_root(repo_root, settings)
    trainer_state = load_json(run_root / "trainer_state.json")
    return trainer_state is not None and trainer_state.get("status") == "completed"


def select_top_backbones(
    results: list[EvaluationResult], *, limit: int = 2
) -> list[BackboneScore]:
    grouped: dict[Path, list[float]] = {}
    for result in results:
        grouped.setdefault(result.model_config, []).append(result.accuracy)
    ranked = [
        BackboneScore(
            model_config=model_config,
            mean_accuracy=mean(scores),
            best_accuracy=max(scores),
        )
        for model_config, scores in grouped.items()
    ]
    ranked.sort(
        key=lambda score: (-score.best_accuracy, -score.mean_accuracy, score.model_config.stem)
    )
    return ranked[:limit]


def select_top_settings_for_backbone(
    results: list[EvaluationResult],
    model_config: Path,
    *,
    limit: int = 4,
) -> list[EvaluationResult]:
    candidates = [result for result in results if result.model_config == model_config]
    candidates.sort(key=lambda result: (-result.accuracy, result.setting_key))
    return candidates[:limit]


def select_finalist_specs(
    screening_results: list[EvaluationResult],
    finalist_dir: Path,
    *,
    backbone_limit: int = 2,
    setting_limit: int = 4,
) -> list[tuple[Path, Path]]:
    finalist_specs: list[tuple[Path, Path]] = []
    for backbone in select_top_backbones(screening_results, limit=backbone_limit):
        for result in select_top_settings_for_backbone(
            screening_results,
            backbone.model_config,
            limit=setting_limit,
        ):
            finalist_specs.append(
                (backbone.model_config, finalist_dir / f"{result.setting_key}.toml")
            )
    return finalist_specs


def select_winner_backbone(finalist_results: list[EvaluationResult]) -> BackboneScore:
    ranked = select_top_backbones(finalist_results, limit=1)
    if not ranked:
        raise ValueError("No finalist results are available.")
    return ranked[0]
