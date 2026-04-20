from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from textvqa_proj.config import Settings, load_settings
from textvqa_proj.orchestration import load_json, training_run_root
from textvqa_proj.utils.io import ensure_dir

SEED_SUFFIX_RE = re.compile(r"-seed\d+$")


@dataclass(frozen=True, slots=True)
class CoreRunRecord:
    config_path: Path
    run_name: str
    family_key: str
    eval_loss: float
    settings: Settings


@dataclass(frozen=True, slots=True)
class FollowupSelection:
    family_key: str
    representative_run_name: str
    representative_config_path: Path
    mean_eval_loss: float
    best_eval_loss: float
    scores: tuple[CoreRunRecord, ...]
    settings: Settings

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_key": self.family_key,
            "representative_run_name": self.representative_run_name,
            "representative_config_path": str(self.representative_config_path),
            "mean_eval_loss": self.mean_eval_loss,
            "best_eval_loss": self.best_eval_loss,
            "learning_rate": self.settings.training.learning_rate,
            "lora_rank": self.settings.lora.rank,
            "lora_alpha": self.settings.lora.alpha,
            "lora_target_modules": list(self.settings.lora.target_modules),
            "scores": [
                {
                    "config_path": str(score.config_path),
                    "run_name": score.run_name,
                    "family_key": score.family_key,
                    "eval_loss": score.eval_loss,
                }
                for score in self.scores
            ],
        }


def core_family_key(run_name: str) -> str:
    return SEED_SUFFIX_RE.sub("", run_name)


def load_completed_core_records(
    repo_root: Path,
    *,
    base_config_paths: list[Path],
    model_config_path: Path,
    training_config_paths: list[Path],
    extra_config_paths: list[Path],
) -> list[CoreRunRecord]:
    records: list[CoreRunRecord] = []
    incomplete: list[str] = []
    missing_eval: list[str] = []

    for training_config_path in training_config_paths:
        config_paths = [
            *base_config_paths,
            model_config_path,
            training_config_path,
            *extra_config_paths,
        ]
        settings = load_settings(config_paths)
        run_root = training_run_root(repo_root, settings)
        state = load_json(run_root / "trainer_state.json")
        if state is None or state.get("status") != "completed":
            incomplete.append(training_config_path.stem)
            continue
        latest_eval = state.get("latest_eval") or {}
        eval_loss = latest_eval.get("eval_loss")
        if eval_loss is None:
            missing_eval.append(training_config_path.stem)
            continue
        records.append(
            CoreRunRecord(
                config_path=training_config_path,
                run_name=settings.run_name,
                family_key=core_family_key(settings.run_name),
                eval_loss=float(eval_loss),
                settings=settings,
            )
        )

    if incomplete:
        raise ValueError(
            "Core matrix is not fully complete yet. Incomplete runs: "
            + ", ".join(sorted(incomplete))
        )
    if missing_eval:
        raise ValueError(
            "Completed core runs are missing eval_loss. Affected runs: "
            + ", ".join(sorted(missing_eval))
        )
    return records


def select_followup_winner(records: list[CoreRunRecord]) -> FollowupSelection:
    if not records:
        raise ValueError("No completed core run records were provided.")

    grouped: dict[str, list[CoreRunRecord]] = {}
    for record in records:
        grouped.setdefault(record.family_key, []).append(record)

    def sort_key(item: tuple[str, list[CoreRunRecord]]) -> tuple[float, float, str]:
        family_key, family_scores = item
        return (
            mean(score.eval_loss for score in family_scores),
            min(score.eval_loss for score in family_scores),
            family_key,
        )

    winner_family_key, winner_scores = min(grouped.items(), key=sort_key)
    representative = next(
        (score for score in winner_scores if score.run_name.endswith("seed07")),
        sorted(winner_scores, key=lambda score: score.run_name)[0],
    )
    return FollowupSelection(
        family_key=winner_family_key,
        representative_run_name=representative.run_name,
        representative_config_path=representative.config_path,
        mean_eval_loss=mean(score.eval_loss for score in winner_scores),
        best_eval_loss=min(score.eval_loss for score in winner_scores),
        scores=tuple(sorted(winner_scores, key=lambda score: score.run_name)),
        settings=representative.settings,
    )


def build_followup_override_toml(selection: FollowupSelection) -> str:
    target_modules = ", ".join(f'"{module}"' for module in selection.settings.lora.target_modules)
    return "\n".join(
        [
            "[metadata]",
            f'selected_core_family = "{selection.family_key}"',
            f'selected_core_run_name = "{selection.representative_run_name}"',
            f"selected_core_mean_eval_loss = {selection.mean_eval_loss}",
            f"selected_core_best_eval_loss = {selection.best_eval_loss}",
            "",
            "[training]",
            f"learning_rate = {selection.settings.training.learning_rate}",
            "",
            "[lora]",
            f"rank = {selection.settings.lora.rank}",
            f"alpha = {selection.settings.lora.alpha}",
            f"dropout = {selection.settings.lora.dropout}",
            f'bias = "{selection.settings.lora.bias}"',
            f"target_modules = [{target_modules}]",
            "",
        ]
    )


def write_followup_override(path: Path, selection: FollowupSelection) -> Path:
    ensure_dir(path.parent)
    path.write_text(build_followup_override_toml(selection), encoding="utf-8")
    return path
