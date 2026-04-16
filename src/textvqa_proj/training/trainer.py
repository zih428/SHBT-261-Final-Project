from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from textvqa_proj.utils.io import atomic_write_json, ensure_dir


@dataclass(slots=True)
class TrainingPaths:
    root: Path

    @property
    def checkpoints_dir(self) -> Path:
        return ensure_dir(self.root / "checkpoints")

    @property
    def state_path(self) -> Path:
        return self.root / "trainer_state.json"


def write_trainer_state(paths: TrainingPaths, payload: dict[str, object]) -> None:
    ensure_dir(paths.root)
    atomic_write_json(paths.state_path, payload)


def latest_checkpoint(paths: TrainingPaths) -> Path | None:
    checkpoints = sorted(paths.checkpoints_dir.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None
