from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from textvqa_proj.utils.io import atomic_write_json, ensure_dir


@dataclass(slots=True)
class TrainingPaths:
    root: Path

    @property
    def checkpoints_dir(self) -> Path:
        return ensure_dir(self.root)

    @property
    def adapter_dir(self) -> Path:
        return ensure_dir(self.root / "adapter")

    @property
    def processor_dir(self) -> Path:
        return ensure_dir(self.root / "processor")

    @property
    def state_path(self) -> Path:
        return self.root / "trainer_state.json"

    @property
    def settings_path(self) -> Path:
        return self.root / "settings.json"


def write_trainer_state(paths: TrainingPaths, payload: dict[str, object]) -> None:
    ensure_dir(paths.root)
    stamped_payload = dict(payload)
    stamped_payload["updated_at"] = datetime.now(tz=UTC).isoformat()
    atomic_write_json(paths.state_path, stamped_payload)


def write_training_settings(paths: TrainingPaths, payload: dict[str, object]) -> None:
    ensure_dir(paths.root)
    if not paths.settings_path.exists():
        atomic_write_json(paths.settings_path, payload)
        return
    existing_payload = json.loads(paths.settings_path.read_text(encoding="utf-8"))
    if existing_payload != payload:
        raise RuntimeError(
            f"Training directory {paths.root} already exists with different settings. "
            "Use a new run_name, training.run_tag, or runtime.run_tag to separate protocols."
        )


def latest_checkpoint(paths: TrainingPaths) -> Path | None:
    checkpoints = sorted(paths.checkpoints_dir.glob("checkpoint-*"))
    return checkpoints[-1] if checkpoints else None
