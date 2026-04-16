from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    from textvqa_proj.config import Settings
    from textvqa_proj.data.dataset import TextVQASample, write_manifest
    from textvqa_proj.inference.runner import ExperimentRunner
    from textvqa_proj.models.fake import FakeAnsweringAdapter

    workspace = Path(tempfile.mkdtemp(prefix="textvqa-smoke-"))
    try:
        manifest_path = workspace / "smoke_manifest.jsonl"
        samples = [
            TextVQASample(
                sample_id="1",
                question="What word is on the sign?",
                image=str(workspace / "dummy.jpg"),
                answers=("open", "open"),
                ocr_tokens=("OPEN",),
            ),
            TextVQASample(
                sample_id="2",
                question="What number is shown?",
                image=str(workspace / "dummy.jpg"),
                answers=("12",),
                ocr_tokens=("12",),
            ),
        ]
        write_manifest(manifest_path, samples)
        settings = Settings()
        settings.data.manifest_path = str(manifest_path)
        settings.runtime.output_root = str(workspace / "runs")
        settings.experiment.batch_size = 2
        settings.experiment.resume = True
        metrics = ExperimentRunner(settings, FakeAnsweringAdapter(settings)).run()
        print(metrics)
    finally:
        shutil.rmtree(workspace)


if __name__ == "__main__":
    main()
