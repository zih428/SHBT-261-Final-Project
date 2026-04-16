from __future__ import annotations

from pathlib import Path


def save_bar_plot(labels: list[str], values: list[float], output_path: Path, *, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to render plots") from exc

    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(labels, values)
    axis.set_title(title)
    axis.set_ylabel("Score")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
