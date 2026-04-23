from __future__ import annotations

import csv
import json
import math
import random
import re
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patheffects as path_effects
from matplotlib import patches
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_ROOT = REPO_ROOT / "paper" / "neurips2026"
FIGURES_DIR = PAPER_ROOT / "figures"
GENERATED_DIR = PAPER_ROOT / "generated"
TABLES_DIR = GENERATED_DIR / "tables"
DATA_DIR = GENERATED_DIR / "data"

SCREENING_DIR = REPO_ROOT / "outputs" / "runs" / "screening"
FINALISTS_DIR = REPO_ROOT / "outputs" / "runs" / "finalists"
APPENDIX_PROMPT_DIR = REPO_ROOT / "outputs" / "runs" / "appendix-prompt-study"
APPENDIX_STRESS_DIR = REPO_ROOT / "outputs" / "runs" / "appendix-stress"
TRAINED_EVAL_DIR = REPO_ROOT / "outputs" / "runs" / "trained_adapters" / "trained-adapter-eval"
TRAINING_DIR = REPO_ROOT / "outputs" / "training"

VAL_MANIFEST = REPO_ROOT / "data" / "cache" / "manifests" / "textvqa_validation.jsonl"
INTERNAL_DEV_MANIFEST = REPO_ROOT / "data" / "cache" / "manifests" / "textvqa_internal_dev.jsonl"
TRAIN_REMAINDER_MANIFEST = REPO_ROOT / "data" / "cache" / "manifests" / "textvqa_train_remainder.jsonl"

BACKBONE_PREFIXES = {
    "qwen2-5-vl-3b-instruct": "Qwen2.5-VL-3B",
    "llava-phi-3-mini-hf": "LLaVA-Phi-3-mini",
    "blip2-opt-2-7b": "BLIP-2 OPT-2.7B",
    "internvl2-5-4b": "InternVL2.5-4B",
    "ocr-lexical": "OCR lexical",
}
BACKBONE_ORDER = [
    "Qwen2.5-VL-3B",
    "LLaVA-Phi-3-mini",
    "BLIP-2 OPT-2.7B",
    "InternVL2.5-4B",
    "OCR lexical",
]
PROMPT_ORDER = [
    "plain",
    "short_answer",
    "ocr_copy_first",
    "ocr_injected",
    "ocr_injected_normalized",
    "ocr_fused",
]
PROMPT_LABELS = {
    "plain": "Plain",
    "short_answer": "Short answer",
    "ocr_copy_first": "OCR copy-first",
    "ocr_injected": "OCR injected",
    "ocr_injected_normalized": "OCR injected\n(normalized)",
    "ocr_fused": "OCR fused",
}

TRAINING_STATE_GLOB = "**/*train-speed-v3/trainer_state.json"

COLOR_BLUE = "#1f5aa6"
COLOR_GREEN = "#3d8b40"
COLOR_PURPLE = "#6f42c1"
COLOR_ORANGE = "#c96a1b"
COLOR_GRAY = "#4d4d4d"


@dataclass
class EvalRow:
    stage: str
    split: str
    model: str
    prompt: str
    path: Path
    metrics: dict[str, float | int | None]


def ensure_dirs() -> None:
    for path in (FIGURES_DIR, TABLES_DIR, DATA_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    with path.open() as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_prediction_map(path: Path) -> dict[str, dict]:
    return {str(record["sample_id"]): record for record in load_jsonl(path)}


def image_id(record: dict) -> str | None:
    metadata = record.get("metadata") or {}
    return record.get("image_id") or metadata.get("image_id")


def latex_escape(text: object) -> str:
    return (
        str(text)
        .replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def normalize_text(text: str) -> str:
    return " ".join("".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text).split())


def parse_screening_or_finalist_run(name: str) -> tuple[str, str]:
    for prefix, label in BACKBONE_PREFIXES.items():
        needle = f"{prefix}-"
        if name.startswith(needle):
            remainder = name[len(needle) :]
            for suffix in ("-internal-dev-mps-tuned-v1", "-validation-mps-tuned-v1"):
                if remainder.endswith(suffix):
                    return label, remainder[: -len(suffix)].replace("-", "_")
    raise ValueError(f"Unrecognized evaluation run name: {name}")


def canonical_screening_rows() -> list[EvalRow]:
    rows: list[EvalRow] = []
    for metrics_path in sorted(SCREENING_DIR.glob("*-mps-tuned-v1/metrics.json")):
        if "archived" in metrics_path.as_posix():
            continue
        run_name = metrics_path.parent.name
        if not any(run_name.startswith(f"{prefix}-") for prefix in BACKBONE_PREFIXES):
            continue
        model, prompt = parse_screening_or_finalist_run(run_name)
        rows.append(
            EvalRow(
                stage="screening",
                split="internal-dev",
                model=model,
                prompt=prompt,
                path=metrics_path.parent,
                metrics=load_json(metrics_path),
            )
        )
    return rows


def finalist_rows() -> list[EvalRow]:
    rows: list[EvalRow] = []
    for metrics_path in sorted(FINALISTS_DIR.glob("*/metrics.json")):
        model, prompt = parse_screening_or_finalist_run(metrics_path.parent.name)
        rows.append(
            EvalRow(
                stage="finalist",
                split="validation",
                model=model,
                prompt=prompt,
                path=metrics_path.parent,
                metrics=load_json(metrics_path),
            )
        )
    return rows


def appendix_rows() -> list[EvalRow]:
    rows: list[EvalRow] = []
    for root, stage in ((APPENDIX_PROMPT_DIR, "appendix-prompt"), (APPENDIX_STRESS_DIR, "appendix-stress")):
        for metrics_path in sorted(root.glob("*/metrics.json")):
            model, prompt = parse_screening_or_finalist_run(metrics_path.parent.name)
            rows.append(
                EvalRow(
                    stage=stage,
                    split="validation",
                    model=model,
                    prompt=prompt,
                    path=metrics_path.parent,
                    metrics=load_json(metrics_path),
                )
            )
    return rows


def training_eval_losses() -> dict[str, float]:
    losses: dict[str, float] = {}
    for state_path in sorted(TRAINING_DIR.glob(TRAINING_STATE_GLOB)):
        state = load_json(state_path)
        run_name = state_path.parent.name
        latest_eval = state.get("latest_eval") or {}
        eval_loss = latest_eval.get("eval_loss")
        if isinstance(eval_loss, (float, int)) and math.isfinite(eval_loss):
            losses[run_name] = float(eval_loss)
    return losses


def parse_trained_slug(run_name: str) -> tuple[str, str]:
    prefix = "qwen2-5-vl-3b-instruct-"
    if not run_name.startswith(prefix):
        raise ValueError(f"Unexpected trained-adapter run: {run_name}")
    remainder = run_name[len(prefix) :]
    for suffix in ("-train-speed-v3-internal-dev-cuda-runpod-v1", "-train-speed-v3-validation-cuda-runpod-v1"):
        if remainder.endswith(suffix):
            return remainder[: -len(suffix)], suffix
    raise ValueError(f"Unexpected trained-adapter suffix: {run_name}")


def slug_to_training_run(slug: str) -> str:
    if slug.startswith("best-assumed-ocr-"):
        return f"qwen2-5-vl-3b-instruct-{slug}-cuda-runpod-v1-train-speed-v3"
    if slug.startswith("best-assumed-"):
        return f"qwen2-5-vl-3b-instruct-{slug}-cuda-runpod-v1-train-speed-v3"
    return f"qwen2-5-vl-3b-instruct-{slug}-cuda-runpod-v1-train-speed-v3"


def trained_rows() -> list[dict[str, object]]:
    eval_losses = training_eval_losses()
    rows: list[dict[str, object]] = []
    for metrics_path in sorted(TRAINED_EVAL_DIR.glob("*/metrics.json")):
        run_name = metrics_path.parent.name
        slug, suffix = parse_trained_slug(run_name)
        split = "validation" if "validation" in suffix else "internal-dev"
        rows.append(
            {
                "slug": slug,
                "split": split,
                "path": metrics_path.parent,
                "metrics": load_json(metrics_path),
                "eval_loss": eval_losses.get(slug_to_training_run(slug)),
            }
        )
    return rows


def trained_label(slug: str) -> str:
    mapping = {
        "all-linear-r16-seed07": "All-linear r16 (seed 07)",
        "all-linear-r16-seed13": "All-linear r16 (seed 13)",
        "all-linear-r32-seed07": "All-linear r32 (seed 07)",
        "all-linear-r32-seed13": "All-linear r32 (seed 13)",
        "attn-r16-seed07": "Attention-only r16 (seed 07)",
        "attn-r16-seed13": "Attention-only r16 (seed 13)",
        "attn-r32-seed07": "Attention-only r32 (seed 07)",
        "attn-r32-seed13": "Attention-only r32 (seed 13)",
        "best-assumed-ocr-off": "OCR ablation: off",
        "best-assumed-ocr-on": "OCR ablation: on",
        "best-assumed-25pct": "Data scaling: 25%",
        "best-assumed-full": "Data scaling: full",
    }
    return mapping[slug]


def trained_short_label(slug: str) -> str:
    mapping = {
        "all-linear-r16-seed07": "All-lin r16 s07",
        "all-linear-r16-seed13": "All-lin r16 s13",
        "all-linear-r32-seed07": "All-lin r32 s07",
        "all-linear-r32-seed13": "All-lin r32 s13",
        "attn-r16-seed07": "Attn r16 s07",
        "attn-r16-seed13": "Attn r16 s13",
        "attn-r32-seed07": "Attn r32 s07",
        "attn-r32-seed13": "Attn r32 s13",
        "best-assumed-ocr-off": "OCR off",
        "best-assumed-ocr-on": "OCR on",
        "best-assumed-25pct": "Scale 25%",
        "best-assumed-full": "Scale full",
    }
    return mapping[slug]


def trained_group(slug: str) -> str:
    if slug.startswith("all-linear") or slug.startswith("attn"):
        return "Core matrix"
    if "ocr" in slug:
        return "OCR ablation"
    return "Data scaling"


def maybe_compute_meteor(predictions_path: Path) -> float | None:
    try:
        from nltk.corpus import wordnet as wn
        from nltk.translate.meteor_score import meteor_score

        wn.synsets("text")
    except Exception:
        return None

    scores: list[float] = []
    with predictions_path.open() as fh:
        for line in fh:
            rec = json.loads(line)
            prediction = (rec.get("normalized_prediction") or rec.get("prediction") or "").strip()
            references = [normalize_text(answer) for answer in rec.get("answers", []) if normalize_text(answer)]
            if not prediction or not references:
                scores.append(0.0)
                continue
            try:
                scores.append(meteor_score([reference.split() for reference in references], prediction.split()))
            except Exception:
                scores.append(0.0)
    return mean(scores) if scores else None


def maybe_load_judge_similarity(run_root: Path) -> float | None:
    metrics_path = run_root / "judge_metrics.json"
    if not metrics_path.exists():
        return None
    metrics = load_json(metrics_path)
    value = metrics.get("judge_similarity")
    if isinstance(value, (float, int)):
        return float(value)
    return None


def paper_visible_run_roots(
    screening: list[EvalRow],
    finalists: list[EvalRow],
    appendix: list[EvalRow],
    trained: list[dict[str, object]],
) -> list[Path]:
    screening_best = []
    for backbone in BACKBONE_ORDER:
        candidates = [row for row in screening if row.model == backbone]
        if not candidates:
            continue
        screening_best.append(max(candidates, key=lambda row: row.metrics["accuracy"]))

    finalists_sorted = sorted(finalists, key=lambda row: row.metrics["accuracy"], reverse=True)
    trained_internal = sorted(
        (row for row in trained if row["split"] == "internal-dev"),
        key=lambda row: row["metrics"]["accuracy"],
        reverse=True,
    )
    tuned_val = next(row for row in trained if row["slug"] == "all-linear-r16-seed13" and row["split"] == "validation")

    selected = [row.path for row in screening_best]
    selected.extend(row.path for row in finalists_sorted)
    selected.extend(row.path for row in appendix)
    selected.extend(Path(row["path"]) for row in trained_internal)
    selected.append(Path(tuned_val["path"]))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in selected:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def percentage(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.2f}"


def percentage_ci(value: float, ci: dict[str, float]) -> str:
    return f"{100 * value:.2f} [{100 * ci['lower']:.2f}, {100 * ci['upper']:.2f}]"


def latex_bold(text: str) -> str:
    return f"\\textbf{{{text}}}"


def is_best(value: float | None, best: float | None) -> bool:
    return value is not None and best is not None and math.isclose(value, best, rel_tol=0.0, abs_tol=1e-12)


def best_max(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return max(values) if values else None


def best_min(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return min(values) if values else None


def percentage_best(value: float | None, best: float | None) -> str:
    rendered = percentage(value)
    return latex_bold(rendered) if is_best(value, best) else rendered


def percentage_ci_best(value: float, ci: dict[str, float], best: float | None) -> str:
    rendered = percentage_ci(value, ci)
    return latex_bold(rendered) if is_best(value, best) else rendered


def decimal_best(value: float | None, best: float | None) -> str:
    rendered = "--" if value is None else f"{value:.4f}"
    return latex_bold(rendered) if is_best(value, best) else rendered


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n")


def sample_id_sort_key(sample_id: str) -> tuple[int, str]:
    try:
        return (0, f"{int(sample_id):012d}")
    except ValueError:
        return (1, sample_id)


def paired_validation_statistics(finalist: EvalRow, tuned_val: dict[str, object]) -> dict[str, object]:
    zero = load_prediction_map(finalist.path / "predictions.jsonl")
    tuned = load_prediction_map(Path(tuned_val["path"]) / "predictions.jsonl")
    common_ids = sorted(set(zero) & set(tuned), key=sample_id_sort_key)
    diffs = [float(tuned[sample_id]["any_match"]) - float(zero[sample_id]["any_match"]) for sample_id in common_ids]

    rng = random.Random(7)
    boot: list[float] = []
    n = len(diffs)
    for _ in range(10000):
        boot.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    boot.sort()

    zero_correct_tuned_wrong = sum(
        1
        for sample_id in common_ids
        if float(zero[sample_id]["any_match"]) == 1.0 and float(tuned[sample_id]["any_match"]) == 0.0
    )
    zero_wrong_tuned_correct = sum(
        1
        for sample_id in common_ids
        if float(zero[sample_id]["any_match"]) == 0.0 and float(tuned[sample_id]["any_match"]) == 1.0
    )
    discordant = zero_correct_tuned_wrong + zero_wrong_tuned_correct
    smaller = min(zero_correct_tuned_wrong, zero_wrong_tuned_correct)
    mcnemar_p = min(
        1.0,
        2.0
        * sum(
            math.comb(discordant, idx) * (0.5**discordant)
            for idx in range(smaller + 1)
        ),
    )
    return {
        "n": n,
        "zero_accuracy": finalist.metrics["accuracy"],
        "zero_accuracy_ci95": finalist.metrics["accuracy_ci95"],
        "tuned_accuracy": tuned_val["metrics"]["accuracy"],
        "tuned_accuracy_ci95": tuned_val["metrics"]["accuracy_ci95"],
        "paired_gain": mean(diffs),
        "paired_gain_ci95": {"lower": boot[250], "upper": boot[9750]},
        "mcnemar_zero_correct_tuned_wrong": zero_correct_tuned_wrong,
        "mcnemar_zero_wrong_tuned_correct": zero_wrong_tuned_correct,
        "mcnemar_p": mcnemar_p,
    }


def question_prefix(question: str) -> str:
    tokens = re.findall(r"[A-Za-z]+", question.lower())
    if not tokens:
        return "other"
    prefix = tokens[0]
    if prefix in {"what", "which", "who", "where", "when", "how", "why", "is", "are", "does", "do", "can"}:
        return prefix
    return "other"


def build_screening_heatmap(screening: list[EvalRow]) -> None:
    matrix = []
    for backbone in BACKBONE_ORDER:
        row = []
        for prompt in PROMPT_ORDER:
            metric = next(
                (item.metrics["accuracy"] for item in screening if item.model == backbone and item.prompt == prompt),
                None,
            )
            row.append(metric if metric is not None else float("nan"))
        matrix.append(row)

    fig, ax = plt.subplots(figsize=(10.8, 4.4), constrained_layout=True)
    im = ax.imshow(matrix, cmap="YlGnBu", vmin=0.0, vmax=0.85, aspect="auto")
    ax.set_xticks(range(len(PROMPT_ORDER)))
    ax.set_xticklabels([PROMPT_LABELS[p] for p in PROMPT_ORDER], fontsize=10)
    ax.set_yticks(range(len(BACKBONE_ORDER)))
    ax.set_yticklabels(BACKBONE_ORDER, fontsize=10)
    ax.set_title("Internal-dev screening accuracy across backbones and OCR-sensitive prompts", fontsize=13, pad=10)
    best_value = max(value for row in matrix for value in row if not math.isnan(value))
    for i, backbone in enumerate(BACKBONE_ORDER):
        for j, prompt in enumerate(PROMPT_ORDER):
            value = matrix[i][j]
            text = "--" if math.isnan(value) else f"{100 * value:.1f}"
            dark_cell = not math.isnan(value) and value >= 0.45
            label = ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold" if is_best(value, best_value) else "normal",
                color="white" if dark_cell else "black",
            )
            if dark_cell:
                label.set_path_effects([path_effects.withStroke(linewidth=1.2, foreground="#172033")])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.ax.set_ylabel("Accuracy (%)", rotation=90)
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_xlabel("Prompt/OCR setting", fontsize=10)
    FIGURES_DIR.joinpath("screening_heatmap.png")
    fig.savefig(FIGURES_DIR / "screening_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_adaptation_summary(finalists: list[EvalRow], trained: list[dict[str, object]]) -> None:
    finalist = next(
        row for row in finalists if row.model == "Qwen2.5-VL-3B" and row.prompt == "short_answer"
    )
    tuned_val = next(row for row in trained if row["slug"] == "all-linear-r16-seed13" and row["split"] == "validation")
    tuned_internal = [row for row in trained if row["split"] == "internal-dev"]

    fig = plt.figure(figsize=(14.8, 5.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.65, 1.28])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel A: headline validation gain.
    labels = ["Zero-shot\nQwen", "Tuned\nLoRA"]
    values = [finalist.metrics["accuracy"], tuned_val["metrics"]["accuracy"]]
    ci_lowers = [
        finalist.metrics["accuracy"] - finalist.metrics["accuracy_ci95"]["lower"],
        tuned_val["metrics"]["accuracy"] - tuned_val["metrics"]["accuracy_ci95"]["lower"],
    ]
    ci_uppers = [
        finalist.metrics["accuracy_ci95"]["upper"] - finalist.metrics["accuracy"],
        tuned_val["metrics"]["accuracy_ci95"]["upper"] - tuned_val["metrics"]["accuracy"],
    ]
    ax0.bar(
        labels,
        values,
        yerr=[ci_lowers, ci_uppers],
        color=[COLOR_BLUE, COLOR_PURPLE],
        alpha=0.92,
        capsize=6,
    )
    ax0.set_ylim(0.72, 0.88)
    ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax0.set_title("Validation headline", fontsize=12)
    gain = (tuned_val["metrics"]["accuracy"] - finalist.metrics["accuracy"]) * 100
    ax0.text(0.5, 0.865, f"+{gain:.2f} points", ha="center", va="top", fontsize=11, fontweight="bold")
    for idx, value in enumerate(values):
        ax0.text(idx, value + 0.004, f"{100 * value:.2f}", ha="center", fontsize=9)
    ax0.tick_params(axis="x", labelsize=9)

    # Panel B: internal-dev trained adapters.
    display_rows = sorted(tuned_internal, key=lambda row: row["metrics"]["accuracy"], reverse=True)
    y_positions = list(range(len(display_rows)))
    colors = []
    for row in display_rows:
        group = trained_group(row["slug"])
        colors.append(
            COLOR_GREEN if group == "Core matrix" else COLOR_ORANGE if group == "Data scaling" else "#8a5a44"
        )
    accuracies = [row["metrics"]["accuracy"] for row in display_rows]
    ax1.barh(y_positions, accuracies, color=colors, alpha=0.9, height=0.58)
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([trained_short_label(row["slug"]) for row in display_rows], fontsize=8.6)
    ax1.set_xlim(0.86, 0.891)
    ax1.invert_yaxis()
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_title("Internal-dev tuned-adapter accuracy", fontsize=12)
    ax1.grid(axis="x", color="#e6e6e6", linewidth=0.8)
    for y_pos, value in zip(y_positions, accuracies):
        ax1.text(value + 0.0004, y_pos, f"{100 * value:.2f}", va="center", fontsize=7.7)

    # Panel C: answer_in_ocr comparison as grouped horizontal bars.
    finalist_breakdown = load_json(finalist.path / "breakdowns.json")["answer_in_ocr"]
    tuned_breakdown = load_json(tuned_val["path"] / "breakdowns.json")["answer_in_ocr"]
    bucket_order = ["absent", "direct", "normalized_only"]
    bucket_labels = ["Absent", "Direct match", "Norm. match"]
    finalist_values = [finalist_breakdown[key]["accuracy"] for key in bucket_order]
    tuned_values = [tuned_breakdown[key]["accuracy"] for key in bucket_order]
    y_positions = [2, 1, 0]
    bar_height = 0.32
    ax2.barh(
        [y + bar_height / 2 for y in y_positions],
        finalist_values,
        height=bar_height,
        color=COLOR_BLUE,
        alpha=0.92,
        label="Zero-shot finalist",
        zorder=2,
    )
    ax2.barh(
        [y - bar_height / 2 for y in y_positions],
        tuned_values,
        height=bar_height,
        color=COLOR_PURPLE,
        alpha=0.92,
        label="Tuned winner",
        zorder=2,
    )
    ax2.set_xlim(0.68, 0.97)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(bucket_labels, fontsize=9)
    ax2.tick_params(axis="y", pad=4)
    ax2.legend(loc="upper right", frameon=False, fontsize=8)
    ax2.set_title("Validation by OCR bucket", fontsize=12)
    ax2.grid(axis="x", color="#ededed", linewidth=0.8, zorder=1)
    for y_pos, f_value, t_value in zip(y_positions, finalist_values, tuned_values):
        ax2.text(f_value + 0.003, y_pos + bar_height / 2, f"{100 * f_value:.1f}", va="center", fontsize=7.5)
        ax2.text(t_value + 0.003, y_pos - bar_height / 2, f"{100 * t_value:.1f}", va="center", fontsize=7.5)

    fig.suptitle(
        "Adaptation summary: tuned Qwen improves both headline validation accuracy and OCR-grounded answer recovery",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.13, wspace=0.56)
    fig.savefig(FIGURES_DIR / "adaptation_summary.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def ocr_bucket(count: int) -> str:
    if count <= 0:
        return "0"
    if count <= 5:
        return "1-5"
    if count <= 15:
        return "6-15"
    if count <= 30:
        return "16-30"
    return "31+"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def build_qualitative_figure() -> None:
    zero_path = FINALISTS_DIR / "qwen2-5-vl-3b-instruct-short-answer-validation-mps-tuned-v1" / "predictions.jsonl"
    tuned_path = TRAINED_EVAL_DIR / "qwen2-5-vl-3b-instruct-all-linear-r16-seed13-train-speed-v3-validation-cuda-runpod-v1" / "predictions.jsonl"

    manifest = {}
    with VAL_MANIFEST.open() as fh:
        for line in fh:
            rec = json.loads(line)
            manifest[rec["sample_id"]] = rec

    zero, tuned = {}, {}
    for path, target in ((zero_path, zero), (tuned_path, tuned)):
        with path.open() as fh:
            for line in fh:
                rec = json.loads(line)
                target[rec["sample_id"]] = rec

    candidates_by_bucket: dict[str, list[dict[str, object]]] = defaultdict(list)
    for sample_id, tuned_rec in tuned.items():
        zero_rec = zero.get(sample_id)
        base_rec = manifest.get(sample_id)
        if zero_rec is None or base_rec is None:
            continue
        if tuned_rec.get("any_match") != 1.0 or zero_rec.get("any_match") == 1.0:
            continue
        bucket = ocr_bucket(int((tuned_rec.get("metadata") or {}).get("ocr_token_count", 0)))
        candidates_by_bucket[bucket].append(
            {
                "sample_id": sample_id,
                "question": base_rec["question"],
                "answers": base_rec["answers"],
                "image": REPO_ROOT / base_rec["image"],
                "zero_prediction": zero_rec.get("prediction", ""),
                "tuned_prediction": tuned_rec.get("prediction", ""),
                "bucket": bucket,
            }
        )

    selected = []
    for bucket in ["1-5", "6-15", "16-30", "31+"]:
        bucket_items = sorted(
            candidates_by_bucket.get(bucket, []),
            key=lambda rec: (len(rec["question"]), len(rec["tuned_prediction"]), rec["sample_id"]),
        )
        if bucket_items:
            selected.append(bucket_items[0])
    selected = selected[:4]

    if len(selected) < 4:
        raise RuntimeError("Unable to find four qualitative improvement cases.")

    fig = plt.figure(figsize=(12.6, 6.9))
    subfigs = fig.subfigures(2, 2, wspace=0.03, hspace=0.08)
    for subfig, rec in zip(subfigs.flat, selected):
        ax_img, ax_txt = subfig.subplots(1, 2, width_ratios=[1.0, 1.3])
        image = Image.open(rec["image"]).convert("RGB")
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title(f"OCR bucket {rec['bucket']}", fontsize=11, loc="left", color=COLOR_GRAY)

        ax_txt.axis("off")
        ax_txt.text(0.0, 0.98, wrap(rec["question"], 34), fontsize=11, fontweight="bold", va="top")
        ax_txt.text(0.0, 0.73, f"Ground truth: {rec['answers'][0]}", fontsize=10)
        zero_box = patches.FancyBboxPatch(
            (0.0, 0.43),
            0.96,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#fdecea",
            edgecolor="#c62828",
            linewidth=1.0,
        )
        tuned_box = patches.FancyBboxPatch(
            (0.0, 0.16),
            0.96,
            0.18,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            facecolor="#e8f5e9",
            edgecolor="#2e7d32",
            linewidth=1.0,
        )
        ax_txt.add_patch(zero_box)
        ax_txt.add_patch(tuned_box)
        ax_txt.text(0.03, 0.56, "Zero-shot finalist", fontsize=9, color="#8d1d1d", fontweight="bold")
        ax_txt.text(0.03, 0.47, wrap(str(rec["zero_prediction"]), 28), fontsize=10)
        ax_txt.text(0.03, 0.29, "Tuned winner", fontsize=9, color="#1b5e20", fontweight="bold")
        ax_txt.text(0.03, 0.20, wrap(str(rec["tuned_prediction"]), 28), fontsize=10)

    fig.suptitle(
        "Representative validation cases where LoRA fine-tuning corrects zero-shot Qwen failures",
        fontsize=13,
        y=0.99,
    )
    fig.savefig(FIGURES_DIR / "qualitative_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_failure_figure() -> None:
    zero_path = (
        FINALISTS_DIR
        / "qwen2-5-vl-3b-instruct-short-answer-validation-mps-tuned-v1"
        / "predictions.jsonl"
    )
    tuned_path = (
        TRAINED_EVAL_DIR
        / "qwen2-5-vl-3b-instruct-all-linear-r16-seed13-train-speed-v3-validation-cuda-runpod-v1"
        / "predictions.jsonl"
    )

    manifest = load_prediction_map(VAL_MANIFEST)
    zero = load_prediction_map(zero_path)
    tuned = load_prediction_map(tuned_path)

    case_specs = [
        ("34608", "Visual time reading"),
        ("34718", "OCR ignored"),
        ("34963", "Partial long answer"),
        ("35120", "Wrong visual target"),
    ]
    selected = []
    for sample_id, failure_mode in case_specs:
        base_rec = manifest[sample_id]
        zero_rec = zero[sample_id]
        tuned_rec = tuned[sample_id]
        ocr_tokens = base_rec.get("ocr_tokens") or []
        selected.append(
            {
                "sample_id": sample_id,
                "mode": failure_mode,
                "question": base_rec["question"],
                "answers": base_rec["answers"],
                "image": REPO_ROOT / base_rec["image"],
                "ocr": ", ".join(str(token) for token in ocr_tokens[:10]) or "N/A",
                "zero_prediction": zero_rec.get("prediction", ""),
                "tuned_prediction": tuned_rec.get("prediction", ""),
            }
        )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(9.6, 7.3),
        gridspec_kw={"width_ratios": [0.9, 1.25, 0.9, 1.25]},
    )
    for index, rec in enumerate(selected):
        row = index // 2
        col = (index % 2) * 2
        ax_img = axes[row, col]
        ax_txt = axes[row, col + 1]
        image = Image.open(rec["image"]).convert("RGB")
        ax_img.imshow(image)
        ax_img.axis("off")
        ax_img.set_title(str(rec["mode"]), fontsize=11.5, loc="left", color=COLOR_GRAY)

        ax_txt.axis("off")
        ax_txt.text(
            0.0,
            0.98,
            wrap(str(rec["question"]), 28),
            fontsize=10.4,
            fontweight="bold",
            va="top",
        )
        ax_txt.text(
            0.0,
            0.72,
            f"Gold: {wrap(str(rec['answers'][0]), 25)}",
            fontsize=9.5,
            va="top",
        )
        ax_txt.text(
            0.0,
            0.56,
            f"OCR: {wrap(str(rec['ocr']), 30)}",
            fontsize=8.8,
            color=COLOR_GRAY,
            va="top",
        )
        ax_txt.text(
            0.0,
            0.29,
            f"Zero-shot: {wrap(str(rec['zero_prediction']), 28)}",
            fontsize=9.3,
            va="top",
            color="#7a1f1f",
        )
        ax_txt.text(
            0.0,
            0.13,
            f"Tuned: {wrap(str(rec['tuned_prediction']), 28)}",
            fontsize=9.3,
            va="top",
            color="#4a148c",
        )

    fig.suptitle(
        "Residual validation failures after LoRA adaptation",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.995,
        top=0.87,
        bottom=0.04,
        wspace=0.26,
        hspace=0.31,
    )
    fig.savefig(FIGURES_DIR / "failure_examples.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_split_audit_table() -> None:
    internal_dev = load_jsonl(INTERNAL_DEV_MANIFEST)
    train_remainder = load_jsonl(TRAIN_REMAINDER_MANIFEST)
    validation = load_jsonl(VAL_MANIFEST)

    train_images = {image_id(record) for record in train_remainder if image_id(record)}
    internal_images = {image_id(record) for record in internal_dev if image_id(record)}
    overlapping_internal = [
        record for record in internal_dev if image_id(record) in train_images
    ]
    nonoverlapping_internal = [
        record for record in internal_dev if image_id(record) not in train_images
    ]

    rows = [
        {
            "pool": "Internal-dev",
            "questions": len(internal_dev),
            "images": len(internal_images),
            "image_relation": f"{len(nonoverlapping_internal)} image-disjoint QA; {len(overlapping_internal)} image-overlap QA",
            "role": "Screening/diagnostics",
        },
        {
            "pool": "Train remainder",
            "questions": len(train_remainder),
            "images": len(train_images),
            "image_relation": "Question-disjoint from internal-dev",
            "role": "LoRA training pool",
        },
        {
            "pool": "Official validation",
            "questions": len(validation),
            "images": len({image_id(record) for record in validation if image_id(record)}),
            "image_relation": "Official held-out split",
            "role": "Final paper-facing evaluation",
        },
    ]
    write_csv(
        DATA_DIR / "split_audit.csv",
        rows,
        ["pool", "questions", "images", "image_relation", "role"],
    )
    with (TABLES_DIR / "split_audit_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{lrrp{0.31\\linewidth}p{0.22\\linewidth}}\n")
        fh.write("\\toprule\n")
        fh.write("Pool & QA & Images & Image relation & Role\\\\\n")
        fh.write("\\midrule\n")
        for row in rows:
            fh.write(
                f"{latex_escape(row['pool'])} & {row['questions']} & {row['images']} & "
                f"{latex_escape(row['image_relation'])} & {latex_escape(row['role'])}\\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")


def build_validation_support_tables(finalist: EvalRow, tuned_val: dict[str, object]) -> None:
    stats = paired_validation_statistics(finalist, tuned_val)
    write_text(DATA_DIR / "validation_significance.json", json.dumps(stats, indent=2))

    finalist_judge = maybe_load_judge_similarity(finalist.path)
    tuned_judge = maybe_load_judge_similarity(Path(tuned_val["path"]))
    best_accuracy = max(float(finalist.metrics["accuracy"]), float(tuned_val["metrics"]["accuracy"]))
    best_consensus = max(float(finalist.metrics["consensus_accuracy"]), float(tuned_val["metrics"]["consensus_accuracy"]))
    best_judge = max(value for value in (finalist_judge, tuned_judge) if value is not None)
    with (TABLES_DIR / "validation_headline_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{lccc}\n")
        fh.write("\\toprule\n")
        fh.write("System & Acc. [95\\% CI] & Cons. & Judge\\\\\n")
        fh.write("\\midrule\n")
        fh.write(
            "Zero-shot Qwen finalist & "
            f"{percentage_ci_best(finalist.metrics['accuracy'], finalist.metrics['accuracy_ci95'], best_accuracy)} & "
            f"{percentage_best(finalist.metrics['consensus_accuracy'], best_consensus)} & "
            f"{percentage_best(finalist_judge, best_judge)}\\\\\n"
        )
        fh.write(
            "Tuned Qwen LoRA & "
            f"{percentage_ci_best(tuned_val['metrics']['accuracy'], tuned_val['metrics']['accuracy_ci95'], best_accuracy)} & "
            f"{percentage_best(tuned_val['metrics']['consensus_accuracy'], best_consensus)} & "
            f"{percentage_best(tuned_judge, best_judge)}\\\\\n"
        )
        fh.write("\\midrule\n")
        gain_ci = stats["paired_gain_ci95"]
        fh.write(
            "Paired gain & "
            f"{percentage(stats['paired_gain'])} [{percentage(gain_ci['lower'])}, {percentage(gain_ci['upper'])}] & "
            "-- & --\\\\\n"
        )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")

    zero_predictions = load_prediction_map(finalist.path / "predictions.jsonl")
    tuned_predictions = load_prediction_map(Path(tuned_val["path"]) / "predictions.jsonl")
    common_ids = sorted(set(zero_predictions) & set(tuned_predictions), key=sample_id_sort_key)
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for sample_id in common_ids:
        prefix = question_prefix(str(zero_predictions[sample_id].get("question", "")))
        grouped[prefix].append(
            (
                float(zero_predictions[sample_id].get("any_match", 0.0)),
                float(tuned_predictions[sample_id].get("any_match", 0.0)),
            )
        )

    rows = []
    for prefix, values in grouped.items():
        if len(values) < 50:
            continue
        zero_acc = mean(pair[0] for pair in values)
        tuned_acc = mean(pair[1] for pair in values)
        rows.append(
            {
                "prefix": prefix,
                "count": len(values),
                "zero_accuracy": zero_acc,
                "tuned_accuracy": tuned_acc,
                "gain": tuned_acc - zero_acc,
            }
        )
    rows.sort(key=lambda row: row["count"], reverse=True)
    write_csv(
        DATA_DIR / "question_prefix_results.csv",
        rows,
        ["prefix", "count", "zero_accuracy", "tuned_accuracy", "gain"],
    )
    with (TABLES_DIR / "question_prefix_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{lrrrr}\n")
        fh.write("\\toprule\n")
        fh.write("Question prefix & Count & Zero-shot & Tuned & Gain\\\\\n")
        fh.write("\\midrule\n")
        best_gain = best_max(rows, "gain")
        for row in rows:
            best_row_accuracy = max(float(row["zero_accuracy"]), float(row["tuned_accuracy"]))
            fh.write(
                f"{latex_escape(row['prefix'])} & {row['count']} & {percentage_best(row['zero_accuracy'], best_row_accuracy)} & "
                f"{percentage_best(row['tuned_accuracy'], best_row_accuracy)} & {percentage_best(row['gain'], best_gain)}\\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")


def build_tables(
    screening: list[EvalRow],
    finalists: list[EvalRow],
    appendix: list[EvalRow],
    trained: list[dict[str, object]],
) -> None:
    screening_best = []
    for backbone in BACKBONE_ORDER:
        candidates = [row for row in screening if row.model == backbone]
        if not candidates:
            continue
        best = max(candidates, key=lambda row: row.metrics["accuracy"])
        screening_best.append(best)

    finalists_sorted = sorted(finalists, key=lambda row: row.metrics["accuracy"], reverse=True)
    strongest_finalist = next(
        row
        for row in finalists_sorted
        if row.model == "Qwen2.5-VL-3B" and row.prompt == "short_answer"
    )
    tuned_val = next(row for row in trained if row["slug"] == "all-linear-r16-seed13" and row["split"] == "validation")
    build_split_audit_table()
    build_validation_support_tables(strongest_finalist, tuned_val)

    main_rows = []
    for row in screening_best:
        meteor = maybe_compute_meteor(row.path / "predictions.jsonl")
        main_rows.append(
            {
                "stage": "Best screening",
                "model": row.model,
                "setting": row.prompt.replace("_", " "),
                "split": row.split,
                "accuracy": row.metrics["accuracy"],
                "consensus_accuracy": row.metrics["consensus_accuracy"],
                "f1": row.metrics["f1"],
                "bleu": row.metrics["bleu"],
                "meteor": meteor,
                "rouge_l": row.metrics["rouge_l"],
                "judge": maybe_load_judge_similarity(row.path),
            }
        )
    for row in finalists_sorted:
        meteor = maybe_compute_meteor(row.path / "predictions.jsonl")
        main_rows.append(
            {
                "stage": "Finalist",
                "model": row.model,
                "setting": row.prompt.replace("_", " "),
                "split": row.split,
                "accuracy": row.metrics["accuracy"],
                "consensus_accuracy": row.metrics["consensus_accuracy"],
                "f1": row.metrics["f1"],
                "bleu": row.metrics["bleu"],
                "meteor": meteor,
                "rouge_l": row.metrics["rouge_l"],
                "judge": maybe_load_judge_similarity(row.path),
            }
        )
    main_rows.append(
        {
            "stage": "Tuned winner",
            "model": "Qwen2.5-VL-3B + LoRA",
            "setting": "all-linear r16 seed13",
            "split": "validation",
            "accuracy": tuned_val["metrics"]["accuracy"],
            "consensus_accuracy": tuned_val["metrics"]["consensus_accuracy"],
            "f1": tuned_val["metrics"]["f1"],
            "bleu": tuned_val["metrics"]["bleu"],
            "meteor": maybe_compute_meteor(tuned_val["path"] / "predictions.jsonl"),
            "rouge_l": tuned_val["metrics"]["rouge_l"],
            "judge": maybe_load_judge_similarity(Path(tuned_val["path"])),
        }
    )
    write_csv(
        DATA_DIR / "main_results.csv",
        main_rows,
        ["stage", "model", "setting", "split", "accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l", "judge"],
    )

    main_validation_rows = [row for row in main_rows if row["split"] == "validation"]
    main_bests = {
        key: best_max(main_validation_rows, key)
        for key in ("accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l", "judge")
    }
    with (TABLES_DIR / "main_results_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{llccccccc}\n")
        fh.write("\\toprule\n")
        fh.write("Stage & Model / setting & Acc. & Cons. & F1 & BLEU & METEOR & ROUGE-L & Judge\\\\\n")
        fh.write("\\midrule\n")
        for row in main_rows:
            label = f"{row['model']} ({row['setting']})"
            comparable_best = main_bests if row["split"] == "validation" else {}
            fh.write(
                f"{latex_escape(row['stage'])} & {latex_escape(label)} & {percentage_best(row['accuracy'], comparable_best.get('accuracy'))} & "
                f"{percentage_best(row['consensus_accuracy'], comparable_best.get('consensus_accuracy'))} & {percentage_best(row['f1'], comparable_best.get('f1'))} & "
                f"{percentage_best(row['bleu'], comparable_best.get('bleu'))} & {percentage_best(row['meteor'], comparable_best.get('meteor'))} & "
                f"{percentage_best(row['rouge_l'], comparable_best.get('rouge_l'))} & {percentage_best(row['judge'], comparable_best.get('judge'))}\\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")

    trained_rows_csv = []
    trained_internal = sorted((row for row in trained if row["split"] == "internal-dev"), key=lambda row: row["metrics"]["accuracy"], reverse=True)
    for row in trained_internal:
        trained_rows_csv.append(
            {
                "group": trained_group(row["slug"]),
                "configuration": trained_label(row["slug"]),
                "eval_loss": row["eval_loss"],
                "accuracy": row["metrics"]["accuracy"],
                "consensus_accuracy": row["metrics"]["consensus_accuracy"],
                "f1": row["metrics"]["f1"],
                "bleu": row["metrics"]["bleu"],
                "meteor": maybe_compute_meteor(row["path"] / "predictions.jsonl"),
                "rouge_l": row["metrics"]["rouge_l"],
                "judge": maybe_load_judge_similarity(Path(row["path"])),
            }
        )
    write_csv(
        DATA_DIR / "trained_adapter_results.csv",
        trained_rows_csv,
        ["group", "configuration", "eval_loss", "accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l", "judge"],
    )
    trained_bests = {
        "eval_loss": best_min(trained_rows_csv, "eval_loss"),
        **{
            key: best_max(trained_rows_csv, key)
            for key in ("accuracy", "consensus_accuracy", "f1", "meteor", "rouge_l", "judge")
        },
    }
    with (TABLES_DIR / "trained_adapter_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{llccccccc}\n")
        fh.write("\\toprule\n")
        fh.write("Group & Configuration & Eval loss & Acc. & Cons. & F1 & METEOR & ROUGE-L & Judge\\\\\n")
        fh.write("\\midrule\n")
        for row in trained_rows_csv:
            fh.write(
                f"{latex_escape(row['group'])} & {latex_escape(row['configuration'])} & {decimal_best(row['eval_loss'], trained_bests['eval_loss'])} & "
                f"{percentage_best(row['accuracy'], trained_bests['accuracy'])} & {percentage_best(row['consensus_accuracy'], trained_bests['consensus_accuracy'])} & "
                f"{percentage_best(row['f1'], trained_bests['f1'])} & {percentage_best(row['meteor'], trained_bests['meteor'])} & "
                f"{percentage_best(row['rouge_l'], trained_bests['rouge_l'])} & {percentage_best(row['judge'], trained_bests['judge'])}\\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")

    appendix_rows_csv = []
    for row in sorted(appendix, key=lambda item: item.metrics["accuracy"], reverse=True):
        appendix_rows_csv.append(
            {
                "branch": "Prompt study" if row.stage == "appendix-prompt" else "Stress test",
                "setting": row.prompt.replace("_", " "),
                "accuracy": row.metrics["accuracy"],
                "consensus_accuracy": row.metrics["consensus_accuracy"],
                "f1": row.metrics["f1"],
                "bleu": row.metrics["bleu"],
                "meteor": maybe_compute_meteor(row.path / "predictions.jsonl"),
                "rouge_l": row.metrics["rouge_l"],
                "judge": maybe_load_judge_similarity(row.path),
            }
        )
    write_csv(
        DATA_DIR / "appendix_results.csv",
        appendix_rows_csv,
        ["branch", "setting", "accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l", "judge"],
    )
    appendix_bests = {
        key: best_max(appendix_rows_csv, key)
        for key in ("accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l", "judge")
    }
    with (TABLES_DIR / "appendix_results_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{llccccccc}\n")
        fh.write("\\toprule\n")
        fh.write("Branch & Setting & Acc. & Cons. & F1 & BLEU & METEOR & ROUGE-L & Judge\\\\\n")
        fh.write("\\midrule\n")
        for row in appendix_rows_csv:
            fh.write(
                f"{latex_escape(row['branch'])} & {latex_escape(row['setting'])} & {percentage_best(row['accuracy'], appendix_bests['accuracy'])} & "
                f"{percentage_best(row['consensus_accuracy'], appendix_bests['consensus_accuracy'])} & {percentage_best(row['f1'], appendix_bests['f1'])} & "
                f"{percentage_best(row['bleu'], appendix_bests['bleu'])} & {percentage_best(row['meteor'], appendix_bests['meteor'])} & "
                f"{percentage_best(row['rouge_l'], appendix_bests['rouge_l'])} & {percentage_best(row['judge'], appendix_bests['judge'])}\\\\\n"
            )
        fh.write("\\bottomrule\n")
        fh.write("\\end{tabular}\n")


def main() -> None:
    ensure_dirs()
    screening = canonical_screening_rows()
    finalists = finalist_rows()
    appendix = appendix_rows()
    trained = trained_rows()

    build_screening_heatmap(screening)
    build_adaptation_summary(finalists, trained)
    build_qualitative_figure()
    build_failure_figure()
    build_tables(screening, finalists, appendix, trained)


if __name__ == "__main__":
    main()
