from __future__ import annotations

import csv
import json
import math
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
        "best-assumed-25pct": "Data scaling: 25 pct",
        "best-assumed-full": "Data scaling: full",
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
        import nltk
        from nltk.translate.meteor_score import meteor_score

        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
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


def percentage(value: float | None) -> str:
    return "--" if value is None else f"{100 * value:.2f}"


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n")


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
    for i, backbone in enumerate(BACKBONE_ORDER):
        for j, prompt in enumerate(PROMPT_ORDER):
            value = matrix[i][j]
            text = "--" if math.isnan(value) else f"{100 * value:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=9, color="black")
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

    fig = plt.figure(figsize=(13.2, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.45, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    # Panel A: headline validation gain.
    labels = ["Zero-shot finalist\n(Qwen short-answer)", "Tuned winner\n(All-linear r16, seed 13)"]
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

    # Panel B: internal-dev trained adapters.
    display_rows = sorted(tuned_internal, key=lambda row: row["metrics"]["accuracy"])
    y_positions = list(range(len(display_rows)))
    colors = []
    for row in display_rows:
        group = trained_group(row["slug"])
        colors.append(
            COLOR_GREEN if group == "Core matrix" else COLOR_ORANGE if group == "Data scaling" else "#8a5a44"
        )
    ax1.hlines(y_positions, xmin=0.86, xmax=[row["metrics"]["accuracy"] for row in display_rows], color="#d9d9d9", linewidth=1.2)
    ax1.scatter(
        [row["metrics"]["accuracy"] for row in display_rows],
        y_positions,
        c=colors,
        s=64,
        zorder=3,
    )
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels([trained_label(row["slug"]) for row in display_rows], fontsize=9)
    ax1.set_xlim(0.86, 0.89)
    ax1.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_title("Internal-dev tuned-adapter accuracy", fontsize=12)
    ax1.grid(axis="x", color="#e6e6e6", linewidth=0.8)

    # Panel C: answer_in_ocr comparison as a dumbbell chart.
    finalist_breakdown = load_json(finalist.path / "breakdowns.json")["answer_in_ocr"]
    tuned_breakdown = load_json(tuned_val["path"] / "breakdowns.json")["answer_in_ocr"]
    bucket_order = ["absent", "direct", "normalized_only"]
    bucket_labels = ["Absent from OCR", "Direct OCR match", "Normalized OCR match"]
    finalist_values = [finalist_breakdown[key]["accuracy"] for key in bucket_order]
    tuned_values = [tuned_breakdown[key]["accuracy"] for key in bucket_order]
    y_positions = [2, 1, 0]
    for y_pos, label, f_value, t_value in zip(y_positions, bucket_labels, finalist_values, tuned_values):
        ax2.hlines(y=y_pos, xmin=f_value, xmax=t_value, color="#d6c7f4", linewidth=3)
        ax2.scatter(f_value, y_pos, color=COLOR_BLUE, s=55, zorder=3)
        ax2.scatter(t_value, y_pos, color=COLOR_PURPLE, s=55, zorder=3)
        ax2.text(f_value - 0.004, y_pos + 0.08, f"{100 * f_value:.1f}", ha="right", va="bottom", fontsize=8)
        ax2.text(t_value + 0.004, y_pos + 0.08, f"{100 * t_value:.1f}", ha="left", va="bottom", fontsize=8)
    ax2.set_xlim(0.70, 0.95)
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(bucket_labels, fontsize=9)
    ax2.legend(
        handles=[
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_BLUE, markersize=8, label="Zero-shot finalist"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PURPLE, markersize=8, label="Tuned winner"),
        ],
        loc="lower right",
        frameon=False,
        fontsize=8,
    )
    ax2.set_title("Validation accuracy by OCR answer bucket", fontsize=12)
    ax2.grid(axis="x", color="#ededed", linewidth=0.8)

    fig.suptitle(
        "Adaptation summary: tuned Qwen improves both headline validation accuracy and OCR-grounded answer recovery",
        fontsize=13,
        y=0.98,
    )
    fig.subplots_adjust(left=0.05, right=0.995, top=0.86, bottom=0.12, wspace=0.32)
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
    tuned_val = next(row for row in trained if row["slug"] == "all-linear-r16-seed13" and row["split"] == "validation")

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
        }
    )
    write_csv(
        DATA_DIR / "main_results.csv",
        main_rows,
        ["stage", "model", "setting", "split", "accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l"],
    )

    with (TABLES_DIR / "main_results_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{llcccccc}\n")
        fh.write("\\toprule\n")
        fh.write("Stage & Model / setting & Acc. & Cons. & F1 & BLEU & METEOR & ROUGE-L\\\\\n")
        fh.write("\\midrule\n")
        for row in main_rows:
            label = f"{row['model']} ({row['setting']})"
            fh.write(
                f"{row['stage']} & {label} & {percentage(row['accuracy'])} & {percentage(row['consensus_accuracy'])} & "
                f"{percentage(row['f1'])} & {percentage(row['bleu'])} & {percentage(row['meteor'])} & {percentage(row['rouge_l'])}\\\\\n"
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
            }
        )
    write_csv(
        DATA_DIR / "trained_adapter_results.csv",
        trained_rows_csv,
        ["group", "configuration", "eval_loss", "accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l"],
    )
    with (TABLES_DIR / "trained_adapter_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{llcccccc}\n")
        fh.write("\\toprule\n")
        fh.write("Group & Configuration & Eval loss & Acc. & Cons. & F1 & METEOR & ROUGE-L\\\\\n")
        fh.write("\\midrule\n")
        for row in trained_rows_csv:
            fh.write(
                f"{row['group']} & {row['configuration']} & {row['eval_loss']:.4f} & {percentage(row['accuracy'])} & "
                f"{percentage(row['consensus_accuracy'])} & {percentage(row['f1'])} & {percentage(row['meteor'])} & {percentage(row['rouge_l'])}\\\\\n"
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
            }
        )
    write_csv(
        DATA_DIR / "appendix_results.csv",
        appendix_rows_csv,
        ["branch", "setting", "accuracy", "consensus_accuracy", "f1", "bleu", "meteor", "rouge_l"],
    )
    with (TABLES_DIR / "appendix_results_table.tex").open("w") as fh:
        fh.write("\\begin{tabular}{llcccccc}\n")
        fh.write("\\toprule\n")
        fh.write("Branch & Setting & Acc. & Cons. & F1 & BLEU & METEOR & ROUGE-L\\\\\n")
        fh.write("\\midrule\n")
        for row in appendix_rows_csv:
            fh.write(
                f"{row['branch']} & {row['setting']} & {percentage(row['accuracy'])} & {percentage(row['consensus_accuracy'])} & "
                f"{percentage(row['f1'])} & {percentage(row['bleu'])} & {percentage(row['meteor'])} & {percentage(row['rouge_l'])}\\\\\n"
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
    build_tables(screening, finalists, appendix, trained)


if __name__ == "__main__":
    main()
