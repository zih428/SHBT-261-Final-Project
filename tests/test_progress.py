from __future__ import annotations

from textvqa_proj.progress import render_progress_report


def test_render_progress_report_mentions_stage_counts() -> None:
    summary = {
        "prep": {
            "internal_dev_external_ocr_rows": 2000,
            "validation_external_ocr_rows": 5000,
        },
        "screening": {
            "counts": {
                "completed": 1,
                "running": 1,
                "pending": 22,
                "failed": 0,
                "other": 0,
                "total": 24,
            },
            "active_run": {
                "label": "qwen25_vl_3b x ocr_fused",
                "processed_count": 86,
                "updated_at": "2026-04-17T01:05:50+00:00",
            },
            "best_completed_run": {
                "label": "qwen25_vl_3b x ocr_copy_first",
                "accuracy": 0.708,
            },
        },
        "screening_baseline": {
            "counts": {
                "completed": 0,
                "running": 0,
                "pending": 6,
                "failed": 0,
                "other": 0,
                "total": 6,
            },
        },
        "finalists": {"status": "blocked until screening completes", "planned_runs": 8},
        "training": {
            "counts": {
                "completed": 0,
                "running": 0,
                "pending": 12,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "blocked until screening/finalist selection completes",
        },
        "appendix": {
            "counts": {
                "completed": 0,
                "running": 0,
                "pending": 8,
                "failed": 0,
                "other": 0,
                "total": 8,
            },
            "status": "blocked until winner backbone is selected",
        },
    }

    report = render_progress_report(summary)

    assert "TextVQA Progress" in report
    assert "1 completed, 1 running, 22 pending" in report
    assert "qwen25_vl_3b x ocr_fused" in report
    assert "accuracy 0.708" in report
