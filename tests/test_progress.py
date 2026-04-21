from __future__ import annotations

from textvqa_proj.progress import render_progress_report, render_training_report


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
                "running": 1,
                "pending": 11,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "active_run": {
                "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                "current_step": 128,
                "max_steps": 1024,
                "checkpoint_step": 0,
                "updated_at": "2026-04-20T18:00:00+00:00",
                "eta_at": "2026-04-20T20:00:00+00:00",
            },
            "runs": [
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                    "status": "running",
                    "current_step": 128,
                    "max_steps": 1024,
                    "checkpoint_step": 0,
                    "updated_at": "2026-04-20T18:00:00+00:00",
                    "eta_at": "2026-04-20T20:00:00+00:00",
                },
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed13",
                    "status": "pending",
                },
            ],
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
    assert "Stages" in report
    assert "Screening" in report
    assert "OCR Baselines" in report
    assert "Screening Highlight" in report
    assert "qwen25_vl_3b x ocr_copy_first" in report
    assert "Accuracy" in report
    assert "0.708" in report
    assert "Training Runs" in report
    assert "128/1024" in report
    assert "Updated (ET)" in report
    assert "Projected Start (ET)" in report
    assert "Projected End (ET)" in report
    assert "Apr 20  2:00 PM" in report
    assert "2h 0m" in report
    assert "2026-04-20T18:00:00+00:00" not in report

    training_report = render_training_report(summary)

    assert "TextVQA Training Progress" in training_report
    assert "Summary" in training_report
    assert "All Runs" in training_report
    assert "core_all_linear_r16_seed07" in training_report
    assert "Projected Start (ET)" in training_report
    assert "Projected End (ET)" in training_report
    assert "Apr 20  2:00 PM" in training_report
    assert "2h 0m" in training_report
    assert "Screening" not in training_report


def test_render_training_report_treats_starting_workers_as_running_stage() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 0,
                "running": 1,
                "pending": 11,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "blocked until finalist selection completes",
            "active_run": None,
            "runs": [
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                    "status": "starting",
                    "latest_log": {"gpu_id": "0"},
                    "updated_at": "2026-04-20T22:39:42+00:00",
                },
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed13",
                    "status": "pending",
                },
            ],
        }
    }

    training_report = render_training_report(summary)

    assert "Summary" in training_report
    assert "blocked until finalist selection completes" not in training_report
    assert "Active Runs" not in training_report
    assert "core_all_linear_r16_seed07" in training_report
    assert "starting" in training_report
    assert "Apr 20  6:39 PM" in training_report


def test_render_training_report_lists_multiple_active_runs() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 0,
                "running": 2,
                "pending": 10,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "runs": [
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                    "status": "running",
                    "current_step": 400,
                    "max_steps": 1024,
                    "updated_at": "2026-04-20T23:39:57+00:00",
                    "eta_at": "2026-04-21T00:50:24+00:00",
                },
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed13",
                    "status": "running",
                    "current_step": 375,
                    "max_steps": 1024,
                    "updated_at": "2026-04-20T23:37:56+00:00",
                    "eta_at": "2026-04-21T00:52:34+00:00",
                },
            ],
        }
    }

    training_report = render_training_report(summary)

    assert "Active Runs" not in training_report
    assert "core_all_linear_r16_seed07" in training_report
    assert "core_all_linear_r16_seed13" in training_report


def test_render_training_report_shows_projected_queue_times() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 0,
                "running": 2,
                "pending": 10,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "runs": [
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                    "status": "running",
                    "current_step": 650,
                    "max_steps": 1024,
                    "checkpoint_step": 512,
                    "updated_at": "2026-04-21T00:08:00+00:00",
                    "eta_at": "2026-04-21T00:50:00+00:00",
                    "projected_start_at": "now",
                    "projected_end_at": "2026-04-21T00:50:00+00:00",
                },
                {
                    "label": "qwen25_vl_3b x all-linear-r32-seed07",
                    "status": "pending",
                    "projected_start_at": "2026-04-21T00:50:00+00:00",
                    "projected_end_at": "2026-04-21T02:45:00+00:00",
                },
            ],
        }
    }

    training_report = render_training_report(summary)

    assert "now" in training_report
    assert "Apr 20  8:50 PM" in training_report
    assert "Apr 20 10:45 PM" in training_report
