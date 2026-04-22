from __future__ import annotations

from datetime import datetime
from pathlib import Path

from textvqa_proj.config import Settings
from textvqa_proj.progress import (
    RunProgress,
    _evaluation_progress,
    _project_training_schedule,
    _training_progress,
    render_progress_report,
    render_training_report,
)


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
                "total_count": 2000,
                "updated_at": "2026-04-17T01:05:50+00:00",
                "eta_at": "2026-04-17T02:05:50+00:00",
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
                    "latest_log": {"loss": 0.81234, "grad_norm": 5.6789},
                    "updated_at": "2026-04-20T18:00:00+00:00",
                    "eta_at": "2026-04-20T20:00:00+00:00",
                },
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed13",
                    "status": "pending",
                },
            ],
            "scheduler": {
                "polled_at": "2026-04-21T23:15:00+00:00",
                "remote_git_head": "abc1234",
                "sync_mode": "disabled-basic-ssh",
                "sync_message": "Artifact sync requires full SSH over exposed TCP.",
                "synced_paths": ["outputs/training"],
                "eval_runs": [
                    {
                        "config_name": "core_all_linear_r16_seed07",
                        "split": "internal_dev",
                        "status": "running",
                        "processed_count": 100,
                        "total_count": 2000,
                        "started_at": "2026-04-21T22:45:00+00:00",
                        "updated_at": "2026-04-21T23:15:00+00:00",
                        "resumed_from_count": 0,
                    }
                ],
                "plan": {
                    "post_train_eval_ready": True,
                    "first_eleven_completed": True,
                    "active_evals": [
                        {
                            "config_name": "core_all_linear_r16_seed07",
                            "split": "internal_dev",
                            "gpu_id": "1",
                            "status": "running",
                        }
                    ],
                    "pending_internal_dev_evals": ["core_all_linear_r32_seed07"],
                    "pending_validation_evals": [],
                    "validation_candidate": "core_all_linear_r16_seed07",
                    "gpus": [
                        {
                            "gpu_id": "0",
                            "assignment_kind": "training",
                            "assignment_label": "best-assumed-full",
                            "utilization_gpu": 62,
                            "memory_used": 28000,
                            "memory_total": 81559,
                        },
                        {
                            "gpu_id": "1",
                            "assignment_kind": "eval",
                            "assignment_label": "core_all_linear_r16_seed07 (internal_dev)",
                            "utilization_gpu": 24,
                            "memory_used": 12000,
                            "memory_total": 81559,
                        },
                    ],
                    "actions": [
                        {
                            "kind": "launch-eval",
                            "gpu_id": "1",
                            "label": "core_all_linear_r32_seed07 internal_dev",
                        }
                    ],
                },
            },
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
    assert "Stage Summary" in report
    assert "Screening" in report
    assert "OCR Baselines" in report
    assert "real VLMs" in report
    assert "Best Screening Run" in report
    assert "qwen25_vl_3b x ocr_copy_first" in report
    assert "Accuracy" in report
    assert "0.708" in report
    assert "Active Evaluations" in report
    assert "86/2000" in report
    assert "1h 0m" in report
    assert "Training Runs" in report
    assert "RunPod Scheduler Status" in report
    assert "RunPod GPU Work" in report
    assert "Post-Train Eval Queue" in report
    assert "Artifact sync" in report
    assert "disabled-basic-ssh" in report
    assert "best-assumed-full" in report
    assert "internal_dev" in report
    assert "Eval queue" in report
    assert "1 running, 1 pending" in report
    assert "100/2000" in report
    assert "128/1024" in report
    assert "Loss" in report
    assert "Grad" in report
    assert "0.8123" in report
    assert "5.679" in report
    assert "Updated (ET)" in report
    assert "Projected Start (ET)" in report
    assert "Projected End (ET)" in report
    assert "Apr 20  2:00 PM" in report
    assert "2h 0m" in report
    assert "2026-04-20T18:00:00+00:00" not in report

    training_report = render_training_report(summary)

    assert "TextVQA Training Progress" in training_report
    assert "Training Overview" in training_report
    assert "Training Runs" in training_report
    assert "RunPod Scheduler Status" in training_report
    assert "RunPod GPU Work" in training_report
    assert "Post-Train Eval Queue" in training_report
    assert "Active training GPUs" in training_report
    assert "GPU" in training_report
    assert "core_all_linear_r16_seed07" in training_report
    assert "Loss" in training_report
    assert "Grad" in training_report
    assert "0.8123" in training_report
    assert "5.679" in training_report
    assert "Projected Start (ET)" in training_report
    assert "Projected End (ET)" in training_report
    assert "Apr 20  2:00 PM" in training_report
    assert "2h 0m" in training_report
    assert "Screening" not in training_report


def test_render_training_report_prefers_live_scheduler_training_snapshot() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 0,
                "running": 0,
                "pending": 12,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "pending",
            "runs": [
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                    "status": "pending",
                }
            ],
            "scheduler": {
                "polled_at": "2026-04-21T23:15:00+00:00",
                "remote_git_head": "abc1234",
                "sync_mode": "disabled-basic-ssh",
                "sync_message": "Artifact sync requires full SSH over exposed TCP.",
                "training": {
                    "counts": {
                        "completed": 4,
                        "running": 2,
                        "pending": 6,
                        "failed": 0,
                        "other": 0,
                        "total": 12,
                    },
                    "status": "running",
                    "runs": [
                        {
                            "config_name": "core_all_linear_r16_seed07",
                            "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                            "status": "completed",
                            "current_step": 1024,
                            "max_steps": 1024,
                            "checkpoint_step": 1024,
                        },
                        {
                            "config_name": "core_attn_r16_seed07",
                            "label": "qwen25_vl_3b x core_attn_r16_seed07",
                            "status": "running",
                            "current_step": 512,
                            "max_steps": 1024,
                        },
                    ],
                },
                "plan": {
                    "post_train_eval_ready": False,
                    "first_eleven_completed": False,
                    "pending_internal_dev_evals": [],
                    "pending_validation_evals": [],
                },
            },
        }
    }

    training_report = render_training_report(summary)

    assert "4" in training_report
    assert "2" in training_report
    assert "core_attn_r16_seed07" in training_report
    assert "512/1024" in training_report


def test_render_training_report_prefers_direct_training_when_newer_than_scheduler() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 4,
                "running": 2,
                "pending": 6,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "runs": [
                {
                    "config_name": "core_attn_r16_seed07",
                    "label": "qwen25_vl_3b x core_attn_r16_seed07",
                    "status": "running",
                    "current_step": 975,
                    "max_steps": 1024,
                    "checkpoint_step": 512,
                    "updated_at": "2026-04-22T02:26:09+00:00",
                },
                {
                    "config_name": "core_attn_r16_seed13",
                    "label": "qwen25_vl_3b x core_attn_r16_seed13",
                    "status": "running",
                    "current_step": 925,
                    "max_steps": 1024,
                    "checkpoint_step": 512,
                    "updated_at": "2026-04-22T02:25:50+00:00",
                },
            ],
            "scheduler": {
                "polled_at": "2026-04-22T01:57:52+00:00",
                "remote_git_head": "b07bd70",
                "training": {
                    "counts": {
                        "completed": 4,
                        "running": 2,
                        "pending": 6,
                        "failed": 0,
                        "other": 0,
                        "total": 12,
                    },
                    "status": "running",
                    "runs": [
                        {
                            "config_name": "core_attn_r16_seed07",
                            "label": "qwen25_vl_3b x core_attn_r16_seed07",
                            "status": "running",
                            "current_step": 650,
                            "max_steps": 1024,
                            "checkpoint_step": 512,
                            "updated_at": "2026-04-22T01:56:44+00:00",
                        },
                        {
                            "config_name": "core_attn_r16_seed13",
                            "label": "qwen25_vl_3b x core_attn_r16_seed13",
                            "status": "running",
                            "current_step": 600,
                            "max_steps": 1024,
                            "checkpoint_step": 512,
                            "updated_at": "2026-04-22T01:56:05+00:00",
                        },
                    ],
                },
                "plan": {
                    "post_train_eval_ready": False,
                    "first_eleven_completed": False,
                    "pending_internal_dev_evals": [],
                    "pending_validation_evals": [],
                },
            },
        }
    }

    training_report = render_training_report(summary)

    assert "975/1024" in training_report
    assert "925/1024" in training_report
    assert "650/1024" not in training_report
    assert "600/1024" not in training_report


def test_render_training_report_prefers_live_gpu_tasks_when_present() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 4,
                "running": 2,
                "pending": 6,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "runs": [
                {
                    "config_name": "core_attn_r16_seed07",
                    "label": "qwen25_vl_3b x core_attn_r16_seed07",
                    "status": "running",
                    "current_step": 975,
                    "max_steps": 1024,
                    "checkpoint_step": 512,
                    "updated_at": "2026-04-22T02:26:09+00:00",
                }
            ],
            "live_gpu_tasks": [
                {
                    "gpu_id": "0",
                    "assignment_kind": "training",
                    "assignment_label": "core_attn_r16_seed13",
                    "utilization_gpu": 54,
                    "memory_used": 27553,
                    "memory_total": 81559,
                },
                {
                    "gpu_id": "1",
                    "assignment_kind": "training",
                    "assignment_label": "core_attn_r16_seed07",
                    "utilization_gpu": 50,
                    "memory_used": 28191,
                    "memory_total": 81559,
                },
            ],
            "scheduler": {
                "polled_at": "2026-04-22T01:57:52+00:00",
                "remote_git_head": "b07bd70",
                "plan": {
                    "gpus": [
                        {
                            "gpu_id": "0",
                            "assignment_kind": "training",
                            "assignment_label": "stale-seed13",
                            "utilization_gpu": 12,
                            "memory_used": 25013,
                            "memory_total": 81559,
                        },
                        {
                            "gpu_id": "1",
                            "assignment_kind": "training",
                            "assignment_label": "stale-seed07",
                            "utilization_gpu": 11,
                            "memory_used": 28165,
                            "memory_total": 81559,
                        },
                    ],
                    "post_train_eval_ready": False,
                    "first_eleven_completed": False,
                    "pending_internal_dev_evals": [],
                    "pending_validation_evals": [],
                },
            },
        }
    }

    training_report = render_training_report(summary)

    assert "core_attn_r16_seed13" in training_report
    assert "54" in training_report
    assert "27553/81559" in training_report
    assert "stale-seed13" not in training_report


def test_render_training_report_lists_active_and_pending_eval_queue() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 11,
                "running": 1,
                "pending": 0,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "runs": [
                {
                    "config_name": "scale_best_assumed_full",
                    "label": "qwen25_vl_3b x scale_best_assumed_full",
                    "status": "running",
                    "current_step": 1075,
                    "max_steps": 4076,
                }
            ],
            "scheduler": {
                "polled_at": "2026-04-22T08:24:44+00:00",
                "remote_git_head": "9eec9da",
                "sync_mode": "full-ssh",
                "synced_paths": [
                    "outputs/training",
                    "outputs/logs/training_matrix",
                ],
                "eval_runs": [
                    {
                        "config_name": "core_all_linear_r16_seed07",
                        "split": "internal_dev",
                        "status": "running",
                        "processed_count": 100,
                        "total_count": 2000,
                        "started_at": "2026-04-22T08:25:06+00:00",
                        "updated_at": "2026-04-22T08:30:06+00:00",
                        "resumed_from_count": 0,
                    }
                ],
                "plan": {
                    "post_train_eval_ready": True,
                    "first_eleven_completed": True,
                    "active_evals": [
                        {
                            "config_name": "core_all_linear_r16_seed07",
                            "split": "internal_dev",
                            "gpu_id": "0",
                            "status": "running",
                        }
                    ],
                    "pending_internal_dev_evals": [
                        "core_all_linear_r16_seed13",
                        "core_all_linear_r32_seed07",
                    ],
                    "pending_validation_evals": [],
                    "gpus": [
                        {
                            "gpu_id": "0",
                            "assignment_kind": "eval",
                            "assignment_label": "core_all_linear_r16_seed07 (internal_dev)",
                            "utilization_gpu": 56,
                            "memory_used": 9149,
                            "memory_total": 81559,
                        },
                        {
                            "gpu_id": "1",
                            "assignment_kind": "training",
                            "assignment_label": "scale_best_assumed_full",
                            "utilization_gpu": 58,
                            "memory_used": 34589,
                            "memory_total": 81559,
                        },
                    ],
                },
            },
        }
    }

    training_report = render_training_report(summary)

    assert "Post-Train Eval Queue" in training_report
    assert "running" in training_report
    assert "pending" in training_report
    assert "core_all_linear_r16_seed07" in training_report
    assert "core_all_linear_r16_seed13" in training_report
    assert "1 running, 2 pending" in training_report
    assert "100/2000" in training_report
    assert "now" in training_report


def test_render_training_report_prefers_live_eval_queue_over_stale_scheduler_snapshot() -> None:
    summary = {
        "training": {
            "counts": {
                "completed": 11,
                "running": 1,
                "pending": 0,
                "failed": 0,
                "other": 0,
                "total": 12,
            },
            "status": "running",
            "runs": [
                {
                    "config_name": "scale_best_assumed_full",
                    "label": "qwen25_vl_3b x scale_best_assumed_full",
                    "status": "running",
                    "current_step": 1075,
                    "max_steps": 4076,
                }
            ],
            "live_gpu_tasks": [
                {
                    "gpu_id": "0",
                    "assignment_kind": "idle",
                    "assignment_label": "-",
                    "utilization_gpu": 0,
                    "memory_used": 0,
                    "memory_total": 81559,
                },
                {
                    "gpu_id": "1",
                    "assignment_kind": "training",
                    "assignment_label": "scale_best_assumed_full",
                    "utilization_gpu": 58,
                    "memory_used": 34589,
                    "memory_total": 81559,
                },
            ],
            "scheduler": {
                "polled_at": "2026-04-22T08:24:44+00:00",
                "remote_git_head": "9eec9da",
                "sync_mode": "full-ssh",
                "synced_paths": [
                    "outputs/training",
                    "outputs/logs/training_matrix",
                ],
                "eval_runs": [
                    {
                        "config_name": "core_all_linear_r16_seed07",
                        "split": "internal_dev",
                        "status": "running",
                        "processed_count": 100,
                        "total_count": 2000,
                        "started_at": "2026-04-22T08:25:06+00:00",
                        "updated_at": "2026-04-22T08:30:06+00:00",
                        "resumed_from_count": 0,
                    }
                ],
                "plan": {
                    "post_train_eval_ready": True,
                    "first_eleven_completed": True,
                    "active_evals": [
                        {
                            "config_name": "core_all_linear_r16_seed07",
                            "split": "internal_dev",
                            "gpu_id": "0",
                            "status": "running",
                        }
                    ],
                    "pending_internal_dev_evals": [
                        "core_all_linear_r16_seed13",
                    ],
                    "pending_validation_evals": [],
                },
            },
        }
    }

    training_report = render_training_report(summary)

    assert "0 running, 1 pending" in training_report
    assert "core_all_linear_r16_seed07 | internal_dev" not in training_report


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

    assert "Training Overview" in training_report
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


def test_render_training_report_surfaces_non_finite_metrics() -> None:
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
            "status": "running",
            "runs": [
                {
                    "label": "qwen25_vl_3b x core_all_linear_r16_seed07",
                    "status": "running",
                    "current_step": 25,
                    "max_steps": 1024,
                    "latest_log": {"loss": 0, "grad_norm": float("nan")},
                    "updated_at": "2026-04-21T21:10:00+00:00",
                }
            ],
        }
    }

    training_report = render_training_report(summary)

    assert "Loss" in training_report
    assert "Grad" in training_report
    assert "0" in training_report
    assert "nan" in training_report


def test_training_progress_prefers_latest_numeric_checkpoint(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "trainer_state.json").write_text(
        """
        {
          "status": "completed",
          "global_step": 1024,
          "max_steps": 1024,
          "checkpoint_step": 512,
          "updated_at": "2026-04-20T23:51:00+00:00"
        }
        """.strip(),
        encoding="utf-8",
    )
    for step in (512, 1024):
        checkpoint_root = run_root / f"checkpoint-{step}"
        checkpoint_root.mkdir()
        (checkpoint_root / "trainer_state.json").write_text(
            f'{{"global_step": {step}, "max_steps": 1024}}',
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "textvqa_proj.progress.load_settings",
        lambda config_paths: Settings(),
    )
    monkeypatch.setattr(
        "textvqa_proj.progress.training_run_root",
        lambda repo_root, settings: run_root,
    )

    progress = _training_progress(tmp_path, [Path("dummy.toml")], config_name="demo")

    assert progress.current_step == 1024
    assert progress.max_steps == 1024
    assert progress.checkpoint_step == 1024


def test_training_progress_downgrades_stale_running_state(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "trainer_state.json").write_text(
        """
        {
          "status": "running",
          "global_step": 512,
          "max_steps": 1024,
          "updated_at": "2026-04-20T19:15:45.625602+00:00"
        }
        """.strip(),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "textvqa_proj.progress.load_settings",
        lambda config_paths: Settings(),
    )
    monkeypatch.setattr(
        "textvqa_proj.progress.training_run_root",
        lambda repo_root, settings: run_root,
    )
    monkeypatch.setattr(
        "textvqa_proj.progress._current_utc",
        lambda: datetime.fromisoformat("2026-04-21T20:58:22+00:00"),
    )

    progress = _training_progress(tmp_path, [Path("dummy.toml")], config_name="demo")

    assert progress.status == "failed"
    assert progress.error is not None
    assert "stale" in progress.error


def test_evaluation_progress_reads_eta_fields(tmp_path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "progress.json").write_text(
        """
        {
          "status": "running",
          "processed_count": 50,
          "total_count": 200,
          "started_at": "2026-04-20T12:00:00+00:00",
          "resumed_from_count": 10,
          "updated_at": "2026-04-20T12:40:00+00:00"
        }
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "textvqa_proj.progress.load_settings",
        lambda config_paths: Settings(),
    )
    monkeypatch.setattr(
        "textvqa_proj.progress.evaluation_run_root",
        lambda repo_root, settings: run_root,
    )

    progress = _evaluation_progress(tmp_path, [Path("dummy-model.toml"), Path("dummy-exp.toml")])

    assert progress.processed_count == 50
    assert progress.total_count == 200
    assert progress.resumed_from_count == 10
    assert progress.eta_at == "2026-04-20T15:10:00+00:00"


def test_project_training_schedule_waits_for_active_runs_without_eta(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        "textvqa_proj.progress.TRAINING_CONFIGS",
        [
            Path("starting-a.toml"),
            Path("starting-b.toml"),
            Path("pending-a.toml"),
            Path("pending-b.toml"),
        ],
    )
    monkeypatch.setattr(
        "textvqa_proj.progress._estimate_training_max_steps",
        lambda repo_root, config_paths: 1024,
    )

    runs = [
        RunProgress(
            label="done-a",
            config_name="done-a",
            status="completed",
            processed_count=0,
            updated_at="2026-04-20T02:50:40+00:00",
            accuracy=None,
            root=tmp_path,
            current_step=1024,
            max_steps=1024,
            started_at="2026-04-20T00:00:00+00:00",
        ),
        RunProgress(
            label="starting-a",
            config_name="starting-a",
            status="running",
            processed_count=0,
            updated_at="2026-04-20T03:00:00+00:00",
            accuracy=None,
            root=tmp_path,
            current_step=0,
            max_steps=1024,
        ),
        RunProgress(
            label="starting-b",
            config_name="starting-b",
            status="starting",
            processed_count=0,
            updated_at="2026-04-20T03:05:00+00:00",
            accuracy=None,
            root=tmp_path,
        ),
        RunProgress(
            label="pending-a",
            config_name="pending-a",
            status="pending",
            processed_count=0,
            updated_at=None,
            accuracy=None,
            root=tmp_path,
        ),
        RunProgress(
            label="pending-b",
            config_name="pending-b",
            status="pending",
            processed_count=0,
            updated_at=None,
            accuracy=None,
            root=tmp_path,
        ),
    ]

    projected_runs = _project_training_schedule(tmp_path, runs, training_overlays=[])
    projected_by_name = {run.config_name: run for run in projected_runs}

    assert projected_by_name["starting-a"].projected_start_at == "now"
    assert projected_by_name["starting-a"].projected_end_at == "2026-04-20T05:50:40+00:00"
    assert projected_by_name["starting-b"].projected_start_at == "now"
    assert projected_by_name["starting-b"].projected_end_at == "2026-04-20T05:55:40+00:00"
    assert projected_by_name["pending-a"].projected_start_at == "2026-04-20T05:50:40+00:00"
    assert projected_by_name["pending-b"].projected_start_at == "2026-04-20T05:55:40+00:00"
