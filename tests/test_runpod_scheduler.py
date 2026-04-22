from __future__ import annotations

from pathlib import Path

from textvqa_proj.runpod_scheduler import build_scheduler_plan, sync_results


def _base_snapshot() -> dict[str, object]:
    training_runs = []
    for config_name in [
        "core_all_linear_r16_seed07",
        "core_all_linear_r16_seed13",
        "core_all_linear_r32_seed07",
        "core_all_linear_r32_seed13",
        "core_attn_r16_seed07",
        "core_attn_r16_seed13",
        "core_attn_r32_seed07",
        "core_attn_r32_seed13",
        "ocr_ablation_off",
        "ocr_ablation_on",
        "scale_best_assumed_25pct",
        "scale_best_assumed_full",
    ]:
        training_runs.append(
            {
                "config_name": config_name,
                "status": "pending",
                "root": f"/remote/{config_name}",
            }
        )
    return {
        "training": {"runs": training_runs},
        "eval_runs": [],
        "gpus": [
            {"gpu_id": "0", "utilization_gpu": 0, "memory_used": 0, "memory_total": 80},
            {"gpu_id": "1", "utilization_gpu": 0, "memory_used": 0, "memory_total": 80},
        ],
        "tmux_sessions": [],
        "active_training": [],
    }


def test_scheduler_prefers_training_resume_when_pending_runs_have_no_active_launcher() -> None:
    snapshot = _base_snapshot()
    snapshot["training"]["runs"][0]["status"] = "completed"

    plan = build_scheduler_plan(snapshot)

    assert plan["actions"][0]["kind"] == "resume-training-core"
    assert plan["post_train_eval_ready"] is False


def test_scheduler_uses_idle_gpu_for_internal_dev_core_eval_once_first_eleven_finish() -> None:
    snapshot = _base_snapshot()
    for run in snapshot["training"]["runs"]:
        if run["config_name"] != "scale_best_assumed_full":
            run["status"] = "completed"
        else:
            run["status"] = "running"
    snapshot["active_training"] = [{"config_name": "scale_best_assumed_full", "gpu_id": "0"}]

    plan = build_scheduler_plan(snapshot)

    assert plan["post_train_eval_ready"] is True
    assert plan["free_gpu_ids"] == ["1"]
    assert plan["actions"] == [
        {
            "kind": "launch-eval",
            "label": "core_all_linear_r16_seed07 internal_dev",
            "gpu_id": "1",
            "session_name": None,
            "lines": [],
        }
    ]


def test_scheduler_skips_internal_dev_tasks_already_running_or_completed() -> None:
    snapshot = _base_snapshot()
    for run in snapshot["training"]["runs"]:
        if run["config_name"] != "scale_best_assumed_full":
            run["status"] = "completed"
        else:
            run["status"] = "running"
    snapshot["active_training"] = [{"config_name": "scale_best_assumed_full", "gpu_id": "0"}]
    snapshot["eval_runs"] = [
        {
            "config_name": "core_all_linear_r16_seed07",
            "split": "internal_dev",
            "status": "running",
            "accuracy": None,
        },
        {
            "config_name": "core_all_linear_r16_seed13",
            "split": "internal_dev",
            "status": "completed",
            "accuracy": 0.8,
        }
    ]

    plan = build_scheduler_plan(snapshot)

    assert plan["actions"][0]["label"] == "core_all_linear_r32_seed07 internal_dev"


def test_scheduler_queues_validation_for_lowest_eval_loss_after_training_and_core_evals_finish() -> None:
    snapshot = _base_snapshot()
    for run in snapshot["training"]["runs"]:
        run["status"] = "completed"
        run["latest_eval"] = {"eval_loss": 0.7}
    snapshot["training"]["runs"][0]["latest_eval"] = {"eval_loss": 0.42}
    snapshot["training"]["runs"][11]["latest_eval"] = {"eval_loss": 0.5}
    for config_name in [
        "core_all_linear_r16_seed07",
        "core_all_linear_r16_seed13",
        "core_all_linear_r32_seed07",
        "core_all_linear_r32_seed13",
        "core_attn_r16_seed07",
        "core_attn_r16_seed13",
        "core_attn_r32_seed07",
        "core_attn_r32_seed13",
    ]:
        snapshot["eval_runs"].append(
            {
                "config_name": config_name,
                "split": "internal_dev",
                "status": "completed",
                "accuracy": 0.75,
            }
        )

    plan = build_scheduler_plan(snapshot)

    assert plan["training_complete"] is True
    assert plan["validation_candidate"] == "core_all_linear_r16_seed07"
    assert plan["actions"][0]["label"] == "core_all_linear_r16_seed07 validation"


def test_scheduler_marks_busy_unattributed_gpu_as_non_idle() -> None:
    snapshot = _base_snapshot()
    for run in snapshot["training"]["runs"]:
        if run["config_name"] != "scale_best_assumed_full":
            run["status"] = "completed"
        else:
            run["status"] = "running"
    snapshot["gpus"][1]["utilization_gpu"] = 40

    plan = build_scheduler_plan(snapshot)

    assert plan["gpus"][1]["assignment_kind"] == "unknown"
    assert plan["free_gpu_ids"] == ["0"]


def test_sync_results_reports_basic_ssh_limitation_without_full_ssh_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RUNPOD_SYNC_HOST", raising=False)
    monkeypatch.delenv("RUNPOD_FULL_SSH_HOST", raising=False)

    sync_state = sync_results(
        tmp_path,
        {
            "sync_paths": {
                "outputs/training": True,
                "outputs/runs/trained_adapters": True,
                "outputs/logs/training_matrix": True,
            }
        },
    )

    assert sync_state["synced_paths"] == []
    assert sync_state["sync_mode"] == "disabled-basic-ssh"
    assert "expose Pod SSH over TCP" in sync_state["sync_message"]


def test_sync_results_uses_snapshot_full_ssh_target_when_available(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RUNPOD_SYNC_HOST", raising=False)
    monkeypatch.delenv("RUNPOD_FULL_SSH_HOST", raising=False)

    calls: list[tuple[str, str]] = []

    def fake_rsync(sync_target, relative_path, repo_root):
        calls.append((sync_target["host"], sync_target["port"]))
        return True

    monkeypatch.setattr("textvqa_proj.runpod_scheduler._rsync_remote_path", fake_rsync)

    sync_state = sync_results(
        tmp_path,
        {
            "sync_paths": {
                "outputs/training": True,
                "outputs/runs/trained_adapters": False,
                "outputs/logs/training_matrix": True,
            },
            "sync_target": {
                "host": "216.243.220.223",
                "port": "16291",
                "user": "root",
            },
        },
    )

    assert sync_state["sync_mode"] == "full-ssh"
    assert sync_state["sync_ready"] is True
    assert sync_state["synced_paths"] == [
        "outputs/training",
        "outputs/logs/training_matrix",
    ]
    assert calls == [("216.243.220.223", "16291"), ("216.243.220.223", "16291")]
