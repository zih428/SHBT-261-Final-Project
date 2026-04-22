from __future__ import annotations

import json
import math
import os
import re
import shlex
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from textvqa_proj.experiment_catalog import TRAINING_CONFIGS
from textvqa_proj.utils.io import atomic_write_json, ensure_dir, json_default

REMOTE_REPO_ROOT = "/workspace/SHBT-261-Final-Project"
STATE_RELATIVE_PATH = Path("outputs/logs/runpod_scheduler/latest_state.json")
SYNC_RELATIVE_PATHS = [
    Path("outputs/training"),
    Path("outputs/runs/trained_adapters"),
    Path("outputs/logs/training_matrix"),
]
POSTEVAL_SESSION_PREFIX = "posteval"
TRAINING_SESSION_PREFIX = "trainresume"
RUNPOD_JSON_START = "__RUNPOD_JSON_START__"
RUNPOD_JSON_END = "__RUNPOD_JSON_END__"
RUNPOD_HOST = "ssh.runpod.io"
RUNPOD_USER = "51avwqd4qoob8t-64411fef"
RUNPOD_KEY = Path.home() / ".ssh/runpod_ed25519"
EVAL_STALE_AFTER = timedelta(minutes=20)
SYNC_HOST_ENV_VARS = ("RUNPOD_SYNC_HOST", "RUNPOD_FULL_SSH_HOST")
SYNC_PORT_ENV_VARS = ("RUNPOD_SYNC_PORT", "RUNPOD_FULL_SSH_PORT")
SYNC_USER_ENV_VARS = ("RUNPOD_SYNC_USER", "RUNPOD_FULL_SSH_USER")
REMOTE_COMMAND_TIMEOUT_SECONDS = 180
RSYNC_TIMEOUT_SECONDS = 300

ALL_TRAINING_CONFIGS = [path.stem for path in TRAINING_CONFIGS]
CORE_TRAINING_CONFIGS = [name for name in ALL_TRAINING_CONFIGS if name.startswith("core_")]
LAST_TRAINING_CONFIG = "scale_best_assumed_full"
FIRST_ELEVEN_CONFIGS = [name for name in ALL_TRAINING_CONFIGS if name != LAST_TRAINING_CONFIG]
POST_TRAIN_EVAL_CONFIGS = list(ALL_TRAINING_CONFIGS)


@dataclass(slots=True)
class SchedulerAction:
    kind: str
    label: str
    gpu_id: str | None = None
    session_name: str | None = None
    lines: list[str] = field(default_factory=list)


def _strip_ansi(text: str) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)


def _extract_marked_json(output: str) -> dict[str, Any]:
    cleaned = _strip_ansi(output)
    start = cleaned.rfind(RUNPOD_JSON_START)
    end = cleaned.rfind(RUNPOD_JSON_END)
    if start == -1 or end == -1 or end <= start:
        raise RuntimeError("RunPod command did not emit marked JSON output.")
    payload = cleaned[start + len(RUNPOD_JSON_START) : end].strip()
    first_brace = payload.find("{")
    last_brace = payload.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace:
        payload = payload[first_brace : last_brace + 1]
    payload = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", payload)
    return json.loads(payload)


def _iso_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _is_finite(value: Any) -> bool:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _session_name_for_eval(config_name: str, split: str, gpu_id: str) -> str:
    return f"{POSTEVAL_SESSION_PREFIX}-gpu{gpu_id}__{config_name}__{split}"


def _parse_eval_session_name(session_name: str) -> dict[str, str] | None:
    prefix = f"{POSTEVAL_SESSION_PREFIX}-gpu"
    if not session_name.startswith(prefix):
        return None
    remainder = session_name[len(prefix) :]
    try:
        gpu_and_rest = remainder.split("__", maxsplit=1)
        gpu_id = gpu_and_rest[0]
        config_name, split = gpu_and_rest[1].rsplit("__", maxsplit=1)
    except (IndexError, ValueError):
        return None
    return {"gpu_id": gpu_id, "config_name": config_name, "split": split}


def _training_runs_by_name(snapshot: dict[str, Any]) -> dict[str, dict[str, Any]]:
    runs = snapshot.get("training", {}).get("runs", [])
    return {
        str(run.get("config_name")): run
        for run in runs
        if isinstance(run, dict) and run.get("config_name")
    }


def _completed_training_configs(snapshot: dict[str, Any]) -> set[str]:
    runs_by_name = _training_runs_by_name(snapshot)
    return {
        name for name, run in runs_by_name.items() if run.get("status") == "completed"
    }


def _first_eleven_completed(snapshot: dict[str, Any]) -> bool:
    completed = _completed_training_configs(snapshot)
    return all(name in completed for name in FIRST_ELEVEN_CONFIGS)


def _training_complete(snapshot: dict[str, Any]) -> bool:
    completed = _completed_training_configs(snapshot)
    return all(name in completed for name in ALL_TRAINING_CONFIGS)


def _active_eval_task_keys(snapshot: dict[str, Any]) -> set[tuple[str, str]]:
    task_keys: set[tuple[str, str]] = set()
    for session_name in snapshot.get("tmux_sessions", []):
        if not isinstance(session_name, str):
            continue
        parsed = _parse_eval_session_name(session_name)
        if parsed is None:
            continue
        task_keys.add((parsed["config_name"], parsed["split"]))
    for run in snapshot.get("eval_runs", []):
        if not isinstance(run, dict):
            continue
        if run.get("status") == "running":
            config_name = run.get("config_name")
            split = run.get("split")
            if isinstance(config_name, str) and isinstance(split, str):
                task_keys.add((config_name, split))
    return task_keys


def _active_eval_tasks(snapshot: dict[str, Any]) -> list[dict[str, str]]:
    tasks: dict[tuple[str, str], dict[str, str]] = {}
    for session_name in snapshot.get("tmux_sessions", []):
        if not isinstance(session_name, str):
            continue
        parsed = _parse_eval_session_name(session_name)
        if parsed is None:
            continue
        task_key = (parsed["config_name"], parsed["split"])
        tasks[task_key] = {
            "config_name": parsed["config_name"],
            "split": parsed["split"],
            "gpu_id": parsed["gpu_id"],
            "status": "running",
        }
    for run in snapshot.get("eval_runs", []):
        if not isinstance(run, dict) or run.get("status") != "running":
            continue
        config_name = run.get("config_name")
        split = run.get("split")
        if not isinstance(config_name, str) or not isinstance(split, str):
            continue
        task_key = (config_name, split)
        existing = tasks.get(task_key, {})
        tasks[task_key] = {
            "config_name": config_name,
            "split": split,
            "gpu_id": str(existing.get("gpu_id") or run.get("gpu_id") or "-"),
            "status": "running",
        }
    return [
        tasks[key]
        for key in sorted(tasks, key=lambda item: (item[1], item[0]))
    ]


def _completed_eval_task_keys(snapshot: dict[str, Any]) -> set[tuple[str, str]]:
    task_keys: set[tuple[str, str]] = set()
    for run in snapshot.get("eval_runs", []):
        if not isinstance(run, dict):
            continue
        if run.get("status") != "completed":
            continue
        config_name = run.get("config_name")
        split = run.get("split")
        if isinstance(config_name, str) and isinstance(split, str):
            task_keys.add((config_name, split))
    return task_keys


def _internal_dev_eval_queue(snapshot: dict[str, Any]) -> list[str]:
    completed = _completed_training_configs(snapshot)
    active_evals = _active_eval_task_keys(snapshot)
    completed_evals = _completed_eval_task_keys(snapshot)
    queue: list[str] = []
    for config_name in POST_TRAIN_EVAL_CONFIGS:
        if config_name not in completed:
            continue
        task_key = (config_name, "internal_dev")
        if task_key in active_evals or task_key in completed_evals:
            continue
        queue.append(config_name)
    return queue


def _validation_candidate(snapshot: dict[str, Any]) -> str | None:
    if not _training_complete(snapshot):
        return None

    completed_internal_dev_scores: dict[str, float] = {}
    for run in snapshot.get("eval_runs", []):
        if not isinstance(run, dict):
            continue
        if run.get("status") != "completed" or run.get("split") != "internal_dev":
            continue
        config_name = run.get("config_name")
        accuracy = run.get("accuracy")
        if not isinstance(config_name, str) or config_name not in POST_TRAIN_EVAL_CONFIGS:
            continue
        if not _is_finite(accuracy):
            continue
        completed_internal_dev_scores[config_name] = float(accuracy)

    runs_by_name = _training_runs_by_name(snapshot)
    if completed_internal_dev_scores:
        ranked_candidates: list[tuple[float, float, str]] = []
        for config_name, accuracy in completed_internal_dev_scores.items():
            run = runs_by_name.get(config_name, {})
            latest_eval = run.get("latest_eval") if isinstance(run, dict) else {}
            eval_loss = (
                float(latest_eval.get("eval_loss"))
                if isinstance(latest_eval, dict) and _is_finite(latest_eval.get("eval_loss"))
                else math.inf
            )
            ranked_candidates.append((-accuracy, eval_loss, config_name))
        ranked_candidates.sort()
        return ranked_candidates[0][2]

    best_name: str | None = None
    best_eval_loss: float | None = None
    for config_name, run in runs_by_name.items():
        if run.get("status") != "completed":
            continue
        latest_eval = run.get("latest_eval")
        if not isinstance(latest_eval, dict):
            continue
        eval_loss = latest_eval.get("eval_loss")
        if not _is_finite(eval_loss):
            continue
        numeric_eval_loss = float(eval_loss)
        if best_eval_loss is None or numeric_eval_loss < best_eval_loss:
            best_eval_loss = numeric_eval_loss
            best_name = config_name
    return best_name


def _validation_eval_queue(snapshot: dict[str, Any]) -> list[str]:
    if _internal_dev_eval_queue(snapshot):
        return []
    candidate = _validation_candidate(snapshot)
    if candidate is None:
        return []
    task_key = (candidate, "validation")
    if task_key in _active_eval_task_keys(snapshot):
        return []
    if task_key in _completed_eval_task_keys(snapshot):
        return []
    return [candidate]


def _has_active_training(snapshot: dict[str, Any]) -> bool:
    runs_by_name = _training_runs_by_name(snapshot)
    return any(
        run.get("status") in {"running", "starting"} for run in runs_by_name.values()
    )


def _pending_training_configs(snapshot: dict[str, Any]) -> list[str]:
    runs_by_name = _training_runs_by_name(snapshot)
    pending = []
    for config_name in ALL_TRAINING_CONFIGS:
        status = runs_by_name.get(config_name, {}).get("status", "pending")
        if status in {"pending", "failed"}:
            pending.append(config_name)
    return pending


def _classify_gpu_status(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    gpus = [dict(gpu) for gpu in snapshot.get("gpus", []) if isinstance(gpu, dict)]
    active_training = snapshot.get("active_training", [])
    tmux_sessions = snapshot.get("tmux_sessions", [])

    training_by_gpu = {
        str(item.get("gpu_id")): str(item.get("config_name"))
        for item in active_training
        if isinstance(item, dict) and item.get("gpu_id") is not None and item.get("config_name")
    }
    eval_by_gpu = {}
    for session_name in tmux_sessions:
        if not isinstance(session_name, str):
            continue
        parsed = _parse_eval_session_name(session_name)
        if parsed is None:
            continue
        eval_by_gpu[parsed["gpu_id"]] = f"{parsed['config_name']} ({parsed['split']})"

    for gpu in gpus:
        gpu_id = str(gpu.get("gpu_id"))
        gpu["assignment_kind"] = "idle"
        gpu["assignment_label"] = "-"
        if gpu_id in training_by_gpu:
            gpu["assignment_kind"] = "training"
            gpu["assignment_label"] = training_by_gpu[gpu_id]
            continue
        if gpu_id in eval_by_gpu:
            gpu["assignment_kind"] = "eval"
            gpu["assignment_label"] = eval_by_gpu[gpu_id]
            continue
        utilization = gpu.get("utilization_gpu")
        try:
            utilization_numeric = int(utilization) if utilization is not None else 0
        except (TypeError, ValueError):
            utilization_numeric = 0
        if utilization_numeric >= 10:
            gpu["assignment_kind"] = "unknown"
            gpu["assignment_label"] = "unattributed CUDA activity"
    return gpus


def _free_gpu_ids(snapshot: dict[str, Any]) -> list[str]:
    return [
        str(gpu.get("gpu_id"))
        for gpu in _classify_gpu_status(snapshot)
        if gpu.get("assignment_kind") == "idle"
    ]


def _remote_training_root(snapshot: dict[str, Any], config_name: str) -> str | None:
    run = _training_runs_by_name(snapshot).get(config_name)
    root = run.get("root") if isinstance(run, dict) else None
    return str(root) if root else None


def build_scheduler_plan(snapshot: dict[str, Any]) -> dict[str, Any]:
    first_eleven_completed = _first_eleven_completed(snapshot)
    training_complete = _training_complete(snapshot)
    active_evals = _active_eval_tasks(snapshot)
    internal_dev_queue = _internal_dev_eval_queue(snapshot)
    validation_queue = _validation_eval_queue(snapshot)
    post_train_eval_ready = first_eleven_completed
    free_gpu_ids = _free_gpu_ids(snapshot)
    actions: list[SchedulerAction] = []
    notes: list[str] = []

    pending_training = _pending_training_configs(snapshot)
    if pending_training and not _has_active_training(snapshot):
        if any(name in CORE_TRAINING_CONFIGS for name in pending_training):
            actions.append(
                SchedulerAction(
                    kind="resume-training-core",
                    label="Resume pending core-matrix training runs",
                )
            )
        else:
            actions.append(
                SchedulerAction(
                    kind="resume-training-followups",
                    label="Resume pending follow-up training runs",
                )
            )
        notes.append("Training takes priority while pending runs remain and no launcher is active.")
    elif post_train_eval_ready and free_gpu_ids:
        if internal_dev_queue:
            for gpu_id, config_name in zip(free_gpu_ids, internal_dev_queue, strict=False):
                actions.append(
                    SchedulerAction(
                        kind="launch-eval",
                        label=f"{config_name} internal_dev",
                        gpu_id=gpu_id,
                    )
                )
        elif validation_queue:
            for gpu_id, config_name in zip(free_gpu_ids, validation_queue, strict=False):
                actions.append(
                    SchedulerAction(
                        kind="launch-eval",
                        label=f"{config_name} validation",
                        gpu_id=gpu_id,
                    )
                )
        else:
            notes.append("No pending post-training eval tasks are ready for an idle GPU.")
    elif not post_train_eval_ready:
        notes.append("Post-training evals stay gated until the first 11 training runs complete.")

    return {
        "first_eleven_completed": first_eleven_completed,
        "training_complete": training_complete,
        "post_train_eval_ready": post_train_eval_ready,
        "free_gpu_ids": free_gpu_ids,
        "active_evals": active_evals,
        "pending_internal_dev_evals": internal_dev_queue,
        "pending_validation_evals": validation_queue,
        "actions": [asdict(action) for action in actions],
        "notes": notes,
        "gpus": _classify_gpu_status(snapshot),
        "validation_candidate": _validation_candidate(snapshot),
    }


def _build_resume_training_lines(kind: str, gpu_ids: list[str]) -> list[str]:
    if not gpu_ids:
        raise RuntimeError("Cannot resume training without any known GPUs.")
    session_name = f"{TRAINING_SESSION_PREFIX}-{datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')}"
    phase_args = (
        "--phase core-matrix"
        if kind == "resume-training-core"
        else "--phase ocr-ablation --phase data-scaling"
    )
    command = " ".join(
        [
            "cd",
            shlex.quote(REMOTE_REPO_ROOT),
            "&&",
            "tmux new-session -d -s",
            shlex.quote(session_name),
            shlex.quote(
                " ".join(
                    [
                        "./.venv/bin/python",
                        "scripts/run_training_matrix_parallel.py",
                        "--config",
                        "configs/runtime_cuda_runpod.toml",
                        "--gpu-ids",
                        ",".join(gpu_ids),
                        "--followup-policy",
                        "auto-from-core",
                        *phase_args.split(),
                    ]
                )
            ),
        ]
    )
    return [command]


def _build_eval_lines(snapshot: dict[str, Any], config_name: str, split: str, gpu_id: str) -> list[str]:
    training_root = _remote_training_root(snapshot, config_name)
    if training_root is None:
        raise RuntimeError(f"Could not resolve remote training root for {config_name}.")
    session_name = _session_name_for_eval(config_name, split, gpu_id)
    eval_command = " ".join(
        [
            "cd",
            shlex.quote(REMOTE_REPO_ROOT),
            "&&",
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(gpu_id)}",
            "&&",
            "export TEXTVQA_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false",
            "&&",
            "./.venv/bin/python -m textvqa_proj.cli evaluate-trained-adapter",
            "--training-root",
            shlex.quote(training_root),
            "--split",
            shlex.quote(split),
            "--output-root",
            "outputs/runs/trained_adapters",
        ]
    )
    return [
        f"cd {shlex.quote(REMOTE_REPO_ROOT)}",
        f"tmux has-session -t {shlex.quote(session_name)} 2>/dev/null && exit 0 || true",
        f"tmux new-session -d -s {shlex.quote(session_name)} {shlex.quote(eval_command)}",
    ]


def _resolve_runpod_ips() -> list[str]:
    ips: list[str] = []
    try:
        info = socket.getaddrinfo(RUNPOD_HOST, 22, proto=socket.IPPROTO_TCP)
    except OSError:
        info = []
    for item in info:
        address = item[4][0]
        if address not in ips:
            ips.append(address)
    for command in (
        ["dscacheutil", "-q", "host", "-a", "name", RUNPOD_HOST],
        ["host", RUNPOD_HOST],
        ["nslookup", RUNPOD_HOST],
    ):
        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        for line in completed.stdout.splitlines():
            if "ip_address:" in line:
                candidate = line.split()[-1]
            elif " has address " in line:
                candidate = line.rsplit(" ", maxsplit=1)[-1]
            elif line.startswith("Address: "):
                candidate = line.split("Address: ", maxsplit=1)[1].strip()
            else:
                continue
            if candidate not in ips:
                ips.append(candidate)
    return ips


def _run_remote_lines(wrapper_path: Path, lines: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(wrapper_path), *lines],
        check=True,
        capture_output=True,
        text=True,
        timeout=REMOTE_COMMAND_TIMEOUT_SECONDS,
    )


def _run_remote_script(wrapper_path: Path, script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(wrapper_path)],
        check=True,
        capture_output=True,
        text=True,
        input=script,
        timeout=REMOTE_COMMAND_TIMEOUT_SECONDS,
    )


def _remote_snapshot_script() -> str:
    return f"""
import json
import os
import subprocess
from pathlib import Path

from textvqa_proj.progress import summarize_project_progress, _latest_training_matrix_status

repo_root = Path.cwd()
summary = summarize_project_progress(
    repo_root,
    training_overlays=[Path("configs/runtime_cuda_runpod.toml")],
)
training_runs = summary["training"]["runs"]
dir_to_config = {{}}
for run in training_runs:
    root = run.get("root")
    config_name = run.get("config_name")
    if root and config_name:
        dir_to_config[Path(root).name] = config_name

eval_runs = []
eval_root = repo_root / "outputs/runs/trained_adapters/trained-adapter-eval"
if eval_root.exists():
    for run_root in sorted(path for path in eval_root.iterdir() if path.is_dir()):
        settings_path = run_root / "settings.json"
        progress_path = run_root / "progress.json"
        metrics_path = run_root / "metrics.json"
        if not settings_path.exists() or not progress_path.exists():
            continue
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {{}}
        )
        adapter_path = settings.get("model", {{}}).get("adapter_path")
        config_name = None
        if adapter_path:
            config_name = dir_to_config.get(Path(adapter_path).parent.name)
        eval_runs.append(
            {{
                "root": str(run_root),
                "config_name": config_name,
                "split": settings.get("experiment", {{}}).get("split"),
                "status": progress.get("status"),
                "processed_count": progress.get("processed_count"),
                "total_count": progress.get("total_count"),
                "started_at": progress.get("started_at"),
                "resumed_from_count": progress.get("resumed_from_count"),
                "updated_at": progress.get("updated_at"),
                "accuracy": metrics.get("accuracy"),
            }}
        )

gpu_rows = []
try:
    gpu_output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
except Exception:
    gpu_output = ""
for line in gpu_output.splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 4:
        continue
    gpu_rows.append(
        {{
            "gpu_id": parts[0],
            "utilization_gpu": int(parts[1]),
            "memory_used": int(parts[2]),
            "memory_total": int(parts[3]),
        }}
    )

try:
    tmux_output = subprocess.check_output(
        ["tmux", "list-sessions", "-F", "#S"],
        text=True,
        stderr=subprocess.DEVNULL,
    )
except Exception:
    tmux_output = ""
tmux_sessions = [line.strip() for line in tmux_output.splitlines() if line.strip()]

training_matrix_status = _latest_training_matrix_status(repo_root) or {{}}
active_training = []
for config_name, item in (training_matrix_status.get("runs") or {{}}).items():
    if item.get("status") not in {{"starting", "running"}}:
        continue
    gpu_id = item.get("gpu_id")
    if gpu_id is None:
        continue
    active_training.append(
        {{
            "config_name": config_name,
            "gpu_id": str(gpu_id),
            "log_path": item.get("log_path"),
            "phase": item.get("phase"),
        }}
    )

git_head = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    cwd=repo_root,
    text=True,
).strip()

payload = {{
    "remote_git_head": git_head,
    "polled_at": "{_iso_now()}",
    "training": summary["training"],
    "eval_runs": eval_runs,
    "gpus": gpu_rows,
    "tmux_sessions": tmux_sessions,
    "active_training": active_training,
    "sync_paths": {{
        "{SYNC_RELATIVE_PATHS[0]}": (repo_root / "{SYNC_RELATIVE_PATHS[0]}").exists(),
        "{SYNC_RELATIVE_PATHS[1]}": (repo_root / "{SYNC_RELATIVE_PATHS[1]}").exists(),
        "{SYNC_RELATIVE_PATHS[2]}": (repo_root / "{SYNC_RELATIVE_PATHS[2]}").exists(),
    }},
    "sync_target": {{
        "host": os.environ.get("RUNPOD_PUBLIC_IP"),
        "port": os.environ.get("RUNPOD_TCP_PORT_22"),
        "user": "root",
    }},
}}
print("{RUNPOD_JSON_START}")
print(json.dumps(payload, sort_keys=True, default=str))
print("{RUNPOD_JSON_END}")
""".strip()


def query_remote_snapshot(repo_root: Path, *, wrapper_path: Path) -> dict[str, Any]:
    completed = _run_remote_lines(
        wrapper_path,
        [
            f"cd {shlex.quote(REMOTE_REPO_ROOT)}",
            "python3 - <<'PY'",
            _remote_snapshot_script(),
            "PY",
        ],
    )
    snapshot = _extract_marked_json(completed.stdout)
    snapshot["plan"] = build_scheduler_plan(snapshot)
    snapshot["polled_at"] = _iso_now()
    snapshot["local_repo_root"] = str(repo_root)
    return snapshot


def _destination_specs() -> list[tuple[str, str]]:
    destinations = [(RUNPOD_HOST, RUNPOD_HOST)]
    for ip in _resolve_runpod_ips():
        destinations.append((ip, ip))
    deduped: list[tuple[str, str]] = []
    seen: set[str] = set()
    for destination, label in destinations:
        if destination in seen:
            continue
        seen.add(destination)
        deduped.append((destination, label))
    return deduped


def _configured_sync_target() -> dict[str, str] | None:
    host = next((os.getenv(name) for name in SYNC_HOST_ENV_VARS if os.getenv(name)), None)
    if not host:
        return None
    port = next(
        (os.getenv(name) for name in SYNC_PORT_ENV_VARS if os.getenv(name)),
        "22",
    )
    user = next(
        (os.getenv(name) for name in SYNC_USER_ENV_VARS if os.getenv(name)),
        "root",
    )
    return {"host": host, "port": port, "user": user}


def _snapshot_sync_target(snapshot: dict[str, Any]) -> dict[str, str] | None:
    target = snapshot.get("sync_target")
    if not isinstance(target, dict):
        return None
    host = str(target.get("host") or "").strip()
    if not host:
        return None
    port = str(target.get("port") or "22").strip() or "22"
    user = str(target.get("user") or "root").strip() or "root"
    return {"host": host, "port": port, "user": user}


def _rsync_remote_path(
    sync_target: dict[str, str],
    relative_path: Path,
    repo_root: Path,
) -> bool:
    local_path = ensure_dir(repo_root / relative_path)
    remote_path = (
        f"{sync_target['user']}@{sync_target['host']}:"
        f"{REMOTE_REPO_ROOT}/{relative_path}/"
    )
    command = [
        "rsync",
        "-az",
        "--partial",
        "-e",
        (
            "ssh "
            f"-i {shlex.quote(str(RUNPOD_KEY))} "
            f"-p {shlex.quote(sync_target['port'])} "
            "-o ConnectTimeout=10"
        ),
        remote_path,
        f"{local_path}/",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=RSYNC_TIMEOUT_SECONDS,
    )
    if result.returncode == 0:
        return True
    stderr = result.stderr.casefold()
    if "no such file or directory" in stderr:
        return True
    return False


def sync_results(repo_root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    sync_target = _configured_sync_target() or _snapshot_sync_target(snapshot)
    if sync_target is None:
        return {
            "synced_paths": [],
            "sync_mode": "disabled-basic-ssh",
            "sync_ready": False,
            "sync_message": (
                "Artifact sync is disabled on proxied ssh.runpod.io access. "
                "Configure RUNPOD_SYNC_HOST/RUNPOD_SYNC_PORT, or expose Pod SSH over TCP, "
                "to enable rsync of training artifacts."
            ),
        }
    synced: list[str] = []
    available_sync_paths = snapshot.get("sync_paths", {})
    for relative_path in SYNC_RELATIVE_PATHS:
        if not available_sync_paths.get(str(relative_path)):
            continue
        if _rsync_remote_path(sync_target, relative_path, repo_root):
            synced.append(str(relative_path))
    message = "No new artifact files were copied in this cycle."
    if synced:
        message = (
            f"Copied updated artifact paths from RunPod over full SSH "
            f"({sync_target['host']}:{sync_target['port']})."
        )
    return {
        "synced_paths": synced,
        "sync_mode": "full-ssh",
        "sync_ready": True,
        "sync_message": message,
    }


def _write_remote_state(wrapper_path: Path, payload: dict[str, Any]) -> None:
    json_payload = json.dumps(payload, indent=2, sort_keys=True, default=json_default)
    script = "\n".join(
        [
            f"cd {shlex.quote(REMOTE_REPO_ROOT)}",
            f"mkdir -p {shlex.quote(str(STATE_RELATIVE_PATH.parent))}",
            f"cat > {shlex.quote(str(STATE_RELATIVE_PATH))} <<'EOF'",
            json_payload,
            "EOF",
            "exit",
        ]
    )
    _run_remote_script(wrapper_path, script)


def _execute_action(
    snapshot: dict[str, Any],
    action: dict[str, Any],
    *,
    wrapper_path: Path,
) -> dict[str, Any]:
    kind = str(action.get("kind"))
    if kind in {"resume-training-core", "resume-training-followups"}:
        lines = _build_resume_training_lines(kind, snapshot["plan"]["free_gpu_ids"])
    elif kind == "launch-eval":
        label = str(action.get("label", ""))
        config_name, split = label.rsplit(" ", maxsplit=1)
        lines = _build_eval_lines(snapshot, config_name, split, str(action.get("gpu_id")))
    else:
        raise RuntimeError(f"Unknown scheduler action: {kind}")
    _run_remote_lines(wrapper_path, lines)
    return {
        "kind": kind,
        "label": action.get("label"),
        "gpu_id": action.get("gpu_id"),
        "executed_at": _iso_now(),
    }


def run_scheduler_cycle(
    repo_root: Path,
    *,
    wrapper_path: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    snapshot = query_remote_snapshot(repo_root, wrapper_path=wrapper_path)
    executed_actions: list[dict[str, Any]] = []
    if not dry_run:
        for action in snapshot["plan"]["actions"]:
            executed_actions.append(
                _execute_action(snapshot, action, wrapper_path=wrapper_path)
            )
        if executed_actions:
            snapshot = query_remote_snapshot(repo_root, wrapper_path=wrapper_path)
        sync_state = sync_results(repo_root, snapshot)
    else:
        sync_state = {
            "synced_paths": [],
            "sync_mode": "dry-run",
            "sync_ready": False,
            "sync_message": "Dry run: skipping artifact sync.",
        }
    state = {
        **snapshot,
        "executed_actions": executed_actions,
        **sync_state,
        "state_written_at": _iso_now(),
    }
    local_state_path = ensure_dir(repo_root / STATE_RELATIVE_PATH.parent) / STATE_RELATIVE_PATH.name
    atomic_write_json(local_state_path, state)
    if not dry_run:
        _write_remote_state(wrapper_path, state)
    return state
