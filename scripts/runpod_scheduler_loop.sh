#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/outputs/logs/runpod_scheduler"
LOOP_LOG="$LOG_DIR/loop.log"
PID_FILE="$LOG_DIR/loop.pid"
DEFAULT_INTERVAL=1800

mkdir -p "$LOG_DIR"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

is_running_pid() {
  local pid="$1"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

run_cycle() {
  {
    echo "[$(timestamp)] scheduler cycle start"
    cd "$REPO_ROOT"
    ./.venv/bin/python scripts/runpod_poll_and_schedule.py
    status=$?
    echo "[$(timestamp)] scheduler cycle end status=$status"
    return "$status"
  } >>"$LOOP_LOG" 2>&1
}

loop_forever() {
  local interval="${1:-$DEFAULT_INTERVAL}"
  echo "$$" >"$PID_FILE"
  trap 'rm -f "$PID_FILE"' EXIT
  while true; do
    run_cycle || true
    sleep "$interval"
  done
}

start_loop() {
  local interval="${1:-$DEFAULT_INTERVAL}"
  if [[ -f "$PID_FILE" ]]; then
    local existing_pid
    existing_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
    if is_running_pid "$existing_pid"; then
      echo "loop already running with pid $existing_pid"
      return 0
    fi
    rm -f "$PID_FILE"
  fi

  nohup /usr/bin/caffeinate -ims /bin/bash -lc \
    "cd \"$REPO_ROOT\" && exec \"$0\" run \"$interval\"" \
    >>"$LOG_DIR/launchd.stdout.log" 2>>"$LOG_DIR/launchd.stderr.log" </dev/null &
  local launcher_pid=$!
  sleep 1
  echo "started scheduler loop launcher pid $launcher_pid"
}

stop_loop() {
  local pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  fi
  if is_running_pid "$pid"; then
    kill "$pid" || true
    sleep 1
  fi
  pkill -f "runpod_scheduler_loop.sh run" 2>/dev/null || true
  pkill -f "/usr/bin/caffeinate -ims" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "stopped scheduler loop"
}

status_loop() {
  local pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  fi
  if is_running_pid "$pid"; then
    echo "running pid=$pid"
    return 0
  fi
  echo "not running"
  return 1
}

usage() {
  cat <<'EOF'
Usage:
  scripts/runpod_scheduler_loop.sh start [interval_seconds]
  scripts/runpod_scheduler_loop.sh run [interval_seconds]
  scripts/runpod_scheduler_loop.sh stop
  scripts/runpod_scheduler_loop.sh status
  scripts/runpod_scheduler_loop.sh once
EOF
}

cmd="${1:-}"
case "$cmd" in
  start)
    start_loop "${2:-$DEFAULT_INTERVAL}"
    ;;
  run)
    loop_forever "${2:-$DEFAULT_INTERVAL}"
    ;;
  stop)
    stop_loop
    ;;
  status)
    status_loop
    ;;
  once)
    run_cycle
    ;;
  *)
    usage
    exit 1
    ;;
esac
