#!/usr/bin/env bash
set -euo pipefail

RUNPOD_USER="${RUNPOD_USER:-51avwqd4qoob8t-64411fef}"
RUNPOD_HOST="${RUNPOD_HOST:-ssh.runpod.io}"
RUNPOD_KEY="${RUNPOD_KEY:-$HOME/.ssh/runpod_ed25519}"

resolve_runpod_ips() {
  local host="$1"

  if command -v dscacheutil >/dev/null 2>&1; then
    dscacheutil -q host -a name "$host" 2>/dev/null | awk '/ip_address: / {print $2}'
  fi

  if command -v host >/dev/null 2>&1; then
    host "$host" 2>/dev/null | awk '/ has address / {print $4}'
  fi

  if command -v nslookup >/dev/null 2>&1; then
    nslookup "$host" 2>/dev/null | awk '/^Address: / {print $2}'
  fi
}

run_ssh() {
  local destination="$1"
  shift
  local -a ssh_args=(
    -tt \
    -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o HostKeyAlias="$RUNPOD_HOST" \
    -i "$RUNPOD_KEY" \
    "$RUNPOD_USER@$destination"
  )

  if (($# == 0)); then
    ssh "${ssh_args[@]}"
    return
  fi

  {
    local line
    for line in "$@"; do
      printf '%s\n' "$line"
    done
    printf 'exit\n'
  } | ssh "${ssh_args[@]}"
}

main() {
  local -a command=()
  if (($# > 0)); then
    command=("$@")
  fi

  if ((${#command[@]} == 0)); then
    if run_ssh "$RUNPOD_HOST"; then
      return 0
    fi
  elif run_ssh "$RUNPOD_HOST" "${command[@]}"; then
    return 0
  fi

  local -a ips=()
  while IFS= read -r ip; do
    [[ -n "$ip" ]] || continue
    ips+=("$ip")
  done < <(resolve_runpod_ips "$RUNPOD_HOST" | sort -u)

  if ((${#ips[@]} == 0)); then
    echo "Failed to resolve fallback IPs for $RUNPOD_HOST" >&2
    return 1
  fi

  local ip
  for ip in "${ips[@]}"; do
    if ((${#command[@]} == 0)); then
      if run_ssh "$ip"; then
        return 0
      fi
    elif run_ssh "$ip" "${command[@]}"; then
      return 0
    fi
  done

  echo "Unable to connect to RunPod via $RUNPOD_HOST or resolved fallback IPs." >&2
  return 1
}

main "$@"
