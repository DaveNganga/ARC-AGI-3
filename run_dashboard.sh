#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")" && pwd)"
next_lock="$project_root/frontend/.next/dev/lock"
runner_pid=""
next_pid=""

stop_previous_next() {
  [[ -f "$next_lock" ]] || return 0

  local stale_pid
  stale_pid="$($project_root/.venv/bin/python -c \
    'import json,sys; print(json.load(open(sys.argv[1])).get("pid", ""))' \
    "$next_lock" 2>/dev/null || true)"

  if [[ "$stale_pid" =~ ^[0-9]+$ ]] && kill -0 "$stale_pid" 2>/dev/null; then
    echo "Stopping previous QAHS dashboard (PID $stale_pid)..."
    kill "$stale_pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$stale_pid" 2>/dev/null || break
      sleep 0.1
    done
  fi

  # This lock belongs only to this project's Next.js development server.
  rm -f "$next_lock"
}

cleanup() {
  [[ -z "$next_pid" ]] || kill -- "-$next_pid" 2>/dev/null || true
  [[ -z "$runner_pid" ]] || kill "$runner_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM
cd "$project_root"
stop_previous_next
.venv/bin/python scripts/export_dashboard_runs.py
.venv/bin/python scripts/local_run_server.py &
runner_pid=$!

cd "$project_root/frontend"
setsid ./node_modules/.bin/next dev &
next_pid=$!
wait "$next_pid"
