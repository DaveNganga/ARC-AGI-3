#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "$0")" && pwd)"
next_lock="$project_root/frontend/.next/dev/lock"
runner_pid=""
next_pid=""
runner_port=""

choose_runner_port() {
  "$project_root/.venv/bin/python" -c '
import socket
for port in range(8765, 8800):
    with socket.socket() as candidate:
        try:
            candidate.bind(("127.0.0.1", port))
        except OSError:
            continue
        print(port)
        break
else:
    raise SystemExit("No local experiment port is available between 8765 and 8799.")
'
}

stop_previous_runner() {
  local stale_pid stale_cwd
  while read -r stale_pid; do
    [[ "$stale_pid" =~ ^[0-9]+$ ]] || continue
    stale_cwd="$(readlink -f "/proc/$stale_pid/cwd" 2>/dev/null || true)"
    [[ "$stale_cwd" == "$project_root" ]] || continue
    echo "Stopping previous local experiment service (PID $stale_pid)..."
    kill "$stale_pid" 2>/dev/null || true
    for _ in {1..30}; do
      kill -0 "$stale_pid" 2>/dev/null || break
      sleep 0.1
    done
  done < <(pgrep -f '[s]cripts/local_run_server.py' 2>/dev/null || true)
}

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
stop_previous_runner
stop_previous_next
runner_port="$(choose_runner_port)"
export QAHS_RUNNER_PORT="$runner_port"
export NEXT_PUBLIC_QAHS_RUNNER_URL="http://127.0.0.1:$runner_port"
echo "Using local experiment port $runner_port."
.venv/bin/python scripts/export_dashboard_runs.py
.venv/bin/python scripts/local_run_server.py &
runner_pid=$!
sleep 0.2
if ! kill -0 "$runner_pid" 2>/dev/null; then
  wait "$runner_pid"
fi

cd "$project_root/frontend"
setsid ./node_modules/.bin/next dev &
next_pid=$!
wait "$next_pid"
