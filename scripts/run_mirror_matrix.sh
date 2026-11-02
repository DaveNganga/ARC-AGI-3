#!/usr/bin/env bash
# Runs the pre-registered Claude mirror matrix (EXPERIMENT_NOTES.md on claude-provider/main).
# After every run: export sanitized traces, commit, and push, so an interrupted session
# loses at most one run. Re-running the script skips (game, seed, budget) already done.
#
# Environment:
#   ARC_API_KEY, ANTHROPIC_API_KEY   required for --source arc
#   ANTHROPIC_EFFORT                 default medium
#   MIRROR_REMOTE                    git remote to push to (default origin; empty = no push)
#   MIRROR_SOURCE                    arc (default) or demo, for testing the loop
#   MIRROR_MATRIX                    override "game seed budget;..." (testing only)
set -uo pipefail
cd "$(dirname "$0")/.."

export ANTHROPIC_EFFORT="${ANTHROPIC_EFFORT:-medium}"
SOURCE="${MIRROR_SOURCE:-arc}"
REMOTE="${MIRROR_REMOTE-origin}"
PY="${PY:-.venv/bin/python}"
QAHS="${QAHS:-.venv/bin/qahs-arc}"
if [ "$SOURCE" = demo ]; then MODEL="Deterministic local reasoner"; else MODEL="${ANTHROPIC_MODEL:-claude-opus-5-5}"; fi
MATRIX="${MIRROR_MATRIX:-ar25 0 20;vc33 0 30;lp85 0 30;r11l 0 20;s5i5 0 20;vc33 1 30;ar25 1 20;r11l 1 20;s5i5 1 20}"
mkdir -p mirror_logs output/arc_runs

already_done() {  # game seed budget -> exit 0 if a trace from $MODEL exists
  "$PY" - "$MODEL" "$@" <<'EOF'
import json, sys
from pathlib import Path
model, game, seed, budget = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
for path in Path("frontend/public/runs").glob("*_comparison.json"):
    d = json.loads(path.read_text(encoding="utf-8"))
    if d.get("model") == model and d.get("game_id") == game and d.get("seed") == seed and d.get("action_budget") == budget:
        sys.exit(0)
sys.exit(1)
EOF
}

redact() {  # strip secret values from a log file
  "$PY" - "$1" <<'EOF'
import os, sys
path = sys.argv[1]
text = open(path, encoding="utf-8", errors="replace").read()
for name in ("ARC_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
    value = os.environ.get(name)
    if value:
        text = text.replace(value, "[redacted]")
open(path, "w", encoding="utf-8").write(text)
EOF
}

publish() {  # message
  git add frontend/public/runs mirror_logs
  git diff --cached --quiet && return 0
  git commit -q -m "$1" || return 0
  [ -z "$REMOTE" ] && return 0
  for attempt in 1 2 3; do
    git push -q "$REMOTE" HEAD && return 0
    echo "push failed (attempt $attempt); retrying in 30s" >&2
    sleep 30
  done
  echo "WARNING: could not push; results are committed locally" >&2
}

IFS=';' read -ra RUNS <<< "$MATRIX"
for run in "${RUNS[@]}"; do
  read -r game seed budget <<< "$run"
  if already_done "$game" "$seed" "$budget"; then
    echo "skip $game seed=$seed budget=$budget (already done)"
    continue
  fi
  log="mirror_logs/${game}_s${seed}_b${budget}_$(date -u +%Y%m%dT%H%M%SZ).log"
  echo "run  $game seed=$seed budget=$budget -> $log"
  before=$(ls output/arc_runs/*.json 2>/dev/null | wc -l)
  $QAHS --source "$SOURCE" --provider anthropic --game "$game" --seed "$seed" \
    --max-steps "$budget" --condition compare >"$log" 2>&1
  status=$?
  echo "exit_status=$status" >>"$log"
  redact "$log"
  after=$(ls output/arc_runs/*.json 2>/dev/null | wc -l)
  if [ "$after" -le "$before" ]; then
    # No trace means a crash outside the episodes (bad key, missing package, outage).
    publish "Mirror run crashed: $game seed $seed (no trace)"
    echo "ABORT: $game seed=$seed produced no trace; see $log" >&2
    tail -20 "$log" >&2
    exit 1
  fi
  "$PY" scripts/export_dashboard_runs.py >>"$log" 2>&1
  publish "Mirror run: $game seed $seed budget $budget ($MODEL)"
done

"$PY" scripts/mirror_compare.py > mirror_logs/compare.md
publish "Mirror study comparison table"
echo "done; comparison in mirror_logs/compare.md"
