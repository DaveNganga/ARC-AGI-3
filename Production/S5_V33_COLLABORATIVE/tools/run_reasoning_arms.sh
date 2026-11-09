#!/usr/bin/env bash
# Run one collaborative arm through the rewriting proxy.
#
#   ARM=control   think left False, leading_rules families required  -> isolates the schema change
#   ARM=reasoning think rewritten to True, same required schema      -> isolates reasoning on top
#
# RUN_COLLABORATIVE_PAPER.sh cannot be reused: it never passes --ollama-url, so it would talk to
# Ollama directly and bypass the proxy. The flag list below is copied from that launcher verbatim
# (lines 7-29) so the only differences from the published run are the two proxy rewrites.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

: "${ARM:=control}"
: "${PORT:=}"
: "${OUT:=}"
case "$ARM" in
  control)   THINK_FLAG="";        PORT="${PORT:-11436}"; OUT="${OUT:-qwen3_8b_reqschema}" ;;
  reasoning) THINK_FLAG="--think"; PORT="${PORT:-11435}"; OUT="${OUT:-qwen3_8b_reasoning}" ;;
  *) echo "ARM must be control or reasoning" >&2; exit 2 ;;
esac

mkdir -p "$OUT"
LOG="$OUT/reasoning_proxy.jsonl"

# The transfer block must complete in one pass. run_s5_benchmark.py:249 reloads the transfer
# source's QAHS state only for --condition qahs, so a collaborative run resumed mid-transfer would
# run the warm episodes without the transferred posterior and change the Transfer score. Same guard
# RUN_COLLABORATIVE_CLAUDE.sh applies; the paper launcher has none.
TRANSFER="$OUT/collaborative_results/collaborative/transfer"
if [ -f "$TRANSFER/source.json" ]; then
  if [ "$(find "$TRANSFER" -maxdepth 1 -name '*.json' | wc -l)" -ne 7 ]; then
    echo "Incomplete transfer block in $TRANSFER: deleting it so it reruns in one pass." >&2
    rm -rf "$TRANSFER"
  fi
fi

python -u tools/reasoning_proxy.py --port "$PORT" $THINK_FLAG --log "$LOG" &
PROXY=$!
trap 'kill "$PROXY" 2>/dev/null || true' EXIT
sleep 3
curl -s "http://127.0.0.1:$PORT/api/tags" >/dev/null || { echo "proxy not answering on $PORT" >&2; exit 1; }
echo "ARM=$ARM proxy pid $PROXY on port $PORT, log $LOG"

python -u run_s5_benchmark.py \
  --condition collaborative \
  --profile paper \
  --experiment all \
  --provider ollama \
  --model qwen3:8b \
  --ollama-url "http://127.0.0.1:$PORT" \
  --max-steps 12 \
  --qahs-particles 384 \
  --qahs-eval-particles 64 \
  --qaoa-reps 1 \
  --qaoa-shots 256 \
  --qaoa-optimizer-maxiter 12 \
  --plan-horizon 8 \
  --beam-width 48 \
  --llm-weight 0.20 \
  --arbitration-margin 0.15 \
  --diagnostic-threshold 0.12 \
  --llm-confidence-weight 0.20 \
  --request-timeout 900 \
  --resume \
  --require-quantum \
  --output-dir "$OUT/collaborative_results" \
  --report "$OUT/s5_16m_collaborative_report.json"
