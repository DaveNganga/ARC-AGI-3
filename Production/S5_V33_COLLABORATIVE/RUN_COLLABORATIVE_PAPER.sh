#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
: "${PROVIDER:=ollama}"
: "${MODEL:=qwen3:8b}"
python -u run_s5_benchmark.py \
  --condition collaborative \
  --profile paper \
  --experiment all \
  --provider "$PROVIDER" \
  --model "$MODEL" \
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
  --output-dir s5_16m_collaborative_results \
  --report s5_16m_collaborative_report.json
