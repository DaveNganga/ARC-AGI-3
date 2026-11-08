#!/usr/bin/env bash
# Claude run of the collaborative condition. Flags are copied from RUN_COLLABORATIVE_PAPER.sh;
# only --provider/--model and the output paths differ. The output paths must differ: the run
# uses --resume, which would otherwise reuse Qwen's episode files from the same paths.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
: "${MODEL:=claude-opus-5-5}"
: "${ANTHROPIC_EFFORT:=medium}"
export ANTHROPIC_EFFORT
OUT="${OUT:-${MODEL//-/_}}"
python -u run_s5_benchmark.py \
  --condition collaborative \
  --profile paper \
  --experiment all \
  --provider anthropic \
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
  --output-dir "$OUT/collaborative_results" \
  --report "$OUT/s5_16m_collaborative_report.json"
# Four-way comparison against the Claude LLM-only and QAHS runs on the same frozen benchmark.
PREV=../S5_16M_STRONGER_AI_PRODUCTION/${OUT}
if [ -f "$PREV/s5_16m_llm_report.json" ]; then
  python compare_s5_collaborative.py --llm "$PREV/s5_16m_llm_report.json" \
    --baseline "$PREV/s5_16m_qahs_report.json" \
    --collaborative "$OUT/s5_16m_collaborative_report.json" \
    --out "$OUT/s5_16m_collaborative_comparison.json"
fi
