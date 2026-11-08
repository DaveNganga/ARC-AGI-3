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
# Resume guard. On --resume, run_s5_benchmark.py reloads the transfer source's QAHS state only
# for --condition qahs (line 248), so a collaborative run resumed between the transfer source and
# its warm episodes would run those warm episodes without the transferred posterior. If the
# transfer block is incomplete, delete it so it reruns as one block, exactly as a fresh run does.
TRANSFER="$OUT/collaborative_results/collaborative/transfer"
if [ -f "$TRANSFER/source.json" ]; then
  if ! python - "$TRANSFER" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
m = json.load(open("benchmark_manifest.json"))
ids = [t["id"] for t in m["transfer"]["targets"][: int(m["profiles"]["paper"]["transfer"])]]
sys.exit(0 if all((d / f"{i}_{k}.json").exists() for i in ids for k in ("cold", "warm")) else 1)
PY
  then
    echo "Incomplete transfer block in $TRANSFER: deleting it so it reruns in one pass." >&2
    rm -rf "$TRANSFER"
  fi
fi
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
  # compare_s5_collaborative.py checks the benchmark and profile but not the model or metric
  # spec, so check those here before pairing the three reports.
  python - "$PREV/s5_16m_llm_report.json" "$PREV/s5_16m_qahs_report.json" "$OUT/s5_16m_collaborative_report.json" <<'PY'
import json, sys
reports = [json.load(open(p)) for p in sys.argv[1:]]
keys = ("provider", "model", "metric_spec_sha256", "manifest_sha256", "profile")
seen = {k: {r.get(k) for r in reports} for k in keys}
bad = {k: v for k, v in seen.items() if len(v) != 1}
if bad:
    sys.exit(f"Reports are not from the same model and benchmark: {bad}")
PY
  python compare_s5_collaborative.py --llm "$PREV/s5_16m_llm_report.json" \
    --baseline "$PREV/s5_16m_qahs_report.json" \
    --collaborative "$OUT/s5_16m_collaborative_report.json" \
    --out "$OUT/s5_16m_collaborative_comparison.json"
fi
