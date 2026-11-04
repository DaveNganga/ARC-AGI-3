#!/usr/bin/env bash
set -euo pipefail
PROVIDER="${PROVIDER:-ollama}"; MODEL="${MODEL:-qwen3:8b}"; OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
python -u run_s5_benchmark.py --condition qahs --profile extended --experiment all --provider "$PROVIDER" --model "$MODEL" --ollama-url "$OLLAMA_URL" --max-steps 12 --request-timeout 900 --resume --require-quantum --qahs-particles 384 --qahs-eval-particles 64 --qaoa-reps 1 --qaoa-shots 256 --qaoa-optimizer-maxiter 12 --plan-horizon 8 --beam-width 48 --llm-weight 0.35 --output-dir s5_16m_extended_results --report s5_16m_qahs_extended_report.json
