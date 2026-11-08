#!/usr/bin/env bash
set -euo pipefail
PROVIDER="${PROVIDER:-ollama}"
MODEL="${MODEL:-qwen3:8b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
COMMON=(--profile smoke --experiment smoke --provider "$PROVIDER" --model "$MODEL" --ollama-url "$OLLAMA_URL" --max-steps 12 --request-timeout 900 --show-grid --resume)
echo '=== LLM-ONLY S5/16M SMOKE ==='
python -u run_s5_benchmark.py --condition llm-only "${COMMON[@]}" --output-dir s5_16m_smoke --report s5_16m_smoke_llm.json
echo '=== QAHS V3.2-S5 24-QUBIT SMOKE ==='
python -u run_s5_benchmark.py --condition qahs "${COMMON[@]}" --require-quantum --qahs-particles 256 --qahs-eval-particles 48 --qaoa-shots 256 --output-dir s5_16m_smoke --report s5_16m_smoke_qahs.json
