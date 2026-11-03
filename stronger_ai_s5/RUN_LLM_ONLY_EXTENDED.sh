#!/usr/bin/env bash
set -euo pipefail
PROVIDER="${PROVIDER:-ollama}"; MODEL="${MODEL:-qwen3:8b}"; OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
python -u run_s5_benchmark.py --condition llm-only --profile extended --experiment all --provider "$PROVIDER" --model "$MODEL" --ollama-url "$OLLAMA_URL" --max-steps 12 --request-timeout 900 --resume --output-dir s5_16m_extended_results --report s5_16m_llm_extended_report.json
