#!/usr/bin/env bash
set -euo pipefail
PROVIDER="${PROVIDER:-ollama}"
MODEL="${MODEL:-qwen3:8b}"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"
python check_env.py --provider "$PROVIDER" --model "$MODEL" --ollama-url "$OLLAMA_URL" --require-quantum
python audit_s5_manifest.py --report s5_manifest_audit.json
python -m unittest -v test_s5_production.py
