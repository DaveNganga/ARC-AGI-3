#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
npm ci --prefix frontend
.venv/bin/python scripts/export_dashboard_runs.py
echo "Setup complete. Start the dashboard with: bash run_dashboard.sh"
