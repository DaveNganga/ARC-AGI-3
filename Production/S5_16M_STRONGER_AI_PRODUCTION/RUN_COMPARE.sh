#!/usr/bin/env bash
set -euo pipefail
LLM_REPORT="${LLM_REPORT:-s5_16m_llm_report.json}"
QAHS_REPORT="${QAHS_REPORT:-s5_16m_qahs_report.json}"
OUT="${OUT:-s5_16m_comparison.json}"
python compare_s5_reports.py --llm "$LLM_REPORT" --qahs "$QAHS_REPORT" --out "$OUT"
