#!/usr/bin/env bash
set -euo pipefail
bash RUN_LLM_ONLY_PAPER.sh
bash RUN_QAHS_PAPER.sh
bash RUN_COMPARE.sh
