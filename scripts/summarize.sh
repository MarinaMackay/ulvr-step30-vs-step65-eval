#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${RUN_ID:-ulvr_vopd_4b_seed42}"
python "$ROOT/eval-kit/summarize.py" --eval-dir "$ROOT/runs/$RUN_ID/eval" \
  --models base step_30 step_65 --out "$ROOT/reports/$RUN_ID/six-benchmarks" --sample 3 --seed 42
python "$ROOT/eval-kit/summarize.py" --eval-dir "$ROOT/runs/$RUN_ID/eval" \
  --models base step_30 step_65 --benchmarks ulvr-validation \
  --out "$ROOT/reports/$RUN_ID/ulvr-validation" --sample 3 --seed 42
