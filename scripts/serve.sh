#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:?Pass base, step_30, or step_65}"
RUN_ID="${RUN_ID:-ulvr_vopd_4b_seed42}"
case "$MODEL" in
  base) MODEL_PATH="$ROOT/models/Qwen3.5-4B" ;;
  step_30|step_65) MODEL_PATH="$ROOT/runs/$RUN_ID/merged/$MODEL" ;;
  *) exit 2 ;;
esac
CMD=(vllm serve "$MODEL_PATH" --served-model-name "$MODEL" --port "${PORT:-18280}"
     --tensor-parallel-size "${TP:-1}" --gpu-memory-utilization 0.85
     --max-model-len 65536 --seed 42)
printf '%q ' "${CMD[@]}"; printf '\n'
if [[ "${2:-}" == --execute ]]; then exec "${CMD[@]}"; fi
echo 'PLAN ONLY. Serving requires an explicit --execute.'
