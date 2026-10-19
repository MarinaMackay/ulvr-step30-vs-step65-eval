#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="${1:?Pass base, step_30 or step_65}"
case "$TAG" in base|step_30|step_65) ;; *) exit 2 ;; esac
MODE="${2:-plan}"
RUN_ID="${RUN_ID:-ulvr_vopd_4b_seed42}"
CKPT="$ROOT/models/Qwen3.5-4B"
[[ "$TAG" == base ]] || CKPT="$ROOT/runs/$RUN_ID/merged/$TAG"
SIX='vstar,zoombench,hrbench-4k,hrbench-8k,mme-realworld,mme-realworld-cn'
SIX="${EVAL_BENCHMARKS:-$SIX}"
if [[ "$MODE" == plan ]]; then
  echo "Full evaluation for $TAG: $SIX plus separate ULVR validation (2558)."
  echo "Set API_BASE, OPENAI_MODEL_ID, JUDGE_API_BASE, JUDGE_MODEL; then use --infer-only, --judge-only or --execute."
  exit 0
fi
[[ "$MODE" == --execute || "$MODE" == --infer-only || "$MODE" == --judge-only ]] || exit 2
: "${API_BASE:?Set actual model API}"
: "${OPENAI_MODEL_ID:?Set actual served model ID}"
if [[ "$MODE" != --infer-only ]]; then
  : "${JUDGE_MODEL:?Set Qwen2.5-72B-Instruct model ID}"
  : "${JUDGE_API_BASE:?Set judge API; use same endpoint for all three checkpoints}"
fi
[[ "$OPENAI_MODEL_ID" == "$TAG" ]] || { echo 'Served model ID must match tag'; exit 2; }
EXTRA=()
[[ "$MODE" == --execute ]] || EXTRA+=("$MODE")
BENCH_GROUPS=("$SIX")
[[ "${SIX_ONLY:-0}" == 1 ]] || BENCH_GROUPS+=(ulvr-validation)
for BENCHES in "${BENCH_GROUPS[@]}"; do
  python "$ROOT/eval-kit/run.py" --tag "$TAG" --checkpoint "$CKPT" \
    --benchmarks "$BENCHES" --data-dir "$ROOT/data/benchmarks" \
    --runs-dir "$ROOT/runs/$RUN_ID/eval" --max-tokens 32768 --workers "${WORKERS:-2}" --thinking False "${EXTRA[@]}"
done
