#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:-ulvr_vopd_4b_seed42}"
STEP="${2:?Pass checkpoint step 30 or 65}"
[[ "$STEP" == 30 || "$STEP" == 65 ]] || exit 2
SRC="$ROOT/runs/$RUN_ID/checkpoints/global_step_$STEP/actor"
DEST="$ROOT/runs/$RUN_ID/merged/step_$STEP"
[[ -d "$SRC" && ! -e "$DEST" ]] || { echo 'Source absent or destination already exists'; exit 2; }
cd "$ROOT/vendor/Vision-OPD"
python -m verl.model_merger merge --backend fsdp --local_dir "$SRC" --target_dir "$DEST"
