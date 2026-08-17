#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
LOG_ROOT="${ROOT}/logs/fullbody4_unadjusted_baseline_then_guidance_20260720"
mkdir -p "${LOG_ROOT}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd "${ROOT}"

run_mode() {
  local mode="$1"
  echo "[mode-start] ${mode} $(date '+%F %T')"

  python scripts/run_cointeract_unadjusted_fullbody4_cycle.py \
    --mode "${mode}" --npu 2 --shard-index 0 --shard-count 3 \
    > "${LOG_ROOT}/${mode}_runner_npu2.log" 2>&1 &
  local pid2=$!

  python scripts/run_cointeract_unadjusted_fullbody4_cycle.py \
    --mode "${mode}" --npu 6 --shard-index 1 --shard-count 3 \
    > "${LOG_ROOT}/${mode}_runner_npu6.log" 2>&1 &
  local pid6=$!

  python scripts/run_cointeract_unadjusted_fullbody4_cycle.py \
    --mode "${mode}" --npu 7 --shard-index 2 --shard-count 3 \
    > "${LOG_ROOT}/${mode}_runner_npu7.log" 2>&1 &
  local pid7=$!

  wait "${pid2}"
  wait "${pid6}"
  wait "${pid7}"
  echo "[mode-done] ${mode} $(date '+%F %T')"
}

run_mode baseline
run_mode stagea30
echo "[all-done] $(date '+%F %T')"
