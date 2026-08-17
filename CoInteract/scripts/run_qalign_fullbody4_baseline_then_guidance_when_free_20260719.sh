#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
LOG_ROOT="${ROOT}/logs/qalign_fullbody4_baseline_then_guidance_20260719"
mkdir -p "${LOG_ROOT}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd "${ROOT}"

npu_free() {
  local npu_id="$1"
  npu-smi info | grep -q "No running processes found in NPU ${npu_id}"
}

wait_for_npus() {
  local a="$1"
  local b="$2"
  while true; do
    if npu_free "${a}" && npu_free "${b}"; then
      echo "[npu-free] ${a},${b} at $(date '+%F %T')"
      return 0
    fi
    echo "[wait-npu] ${a},${b} not both free at $(date '+%F %T')"
    sleep 120
  done
}

run_mode() {
  local mode="$1"
  echo "[mode-start] ${mode} at $(date '+%F %T')"
  python scripts/run_cointeract_qalign_fullbody4_cycle.py \
    --mode "${mode}" \
    --npu 6 \
    --shard-index 0 \
    --shard-count 2 \
    > "${LOG_ROOT}/${mode}_runner_npu6.log" 2>&1 &
  local pid6=$!

  python scripts/run_cointeract_qalign_fullbody4_cycle.py \
    --mode "${mode}" \
    --npu 7 \
    --shard-index 1 \
    --shard-count 2 \
    > "${LOG_ROOT}/${mode}_runner_npu7.log" 2>&1 &
  local pid7=$!

  wait "${pid6}"
  wait "${pid7}"
  echo "[mode-done] ${mode} at $(date '+%F %T')"
}

wait_for_npus 6 7
run_mode baseline

wait_for_npus 6 7
run_mode stagea30

echo "[all-done] $(date '+%F %T')"
