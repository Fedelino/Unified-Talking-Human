#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
LOG_DIR="${ROOT}/logs/qalign_fb01_talkinghuman_motions_skip12_launcher_20260721_npu5"
OUT_NAME="qalign_fb01_talkinghuman_motions_skip12_baseline_20260721"
CSV_PATH="${ROOT}/examples/cointeract_qalign_fb01_talkinghuman_motions_skip12_20260721.csv"
mkdir -p "${LOG_DIR}"

cd "${ROOT}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

echo "[infer-start] $(date '+%F %T')"
ASCEND_RT_VISIBLE_DEVICES=5 python scripts/run_cointeract_qalign_fb01_talkinghuman_motions_cycle.py \
  --npu 5 \
  --shard-index 0 \
  --shard-count 1 \
  --face-reference-guidance-scale 0.0 \
  --csv-path "${CSV_PATH}" \
  --output-name "${OUT_NAME}" \
  > "${LOG_DIR}/infer.log" 2>&1
echo "[infer-done] $(date '+%F %T')"
