#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
LOG_DIR="${ROOT}/logs/qalign_fb01_talkinghuman_motions_launcher_20260721_npu5"
mkdir -p "${LOG_DIR}"

cd "${ROOT}"

echo "[extract-start] $(date '+%F %T')"
ASCEND_RT_VISIBLE_DEVICES=5 bash scripts/run_extract_qalign_fb01_talkinghuman_motions_20260721.sh \
  > "${LOG_DIR}/extract.log" 2>&1
echo "[extract-done] $(date '+%F %T')"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

echo "[infer-start] $(date '+%F %T')"
ASCEND_RT_VISIBLE_DEVICES=5 python scripts/run_cointeract_qalign_fb01_talkinghuman_motions_cycle.py \
  --npu 5 \
  --shard-index 0 \
  --shard-count 1 \
  --face-reference-guidance-scale 0.0 \
  > "${LOG_DIR}/infer.log" 2>&1
echo "[infer-done] $(date '+%F %T')"
