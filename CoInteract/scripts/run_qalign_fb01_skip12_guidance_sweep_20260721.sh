#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
CSV_PATH="${ROOT}/examples/cointeract_qalign_fb01_talkinghuman_motions_skip12_20260721.csv"
LAUNCH_LOG_DIR="${ROOT}/logs/qalign_fb01_skip12_guidance_sweep_launcher_20260721"
mkdir -p "${LAUNCH_LOG_DIR}"

cd "${ROOT}"
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

run_scale() {
  local scale="$1"
  local scale_name="$2"
  local npu="$3"
  local output_name="qalign_fb01_talkinghuman_motions_skip12_guidance_${scale_name}_20260721"
  local log_path="${LAUNCH_LOG_DIR}/${output_name}_npu${npu}.log"
  echo "[start] scale=${scale} npu=${npu} output=${output_name} $(date '+%F %T')"
  ASCEND_RT_VISIBLE_DEVICES="${npu}" python scripts/run_cointeract_qalign_fb01_talkinghuman_motions_cycle.py \
    --npu "${npu}" \
    --shard-index 0 \
    --shard-count 1 \
    --face-reference-guidance-scale "${scale}" \
    --csv-path "${CSV_PATH}" \
    --output-name "${output_name}" \
    > "${log_path}" 2>&1
  echo "[done] scale=${scale} npu=${npu} output=${output_name} $(date '+%F %T')"
}

run_scale "1.0" "scale1p0" "1" &
pid1=$!
run_scale "3.0" "scale3p0" "2" &
pid2=$!
run_scale "5.0" "scale5p0" "5" &
pid3=$!
run_scale "7.0" "scale7p0" "7" &
pid4=$!

wait "${pid1}"
wait "${pid2}"
wait "${pid3}"
wait "${pid4}"

echo "[all-done] $(date '+%F %T')"
