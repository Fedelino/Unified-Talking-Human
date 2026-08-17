#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <npu_id> <shard_index> <shard_count>" >&2
  exit 2
fi

NPU_ID="$1"
SHARD_INDEX="$2"
SHARD_COUNT="$3"

ROOT="/data1/workspace/linxinliang/CoInteract"
CSV_PATH="${ROOT}/examples/cointeract_fullbody4_aligned_motions_20260718.csv"
OUT_DIR="${ROOT}/output_videos/fullbody4_aligned_posefreq_baseline_20260718"
LOG_DIR="${ROOT}/logs/fullbody4_aligned_posefreq_baseline_20260718"
TMP_DIR="${ROOT}/tmp/fullbody4_aligned_posefreq_baseline_20260718/shard_${SHARD_INDEX}_of_${SHARD_COUNT}"

mkdir -p "${OUT_DIR}" "${LOG_DIR}" "${TMP_DIR}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

cd "${ROOT}"

python - "${CSV_PATH}" "${TMP_DIR}" "${SHARD_INDEX}" "${SHARD_COUNT}" <<'PY' | while IFS=$'\t' read -r sample_id one_csv; do
import csv
import os
import sys

csv_path, tmp_dir, shard_index, shard_count = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
os.makedirs(tmp_dir, exist_ok=True)

with open(csv_path, newline="") as f:
    rows = list(csv.DictReader(f))

for i, row in enumerate(rows):
    if i % shard_count != shard_index:
        continue
    sample_id = row["sample_id"]
    one_csv = os.path.join(tmp_dir, f"{sample_id}.csv")
    with open(one_csv, "w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerow(row)
    print(f"{sample_id}\t{one_csv}")
PY
  out_video="${OUT_DIR}/${sample_id}.mp4"
  case_log="${LOG_DIR}/${sample_id}_npu${NPU_ID}.log"

  if [[ -s "${out_video}" ]]; then
    echo "[skip] ${sample_id}: output exists"
    continue
  fi

  echo "[run] npu=${NPU_ID} shard=${SHARD_INDEX}/${SHARD_COUNT} sample=${sample_id}"
  set +e
  ASCEND_RT_VISIBLE_DEVICES="${NPU_ID}" \
  TOKENIZERS_PARALLELISM=false \
  python batch_infer.py \
    --csv_path "${one_csv}" \
    --output_dir "${OUT_DIR}" \
    --height 832 \
    --width 480 \
    --num_frames 80 \
    --num_clips 1 \
    --num_inference_steps 40 \
    --cfg_scale 7.0 \
    --sigma_shift 7.0 \
    --reference_compose_mode stretch \
    --pose_align_mode bbox \
    --pose_align_target_fps 25.0 \
    --face_reference_guidance_scale 0.0 \
    > "${case_log}" 2>&1
  status=$?
  set -e
  echo "[done] sample=${sample_id} status=${status} log=${case_log}"
  sleep 8
done
