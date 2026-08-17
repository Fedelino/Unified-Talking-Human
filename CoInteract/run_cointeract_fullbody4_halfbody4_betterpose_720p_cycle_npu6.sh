#!/usr/bin/env bash
set -euo pipefail

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd /data1/workspace/linxinliang/CoInteract

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

FULLBODY_CSV="/data1/workspace/linxinliang/CoInteract/examples/cointeract_fullbody4_betterpose_720p_20260626.csv"
HALFBODY_CSV="/data1/workspace/linxinliang/CoInteract/examples/cointeract_halfbody4_betterpose_720p_20260626.csv"
FULLBODY_OUT="/data1/workspace/linxinliang/CoInteract/output_videos/side_by_side_fullbody4_cointeract_og_betterpose_720p_20260626"
HALFBODY_OUT="/data1/workspace/linxinliang/CoInteract/output_videos/side_by_side_halfbody4_cointeract_og_betterpose_720p_20260626"
LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/side_by_side_fullbody4_halfbody4_cointeract_og_betterpose_720p_cycle_20260626.log"
TMP_ROOT="/tmp/side_by_side_fullbody4_halfbody4_cointeract_og_betterpose_720p_cycle_20260626"
COOLDOWN_SECONDS=120
MAX_ATTEMPTS=3

mkdir -p "$FULLBODY_OUT" "$HALFBODY_OUT" "$(dirname "$LOG_PATH")" "$TMP_ROOT"

run_csv_cycle() {
  local csv_path="$1"
  local output_dir="$2"
  local tmp_dir="$3"
  local label="$4"

  mkdir -p "$output_dir" "$tmp_dir"

  mapfile -t ROW_CSVS < <(python - "$csv_path" "$output_dir" "$tmp_dir" <<'PY'
import csv
import os
import re
import sys

csv_path, output_dir, tmp_dir = sys.argv[1:4]
os.makedirs(tmp_dir, exist_ok=True)
for name in os.listdir(tmp_dir):
    path = os.path.join(tmp_dir, name)
    if os.path.isfile(path):
        os.remove(path)

with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = list(reader)

def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)

for idx, row in enumerate(rows, start=1):
    sample_id = row["sample_id"].strip()
    output_path = os.path.join(output_dir, f"{sample_id}.mp4")
    if os.path.exists(output_path):
        continue
    row_csv = os.path.join(tmp_dir, f"{idx:02d}_{safe_name(sample_id)}.csv")
    with open(row_csv, "w", newline="", encoding="utf-8") as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(row_csv)
PY
)

  if [ "${#ROW_CSVS[@]}" -eq 0 ]; then
      echo "[$(date '+%F %T')] ${label}: all outputs already exist." | tee -a "$LOG_PATH"
      return
  fi

  for row_csv in "${ROW_CSVS[@]}"; do
      sample_name="$(basename "$row_csv" .csv)"
      attempt=1
      while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
          echo "[$(date '+%F %T')] ${label}: running ${sample_name}, attempt ${attempt}" | tee -a "$LOG_PATH"
          if python batch_infer_qilin_og_m2v.py \
              --base_model_path /data1/Wan-AI/wan22_s2v \
              --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
              --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
              --csv_path "$row_csv" \
              --data_base_path /data1/workspace/linxinliang/CoInteract \
              --output_dir "$output_dir" \
              --height 1280 \
              --width 720 \
              --num_frames 80 \
              --num_clips 1 \
              --num_inference_steps 40 \
              --cfg_scale 7.0 >>"$LOG_PATH" 2>&1; then
              echo "[$(date '+%F %T')] ${label}: completed ${sample_name}" | tee -a "$LOG_PATH"
              break
          fi
          echo "[$(date '+%F %T')] ${label}: failed ${sample_name} on attempt ${attempt}; cooling down ${COOLDOWN_SECONDS}s" | tee -a "$LOG_PATH"
          sleep "$COOLDOWN_SECONDS"
          attempt=$((attempt + 1))
      done

      if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
          echo "[$(date '+%F %T')] ${label}: giving up on ${sample_name} after ${MAX_ATTEMPTS} attempts" | tee -a "$LOG_PATH"
      fi

      echo "[$(date '+%F %T')] ${label}: per-case cooldown ${COOLDOWN_SECONDS}s before next launch" | tee -a "$LOG_PATH"
      sleep "$COOLDOWN_SECONDS"
  done
}

echo "===== CoInteract fullbody4+halfbody4 betterpose 720p cycle start $(date '+%F %T') =====" | tee -a "$LOG_PATH"
echo "visible_npu=${ASCEND_RT_VISIBLE_DEVICES}" | tee -a "$LOG_PATH"

run_csv_cycle "$FULLBODY_CSV" "$FULLBODY_OUT" "$TMP_ROOT/fullbody" "fullbody4_betterpose_720p"
run_csv_cycle "$HALFBODY_CSV" "$HALFBODY_OUT" "$TMP_ROOT/halfbody" "halfbody4_betterpose_720p"

echo "===== CoInteract fullbody4+halfbody4 betterpose 720p cycle end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
