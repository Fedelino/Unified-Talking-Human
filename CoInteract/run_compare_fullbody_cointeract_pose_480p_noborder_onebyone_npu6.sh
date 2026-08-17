#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES=6
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

CSV_PATH="/data1/workspace/linxinliang/CoInteract/examples/compare_fullbody_talkinghuman_480p_noborder.csv"
OUTPUT_DIR="/data1/workspace/linxinliang/CoInteract/output_videos/compare_fullbody_cointeract_pose_20260624_480p_noborder"
LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/compare_fullbody_cointeract_pose_20260624_480p_noborder.log"
TMP_DIR="/tmp/compare_fullbody_cointeract_pose_20260624_480p_noborder_rows"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_PATH")" "$TMP_DIR"

prepare_row_csvs() {
python - "$CSV_PATH" "$OUTPUT_DIR" "$TMP_DIR" <<'PY'
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
}

echo "===== compare fullbody cointeract 480p noborder start $(date '+%F %T') =====" | tee -a "$LOG_PATH"

mapfile -t ROW_CSVS < <(prepare_row_csvs)
if [ "${#ROW_CSVS[@]}" -eq 0 ]; then
    echo "[$(date '+%F %T')] all outputs already exist." | tee -a "$LOG_PATH"
    exit 0
fi

for row_csv in "${ROW_CSVS[@]}"; do
    sample_name="$(basename "$row_csv" .csv)"
    attempt=1
    while [ "$attempt" -le 3 ]; do
        echo "[$(date '+%F %T')] running ${sample_name}, attempt ${attempt}" | tee -a "$LOG_PATH"
        if python batch_infer.py \
            --base_model_path /data1/Wan-AI/wan22_s2v \
            --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
            --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
            --csv_path "$row_csv" \
            --data_base_path /data1/workspace/linxinliang/CoInteract \
            --output_dir "$OUTPUT_DIR" \
            --height 832 \
            --width 480 \
            --num_frames 80 \
            --num_clips 1 \
            --num_inference_steps 40 \
            --cfg_scale 7.0 >>"$LOG_PATH" 2>&1; then
            echo "[$(date '+%F %T')] completed ${sample_name}" | tee -a "$LOG_PATH"
            break
        fi
        echo "[$(date '+%F %T')] failed ${sample_name} on attempt ${attempt}; cooling down 180s" | tee -a "$LOG_PATH"
        sleep 180
        attempt=$((attempt + 1))
    done

    if [ "$attempt" -gt 3 ]; then
        echo "[$(date '+%F %T')] giving up on ${sample_name} after 3 attempts" | tee -a "$LOG_PATH"
    fi
done

echo "===== compare fullbody cointeract 480p noborder end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
