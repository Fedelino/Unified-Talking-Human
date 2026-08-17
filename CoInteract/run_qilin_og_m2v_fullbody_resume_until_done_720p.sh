#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

export ASCEND_RT_VISIBLE_DEVICES=7

CSV_PATH="/data1/workspace/linxinliang/CoInteract/examples/m2v_og/cointeract_qilin_og_m2v_fullbody.csv"
OUTPUT_DIR="/data1/workspace/linxinliang/CoInteract/output_videos/qilin_og_m2v_fullbody_720p_20260622"
LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/qilin_og_m2v_fullbody_720p_20260622_resume.log"
TMP_CSV="/tmp/cointeract_qilin_og_m2v_fullbody_missing.csv"
RUN_SIG="batch_infer_qilin_og_m2v.py.*--output_dir ${OUTPUT_DIR}"

all_done() {
python - "$CSV_PATH" "$OUTPUT_DIR" <<'PY'
import csv
import os
import sys

csv_path, output_dir = sys.argv[1], sys.argv[2]
with open(csv_path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

missing = []
for row in rows:
    sample_id = row["sample_id"].strip()
    if not os.path.exists(os.path.join(output_dir, f"{sample_id}.mp4")):
        missing.append(sample_id)

if missing:
    print("MISSING", *missing)
    raise SystemExit(1)
print("DONE")
PY
}

write_missing_csv() {
python - "$CSV_PATH" "$OUTPUT_DIR" "$TMP_CSV" <<'PY'
import csv
import os
import sys

csv_path, output_dir, tmp_csv = sys.argv[1:4]
with open(csv_path, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames
    rows = [row for row in reader if not os.path.exists(os.path.join(output_dir, f"{row['sample_id'].strip()}.mp4"))]

with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(tmp_csv)
print(f"rows={len(rows)}")
PY
}

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_PATH")"

while true; do
    if pgrep -f "$RUN_SIG" >/dev/null; then
        echo "[$(date '+%F %T')] fullbody OG run still active; waiting..." | tee -a "$LOG_PATH"
        sleep 120
        continue
    fi

    if all_done >>"$LOG_PATH" 2>&1; then
        echo "[$(date '+%F %T')] all fullbody OG outputs are present." | tee -a "$LOG_PATH"
        exit 0
    fi

    echo "[$(date '+%F %T')] resuming missing fullbody OG samples..." | tee -a "$LOG_PATH"
    write_missing_csv >>"$LOG_PATH" 2>&1

    python batch_infer_qilin_og_m2v.py \
        --base_model_path /data1/Wan-AI/wan22_s2v \
        --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
        --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors \
        --csv_path "$TMP_CSV" \
        --data_base_path /data1/workspace/linxinliang/CoInteract \
        --output_dir "$OUTPUT_DIR" \
        --height 1280 \
        --width 720 \
        --num_frames 80 \
        --num_clips 1 \
        --num_inference_steps 40 \
        --cfg_scale 7.0 >>"$LOG_PATH" 2>&1
done
