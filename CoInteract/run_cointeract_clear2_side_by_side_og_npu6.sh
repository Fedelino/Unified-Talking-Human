#!/usr/bin/env bash
set -euo pipefail

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd /data1/workspace/linxinliang/CoInteract

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-6}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

CSV_PATH="/data1/workspace/linxinliang/CoInteract/examples/cointeract_fullbody_clear2_m2v.csv"
OUTPUT_DIR="/data1/workspace/linxinliang/CoInteract/output_videos/side_by_side_clear2_cointeract_og_20260624"
LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/side_by_side_clear2_cointeract_og_20260624.log"
TMP_DIR="/tmp/side_by_side_clear2_cointeract_og_20260624_rows"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_PATH")" "$TMP_DIR"
rm -f "$TMP_DIR"/*.csv 2>/dev/null || true

prepare_row_csvs() {
python - "$CSV_PATH" "$OUTPUT_DIR" "$TMP_DIR" <<'PY'
import csv
import os
import re
import sys

csv_path, output_dir, tmp_dir = sys.argv[1:4]
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

echo "===== CoInteract clear2 side-by-side OG start $(date '+%F %T') =====" | tee -a "$LOG_PATH"

mapfile -t ROW_CSVS < <(prepare_row_csvs)
if [ "${#ROW_CSVS[@]}" -eq 0 ]; then
    echo "[$(date '+%F %T')] all outputs already exist." | tee -a "$LOG_PATH"
    exit 0
fi

for row_csv in "${ROW_CSVS[@]}"; do
    sample_name="$(basename "$row_csv" .csv)"
    echo "[$(date '+%F %T')] running ${sample_name}" | tee -a "$LOG_PATH"
    python batch_infer_qilin_og_m2v.py \
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
        --cfg_scale 7.0 >>"$LOG_PATH" 2>&1
    echo "[$(date '+%F %T')] completed ${sample_name}" | tee -a "$LOG_PATH"
done

echo "===== CoInteract clear2 side-by-side OG end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
