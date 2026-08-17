#!/usr/bin/env bash
set -euo pipefail

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd /data1/workspace/linxinliang/CoInteract

export TOKENIZERS_PARALLELISM=false
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-7}"
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

LORA_PATH="${LORA_PATH:-/data1/workspace/linxinliang/CoInteract/output/ubcfashion_tiktok_pose_full/version_0/step-500.safetensors}"
OUTPUT_DIR="/data1/workspace/linxinliang/CoInteract/output_videos/finetune_stage1_step500_m2v_eval_20260710"
LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/finetune_stage1_step500_m2v_eval_20260710.log"
TMP_DIR="/tmp/finetune_stage1_step500_m2v_eval_20260710_rows"

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_PATH")" "$TMP_DIR"
rm -f "$TMP_DIR"/*.csv

python - "$TMP_DIR" <<'PY'
import csv
import re
import sys
from pathlib import Path

tmp_dir = Path(sys.argv[1])
cases = [
    (Path("examples/cointeract_fullbody4_motion2_m2v_prompted_20260628.csv"), 0),
    (Path("examples/cointeract_halfbody4_betterpose_720p_20260626.csv"), 0),
]

def safe(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", text)

for csv_path, row_index in cases:
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames
    row = rows[row_index]
    sample_id = f"{row['sample_id'].strip()}_stage1_step500"
    row["sample_id"] = sample_id
    out_path = tmp_dir / f"{safe(sample_id)}.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(out_path)
PY

echo "===== Stage1 step500 M2V eval start $(date '+%F %T') =====" | tee -a "$LOG_PATH"
echo "visible_npu=${ASCEND_RT_VISIBLE_DEVICES}" | tee -a "$LOG_PATH"
echo "lora_path=${LORA_PATH}" | tee -a "$LOG_PATH"

for row_csv in "$TMP_DIR"/*.csv; do
    sample_name="$(basename "$row_csv" .csv)"
    echo "[$(date '+%F %T')] running ${sample_name}" | tee -a "$LOG_PATH"
    python batch_infer_qilin_og_m2v.py \
        --base_model_path /data1/Wan-AI/wan22_s2v \
        --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
        --lora_path "$LORA_PATH" \
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
    sleep 60
done

echo "===== Stage1 step500 M2V eval end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
