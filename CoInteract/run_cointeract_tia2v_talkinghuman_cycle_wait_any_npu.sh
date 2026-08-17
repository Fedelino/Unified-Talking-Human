#!/usr/bin/env bash
set -euo pipefail

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate interact_avatar

cd /data1/workspace/linxinliang/CoInteract

export TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128

CSV_PATH="/data1/workspace/linxinliang/CoInteract/examples/cointeract_tia2v_talkinghuman_20260629.csv"
OUTPUT_DIR="/data1/workspace/linxinliang/CoInteract/output_videos/tia2v_talkinghuman_cointeract_og_20260629"
LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/tia2v_talkinghuman_cointeract_og_cycle_20260629.log"
TMP_ROOT="/tmp/tia2v_talkinghuman_cointeract_og_cycle_20260629"
COOLDOWN_SECONDS=90
WAIT_SECONDS=120
MAX_ATTEMPTS=3
CANDIDATE_NPUS=(0 1 2 3 4 5 6 7)

mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG_PATH")" "$TMP_ROOT"

mapfile -t ROW_CSVS < <(python - "$CSV_PATH" "$OUTPUT_DIR" "$TMP_ROOT" <<'PY'
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

pick_free_npu() {
    local table
    table="$(npu-smi info 2>/dev/null || true)"
    for npu_id in "${CANDIDATE_NPUS[@]}"; do
        if grep -q "No running processes found in NPU ${npu_id}" <<<"$table"; then
            echo "$npu_id"
            return 0
        fi
    done
    return 1
}

wait_for_any_free_npu() {
    while true; do
        local free_npu
        if free_npu="$(pick_free_npu)"; then
            echo "$free_npu"
            return 0
        fi
        echo "[$(date '+%F %T')] tia2v_talkinghuman: waiting for any free NPU among ${CANDIDATE_NPUS[*]}" | tee -a "$LOG_PATH"
        sleep "$WAIT_SECONDS"
    done
}

echo "===== CoInteract T+A+I2V TalkingHuman cycle start $(date '+%F %T') =====" | tee -a "$LOG_PATH"
echo "candidate_npus=${CANDIDATE_NPUS[*]}" | tee -a "$LOG_PATH"

if [ "${#ROW_CSVS[@]}" -eq 0 ]; then
    echo "[$(date '+%F %T')] tia2v_talkinghuman: all outputs already exist." | tee -a "$LOG_PATH"
else
    for row_csv in "${ROW_CSVS[@]}"; do
        sample_name="$(basename "$row_csv" .csv)"
        attempt=1
        while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
            selected_npu="$(wait_for_any_free_npu)"
            echo "[$(date '+%F %T')] tia2v_talkinghuman: running ${sample_name} on NPU ${selected_npu}, attempt ${attempt}" | tee -a "$LOG_PATH"
            if env ASCEND_RT_VISIBLE_DEVICES="${selected_npu}" python batch_infer_qilin_og_tia2v.py \
                --base_model_path /data1/Wan-AI/wan22_s2v \
                --audio_encoder_path /data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large \
                --lora_path /data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint.safetensors \
                --csv_path "$row_csv" \
                --data_base_path /data1/workspace/linxinliang/CoInteract \
                --output_dir "$OUTPUT_DIR" \
                --height 832 \
                --width 480 \
                --num_frames 80 \
                --num_clips 3 \
                --num_inference_steps 40 \
                --cfg_scale 7.0 >>"$LOG_PATH" 2>&1; then
                echo "[$(date '+%F %T')] tia2v_talkinghuman: completed ${sample_name}" | tee -a "$LOG_PATH"
                break
            fi
            echo "[$(date '+%F %T')] tia2v_talkinghuman: failed ${sample_name} on attempt ${attempt}; cooling down ${COOLDOWN_SECONDS}s" | tee -a "$LOG_PATH"
            sleep "$COOLDOWN_SECONDS"
            attempt=$((attempt + 1))
        done

        if [ "$attempt" -gt "$MAX_ATTEMPTS" ]; then
            echo "[$(date '+%F %T')] tia2v_talkinghuman: giving up on ${sample_name} after ${MAX_ATTEMPTS} attempts" | tee -a "$LOG_PATH"
        fi

        echo "[$(date '+%F %T')] tia2v_talkinghuman: per-case cooldown ${COOLDOWN_SECONDS}s before next launch" | tee -a "$LOG_PATH"
        sleep "$COOLDOWN_SECONDS"
    done
fi

echo "===== CoInteract T+A+I2V TalkingHuman cycle end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
