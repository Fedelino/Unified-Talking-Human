#!/usr/bin/env bash
set -euo pipefail

ROOT="/data1/workspace/linxinliang/CoInteract"
CSV_PATH="$ROOT/examples/cointeract_tia2v_3cases_30s_20260702.csv"
OUT_DIR="$ROOT/output_videos/tia2v_3cases_30s_20260702"
LOG_PATH="$ROOT/logs/tia2v_3cases_30s_20260702.log"
TMP_DIR="/tmp/tia2v_3cases_30s_20260702"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"
VISIBLE_NPU="7"

mkdir -p "$OUT_DIR" "$TMP_DIR" "$(dirname "$LOG_PATH")"

{
  echo "===== CoInteract TIA2V 3-case 30s start $(date '+%F %T') ====="
  echo "visible_npu=$VISIBLE_NPU"
  echo "csv_path=$CSV_PATH"
  echo "output_dir=$OUT_DIR"
} | tee -a "$LOG_PATH"

"$PYTHON_BIN" - <<'PY'
import pandas as pd
from pathlib import Path

csv_path = Path("/data1/workspace/linxinliang/CoInteract/examples/cointeract_tia2v_3cases_30s_20260702.csv")
tmp_dir = Path("/tmp/tia2v_3cases_30s_20260702")
tmp_dir.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(csv_path)
for idx, row in df.iterrows():
    sample_id = str(row["sample_id"]).strip()
    out_csv = tmp_dir / f"{idx + 1:02d}_{sample_id}.csv"
    row.to_frame().T.to_csv(out_csv, index=False)
    print(out_csv)
PY

for case_csv in "$TMP_DIR"/*.csv; do
  sample_name="$(basename "$case_csv" .csv)"
  save_path="$OUT_DIR/${sample_name#*_}.mp4"

  if [[ -f "$save_path" ]]; then
    echo "[$(date '+%F %T')] Skip (exists): $sample_name" | tee -a "$LOG_PATH"
    continue
  fi

  echo "[$(date '+%F %T')] Running: $sample_name on NPU $VISIBLE_NPU" | tee -a "$LOG_PATH"
  env ASCEND_RT_VISIBLE_DEVICES="$VISIBLE_NPU" \
    "$PYTHON_BIN" "$ROOT/batch_infer_qilin_og_tia2v.py" \
      --base_model_path /data1/Wan-AI/wan22_s2v \
      --audio_encoder_path "$ROOT/models/chinese-wav2vec2-large" \
      --lora_path "$ROOT/models/CoInteract/checkpoint.safetensors" \
      --csv_path "$case_csv" \
      --output_dir "$OUT_DIR" \
      --height 832 \
      --width 480 \
      --num_frames 76 \
      --num_clips 10 \
      --num_inference_steps 40 \
      --cfg_scale 7.0 \
      --max_audio_seconds 30 \
      >> "$LOG_PATH" 2>&1

  echo "[$(date '+%F %T')] Finished: $sample_name" | tee -a "$LOG_PATH"
  sleep 20
done

echo "===== CoInteract TIA2V 3-case 30s end $(date '+%F %T') =====" | tee -a "$LOG_PATH"
