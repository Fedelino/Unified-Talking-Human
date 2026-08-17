#!/usr/bin/env bash
set -euo pipefail

LOG_PATH="/data1/workspace/linxinliang/CoInteract/logs/qilin_og_m2v_halfbody_refimg4_wait_20260623.log"
FULLFRAME_RUNNER_SIG="run_qilin_og_m2v_halfbody_fullframe_onebyone_720p.sh"
RUNNER="/data1/workspace/linxinliang/CoInteract/run_qilin_og_m2v_halfbody_refimg4_onebyone_720p.sh"

mkdir -p "$(dirname "$LOG_PATH")"

is_npu7_busy() {
python - <<'PY'
import re
import subprocess
import sys

text = subprocess.check_output(["npu-smi", "info"], text=True, errors="ignore")
busy = False
for line in text.splitlines():
    if re.match(r"^\|\s*7\s+0\s+\|\s*\d+\s+\|", line):
        busy = True
        break
sys.exit(0 if busy else 1)
PY
}

echo "[$(date '+%F %T')] waiting for fullframe halfbody runner to finish..." | tee -a "$LOG_PATH"
while pgrep -f "$FULLFRAME_RUNNER_SIG" >/dev/null; do
    echo "[$(date '+%F %T')] fullframe halfbody runner still active; sleeping 120s" | tee -a "$LOG_PATH"
    sleep 120
done

echo "[$(date '+%F %T')] fullframe halfbody runner is gone; waiting for NPU 7 to be free..." | tee -a "$LOG_PATH"
while is_npu7_busy; do
    echo "[$(date '+%F %T')] NPU 7 still busy; sleeping 120s" | tee -a "$LOG_PATH"
    sleep 120
done

echo "[$(date '+%F %T')] launching ref_img/halfbody 4-case runner" | tee -a "$LOG_PATH"
bash "$RUNNER" >>"$LOG_PATH" 2>&1
