#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 <watch_pid> <target_npu> [log_file]" >&2
  exit 1
fi

WATCH_PID="$1"
TARGET_NPU="$2"
LOG_FILE="${3:-/tmp/watch_and_hold_npu.log}"
PYTHON_BIN="/data1/miniconda3/envs/interact_avatar/bin/python"

timestamp() {
  date '+%F %T'
}

echo "[$(timestamp)] watcher started for pid=${WATCH_PID}, target_npu=${TARGET_NPU}" | tee -a "$LOG_FILE"

while kill -0 "$WATCH_PID" 2>/dev/null; do
  echo "[$(timestamp)] pid ${WATCH_PID} still running; waiting before holding NPU ${TARGET_NPU}" | tee -a "$LOG_FILE"
  sleep 60
done

echo "[$(timestamp)] pid ${WATCH_PID} exited; launching NPU ${TARGET_NPU} holder" | tee -a "$LOG_FILE"

ASCEND_RT_VISIBLE_DEVICES="$TARGET_NPU" nohup "$PYTHON_BIN" - <<'PY' >> "$LOG_FILE" 2>&1 &
import os
import time

import torch
import torch_npu

torch.npu.set_device("npu:0")

blocks = []
block_shape = (512, 1024, 1024)  # about 1 GiB per fp16 block
for idx in range(4):
    try:
        blocks.append(torch.empty(block_shape, dtype=torch.float16, device="npu"))
        print(f"allocated block {idx + 1}/4 on visible NPU {os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}", flush=True)
    except Exception as exc:
        print(f"allocation stopped after {idx} blocks: {exc}", flush=True)
        break

print(
    f"holder active on visible_npu={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')} "
    f"with {len(blocks)} blocks",
    flush=True,
)

while True:
    for tensor in blocks:
        tensor.add_(0)
    time.sleep(300)
PY

HOLDER_PID=$!
echo "[$(timestamp)] holder launched with pid=${HOLDER_PID} on NPU ${TARGET_NPU}" | tee -a "$LOG_FILE"
