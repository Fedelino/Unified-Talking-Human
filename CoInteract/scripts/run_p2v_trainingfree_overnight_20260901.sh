#!/usr/bin/env bash
set -euo pipefail

ROOT=/data1/workspace/linxinliang/CoInteract
LOG_DIR="$ROOT/logs/p2v_trainingfree_overnight_20260901"
PY=/data1/miniconda3/envs/cointeract/bin/python
NPUS="${NPUS:-auto}"

source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd "$ROOT"
mkdir -p "$LOG_DIR"

echo "[prepare] assets" | tee "$LOG_DIR/overnight_launcher.log"
"$PY" scripts/run_p2v_trainingfree_overnight_20260901.py --prepare-only \
  > "$LOG_DIR/prepare_all.log" 2>&1

if [[ "$NPUS" == "auto" ]]; then
  mapfile -t FREE_NPUS < <(npu-smi info | awk '
    /^[|] [0-7][[:space:]]+910B2/ {npu=$2}
    /[0-9]+[[:space:]]*\\/[[:space:]]*65536/ {
      split($0,a,"|"); split(a[3],b,"/"); gsub(/[^0-9]/,"",b[1]);
      if (b[1] + 0 < 6000) print npu
    }' | sort -n | uniq)
else
  IFS=',' read -r -a FREE_NPUS <<< "$NPUS"
fi

if [[ ${#FREE_NPUS[@]} -eq 0 ]]; then
  echo "[error] no free NPUs found" | tee -a "$LOG_DIR/overnight_launcher.log"
  exit 1
fi

PARTITIONS=${#FREE_NPUS[@]}
echo "[launch] npu_count=$PARTITIONS npus=${FREE_NPUS[*]}" | tee -a "$LOG_DIR/overnight_launcher.log"
for idx in "${!FREE_NPUS[@]}"; do
  npu="${FREE_NPUS[$idx]}"
  (
    "$PY" scripts/run_p2v_trainingfree_overnight_20260901.py \
      --skip-prepare --partition "$idx" --partitions "$PARTITIONS" --npu "$npu"
  ) > "$LOG_DIR/partition_${idx}_npu${npu}.log" 2>&1 &
  echo $! > "$LOG_DIR/partition_${idx}_npu${npu}.pid"
  echo "[launch] partition=$idx npu=$npu pid=$(cat "$LOG_DIR/partition_${idx}_npu${npu}.pid")" | tee -a "$LOG_DIR/overnight_launcher.log"
done

(
  set +e
  for idx in "${!FREE_NPUS[@]}"; do
    npu="${FREE_NPUS[$idx]}"
    pid_file="$LOG_DIR/partition_${idx}_npu${npu}.pid"
    wait "$(cat "$pid_file")"
  done
  "$PY" scripts/run_p2v_trainingfree_overnight_20260901.py --eval-only
) > "$LOG_DIR/eval_after_all.log" 2>&1 &
echo $! > "$LOG_DIR/eval_after_all.pid"
echo "[watcher] eval_pid=$(cat "$LOG_DIR/eval_after_all.pid")" | tee -a "$LOG_DIR/overnight_launcher.log"
