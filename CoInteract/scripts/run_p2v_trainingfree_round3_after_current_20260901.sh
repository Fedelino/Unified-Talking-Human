#!/usr/bin/env bash
set -u

cd /data1/workspace/linxinliang/CoInteract || exit 1
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

LOG_DIR="logs/p2v_trainingfree_overnight_20260901/round3"
mkdir -p "$LOG_DIR"

echo "[round3 watcher] waiting for existing overnight queue to finish: $(date)"
while pgrep -af 'run_p2v_trainingfree_overnight_20260901.py|batch_infer.py .*p2v_trainingfree_overnight_20260901|id_face_retarget.py .*p2v_trainingfree_overnight_20260901_assets' | grep -v 'run_p2v_trainingfree_round3_after_current' >/dev/null 2>&1; do
  sleep 180
done

if [[ ! -f output_videos/p2v_trainingfree_overnight_20260901_eval/summary.csv ]]; then
  echo "[round3 watcher] previous eval summary missing; running previous eval first: $(date)"
  /data1/miniconda3/envs/cointeract/bin/python scripts/run_p2v_trainingfree_overnight_20260901.py --eval-only > "$LOG_DIR/previous_eval_before_round3.log" 2>&1
fi

echo "[round3 watcher] preparing round3 assets: $(date)"
/data1/miniconda3/envs/cointeract/bin/python scripts/run_p2v_trainingfree_round3_20260901.py --prepare-only > "$LOG_DIR/round3_prepare.log" 2>&1

PARTITIONS=8
for idx in $(seq 0 7); do
  nohup /data1/miniconda3/envs/cointeract/bin/python scripts/run_p2v_trainingfree_round3_20260901.py \
    --skip-prepare --partition "$idx" --partitions "$PARTITIONS" --npu "$idx" \
    > "$LOG_DIR/round3_partition_${idx}_npu${idx}.log" 2>&1 &
  echo $! > "$LOG_DIR/round3_partition_${idx}_npu${idx}.pid"
done

for pidfile in "$LOG_DIR"/round3_partition_*.pid; do
  pid="$(cat "$pidfile")"
  while kill -0 "$pid" 2>/dev/null; do
    sleep 180
  done
done

echo "[round3 watcher] evaluating round3: $(date)"
/data1/miniconda3/envs/cointeract/bin/python scripts/run_p2v_trainingfree_round3_20260901.py --eval-only > "$LOG_DIR/round3_eval.log" 2>&1
echo "[round3 watcher] complete: $(date)"
