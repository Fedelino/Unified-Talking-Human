#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

LOG_DIR=logs/p2v_trainingfree_overnight_20260901/round4
mkdir -p "$LOG_DIR"

echo "[watch] waiting for round4 generation jobs..."
while true; do
  COUNT=$(pgrep -af 'run_p2v_trainingfree_round4_20260901.py --skip-prepare|p2v_trainingfree_overnight_20260901_round4' | grep -v watch_round4_eval_blocks | wc -l || true)
  echo "[watch] active=$COUNT time=$(date '+%F %T')"
  if [[ "$COUNT" -eq 0 ]]; then
    break
  fi
  sleep 120
done

echo "[watch] running round4 eval and keyframe postprocess..."
python scripts/run_p2v_trainingfree_round4_20260901.py --eval-only \
  > "$LOG_DIR/eval_driver.log" 2>&1

echo "[watch] rebuilding presentation video blocks..."
python scripts/make_p2v_presentation_video_blocks_20260902.py \
  > "$LOG_DIR/presentation_blocks_all.log" 2>&1

tar -C output_videos -czf output_videos/p2v_trainingfree_overnight_20260901_presentation_blocks.tar.gz \
  p2v_trainingfree_overnight_20260901_presentation_blocks

echo "[watch] done $(date '+%F %T')"
