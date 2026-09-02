#!/usr/bin/env bash
set -euo pipefail

cd /data1/workspace/linxinliang/CoInteract
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract

mkdir -p logs/p2v_trainingfree_overnight_20260901/round4

if [[ "${ROUND4_PREPARE_POSES:-0}" == "1" ]]; then
  python scripts/run_p2v_trainingfree_round4_20260901.py --prepare-only \
    > logs/p2v_trainingfree_overnight_20260901/round4/prepare_driver.log 2>&1
else
  echo "Skipping slow pose preparation. Set ROUND4_PREPARE_POSES=1 to enable yaw-gated geometry variants."
fi

NPUS=(${ROUND4_NPUS:-0 1 2 3 4 5 6 7})
PARTITIONS=${#NPUS[@]}

for IDX in "${!NPUS[@]}"; do
  NPU="${NPUS[$IDX]}"
  LOG="logs/p2v_trainingfree_overnight_20260901/round4/partition_${IDX}_npu${NPU}.log"
  nohup python scripts/run_p2v_trainingfree_round4_20260901.py \
    --skip-prepare \
    --npu "${NPU}" \
    --partition "${IDX}" \
    --partitions "${PARTITIONS}" \
    > "${LOG}" 2>&1 &
  echo "launched partition=${IDX}/${PARTITIONS} npu=${NPU} pid=$! log=${LOG}"
done

echo "When generation finishes, run:"
echo "python scripts/run_p2v_trainingfree_round4_20260901.py --eval-only"
