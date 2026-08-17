#!/bin/bash
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd /data1/workspace/linxinliang/CoInteract
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-2}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export OUTPUT_PATH=./output/consisid_v2_ogfrozen_20260714
export SAVE_STEPS=50
export NUM_EPOCHS=5
bash scripts/train_consisid.sh
echo "EXIT_CODE=$?"
