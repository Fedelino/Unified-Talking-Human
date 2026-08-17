#!/bin/bash
source /data1/miniconda3/etc/profile.d/conda.sh
conda activate cointeract
cd /data1/workspace/linxinliang/CoInteract
export ASCEND_RT_VISIBLE_DEVICES=${DEV:-3}
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
python infer_consisid.py   --csv_path examples/cointeract_fullbody4_motion1_m2v_prompted_20260628.csv   --consisid_ckpt output/consisid_v2_ogfrozen_20260714/version_0/step-150.safetensors   --arcface_cache models/arcface/motion1_refs_arcface.npz   --output_dir output_videos/consisid_v2_motion1_step150_20260714   --height 480 --width 320 --num_frames 81 --num_clips 1 --num_inference_steps 30
echo "EXIT_CODE=$?"
