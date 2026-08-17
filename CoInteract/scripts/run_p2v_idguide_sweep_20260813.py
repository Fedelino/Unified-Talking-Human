import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
CSV = "examples/th_fullbody_001_custom_motion2_noaudio_baseline_20260716.csv"
DATE = "20260813"

COMMON_ARGS = [
    "--base_model_path", "/data1/Wan-AI/wan22_s2v",
    "--audio_encoder_path", "/data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large",
    "--lora_path", "/data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors",
    "--csv_path", CSV,
    "--data_base_path", "/data1/workspace/linxinliang/CoInteract",
    "--height", "832",
    "--width", "480",
    "--num_frames", "80",
    "--num_clips", "1",
    "--num_inference_steps", "40",
    "--cfg_scale", "7.0",
    "--sigma_shift", "7.0",
    "--reference_compose_mode", "stretch",
    "--no_resize_output_to_reference",
    "--arcface_guidance_scale", "0.0",
]

VARIANTS = [
    {
        "name": "baseline_noguidance",
        "npu": "0",
        "extra": [
            "--face_reference_guidance_scale", "0.0",
        ],
    },
    {
        "name": "zero_scale1",
        "npu": "1",
        "extra": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "zero",
        ],
    },
    {
        "name": "latblur_k5_scale1",
        "npu": "2",
        "extra": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
        ],
    },
    {
        "name": "latblur_k5_samg_scale1",
        "npu": "3",
        "extra": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
            "--face_reference_guidance_samg",
            "--face_reference_guidance_samg_min_mult", "0.5",
            "--face_reference_guidance_samg_max_mult", "1.5",
        ],
    },
    {
        "name": "latblur_k5_samg_apg025_scale1",
        "npu": "4",
        "extra": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
            "--face_reference_guidance_samg",
            "--face_reference_guidance_samg_min_mult", "0.5",
            "--face_reference_guidance_samg_max_mult", "1.5",
            "--face_reference_guidance_apg_eta", "0.25",
        ],
    },
]


def main() -> None:
    (ROOT / "logs").mkdir(exist_ok=True)
    for variant in VARIANTS:
        out_dir = ROOT / "output_videos" / f"p2v_idguide_{variant['name']}_motion2_{DATE}"
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = ROOT / "logs" / f"p2v_idguide_{variant['name']}_motion2_{DATE}.log"

        args = COMMON_ARGS + ["--output_dir", str(out_dir)] + variant["extra"]
        quoted_args = " ".join(shlex.quote(arg) for arg in args)
        command = (
            "source /data1/miniconda3/etc/profile.d/conda.sh && "
            "conda activate cointeract && "
            f"cd {shlex.quote(str(ROOT))} && "
            "export TOKENIZERS_PARALLELISM=false && "
            "export PYTHONUNBUFFERED=1 && "
            "export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8 && "
            f"export ASCEND_RT_VISIBLE_DEVICES={shlex.quote(variant['npu'])} && "
            f"python batch_infer.py {quoted_args}"
        )

        with log_path.open("ab") as log_file:
            log_file.write(f"\n===== launch {variant['name']} on NPU {variant['npu']} =====\n".encode())
            process = subprocess.Popen(
                ["bash", "-lc", command],
                cwd=str(ROOT),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        print(f"{variant['name']}\tnpu={variant['npu']}\tpid={process.pid}\tlog={log_path}\tout={out_dir}")


if __name__ == "__main__":
    main()
