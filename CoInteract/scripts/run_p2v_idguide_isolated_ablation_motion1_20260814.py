import csv
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
DATE = "20260814"
CSV_PATH = ROOT / "examples" / "th_fullbody_001_custom_motion1_noaudio_baseline_20260814.csv"
NPUS = ["0", "1", "2", "5", "6", "7"]

REFERENCE = "/data1/workspace/linxinliang/InteractAvatar/InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
POSE = "/data1/workspace/linxinliang/InteractAvatar/InterDemo/custom_motion/dwpose/motion1_pose.mp4"
PROMPT = "A full-body person follows the provided motion with stable facial identity, stable eyes, stable nose, stable lips, stable jawline, and stable whole-body proportions."

COMMON_ARGS = [
    "--base_model_path", "/data1/Wan-AI/wan22_s2v",
    "--audio_encoder_path", "/data1/workspace/linxinliang/CoInteract/models/chinese-wav2vec2-large",
    "--lora_path", "/data1/workspace/linxinliang/CoInteract/models/CoInteract/checkpoint_pose.safetensors",
    "--csv_path", str(CSV_PATH.relative_to(ROOT)),
    "--data_base_path", str(ROOT),
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
        "args": ["--face_reference_guidance_scale", "0.0"],
    },
    {
        "name": "vv_zero_scale1",
        "args": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "zero",
        ],
    },
    {
        "name": "vv_latblur_k5_scale1",
        "args": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
        ],
    },
    {
        "name": "vv_latblur_k5_samg_default_scale1",
        "args": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
            "--face_reference_guidance_samg",
            "--face_reference_guidance_samg_min_mult", "0.5",
            "--face_reference_guidance_samg_max_mult", "1.5",
        ],
    },
    {
        "name": "vv_latblur_k5_samg_strong_scale1",
        "args": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
            "--face_reference_guidance_samg",
            "--face_reference_guidance_samg_min_mult", "0.25",
            "--face_reference_guidance_samg_max_mult", "2.0",
        ],
    },
    {
        "name": "vv_latblur_k5_apg025_scale1",
        "args": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
            "--face_reference_guidance_apg_eta", "0.25",
        ],
    },
    {
        "name": "vv_latblur_k5_apg0_scale1",
        "args": [
            "--face_reference_guidance_scale", "1.0",
            "--face_reference_guidance_counterfactual", "latent_blur",
            "--face_reference_guidance_blur_kernel", "5",
            "--face_reference_guidance_apg_eta", "0.0",
        ],
    },
    {
        "name": "vv_latblur_k5_samg_default_apg025_scale1",
        "args": [
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


def ensure_csv() -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sample_id", "prompt", "audio", "person_image", "product_image", "pose_video"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "th_fullbody_001_custom_motion1_noaudio_baseline",
                "prompt": PROMPT,
                "audio": "",
                "person_image": REFERENCE,
                "product_image": "",
                "pose_video": POSE,
            }
        )


def build_command(variant, out_dir: Path) -> list[str]:
    args = COMMON_ARGS + variant["args"] + ["--output_dir", str(out_dir)]
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    script = (
        "source /data1/miniconda3/etc/profile.d/conda.sh && "
        "conda activate cointeract && "
        f"cd {shlex.quote(str(ROOT))} && "
        "export TOKENIZERS_PARALLELISM=false && "
        "export PYTHONUNBUFFERED=1 && "
        "export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256,garbage_collection_threshold:0.8 && "
        "python batch_infer.py "
        + quoted_args
    )
    return ["bash", "-lc", script]


def launch(variant, npu: str):
    out_dir = ROOT / "output_videos" / f"p2v_idguide_motion1_{variant['name']}_{DATE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"p2v_idguide_motion1_{variant['name']}_{DATE}.log"
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = npu
    with log_path.open("ab") as log_file:
        log_file.write(f"\n===== launch {variant['name']} on NPU {npu} =====\n".encode())
        process = subprocess.Popen(
            build_command(variant, out_dir),
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    print(f"START\t{variant['name']}\tnpu={npu}\tpid={process.pid}\tlog={log_path}\tout={out_dir}", flush=True)
    return process, variant, npu


def main() -> None:
    ensure_csv()
    (ROOT / "logs").mkdir(exist_ok=True)
    queue = list(VARIANTS)
    running = []

    while queue or running:
        while queue and len(running) < len(NPUS):
            used = {item[2] for item in running}
            npu = next(n for n in NPUS if n not in used)
            running.append(launch(queue.pop(0), npu))

        time.sleep(20)
        still_running = []
        for process, variant, npu in running:
            code = process.poll()
            if code is None:
                still_running.append((process, variant, npu))
                continue
            print(f"DONE\t{variant['name']}\tnpu={npu}\treturncode={code}", flush=True)
        running = still_running


if __name__ == "__main__":
    main()
