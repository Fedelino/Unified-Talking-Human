import csv
import os
import shlex
import subprocess
import time
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
DATE = "20260815"
CSV_PATH = ROOT / "examples" / "th_fullbody_001_custom_motion1_refkv_strong_20260815.csv"
NPUS = ["0", "1", "2"]

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
    "--face_reference_guidance_scale", "0.0",
    "--reference_kv_guidance_mode", "head_attn",
    "--reference_kv_guidance_blocks", "10:22",
    "--reference_kv_guidance_start_t", "0.05",
    "--reference_kv_guidance_end_t", "0.75",
]

VARIANTS = [
    ("refkv_headattn_s025", "0.25"),
    ("refkv_headattn_s050", "0.50"),
    ("refkv_headattn_s100", "1.00"),
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
                "sample_id": "th_fullbody_001_custom_motion1_refkv_strong",
                "prompt": PROMPT,
                "audio": "",
                "person_image": REFERENCE,
                "product_image": "",
                "pose_video": POSE,
            }
        )


def build_command(scale: str, out_dir: Path) -> list[str]:
    args = COMMON_ARGS + ["--reference_kv_guidance_scale", scale, "--output_dir", str(out_dir)]
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


def launch(name: str, scale: str, npu: str):
    out_dir = ROOT / "output_videos" / f"p2v_refkv_strong_prevconfig_motion1_{name}_{DATE}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = ROOT / "logs" / f"p2v_refkv_strong_prevconfig_motion1_{name}_{DATE}.log"
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = npu
    env["ASCEND_VISIBLE_DEVICES"] = npu
    with log_path.open("ab") as log_file:
        log_file.write(f"\n===== launch {name} scale={scale} on NPU {npu} =====\n".encode())
        process = subprocess.Popen(
            build_command(scale, out_dir),
            cwd=str(ROOT),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
    print(f"START\t{name}\tscale={scale}\tnpu={npu}\tpid={process.pid}\tlog={log_path}\tout={out_dir}", flush=True)
    return process, name, npu


def main() -> None:
    ensure_csv()
    (ROOT / "logs").mkdir(exist_ok=True)
    queue = list(VARIANTS)
    running = []

    while queue or running:
        while queue and len(running) < len(NPUS):
            used = {item[2] for item in running}
            npu = next(n for n in NPUS if n not in used)
            name, scale = queue.pop(0)
            running.append(launch(name, scale, npu))

        time.sleep(20)
        still_running = []
        for process, name, npu in running:
            code = process.poll()
            if code is None:
                still_running.append((process, name, npu))
                continue
            print(f"DONE\t{name}\tnpu={npu}\treturncode={code}", flush=True)
        running = still_running


if __name__ == "__main__":
    main()
