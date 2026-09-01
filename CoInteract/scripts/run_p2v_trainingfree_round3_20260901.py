#!/usr/bin/env python3
"""Round3 training-free CoInteract P2V identity experiments.

This add-on intentionally waits behind the 20260901 overnight queue. It keeps
CoInteract frozen and only tests isolated inference-time or postprocess variants.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
IA_ROOT = Path("/data1/workspace/linxinliang/InteractAvatar")
REF = IA_ROOT / "InterDemo/TalkingHumanDemo_fullbody/test_data/ref_img/wholebody/fullImage1a3b5329d192f028c14bc03d20c24afa629a3123.jpg"
PROMPT = "A full-body person follows the provided motion with stable facial identity, stable eyes, stable nose, stable lips, stable jawline, and stable whole-body proportions."
BASE_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901"
EVAL_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_eval_round3"
LOG_DIR = ROOT / "logs/p2v_trainingfree_overnight_20260901/round3"
ASSET_DIR = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_assets/round3"
CSV_DIR = ROOT / "examples/p2v_trainingfree_overnight_20260901_round3"
ARC = ROOT / "models/arcface/w600k_r50.onnx"

POSES = {
    "m1": {
        "pose": IA_ROOT / "InterDemo/custom_motion/dwpose/motion1_pose.mp4",
        "raw": IA_ROOT / "InterDemo/custom_motion/raw/motion1.mp4",
    },
    "m2": {
        "pose": IA_ROOT / "InterDemo/custom_motion/dwpose/motion2_pose.mp4",
        "raw": IA_ROOT / "InterDemo/custom_motion/raw/motion2.mp4",
    },
    "handwave": {
        "pose": IA_ROOT / "InterDemo/simple_motion_pose_bank_20260814/dwpose_qilin_aligned/th_motion1_handwave_qilin_aligned_pose.mp4",
        "raw": IA_ROOT / "InterDemo/simple_motion_pose_bank_20260814/raw_25fps_max4s/th_motion1_handwave.mp4",
    },
}


@dataclass
class Variant:
    name: str
    motion: str
    person_image: Path
    pose_video: Path
    args: dict[str, str | float | int | bool] = field(default_factory=dict)


def run(cmd: list[str], log_path: Path | None = None, env: dict[str, str] | None = None, check: bool = True):
    print("[cmd]", " ".join(shlex.quote(str(x)) for x in cmd), flush=True)
    if log_path is None:
        return subprocess.run(cmd, cwd=ROOT, env=env, check=check)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("# " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n\n")
        log.flush()
        return subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True, check=check)


def write_csv(path: Path, sample_id: str, person_image: Path, pose_video: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "prompt", "audio", "person_image", "product_image", "pose_video"])
        writer.writeheader()
        writer.writerow({
            "sample_id": sample_id,
            "prompt": PROMPT,
            "audio": "",
            "person_image": str(person_image),
            "product_image": "",
            "pose_video": str(pose_video),
        })


def probe_optional_methods() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    candidates = {
        "CodeFormer": [ROOT / "CodeFormer", Path("/data1/workspace/linxinliang/CodeFormer")],
        "GFPGAN": [ROOT / "GFPGAN", Path("/data1/workspace/linxinliang/GFPGAN")],
        "RestoreFormer": [ROOT / "RestoreFormer", Path("/data1/workspace/linxinliang/RestoreFormer")],
        "DECA": [ROOT / "DECA", Path("/data1/workspace/linxinliang/DECA")],
        "Stand-In": [ROOT / "Stand-In", Path("/data1/workspace/linxinliang/Stand-In")],
        "MagicMirror": [ROOT / "MagicMirror", Path("/data1/workspace/linxinliang/MagicMirror")],
    }
    lines = []
    for name, paths in candidates.items():
        found = [str(p) for p in paths if p.exists()]
        lines.append(f"{name}: {'FOUND ' + '; '.join(found) if found else 'unavailable - skipped'}")
    lines.append("BachVid-lite headcache: unavailable - no existing CoInteract intermediate-cache hook was found.")
    (LOG_DIR / "round3_optional_method_probe.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_assets() -> None:
    for path in (LOG_DIR, ASSET_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)
    probe_optional_methods()
    for motion, paths in POSES.items():
        raw = paths["raw"]
        if not raw.exists():
            print(f"[skip yaw prepare] missing raw motion for {motion}: {raw}", flush=True)
            continue
        for blend, yaw in ((0.25, 20.0), (0.35, 35.0)):
            tag = f"{motion}_yawgated_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_pose.mp4"
            run([
                sys.executable,
                "scripts/id_face_retarget_yawgated_20260901.py",
                "--driving_video", str(raw),
                "--reference_image", str(REF),
                "--out_video", str(ASSET_DIR / "pose_variants" / tag),
                "--height", "832",
                "--width", "480",
                "--fps", "25",
                "--max_frames", "80",
                "--blend", str(blend),
                "--yaw_threshold", str(yaw),
                "--yaw_softness", "8",
            ], LOG_DIR / f"prepare_{tag}.log", check=False)


def pose_exists(motion: str) -> bool:
    return POSES[motion]["pose"].exists()


def build_variants() -> list[Variant]:
    variants: list[Variant] = []
    motions = [m for m in ("m1", "m2", "handwave") if pose_exists(m)]

    if "handwave" in motions:
        variants.append(Variant("round3_baseline_handwave", "handwave", REF, POSES["handwave"]["pose"]))

    for motion in motions:
        for blend, yaw in ((0.25, 20), (0.35, 35)):
            pose = ASSET_DIR / "pose_variants" / f"{motion}_yawgated_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_pose.mp4"
            if pose.exists():
                variants.append(Variant(f"round3_yawgated_blend{int(blend * 100):02d}_yaw{int(yaw):02d}_{motion}", motion, REF, pose))

    guidance = [
        ("round3_dv_zero_scale1p0", 1.0, "zero", 5, False, 1.0),
        ("round3_dv_latblur7_scale1p0", 1.0, "latent_blur", 7, False, 1.0),
        ("round3_dv_latblur7_samg_scale1p5", 1.5, "latent_blur", 7, True, 1.0),
        ("round3_dv_latblur7_samg_apg025_scale2p0", 2.0, "latent_blur", 7, True, 0.25),
    ]
    for motion in motions:
        for name, scale, counterfactual, kernel, samg, apg_eta in guidance:
            extra: dict[str, str | float | int | bool] = {
                "face_reference_guidance_scale": scale,
                "face_reference_guidance_counterfactual": counterfactual,
                "face_reference_guidance_blur_kernel": kernel,
                "face_reference_guidance_apg_eta": apg_eta,
                "face_reference_guidance_start_t": 0.05,
                "face_reference_guidance_end_t": 0.75,
            }
            if samg:
                extra.update({
                    "face_reference_guidance_samg": True,
                    "face_reference_guidance_samg_min_mult": 0.5,
                    "face_reference_guidance_samg_max_mult": 1.5,
                })
            variants.append(Variant(f"{name}_{motion}", motion, REF, POSES[motion]["pose"], extra))
    return variants


def batch_command(csv_path: Path, out_dir: Path, extra: dict[str, str | float | int | bool]) -> list[str]:
    cmd = [
        sys.executable,
        "batch_infer.py",
        "--base_model_path", "./models/Wan2.2-S2V-14B",
        "--audio_encoder_path", "./models/chinese-wav2vec2-large",
        "--csv_path", str(csv_path),
        "--height", "832",
        "--width", "480",
        "--num_frames", "80",
        "--num_clips", "1",
        "--num_inference_steps", "40",
        "--cfg_scale", "7.0",
        "--sigma_shift", "7.0",
        "--lora_path", "./models/CoInteract/checkpoint_pose.safetensors",
        "--reference_compose_mode", "stretch",
        "--output_dir", str(out_dir),
        "--no_resize_output_to_reference",
    ]
    for key, value in extra.items():
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])
    return cmd


def selected_variants(variants: list[Variant], partition: int, partitions: int) -> list[Variant]:
    if partitions <= 1:
        return variants
    return [variant for idx, variant in enumerate(variants) if idx % partitions == partition]


def run_generation(args) -> None:
    env = os.environ.copy()
    env["ASCEND_RT_VISIBLE_DEVICES"] = str(args.npu)
    env["ASCEND_VISIBLE_DEVICES"] = str(args.npu)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_NPU_ALLOC_CONF", "max_split_size_mb:256,garbage_collection_threshold:0.8")
    for variant in selected_variants(build_variants(), args.partition, args.partitions):
        csv_path = CSV_DIR / f"{variant.name}.csv"
        out_dir = BASE_OUT / variant.name
        write_csv(csv_path, variant.name, variant.person_image, variant.pose_video)
        if (out_dir / f"{variant.name}.mp4").exists():
            print(f"[skip exists] {variant.name}", flush=True)
            continue
        run(batch_command(csv_path, out_dir, variant.args), LOG_DIR / f"{variant.name}.log", env=env, check=False)


def eval_one(video: Path, reference: Path, name: str, rows: list[dict[str, str]]) -> None:
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    txt_path = EVAL_OUT / f"{name}.txt"
    csv_path = EVAL_OUT / f"{name}.csv"
    sheet_path = EVAL_OUT / f"{name}_sheet.jpg"
    proc = subprocess.run([
        sys.executable,
        "eval/id_drift_metric.py",
        "--video", str(video),
        "--reference", str(reference),
        "--arcface_onnx", str(ARC),
        "--detector", "insightface",
        "--every", "5",
        "--out_csv", str(csv_path),
    ], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
    txt_path.write_text(proc.stdout, encoding="utf-8")
    subprocess.run([
        sys.executable,
        "scripts/make_frame_sheet.py",
        "--video", str(video),
        "--out", str(sheet_path),
        "--frames", "0,25,50,75",
    ], cwd=ROOT, check=False)
    row = {"name": name, "video": str(video)}
    for pattern, keys in [
        (r"cos_ref\s+mean/min:\s+([0-9.+-]+)\s*/\s*([0-9.+-]+)", ("cos_mean", "cos_min")),
        (r"cos_ref\s+first/last:\s+([0-9.+-]+)\s*/\s*([0-9.+-]+)", ("cos_first", "cos_last")),
        (r"drift slope \(cos/s\):\s+([0-9.+-]+)", ("slope",)),
        (r"face detect(?:ion)? rate:\s+([0-9.+-]+)", ("detect_rate",)),
    ]:
        match = re.search(pattern, proc.stdout)
        if match:
            for idx, key in enumerate(keys, start=1):
                row[key] = match.group(idx)
    rows.append(row)


def baseline_video_for_motion(motion: str) -> Path | None:
    candidates = [
        BASE_OUT / f"00_baseline_{motion}" / f"00_baseline_{motion}.mp4",
        BASE_OUT / f"round3_baseline_{motion}" / f"round3_baseline_{motion}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def run_postprocess_and_eval() -> None:
    rows: list[dict[str, str]] = []
    for motion in ("m1", "m2", "handwave"):
        baseline = baseline_video_for_motion(motion)
        if baseline is None:
            continue
        for mode, strength, name_part in [
            ("restore", 0.25, "temporal_face_restore025"),
            ("restore", 0.40, "temporal_face_restore040"),
            ("color_anchor", 0.35, "temporal_color_anchor035"),
        ]:
            post_name = f"round3_{name_part}_{motion}"
            out_video = BASE_OUT / post_name / f"{post_name}.mp4"
            if not out_video.exists():
                run([
                    sys.executable,
                    "scripts/temporal_face_stabilize_20260901.py",
                    "--video", str(baseline),
                    "--out", str(out_video),
                    "--mode", mode,
                    "--strength", str(strength),
                ], LOG_DIR / f"{post_name}.log", check=False)

    for video in sorted(BASE_OUT.glob("round3*/*.mp4")):
        eval_one(video, REF, video.parent.name, rows)
    for motion in ("m1", "m2"):
        baseline = baseline_video_for_motion(motion)
        if baseline is not None:
            eval_one(baseline, REF, f"baseline_compare_{motion}", rows)

    summary_path = EVAL_OUT / "summary.csv"
    fields = ["name", "cos_mean", "cos_min", "cos_first", "cos_last", "slope", "detect_rate", "video"]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--npu", default="7")
    parser.add_argument("--partition", type=int, default=0)
    parser.add_argument("--partitions", type=int, default=1)
    args = parser.parse_args()
    os.chdir(ROOT)
    for path in (BASE_OUT, EVAL_OUT, LOG_DIR, ASSET_DIR, CSV_DIR):
        path.mkdir(parents=True, exist_ok=True)
    if args.eval_only:
        run_postprocess_and_eval()
        return
    if not args.skip_prepare:
        prepare_assets()
    if args.prepare_only:
        return
    run_generation(args)


if __name__ == "__main__":
    main()
