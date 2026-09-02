#!/usr/bin/env python3
"""Create tiled MP4 comparison blocks for P2V presentation results.

Each output video plays multiple method results synchronized over time, with a
compact label and ArcFace summary per tile. This is easier to show in slides or
meetings than separate still-frame sheets.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/data1/workspace/linxinliang/CoInteract")
BASE_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901"
ROUND4_OUT = BASE_OUT / "round4"
EVALS = [
    ROOT / "output_videos/p2v_trainingfree_overnight_20260901_eval/summary.csv",
    ROOT / "output_videos/p2v_trainingfree_overnight_20260901_eval_round3/summary.csv",
    ROOT / "output_videos/p2v_trainingfree_overnight_20260901_eval_round4/summary.csv",
]
DEFAULT_OUT = ROOT / "output_videos/p2v_trainingfree_overnight_20260901_presentation_blocks"


@dataclass(frozen=True)
class ClipSpec:
    key: str
    label: str


GROUPS: dict[str, list[ClipSpec]] = {
    "motion1_identity_methods": [
        ClipSpec("00_baseline_m1", "baseline"),
        ClipSpec("a_refprep_ref_face_sharp040_m1", "ref face sharpen"),
        ClipSpec("b_facekv_s0p05_m1", "face-only ref K/V"),
        ClipSpec("c_idface_blend15_m1", "light ID face pose"),
        ClipSpec("round3_dv_zero_scale1p0_m1", "delta-v zero"),
        ClipSpec("round3_dv_latblur7_samg_apg025_scale2p0_m1", "delta-v blur+SAMG+APG"),
        ClipSpec("round3_yawgated_blend35_yaw35_m1", "yaw-gated geometry"),
        ClipSpec("round3_temporal_face_restore025_m1", "post face restore"),
    ],
    "motion2_identity_methods": [
        ClipSpec("00_baseline_m2", "baseline"),
        ClipSpec("a_refprep_ref_face_sharp040_m2", "ref face sharpen"),
        ClipSpec("b_facekv_s0p05_m2", "face-only ref K/V"),
        ClipSpec("c_idface_blend35_m2", "light ID face pose"),
        ClipSpec("round3_dv_zero_scale1p0_m2", "delta-v zero"),
        ClipSpec("round3_dv_latblur7_scale1p0_m2", "delta-v latent blur"),
        ClipSpec("round3_dv_latblur7_samg_scale1p5_m2", "delta-v + SAMG"),
        ClipSpec("round3_dv_latblur7_samg_apg025_scale2p0_m2", "delta-v + SAMG+APG"),
        ClipSpec("round3_temporal_face_restore025_m2", "post face restore"),
    ],
    "round3_guidance_m1": [
        ClipSpec("00_baseline_m1", "baseline"),
        ClipSpec("round3_dv_zero_scale1p0_m1", "zero weak-ID s1"),
        ClipSpec("round3_dv_latblur7_scale1p0_m1", "latent blur s1"),
        ClipSpec("round3_dv_latblur7_samg_scale1p5_m1", "blur+SAMG s1.5"),
        ClipSpec("round3_dv_latblur7_samg_apg025_scale2p0_m1", "blur+SAMG+APG s2"),
    ],
    "round3_guidance_m2": [
        ClipSpec("00_baseline_m2", "baseline"),
        ClipSpec("round3_dv_zero_scale1p0_m2", "zero weak-ID s1"),
        ClipSpec("round3_dv_latblur7_scale1p0_m2", "latent blur s1"),
        ClipSpec("round3_dv_latblur7_samg_scale1p5_m2", "blur+SAMG s1.5"),
        ClipSpec("round3_dv_latblur7_samg_apg025_scale2p0_m2", "blur+SAMG+APG s2"),
    ],
    "round4_m1_when_available": [
        ClipSpec("r4_baseline_m1", "r4 baseline"),
        ClipSpec("r4_dv_full_scale2p5_m1", "guidance s2.5"),
        ClipSpec("r4_dv_detail_scale3p0_m1", "detail-window s3"),
        ClipSpec("r4_dv_broad_scale3p0_m1", "broad-window s3"),
        ClipSpec("r4_dv_noapg_scale2p0_m1", "no APG s2"),
        ClipSpec("r4_dv_coarsefine_scale2p0_m1", "coarse-fine s2"),
        ClipSpec("r4_kfanchor_video_s0p2_m1", "keyframe video anchor"),
        ClipSpec("r4_kfanchor_reference_s0p12_m1", "keyframe ref anchor"),
    ],
    "round4_m2_when_available": [
        ClipSpec("r4_baseline_m2", "r4 baseline"),
        ClipSpec("r4_dv_full_scale2p5_m2", "guidance s2.5"),
        ClipSpec("r4_dv_detail_scale3p0_m2", "detail-window s3"),
        ClipSpec("r4_dv_broad_scale3p0_m2", "broad-window s3"),
        ClipSpec("r4_dv_noapg_scale2p0_m2", "no APG s2"),
        ClipSpec("r4_dv_coarsefine_scale2p0_m2", "coarse-fine s2"),
        ClipSpec("r4_kfanchor_video_s0p2_m2", "keyframe video anchor"),
        ClipSpec("r4_kfanchor_reference_s0p12_m2", "keyframe ref anchor"),
    ],
}


def load_metrics() -> dict[str, dict[str, str]]:
    metrics: dict[str, dict[str, str]] = {}
    for path in EVALS:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                name = row.get("name", "")
                if name:
                    metrics[name] = row
    return metrics


def find_video(key: str) -> Path | None:
    candidates = [
        BASE_OUT / key / f"{key}.mp4",
        ROUND4_OUT / key / f"{key}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = sorted(BASE_OUT.glob(f"**/{key}.mp4")) + sorted(BASE_OUT.glob(f"**/*{key}*.mp4"))
    return matches[0] if matches else None


def open_caps(specs: list[ClipSpec]) -> list[tuple[ClipSpec, Path, cv2.VideoCapture]]:
    opened = []
    for spec in specs:
        path = find_video(spec.key)
        if path is None:
            print(f"[missing] {spec.key}")
            continue
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            print(f"[bad video] {path}")
            continue
        opened.append((spec, path, cap))
    return opened


def metric_label(key: str, metrics: dict[str, dict[str, str]]) -> str:
    row = metrics.get(key, {})
    mean = row.get("cos_mean", "")
    minv = row.get("cos_min", "")
    last = row.get("cos_last", "")
    if mean:
        return f"Arc mean {mean} | min {minv} | last {last}"
    return "Arc pending"


def read_frame_at(cap: cv2.VideoCapture, idx: int, fallback_shape: tuple[int, int]) -> np.ndarray:
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    use_idx = max(0, min(idx, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, use_idx)
    ok, frame = cap.read()
    if ok and frame is not None:
        return frame
    h, w = fallback_shape
    return np.zeros((h, w, 3), np.uint8)


def letterbox(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tile_w, tile_h = size
    h, w = frame.shape[:2]
    scale = min(tile_w / max(w, 1), tile_h / max(h, 1))
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((tile_h, tile_w, 3), np.uint8)
    x, y = (tile_w - nw) // 2, (tile_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def draw_header(tile: np.ndarray, title: str, subtitle: str, frame_idx: int) -> np.ndarray:
    header_h = 58
    out = np.zeros((tile.shape[0] + header_h, tile.shape[1], 3), np.uint8)
    out[header_h:] = tile
    cv2.putText(out, title[:34], (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, subtitle[:46], (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (210, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"f{frame_idx:03d}", (tile.shape[1] - 58, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 255, 180), 1, cv2.LINE_AA)
    return out


def make_block(group_name: str, specs: list[ClipSpec], out_dir: Path, metrics: dict[str, dict[str, str]], tile_width: int) -> None:
    opened = open_caps(specs)
    if len(opened) < 2:
        print(f"[skip group] {group_name}: only {len(opened)} video(s)")
        for _, _, cap in opened:
            cap.release()
        return

    fps = min((cap.get(cv2.CAP_PROP_FPS) or 25.0) for _, _, cap in opened)
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1 for _, _, cap in opened)
    heights = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 832 for _, _, cap in opened]
    widths = [int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 480 for _, _, cap in opened]
    aspect = float(np.median(widths)) / max(1.0, float(np.median(heights)))
    tile_h = int(round(tile_width / max(aspect, 0.1)))
    tile_h = max(240, min(tile_h, 540))
    label_tile_h = tile_h + 58

    cols = min(3, len(opened))
    rows = int(math.ceil(len(opened) / cols))
    canvas_w = cols * tile_width
    canvas_h = rows * label_tile_h
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group_name}.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (canvas_w, canvas_h))

    for frame_idx in range(total):
        canvas = np.zeros((canvas_h, canvas_w, 3), np.uint8)
        for i, (spec, _path, cap) in enumerate(opened):
            frame = read_frame_at(cap, frame_idx, (832, 480))
            tile = letterbox(frame, (tile_width, tile_h))
            title = f"{i + 1}. {spec.label}"
            subtitle = metric_label(spec.key, metrics)
            tile = draw_header(tile, title, subtitle, frame_idx)
            r, c = divmod(i, cols)
            y, x = r * label_tile_h, c * tile_width
            canvas[y:y + label_tile_h, x:x + tile_width] = tile
        writer.write(canvas)

    writer.release()
    for _, _, cap in opened:
        cap.release()
    print(f"[done] {out_path} videos={len(opened)} frames={total} fps={fps:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default=str(DEFAULT_OUT))
    parser.add_argument("--tile_width", type=int, default=320)
    parser.add_argument("--groups", default="all", help="Comma-separated group names, or all")
    args = parser.parse_args()

    metrics = load_metrics()
    names = list(GROUPS) if args.groups == "all" else [x.strip() for x in args.groups.split(",") if x.strip()]
    out_dir = Path(args.out_dir)
    for name in names:
        if name not in GROUPS:
            print(f"[unknown group] {name}")
            continue
        make_block(name, GROUPS[name], out_dir, metrics, args.tile_width)


if __name__ == "__main__":
    main()
