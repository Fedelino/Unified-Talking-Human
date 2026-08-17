#!/usr/bin/env python3
"""Extract Qilin-style aligned DWPose video, with optional FPS resampling.

This intentionally aligns detected keypoints before drawing the stickman. It does
not resize or bbox-warp an already-rendered pose video.
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


INTERACTAVATAR_ROOT = Path("/data1/workspace/linxinliang/InteractAvatar")
DWPOSE_ROOT = INTERACTAVATAR_ROOT / "DWPose"
os.chdir(INTERACTAVATAR_ROOT)
sys.path.insert(0, str(INTERACTAVATAR_ROOT))
sys.path.insert(0, str(DWPOSE_ROOT))

try:
    from DWPose.skeleton_extraction import draw_pose
    from DWPose.dwpose_utils.dwpose_detector import dwpose_detector_aligned
except ImportError:
    from skeleton_extraction import draw_pose
    from dwpose_utils.dwpose_detector import dwpose_detector_aligned


def read_video_resampled(video_path: Path, target_fps: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1e-3:
        source_fps = target_fps

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames:
        raise RuntimeError(f"no frames decoded from video: {video_path}")

    duration = len(frames) / source_fps
    target_count = max(1, int(round(duration * target_fps)))
    positions = np.arange(target_count, dtype=np.float32) * (source_fps / target_fps)
    indices = np.clip(np.round(positions).astype(np.int32), 0, len(frames) - 1)
    return [frames[i] for i in indices], source_fps, duration


def compute_align_params(detected_poses, ref_body, ref_keypoint_id, height: int, width: int):
    valid_bodies = [
        p["bodies"]["candidate"] for p in detected_poses if p["bodies"]["candidate"].shape[0] == 18
    ]
    if len(valid_bodies) == 0 or len(ref_keypoint_id) == 0:
        return np.array([1.0, 1.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)

    detected_bodies = np.stack(valid_bodies)[:, ref_keypoint_id]
    ay, by = np.polyfit(
        detected_bodies[:, :, 1].flatten(),
        np.tile(ref_body[:, 1], len(detected_bodies)),
        1,
    )
    fh = height
    fw = width
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    return np.array([ax, ay], dtype=np.float32), np.array([bx, by], dtype=np.float32)


def thicken_stickman(pose_img_hwc_rgb: np.ndarray, thickness: int):
    if thickness <= 1:
        return pose_img_hwc_rgb
    kernel = np.ones((int(thickness), int(thickness)), dtype=np.uint8)
    return cv2.dilate(pose_img_hwc_rgb, kernel, iterations=1)


def write_video(frames_rgb, output_video: Path, fps: float, width: int, height: int):
    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    chosen_codec = None
    for codec in ("mp4v", "avc1", "MJPG"):
        candidate = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*codec), float(fps), (width, height))
        if candidate.isOpened():
            writer = candidate
            chosen_codec = codec
            break
        candidate.release()
    if writer is None:
        raise RuntimeError(f"failed to open video writer: {output_video}")

    print(f"[write] {output_video} codec={chosen_codec} fps={fps:.3f} size={width}x{height} frames={len(frames_rgb)}")
    for frame_rgb in frames_rgb:
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
    writer.release()


def extract_aligned_pose(input_video: Path, ref_image: Path, output_video: Path, target_fps: float, thickness: int, draw_face: bool):
    ref_bgr = cv2.imread(str(ref_image))
    if ref_bgr is None:
        raise RuntimeError(f"failed to read ref image: {ref_image}")
    ref_rgb = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)
    out_height, out_width = ref_rgb.shape[:2]

    frames_rgb, source_fps, duration = read_video_resampled(input_video, target_fps)
    print(
        f"[read] {input_video} source_fps={source_fps:.3f} duration={duration:.3f}s "
        f"-> target_fps={target_fps:.3f}, frames={len(frames_rgb)}"
    )

    ref_pose = dwpose_detector_aligned(ref_rgb)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [
        i
        for i in ref_keypoint_id
        if len(ref_pose["bodies"]["subset"]) > 0 and ref_pose["bodies"]["subset"][0][i] >= 0.0
    ]
    ref_body = ref_pose["bodies"]["candidate"][ref_keypoint_id] if ref_keypoint_id else np.empty((0, 2))

    detected_poses = []
    for frame_rgb in tqdm(frames_rgb, desc="Detecting DWPose", unit="frame"):
        detected_poses.append(dwpose_detector_aligned(frame_rgb))

    a, b = compute_align_params(detected_poses, ref_body, ref_keypoint_id, out_height, out_width)
    print(f"[align] a={a.tolist()} b={b.tolist()} ref_points={len(ref_keypoint_id)}")

    rendered = []
    for detected_pose in tqdm(detected_poses, desc="Rendering aligned pose", unit="frame"):
        pose = copy.deepcopy(detected_pose)
        pose["bodies"]["candidate"] = pose["bodies"]["candidate"] * a + b
        pose["faces"] = pose["faces"] * a + b
        pose["hands"] = pose["hands"] * a + b
        pose_img_chw = draw_pose(pose, out_height, out_width, draw_face=draw_face)
        pose_img_hwc = np.transpose(pose_img_chw, (1, 2, 0))
        rendered.append(thicken_stickman(pose_img_hwc, thickness=thickness))

    write_video(rendered, output_video, target_fps, out_width, out_height)


def main():
    parser = argparse.ArgumentParser(description="Qilin-style aligned DWPose extraction with target-FPS resampling.")
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--ref-image", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=25.0)
    parser.add_argument("--thickness", type=int, default=1)
    parser.add_argument("--draw-face", action="store_true")
    args = parser.parse_args()

    extract_aligned_pose(
        input_video=args.input_video,
        ref_image=args.ref_image,
        output_video=args.output_video,
        target_fps=args.target_fps,
        thickness=max(1, int(args.thickness)),
        draw_face=args.draw_face,
    )


if __name__ == "__main__":
    main()
