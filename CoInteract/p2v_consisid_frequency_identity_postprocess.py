#!/usr/bin/env python3
"""
ConsisID-inspired frequency identity pass for already-generated P2V videos.

This does not run the official ConsisID pipeline. ConsisID itself is IPT2V and
does not consume a pose video. Here we borrow the practical idea that identity
is partly preserved by anchoring stable facial detail/frequency information:
for each generated frame, use the aligned DWPose face region to add reference
face high-frequency detail back into the generated face with a soft mask.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Box:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return max(1, self.x2 - self.x1)

    @property
    def h(self) -> int:
        return max(1, self.y2 - self.y1)


def clamp_box(box: Box, width: int, height: int) -> Box:
    x1 = max(0, min(width - 1, int(round(box.x1))))
    y1 = max(0, min(height - 1, int(round(box.y1))))
    x2 = max(x1 + 1, min(width, int(round(box.x2))))
    y2 = max(y1 + 1, min(height, int(round(box.y2))))
    return Box(x1, y1, x2, y2)


def expand_box(box: Box, width: int, height: int, scale_x: float, scale_y: float) -> Box:
    cx = (box.x1 + box.x2) * 0.5
    cy = (box.y1 + box.y2) * 0.5
    bw = box.w * scale_x
    bh = box.h * scale_y
    return clamp_box(Box(cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2), width, height)


def non_black_bbox(image_bgr: np.ndarray, threshold: int = 18) -> Box | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.where(gray > threshold)
    if len(xs) == 0:
        return None
    return Box(int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)


def main_pose_body_bbox(pose_bgr: np.ndarray) -> Box | None:
    """Return the dominant/central pose component bbox.

    Whole-body DWPose visualizations may contain background people. A plain
    non-black bbox can accidentally include all of them, so we dilate skeleton
    strokes into connected components and choose the largest body-like region.
    """
    h, w = pose_bgr.shape[:2]
    gray = cv2.cvtColor(pose_bgr, cv2.COLOR_BGR2GRAY)
    mask = (gray > 18).astype(np.uint8) * 255
    kernel_h = max(15, int(round(h * 0.018)) | 1)
    kernel_w = max(15, int(round(w * 0.018)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_w, kernel_h))
    mask = cv2.dilate(mask, kernel, iterations=2)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return non_black_bbox(pose_bgr)

    image_center_x = w * 0.5
    best_score = -1.0
    best_box: Box | None = None
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area < max(64, h * w * 0.0003):
            continue
        box = Box(int(x), int(y), int(x + bw), int(y + bh))
        aspect_score = min(2.5, max(0.2, box.h / float(max(1, box.w))))
        center_penalty = abs(((box.x1 + box.x2) * 0.5) - image_center_x) / max(1.0, image_center_x)
        score = float(area) * aspect_score * (1.0 - 0.35 * center_penalty)
        if score > best_score:
            best_score = score
            best_box = box
    return best_box


def pose_face_bbox(pose_bgr: np.ndarray) -> Box | None:
    """Estimate face bbox from a DWPose visualization.

    DWPose face landmarks are concentrated near the top of the body bbox. When
    landmark colors are unavailable/too sparse, fall back to a head-sized box
    inferred from the body bbox.
    """
    h, w = pose_bgr.shape[:2]
    body = main_pose_body_bbox(pose_bgr)
    if body is None:
        return None

    # Approximate head from the dominant full-body bbox.
    #
    # The full-body DWPose files used in this experiment do not reliably draw
    # dense face landmarks. The highest colored pixels are often raised
    # arms/hands, not face points, so a landmark-top bbox grabs hair/shoulder
    # or background. A torso-centered head estimate is more stable for this
    # training-free post-process.
    head_h = max(16, int(body.h * 0.105))
    head_w = max(16, int(body.h * 0.085))
    cx = (body.x1 + body.x2) * 0.5
    cy = body.y1 + body.h * 0.095
    return clamp_box(Box(cx - head_w / 2, cy - head_h / 2, cx + head_w / 2, cy + head_h / 2), w, h)


def soft_ellipse_mask(height: int, width: int, blur: int = 17) -> np.ndarray:
    mask = np.zeros((height, width), np.float32)
    center = (width // 2, height // 2)
    axes = (max(1, int(width * 0.46)), max(1, int(height * 0.48)))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    blur = max(3, int(blur) | 1)
    mask = cv2.GaussianBlur(mask, (blur, blur), 0)
    return mask[..., None]


def load_haar_face_detector():
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    return None if detector.empty() else detector


def detect_haar_faces(image_bgr: np.ndarray, detector) -> list[Box]:
    if detector is None:
        return []
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(28, 28),
    )
    out: list[Box] = []
    h, w = image_bgr.shape[:2]
    for x, y, bw, bh in faces:
        out.append(clamp_box(Box(int(x), int(y), int(x + bw), int(y + bh)), w, h))
    return out


def choose_face_box(faces: list[Box], width: int, height: int, previous: Box | None = None) -> Box | None:
    if not faces:
        return None

    image_center_x = width * 0.5
    if previous is not None:
        prev_cx = (previous.x1 + previous.x2) * 0.5
        prev_cy = (previous.y1 + previous.y2) * 0.5
        best = None
        best_dist = 1e18
        for face in faces:
            cx = (face.x1 + face.x2) * 0.5
            cy = (face.y1 + face.y2) * 0.5
            dist = ((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best = face
        if best is not None:
            max_jump = max(previous.w, previous.h) * 2.4
            if best_dist <= max_jump:
                return best
        return None

    best_score = -1e18
    best_face = None
    for face in faces:
        cx = (face.x1 + face.x2) * 0.5
        cy = (face.y1 + face.y2) * 0.5
        area = face.w * face.h
        center_penalty = abs(cx - image_center_x) / max(1.0, image_center_x)
        # Prefer the main subject: large face, near image center, upper half.
        vertical_penalty = max(0.0, (cy - height * 0.55) / max(1.0, height * 0.45))
        score = area * (1.0 - 0.35 * center_penalty - 0.45 * vertical_penalty)
        if score > best_score:
            best_score = score
            best_face = face
    return best_face


def transfer_frequency(frame_face: np.ndarray, ref_face: np.ndarray, alpha: float, color_alpha: float) -> np.ndarray:
    ref_face = cv2.resize(ref_face, (frame_face.shape[1], frame_face.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    frame_f = frame_face.astype(np.float32)
    ref_f = ref_face.astype(np.float32)

    sigma = max(2.0, min(frame_face.shape[:2]) / 18.0)
    ref_low = cv2.GaussianBlur(ref_f, (0, 0), sigma)
    ref_high = ref_f - ref_low
    enhanced = frame_f + float(alpha) * ref_high

    if color_alpha > 0:
        # A very weak low-frequency color pull helps lips/skin tone without fully pasting the reference.
        frame_low = cv2.GaussianBlur(frame_f, (0, 0), sigma)
        enhanced = enhanced + float(color_alpha) * (ref_low - frame_low)

    return np.clip(enhanced, 0, 255).astype(np.uint8)


def process_video(args: argparse.Namespace) -> None:
    ref = cv2.imread(args.reference_image)
    ref_pose = cv2.imread(args.reference_pose)
    if ref is None:
        raise FileNotFoundError(args.reference_image)
    if ref_pose is None:
        raise FileNotFoundError(args.reference_pose)

    cap = cv2.VideoCapture(args.input_video)
    pose_cap = cv2.VideoCapture(args.aligned_pose)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open input video: {args.input_video}")
    if not pose_cap.isOpened():
        raise RuntimeError(f"failed to open aligned pose video: {args.aligned_pose}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ref = cv2.resize(ref, (width, height), interpolation=cv2.INTER_LANCZOS4)
    ref_pose = cv2.resize(ref_pose, (width, height), interpolation=cv2.INTER_LINEAR)
    face_detector = load_haar_face_detector()
    ref_box = choose_face_box(detect_haar_faces(ref, face_detector), width, height)
    if ref_box is None:
        ref_box = pose_face_bbox(ref_pose)
    if ref_box is None:
        raise RuntimeError("failed to estimate reference face bbox")
    ref_box = expand_box(ref_box, width, height, args.reference_face_expand, args.reference_face_expand)
    ref_face = ref[ref_box.y1 : ref_box.y2, ref_box.x1 : ref_box.x2].copy()

    os.makedirs(os.path.dirname(args.output_video), exist_ok=True)
    temp_video = args.output_video
    mux_audio = args.copy_audio and args.output_video.endswith(".mp4")
    if mux_audio:
        fd, temp_video = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)

    writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open writer: {temp_video}")

    compare_writer = None
    if args.compare_video:
        os.makedirs(os.path.dirname(args.compare_video), exist_ok=True)
        compare_writer = cv2.VideoWriter(args.compare_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * 2, height))

    frame_idx = 0
    last_box: Box | None = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        pose_ok, pose = pose_cap.read()
        if not pose_ok:
            pose = None

        out = frame.copy()
        faces = detect_haar_faces(frame, face_detector)
        box = choose_face_box(faces, width, height, previous=last_box)
        if box is None and last_box is not None:
            box = last_box
        if box is None:
            box = pose_face_bbox(pose) if pose is not None else None
        if box is None:
            box = last_box
        if box is not None:
            box = expand_box(box, width, height, args.face_expand_x, args.face_expand_y)
            face = frame[box.y1 : box.y2, box.x1 : box.x2]
            if face.size:
                enhanced_face = transfer_frequency(face, ref_face, args.alpha, args.color_alpha)
                mask = soft_ellipse_mask(face.shape[0], face.shape[1], args.mask_blur)
                blended = (enhanced_face.astype(np.float32) * mask + face.astype(np.float32) * (1.0 - mask)).astype(np.uint8)
                out[box.y1 : box.y2, box.x1 : box.x2] = blended
                last_box = box

        writer.write(out)
        if compare_writer is not None:
            compare_writer.write(np.concatenate([frame, out], axis=1))
        frame_idx += 1

    cap.release()
    pose_cap.release()
    writer.release()
    if compare_writer is not None:
        compare_writer.release()

    if mux_audio:
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            temp_video,
            "-i",
            args.input_video,
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "copy",
            "-shortest",
            args.output_video,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.remove(temp_video)

    print(f"processed_frames={frame_idx}")
    print(f"reference_face_box={ref_box}")
    print(f"output_video={args.output_video}")
    if args.compare_video:
        print(f"compare_video={args.compare_video}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_video", required=True)
    parser.add_argument("--aligned_pose", required=True)
    parser.add_argument("--reference_image", required=True)
    parser.add_argument("--reference_pose", required=True)
    parser.add_argument("--output_video", required=True)
    parser.add_argument("--compare_video", default="")
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--color_alpha", type=float, default=0.08)
    parser.add_argument("--face_expand_x", type=float, default=1.15)
    parser.add_argument("--face_expand_y", type=float, default=1.10)
    parser.add_argument("--reference_face_expand", type=float, default=1.20)
    parser.add_argument("--mask_blur", type=int, default=25)
    parser.add_argument("--copy_audio", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    process_video(parse_args())
