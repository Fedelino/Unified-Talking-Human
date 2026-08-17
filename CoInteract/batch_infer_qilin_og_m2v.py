"""
CoInteract Batch Inference Script

Generate speech-driven human-object interaction videos from a CSV file.
Each row in the CSV should contain: person_image and pose_video columns.
Optional columns: sample_id, prompt, audio, product_image, scale, prompt2, prompt3.
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ascend NPU support: monkey-patch torch.cuda.* -> torch.npu.*, nccl -> hccl, etc.
# Must run before any torch.cuda usage. No-op on non-NPU machines.
try:
    import torch_npu  # noqa: F401
    from torch_npu.contrib import transfer_to_npu  # noqa: F401
except ImportError:
    pass

import argparse
import math
import shutil
import subprocess
import tempfile
import wave
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import librosa
import pandas as pd
from pathlib import Path
import torchvision.transforms.functional as TF
import cv2

from diffsynth import VideoData, save_video_with_audio
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig, WanVideoUnit_S2V
from diffsynth.models.utils import load_state_dict

# Auto-detect device: NPU > CUDA > CPU
DEVICE = "npu" if torch.npu.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="CoInteract Batch Inference")
    # Model paths
    parser.add_argument("--base_model_path", type=str, default="./models/Wan2.2-S2V-14B",
                        help="Path to Wan2.2-S2V-14B base model directory")
    parser.add_argument("--audio_encoder_path", type=str, default="./models/chinese-wav2vec2-large",
                        help="Path to chinese-wav2vec2-large model directory")
    parser.add_argument("--lora_path", type=str, default="./models/CoInteract/checkpoint.safetensors",
                        help="Path to LoRA checkpoint (safetensors)")
    parser.add_argument("--lora_alpha", type=float, default=1.0,
                        help="LoRA alpha scale factor")
    # MoE config
    parser.add_argument("--use_moe", action="store_true", default=True,
                        help="Enable Human-Aware MoE FFN")
    parser.add_argument("--no_use_moe", action="store_false", dest="use_moe",
                        help="Disable Human-Aware MoE FFN for ablation.")
    parser.add_argument("--expert_hidden_dim", type=int, default=256,
                        help="Hidden dimension for MoE expert networks")
    parser.add_argument("--use_audio_face_mask", action="store_true", default=False,
                        help="Enable Audio Face Mask (audio controls face region only)")
    # Generation config
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to input CSV file")
    # Project root: directory where this script lives
    _PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--data_base_path", type=str, default=_PROJECT_ROOT,
                        help="Base path for resolving relative paths in CSV (default: project root)")
    parser.add_argument("--output_dir", type=str, default="./output_videos",
                        help="Output directory for generated videos")
    parser.add_argument("--height", type=int, default=832)
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--num_frames", type=int, default=80,
                        help="Number of frames per clip (80 frames = 3.24 seconds at 25fps)")
    parser.add_argument("--num_clips", type=int, default=3,
                        help="Number of clips to generate per sample")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sigma_shift", type=float, default=7.0,
                        help="Noise schedule time shift parameter")
    parser.add_argument("--negative_prompt", type=str,
                        default="Blurry, worst quality, blurred details, static frame, "
                                "violent emotions, rapid hand shaking, subtitles, ugly, "
                                "deformed, extra fingers, poorly drawn hands, poorly drawn face")
    parser.add_argument("--pose_align_mode", type=str, default="none", choices=["none", "bbox"],
                        help="Spatial pose preprocessing mode. Use 'none' to avoid bbox-driven camera drift.")
    parser.add_argument("--identity_layout", type=str, default="single", choices=["single", "inset_triptych"],
                        help="Training-free identity reference layout. 'inset_triptych' uses full-body + upper-body + face crops.")
    parser.add_argument("--reference_compose_mode", type=str, default="pad", choices=["pad", "stretch"],
                        help="How to map a single reference image to the model canvas. 'stretch' matches OG CoInteract behavior.")
    parser.add_argument("--reference_preprocess_mode", type=str, default="none",
                        choices=["none", "face_boost", "face_upper_boost", "face_only"],
                        help="Training-free identity preprocessing applied before the single reference image is sent to the model.")
    parser.add_argument("--identity_inset_scale", type=float, default=1.0,
                        help="Scale factor for upper-body/face inset panels. Values >1 allocate more canvas pixels to identity details.")
    parser.add_argument("--identity_crop_enhance", type=str, default="none", choices=["none", "sharpen", "upscale_sharpen"],
                        help="Training-free enhancement for upper-body/face crops before composing the identity board.")
    parser.add_argument("--save_identity_debug", action="store_true", default=False,
                        help="Save the constructed identity board image for debugging.")
    parser.add_argument("--save_reference_debug", action="store_true", default=False,
                        help="Save the preprocessed single reference image used for conditioning.")
    parser.add_argument("--initial_motion_video_path", type=str, default=None,
                        help="Optional generated video whose last frames initialize S2V motion memory.")
    parser.add_argument("--initial_motion_frames", type=int, default=73,
                        help="Number of frames to keep from initial_motion_video_path for motion memory.")
    return parser.parse_args()


def resize_and_pad(image: Image.Image, target_height: int, target_width: int,
                   pad_color=(0, 0, 0)) -> Image.Image:
    """
    Resize image preserving aspect ratio, then pad to target size (center-aligned).
    Consistent with training-time ImageResizeAndPad preprocessing.
    """
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    scale = scale / 1.25  # Scale down slightly to avoid cropping artifacts

    new_width = round(width * scale)
    new_height = round(height * scale)

    interpolation = (TF.InterpolationMode.LANCZOS if scale < 1
                     else TF.InterpolationMode.BILINEAR)
    image = TF.resize(image, (new_height, new_width), interpolation=interpolation)

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    pad_top = (target_height - new_height) // 2
    pad_bottom = target_height - new_height - pad_top
    image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)

    return image


def resize_stretch(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    interpolation = (
        Image.Resampling.LANCZOS
        if image.width > target_width or image.height > target_height
        else Image.Resampling.BILINEAR
    )
    return image.resize((target_width, target_height), interpolation)


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int):
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def expand_box(box, image_width: int, image_height: int, scale_x: float = 1.0, scale_y: float = 1.0):
    x1, y1, x2, y2 = box
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bw = max(2.0, float(x2 - x1))
    bh = max(2.0, float(y2 - y1))
    half_w = 0.5 * bw * scale_x
    half_h = 0.5 * bh * scale_y
    return clamp_box(
        int(round(cx - half_w)),
        int(round(cy - half_h)),
        int(round(cx + half_w)),
        int(round(cy + half_h)),
        image_width,
        image_height,
    )


def infer_body_bbox_from_pose_image(ref_pose_image_path: str | None, image_size):
    if ref_pose_image_path is None or not os.path.exists(ref_pose_image_path):
        return None
    ref_pose = cv2.imread(ref_pose_image_path)
    if ref_pose is None:
        return None
    image_width, image_height = image_size
    ref_pose = cv2.resize(ref_pose, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    return compute_pose_bbox(ref_pose)


def detect_face_bbox(image: Image.Image):
    try:
        cascade_dir = getattr(cv2.data, "haarcascades", None)
        if cascade_dir is None:
            return None
        cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            return None
        detector = cv2.CascadeClassifier(cascade_path)
        if detector.empty():
            return None
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(24, 24))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        return int(x), int(y), int(x + w), int(y + h)
    except Exception:
        return None


def derive_identity_crops(image: Image.Image, ref_pose_image_path: str | None):
    image_width, image_height = image.size
    body_box = infer_body_bbox_from_pose_image(ref_pose_image_path, image.size)
    if body_box is None:
        body_box = (0, 0, image_width, image_height)
    body_box = expand_box(body_box, image_width, image_height, scale_x=1.12, scale_y=1.08)

    bx1, by1, bx2, by2 = body_box
    bw = max(2, bx2 - bx1)
    bh = max(2, by2 - by1)
    cx = 0.5 * (bx1 + bx2)

    upper_half_h = int(round(bh * 0.34))
    upper_box = clamp_box(
        int(round(cx - bw * 0.42)),
        int(round(by1 + bh * 0.02)),
        int(round(cx + bw * 0.42)),
        int(round(by1 + bh * 0.02 + upper_half_h)),
        image_width,
        image_height,
    )
    upper_box = expand_box(upper_box, image_width, image_height, scale_x=1.18, scale_y=1.18)

    face_box = detect_face_bbox(image)
    if face_box is None:
        face_box = clamp_box(
            int(round(cx - bw * 0.18)),
            int(round(by1 + bh * 0.02)),
            int(round(cx + bw * 0.18)),
            int(round(by1 + bh * 0.30)),
            image_width,
            image_height,
        )
    face_box = expand_box(face_box, image_width, image_height, scale_x=1.45, scale_y=1.45)

    upper_crop = image.crop(upper_box)
    face_crop = image.crop(face_box)
    return upper_crop, face_crop


def derive_identity_boxes(image: Image.Image, ref_pose_image_path: str | None):
    image_width, image_height = image.size
    body_box = infer_body_bbox_from_pose_image(ref_pose_image_path, image.size)
    if body_box is None:
        body_box = (0, 0, image_width, image_height)
    body_box = expand_box(body_box, image_width, image_height, scale_x=1.12, scale_y=1.08)

    bx1, by1, bx2, by2 = body_box
    bw = max(2, bx2 - bx1)
    bh = max(2, by2 - by1)
    cx = 0.5 * (bx1 + bx2)

    upper_half_h = int(round(bh * 0.34))
    upper_box = clamp_box(
        int(round(cx - bw * 0.42)),
        int(round(by1 + bh * 0.02)),
        int(round(cx + bw * 0.42)),
        int(round(by1 + bh * 0.02 + upper_half_h)),
        image_width,
        image_height,
    )
    upper_box = expand_box(upper_box, image_width, image_height, scale_x=1.18, scale_y=1.18)

    face_box = detect_face_bbox(image)
    if face_box is None:
        face_box = clamp_box(
            int(round(cx - bw * 0.18)),
            int(round(by1 + bh * 0.02)),
            int(round(cx + bw * 0.18)),
            int(round(by1 + bh * 0.30)),
            image_width,
            image_height,
        )
    face_box = expand_box(face_box, image_width, image_height, scale_x=1.45, scale_y=1.45)
    return body_box, upper_box, face_box


def enhance_identity_crop(image: Image.Image, mode: str) -> Image.Image:
    if mode == "none":
        return image
    enhanced = image.convert("RGB")
    if mode == "upscale_sharpen":
        enhanced = enhanced.resize(
            (max(2, enhanced.width * 2), max(2, enhanced.height * 2)),
            Image.Resampling.LANCZOS,
        )
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.8, percent=175, threshold=3))
    enhanced = ImageEnhance.Contrast(enhanced).enhance(1.06)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(1.15)
    return enhanced


def soft_paste(base_image: Image.Image, patch_image: Image.Image, box, blur_radius: int = 12) -> Image.Image:
    x1, y1, x2, y2 = box
    target_w = max(1, x2 - x1)
    target_h = max(1, y2 - y1)
    patch = patch_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    mask = Image.new("L", (target_w, target_h), color=255)
    if blur_radius > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    base = base_image.copy()
    base.paste(patch, (x1, y1), mask)
    return base


def preprocess_reference_image(
    image: Image.Image,
    ref_pose_image_path: str | None,
    preprocess_mode: str,
) -> Image.Image:
    image = image.convert("RGB")
    if preprocess_mode == "none":
        return image

    _, upper_box, face_box = derive_identity_boxes(image, ref_pose_image_path)

    if preprocess_mode == "face_only":
        face_crop = image.crop(face_box)
        return face_crop.resize(image.size, Image.Resampling.LANCZOS)

    result = image.copy()
    face_crop = enhance_identity_crop(image.crop(face_box), "upscale_sharpen")
    face_blur = max(8, int(round(min(face_box[2] - face_box[0], face_box[3] - face_box[1]) * 0.08)))
    result = soft_paste(result, face_crop, face_box, blur_radius=face_blur)

    if preprocess_mode == "face_upper_boost":
        upper_crop = enhance_identity_crop(image.crop(upper_box), "sharpen")
        upper_blur = max(10, int(round(min(upper_box[2] - upper_box[0], upper_box[3] - upper_box[1]) * 0.06)))
        result = soft_paste(result, upper_crop, upper_box, blur_radius=upper_blur)

    return result


def build_identity_reference_image(
    image: Image.Image,
    ref_pose_image_path: str | None,
    target_height: int,
    target_width: int,
    identity_layout: str,
    reference_compose_mode: str = "pad",
    reference_preprocess_mode: str = "none",
    identity_inset_scale: float = 1.0,
    identity_crop_enhance: str = "none",
):
    image = preprocess_reference_image(image, ref_pose_image_path, reference_preprocess_mode)
    if reference_compose_mode == "stretch":
        base = resize_stretch(image, target_height=target_height, target_width=target_width)
    else:
        base = resize_and_pad(image, target_height=target_height, target_width=target_width, pad_color=(0, 0, 0))
    if identity_layout == "single":
        return base

    upper_crop, face_crop = derive_identity_crops(image, ref_pose_image_path)
    upper_crop = enhance_identity_crop(upper_crop, identity_crop_enhance)
    face_crop = enhance_identity_crop(face_crop, identity_crop_enhance)
    board = base.copy()

    inset_scale = max(0.6, float(identity_inset_scale))
    right_panel_width = max(72, int(round(target_width * 0.28 * inset_scale)))
    upper_panel_height = max(96, int(round(target_height * 0.34 * inset_scale)))
    face_panel_height = max(80, int(round(target_height * 0.24 * inset_scale)))
    right_panel_width = min(right_panel_width, int(round(target_width * 0.44)))
    upper_panel_height = min(upper_panel_height, int(round(target_height * 0.50)))
    face_panel_height = min(face_panel_height, int(round(target_height * 0.34)))
    margin = max(10, int(round(target_width * 0.025)))
    border = 4

    upper_panel = resize_and_pad(
        upper_crop,
        target_height=upper_panel_height,
        target_width=right_panel_width,
        pad_color=(8, 8, 8),
    )
    face_panel = resize_and_pad(
        face_crop,
        target_height=face_panel_height,
        target_width=right_panel_width,
        pad_color=(8, 8, 8),
    )

    def framed(panel: Image.Image):
        return ImageOps.expand(panel, border=border, fill=(245, 245, 245))

    upper_panel = framed(upper_panel)
    face_panel = framed(face_panel)

    upper_x = target_width - upper_panel.width - margin
    upper_y = margin
    face_x = target_width - face_panel.width - margin
    face_y = min(target_height - face_panel.height - margin, upper_y + upper_panel.height + margin)

    board.paste(upper_panel, (upper_x, upper_y))
    board.paste(face_panel, (face_x, face_y))
    return board


def infer_ref_pose_image_path(image_path: str) -> str | None:
    path = Path(image_path)
    path_str = str(path)
    if "/ref_img/" in path_str:
        candidate = path_str.replace("/ref_img/", "/ref_pose/")
    elif "\\ref_img\\" in path_str:
        candidate = path_str.replace("\\ref_img\\", "\\ref_pose\\")
    else:
        return None
    candidate_path = Path(candidate)
    return str(candidate_path.with_name(f"{candidate_path.stem}_pose.png"))


def compute_pose_bbox(image_bgr: np.ndarray, threshold: int = 10):
    if image_bgr is None:
        return None
    mask = image_bgr.max(axis=2) > threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def align_pose_frame_to_reference(frame_bgr: np.ndarray, ref_bbox, target_width: int, target_height: int) -> np.ndarray:
    resized = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    src_bbox = compute_pose_bbox(resized)
    if src_bbox is None or ref_bbox is None:
        return resized

    src_x1, src_y1, src_x2, src_y2 = src_bbox
    ref_x1, ref_y1, ref_x2, ref_y2 = ref_bbox
    src_w = max(1, src_x2 - src_x1 + 1)
    src_h = max(1, src_y2 - src_y1 + 1)
    ref_w = max(1, ref_x2 - ref_x1 + 1)
    ref_h = max(1, ref_y2 - ref_y1 + 1)

    scale = min(ref_h / src_h, ref_w / src_w) * 0.98
    new_w = max(1, int(round(target_width * scale)))
    new_h = max(1, int(round(target_height * scale)))
    scaled = cv2.resize(resized, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    scaled_bbox = compute_pose_bbox(scaled)
    if scaled_bbox is None:
        return resized

    scaled_x1, scaled_y1, scaled_x2, scaled_y2 = scaled_bbox
    scaled_cx = 0.5 * (scaled_x1 + scaled_x2)
    scaled_bottom = scaled_y2

    ref_cx = 0.5 * (ref_x1 + ref_x2)
    ref_bottom = ref_y2

    offset_x = int(round(ref_cx - scaled_cx))
    offset_y = int(round(ref_bottom - scaled_bottom))

    aligned = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    src_left = max(0, -offset_x)
    src_top = max(0, -offset_y)
    dst_left = max(0, offset_x)
    dst_top = max(0, offset_y)
    copy_w = min(new_w - src_left, target_width - dst_left)
    copy_h = min(new_h - src_top, target_height - dst_top)
    if copy_w <= 0 or copy_h <= 0:
        return resized

    aligned[dst_top:dst_top + copy_h, dst_left:dst_left + copy_w] = scaled[src_top:src_top + copy_h, src_left:src_left + copy_w]
    return aligned


def _nearest_multiple_of_four(value: float) -> int:
    return max(4, int(round(value / 4.0) * 4))


def prepare_pose_video_for_reference(
    pose_video_path: str,
    ref_pose_image_path: str | None,
    target_width: int,
    target_height: int,
    target_fps: float,
    base_num_frames: int,
    pose_align_mode: str = "none",
    debug_pose_path: str | None = None,
):
    cap = cv2.VideoCapture(pose_video_path)
    if not cap.isOpened():
        raise RuntimeError(f"failed to open pose video: {pose_video_path}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 1e-3:
        source_fps = target_fps

    source_frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        source_frames.append(frame)
    cap.release()

    if not source_frames:
        raise RuntimeError(f"pose video has no readable frames: {pose_video_path}")

    duration_seconds = len(source_frames) / source_fps
    resampled_count = max(1, int(round(duration_seconds * target_fps)))
    src_positions = np.arange(resampled_count, dtype=np.float32) * (source_fps / target_fps)
    src_indices = np.clip(np.round(src_positions).astype(np.int32), 0, len(source_frames) - 1)
    resampled_frames = [source_frames[idx] for idx in src_indices]

    effective_num_clips = max(1, int(math.ceil(len(resampled_frames) / float(base_num_frames))))
    effective_num_frames = _nearest_multiple_of_four(len(resampled_frames) / float(effective_num_clips))
    total_required_frames = effective_num_clips * effective_num_frames

    if len(resampled_frames) < total_required_frames:
        resampled_frames.extend([resampled_frames[-1]] * (total_required_frames - len(resampled_frames)))
    elif len(resampled_frames) > total_required_frames:
        resampled_frames = resampled_frames[:total_required_frames]

    ref_bbox = None
    if pose_align_mode == "bbox" and ref_pose_image_path and os.path.exists(ref_pose_image_path):
        ref_pose = cv2.imread(ref_pose_image_path)
        if ref_pose is not None:
            ref_pose = cv2.resize(ref_pose, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            ref_bbox = compute_pose_bbox(ref_pose)

    if pose_align_mode == "bbox" and ref_bbox is not None:
        aligned_frames = [
            align_pose_frame_to_reference(frame, ref_bbox, target_width, target_height)
            for frame in resampled_frames
        ]
    else:
        aligned_frames = [
            cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
            for frame in resampled_frames
        ]

    if debug_pose_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        debug_pose_path = temp_file.name
        temp_file.close()

    writer = cv2.VideoWriter(
        debug_pose_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(target_fps),
        (target_width, target_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open pose writer: {debug_pose_path}")
    for frame in aligned_frames:
        writer.write(frame)
    writer.release()

    return {
        "pose_video_path": debug_pose_path,
        "num_clips": effective_num_clips,
        "num_frames": effective_num_frames,
        "duration_seconds": total_required_frames / float(target_fps),
        "source_duration_seconds": duration_seconds,
        "source_fps": source_fps,
        "target_frame_count": total_required_frames,
        "ref_pose_image_path": ref_pose_image_path,
        "pose_align_mode": pose_align_mode,
    }


def build_silent_audio(duration_seconds: float, sample_rate: int) -> np.ndarray:
    num_samples = max(1, int(round(duration_seconds * sample_rate)))
    return np.zeros(num_samples, dtype=np.float32)


def write_audio_wav(audio_path: str, audio_array: np.ndarray, sample_rate: int) -> str:
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    pcm = (audio_array * 32767.0).astype(np.int16)
    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return audio_path


def resize_video_to_resolution(video_path: str, target_width: int, target_height: int) -> None:
    if target_width <= 0 or target_height <= 0 or not os.path.exists(video_path):
        return

    temp_output = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_output.close()
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vf",
        f"scale={target_width}:{target_height}",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        temp_output.name,
    ]
    try:
        subprocess.run(
            ffmpeg_cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            os.replace(temp_output.name, video_path)
        except OSError:
            shutil.move(temp_output.name, video_path)
    finally:
        if os.path.exists(temp_output.name):
            os.remove(temp_output.name)


def normalize_motion_memory(frames, motion_frames: int):
    if not frames:
        return []
    frames = list(frames)[-max(1, int(motion_frames)):]
    if len(frames) < motion_frames:
        frames = [frames[0]] * (motion_frames - len(frames)) + frames
    return frames


def load_initial_motion_frames(video_path: str | None, target_height: int, target_width: int, motion_frames: int):
    if video_path is None or not str(video_path).strip():
        return []
    if not os.path.exists(video_path):
        print(f"[motion_memory] missing initial motion video: {video_path}")
        return []

    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_bgr = cv2.resize(frame_bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()

    if not frames:
        print(f"[motion_memory] no frames decoded from {video_path}")
        return []
    frames = normalize_motion_memory(frames, motion_frames)
    print(f"[motion_memory] loaded {len(frames)} initial motion frames from {video_path}")
    return frames


def prepare_audio_conditioning(audio_path, audio_sample_rate, num_frames, fps, num_clips, duration_override_seconds=None):
    if audio_path is not None and str(audio_path).strip():
        if os.path.exists(audio_path):
            input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
            return input_audio, sample_rate, audio_path, None
        print(f"[audio] missing audio path {audio_path}; falling back to silence")

    effective_clips = max(1, int(num_clips) if num_clips is not None else 1)
    duration_seconds = max(num_frames * effective_clips / float(fps), 1.0 / float(fps))
    if duration_override_seconds is not None:
        duration_seconds = max(float(duration_override_seconds), 1.0 / float(fps))
    input_audio = build_silent_audio(duration_seconds, audio_sample_rate)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_audio.close()
    write_audio_wav(temp_audio.name, input_audio, audio_sample_rate)
    print(f"[audio] no audio provided; generated silent conditioning track: {temp_audio.name}")
    return input_audio, audio_sample_rate, temp_audio.name, temp_audio.name


def speech_to_video(pipe, prompts, person_image, audio_path,
                    product_image=None, product_image_scale=1.0,
                    negative_prompt="", num_clips=None,
                    audio_sample_rate=16000, pose_video_path=None,
                    num_frames=80, height=448, width=832,
                    num_inference_steps=40, fps=25, motion_frames=73,
                    save_path=None, sigma_shift=5.0, cfg_scale=5.0,
                    duration_override_seconds=None,
                    initial_motion_video_path=None):
    """
    Generate video from speech audio and reference image.
    
    Args:
        prompts: list of prompt strings, one per clip. If fewer prompts than clips,
                 the last prompt is reused for remaining clips.
    """
    input_audio, sample_rate, audio_path_for_save, temp_audio_path = prepare_audio_conditioning(
        audio_path=audio_path,
        audio_sample_rate=audio_sample_rate,
        num_frames=num_frames,
        fps=fps,
        num_clips=num_clips,
        duration_override_seconds=duration_override_seconds,
    )
    pose_video = (VideoData(pose_video_path, height=height, width=width)
                  if pose_video_path else None)

    audio_embeds, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
        pipe=pipe,
        input_audio=input_audio,
        audio_sample_rate=sample_rate,
        s2v_pose_video=pose_video,
        num_frames=num_frames + 1,
        height=height,
        width=width,
        fps=fps,
    )
    num_repeat = min(num_repeat, num_clips) if num_clips is not None else num_repeat
    print(f"Generating {num_repeat} video clip(s)...")

    motion_videos = load_initial_motion_frames(
        initial_motion_video_path,
        target_height=height,
        target_width=width,
        motion_frames=motion_frames,
    )
    video = []

    try:
        for r in range(num_repeat):
            current_prompt = prompts[min(r, len(prompts) - 1)]

            s2v_pose_latents = pose_latents[r] if pose_latents is not None else None
            current_clip = pipe(
                prompt=current_prompt,
                person_image=person_image,
                product_image=product_image,
                product_image_scale=product_image_scale,
                negative_prompt=negative_prompt,
                seed=r,
                num_frames=num_frames + 1,
                height=height,
                width=width,
                audio_embeds=audio_embeds[r],
                s2v_pose_latents=s2v_pose_latents,
                motion_video=motion_videos,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                cfg_scale=cfg_scale,
            )
            current_clip = current_clip[-num_frames:]

            overlap_frames_num = min(motion_frames, len(current_clip))
            motion_videos = normalize_motion_memory(
                motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:],
                motion_frames,
            )
            video.extend(current_clip)
            save_video_with_audio(video, save_path, audio_path_for_save, fps=25, quality=5)
            print(f"  Processed clip {r+1}/{num_repeat}")
    finally:
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return video


def load_moe_weights(model, state_dict, device=DEVICE, dtype=torch.bfloat16):
    """Load MoE-specific weights (router, hand_expert, face_expert) from checkpoint."""
    moe_keys = ['router', 'hand_expert', 'face_expert']
    moe_state_dict = {}

    for key in state_dict:
        if not any(k in key for k in moe_keys):
            continue
        model_key = key[len('diffusion_model.'):] if key.startswith('diffusion_model.') else key
        moe_state_dict[model_key] = state_dict[key].to(device=device, dtype=dtype)

    if moe_state_dict:
        model.load_state_dict(moe_state_dict, strict=False)
        print(f"[MoE] Loaded {len(moe_state_dict)} MoE weights")
    else:
        print("[MoE] Warning: No MoE weights found in checkpoint")

    return len(moe_state_dict)


def load_model(args):
    """Load pipeline with all model components."""
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=DEVICE,
        model_configs=[
            ModelConfig(path=[
                f"{args.base_model_path}/diffusion_pytorch_model-00001-of-00004.safetensors",
                f"{args.base_model_path}/diffusion_pytorch_model-00002-of-00004.safetensors",
                f"{args.base_model_path}/diffusion_pytorch_model-00003-of-00004.safetensors",
                f"{args.base_model_path}/diffusion_pytorch_model-00004-of-00004.safetensors",
            ]),
            ModelConfig(path=f"{args.base_model_path}/models_t5_umt5-xxl-enc-bf16.pth"),
            ModelConfig(path=f"{args.audio_encoder_path}/pytorch_model.bin"),
            ModelConfig(path=f"{args.base_model_path}/Wan2.1_VAE.pth"),
        ],
        tokenizer_config=ModelConfig(path=f"{args.base_model_path}/google/umt5-xxl"),
        audio_processor_config=ModelConfig(path=args.audio_encoder_path),
    )

    # Enable Human-Aware MoE FFN
    if args.use_moe:
        if hasattr(pipe.dit, 'enable_moe_ffn'):
            pipe.dit.enable_moe_ffn(expert_hidden_dim=args.expert_hidden_dim)
            print(f"[MoE] Enabled with expert_hidden_dim={args.expert_hidden_dim}")

    # Enable Audio Face Mask
    if args.use_audio_face_mask and args.use_moe:
        if hasattr(pipe.dit, 'set_audio_face_mask_config'):
            pipe.dit.set_audio_face_mask_config(
                use_audio_face_mask=True,
                audio_mask_train_source="router",
            )
            print("[Audio Face Mask] Enabled (source=router)")

    # Load LoRA + MoE weights
    if args.lora_path is not None:
        print(f"Loading checkpoint: {args.lora_path}")
        pipe.load_lora(module=pipe.dit, lora_config=args.lora_path, alpha=args.lora_alpha)
        print(f"  LoRA loaded (alpha={args.lora_alpha})")

        if args.use_moe:
            ckpt_state = load_state_dict(args.lora_path, torch_dtype=torch.bfloat16, device=DEVICE)
            load_moe_weights(pipe.dit, ckpt_state, device=DEVICE, dtype=torch.bfloat16)
            pipe.dit = pipe.dit.to(device=DEVICE, dtype=torch.bfloat16)

    # Enable VRAM management: automatically offload idle models (text_encoder, vae,
    # audio_encoder) to CPU and only keep the active model on GPU.
    pipe.vram_management_enabled = True

    return pipe


def process_batch(args, pipe):
    """Process all samples from the CSV file."""
    os.makedirs(args.output_dir, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    total = len(df)
    print(f"Total samples: {total}")

    for idx, row in df.iterrows():
        try:
            audio_rel = row['audio'] if 'audio' in df.columns else None
            image_rel = row['person_image']

            # Read prompts: prompt (required), prompt2/prompt3 (optional)
            prompt_val = row.get('prompt', '') if 'prompt' in df.columns else ''
            prompt1 = '' if pd.isna(prompt_val) else str(prompt_val).strip()
            prompt2 = str(row['prompt2']).strip() if 'prompt2' in df.columns and pd.notna(row.get('prompt2')) else None
            prompt3 = str(row['prompt3']).strip() if 'prompt3' in df.columns and pd.notna(row.get('prompt3')) else None

            # Build prompts list: [prompt1] or [prompt1, prompt2] or [prompt1, prompt2, prompt3]
            prompts = [prompt1]
            if prompt2:
                prompts.append(prompt2)
            if prompt3:
                prompts.append(prompt3)

            # Resolve paths
            audio_path = None
            audio_rel_str = None
            if audio_rel is not None and pd.notna(audio_rel) and str(audio_rel).strip():
                audio_rel_str = str(audio_rel).strip()
                audio_path = (audio_rel_str if os.path.isabs(audio_rel_str)
                              else os.path.join(args.data_base_path, audio_rel_str))
            image_path = (image_rel if os.path.isabs(image_rel)
                          else os.path.join(args.data_base_path, image_rel))

            # Product reference image (optional column)
            product_ref_path = None
            if 'product_image' in df.columns:
                val = row.get('product_image')
                if val is not None and pd.notna(val) and str(val).strip():
                    product_ref_path = (str(val) if os.path.isabs(str(val))
                                        else os.path.join(args.data_base_path, str(val)))

            product_image_scale = float(row.get('scale', 1.0)) if 'scale' in df.columns else 1.0

            # Pose-driven reference video (optional column)
            pose_video_path = None
            if 'pose_video' in df.columns:
                val = row.get('pose_video')
                if val is not None and pd.notna(val) and str(val).strip():
                    pose_video_path = (str(val) if os.path.isabs(str(val))
                                       else os.path.join(args.data_base_path, str(val)))

            sample_name = (str(row.get("sample_id")).strip()
                           if "sample_id" in df.columns and pd.notna(row.get("sample_id")) and str(row.get("sample_id")).strip()
                           else Path(audio_rel_str or image_rel).stem)
            save_path = os.path.join(args.output_dir, f"{sample_name}.mp4")

            if os.path.exists(save_path):
                print(f"[{idx+1}/{total}] Skip (exists): {sample_name}")
                continue

            print(f"\n[{idx+1}/{total}] Processing: {sample_name}")
            original_person_image = Image.open(image_path).convert("RGB")
            reference_width, reference_height = original_person_image.size
            ref_pose_image_path = infer_ref_pose_image_path(image_path)
            person_image = build_identity_reference_image(
                original_person_image,
                ref_pose_image_path=ref_pose_image_path,
                target_height=args.height,
                target_width=args.width,
                identity_layout=args.identity_layout,
                reference_compose_mode=args.reference_compose_mode,
                reference_preprocess_mode=args.reference_preprocess_mode,
                identity_inset_scale=args.identity_inset_scale,
                identity_crop_enhance=args.identity_crop_enhance,
            )
            if args.save_identity_debug:
                debug_img_path = os.path.join(args.output_dir, f"{sample_name}__identity_board.png")
                person_image.save(debug_img_path)
            if args.save_reference_debug:
                ref_debug = preprocess_reference_image(
                    original_person_image,
                    ref_pose_image_path=ref_pose_image_path,
                    preprocess_mode=args.reference_preprocess_mode,
                )
                ref_debug_path = os.path.join(args.output_dir, f"{sample_name}__reference_debug.png")
                ref_debug.save(ref_debug_path)
            effective_num_clips = args.num_clips
            effective_num_frames = args.num_frames
            effective_pose_video_path = pose_video_path
            duration_override_seconds = None
            cleanup_paths = []

            if pose_video_path:
                if 'ref_pose_image' in df.columns:
                    val = row.get('ref_pose_image')
                    if val is not None and pd.notna(val) and str(val).strip():
                        ref_pose_image_path = (str(val) if os.path.isabs(str(val))
                                               else os.path.join(args.data_base_path, str(val)))

                aligned_pose_debug_path = os.path.join(args.output_dir, f"{sample_name}__aligned_pose.mp4")
                pose_plan = prepare_pose_video_for_reference(
                    pose_video_path=pose_video_path,
                    ref_pose_image_path=ref_pose_image_path,
                    target_width=args.width,
                    target_height=args.height,
                    target_fps=25.0,
                    base_num_frames=args.num_frames,
                    pose_align_mode=args.pose_align_mode,
                    debug_pose_path=aligned_pose_debug_path,
                )
                effective_pose_video_path = pose_plan["pose_video_path"]
                effective_num_clips = pose_plan["num_clips"]
                effective_num_frames = pose_plan["num_frames"]
                duration_override_seconds = pose_plan["duration_seconds"]
                print(
                    "[pose_align] "
                    f"mode={pose_plan['pose_align_mode']}, "
                    f"source_fps={pose_plan['source_fps']:.3f}, "
                    f"source_duration={pose_plan['source_duration_seconds']:.2f}s, "
                    f"aligned_frames={pose_plan['target_frame_count']}, "
                    f"num_frames={effective_num_frames}, "
                    f"num_clips={effective_num_clips}, "
                    f"ref_pose={'yes' if pose_plan['ref_pose_image_path'] and os.path.exists(pose_plan['ref_pose_image_path']) else 'no'}"
                )

            product_image = None
            if product_ref_path and os.path.exists(product_ref_path):
                product_image = resize_and_pad(
                    Image.open(product_ref_path).convert("RGB"),
                    target_height=args.height, target_width=args.width,
                    pad_color=(0, 0, 0)
                )

            speech_to_video(
                pipe=pipe,
                prompts=prompts,
                person_image=person_image,
                product_image=product_image,
                product_image_scale=product_image_scale,
                audio_path=audio_path,
                pose_video_path=effective_pose_video_path,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_frames=effective_num_frames,
                num_clips=effective_num_clips,
                num_inference_steps=args.num_inference_steps,
                save_path=save_path,
                sigma_shift=args.sigma_shift,
                cfg_scale=args.cfg_scale,
                duration_override_seconds=duration_override_seconds,
                initial_motion_video_path=args.initial_motion_video_path,
            )
            resize_video_to_resolution(save_path, reference_width, reference_height)
            print(f"  Saved: {save_path}")

        except Exception as e:
            print(f"[{idx+1}/{total}] Error: {e}")
            continue

    print(f"\nDone! Output: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    print("Loading models...")
    pipe = load_model(args)
    print("\nStarting batch inference...")
    process_batch(args, pipe)
