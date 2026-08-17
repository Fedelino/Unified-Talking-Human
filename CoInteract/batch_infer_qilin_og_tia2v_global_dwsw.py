"""
CoInteract Batch Inference Script

Generate speech-driven human-object interaction videos from a CSV file.
Each row in the CSV should contain: audio, person_image, prompt columns.
Optional columns: prompt2, prompt3, product_image, scale, pose_video.
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
import tempfile
import wave
import torch
import numpy as np
from PIL import Image
import librosa
import pandas as pd
from pathlib import Path
import torchvision.transforms.functional as TF

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
    parser.add_argument("--max_audio_seconds", type=float, default=30.0,
                        help="Hard cap for source audio duration before feature extraction")
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--cfg_scale_audio", type=float, default=4.0,
                        help="Audio guidance scale for StableAvatar-style audio native guidance.")
    parser.add_argument("--enable_stableavatar_audio_guidance", action="store_true", default=False,
                        help="Use StableAvatar-style separated audio guidance: uncond -> prompt/no-audio -> prompt+audio.")
    parser.add_argument("--sigma_shift", type=float, default=7.0,
                        help="Noise schedule time shift parameter")
    parser.add_argument("--enable_stableavatar_long_fusion", action="store_true", default=False,
                        help="Use StableAvatar-inspired weighted overlap fusion across long-video clip boundaries.")
    parser.add_argument("--overlap_frames", type=int, default=40,
                        help="Number of raw video frames to overlap and fuse between neighboring clips.")
    parser.add_argument("--overlap_weight_scheme", type=str, default="log", choices=["uniform", "log"],
                        help="Weight ramp for long-video overlap fusion.")
    parser.add_argument("--clip_seed_mode", type=str, default="fixed", choices=["fixed", "incremental"],
                        help="Per-clip seed strategy. 'fixed' is more stable for long videos.")
    parser.add_argument("--base_seed", type=int, default=0,
                        help="Base seed used for long-video generation.")
    parser.add_argument("--enable_latent_dwsw", action="store_true", default=False,
                        help="Enable S2V-aware latent sliding-window merging inside each denoising step.")
    parser.add_argument("--latent_window_size", type=int, default=20,
                        help="Target-latent window size for scheduler-level DWSW. 20 target latents + 1 ref latent matches an 81-frame S2V clip.")
    parser.add_argument("--latent_overlap", type=int, default=10,
                        help="Latent target-frame overlap between adjacent DWSW windows.")
    parser.add_argument("--scheduler_global_weight_scheme", type=str, default="uniform", choices=["uniform", "log"],
                        help="StableAvatar-style scheduler-level latent blend weights.")
    parser.add_argument("--use_static_reference_motion", action="store_true", default=False,
                        help="Use 73 repeated reference frames as CoInteract motion memory to anchor layout in TIA2V.")
    parser.add_argument("--enable_true_global_dwsw", action="store_true", default=True,
                        help="Run one full-length latent trajectory with DWSW instead of per-clip generation.")
    parser.add_argument("--disable_global_dwsw", action="store_true", default=False,
                        help="In true-global mode, call the model on the whole latent timeline without DWSW tiling. This is mostly for short OOM-risk tests.")
    parser.add_argument("--negative_prompt", type=str,
                        default="Blurry, worst quality, blurred details, static frame, "
                                "violent emotions, rapid hand shaking, subtitles, ugly, "
                                "deformed, extra fingers, poorly drawn hands, poorly drawn face")
    return parser.parse_args()


def resize_and_pad(image: Image.Image, target_height: int, target_width: int,
                   pad_color=(0, 0, 0), extra_scale: float = 1.0,
                   v_align: str = "center") -> Image.Image:
    """
    Resize image preserving aspect ratio, then pad to target size (center-aligned).
    Consistent with training-time ImageResizeAndPad preprocessing.
    """
    width, height = image.size
    scale = min(target_width / width, target_height / height)
    scale = scale / 1.25  # Scale down slightly to avoid cropping artifacts
    scale = scale * max(float(extra_scale), 1e-3)

    new_width = round(width * scale)
    new_height = round(height * scale)

    interpolation = (TF.InterpolationMode.LANCZOS if scale < 1
                     else TF.InterpolationMode.BILINEAR)
    image = TF.resize(image, (new_height, new_width), interpolation=interpolation)

    pad_left = (target_width - new_width) // 2
    pad_right = target_width - new_width - pad_left
    remaining_height = target_height - new_height
    if v_align == "bottom":
        pad_top = remaining_height
        pad_bottom = 0
    elif v_align == "top":
        pad_top = 0
        pad_bottom = remaining_height
    else:
        pad_top = remaining_height // 2
        pad_bottom = remaining_height - pad_top
    image = TF.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=pad_color)

    return image


def build_silent_audio(duration_seconds: float, sample_rate: int) -> np.ndarray:
    num_samples = max(1, int(round(duration_seconds * sample_rate)))
    return np.zeros(num_samples, dtype=np.float32)


def write_audio_wav(audio_path: str, audio_array: np.ndarray, sample_rate: int) -> str:
    audio_array = np.asarray(audio_array, dtype=np.float32)
    audio_array = np.clip(audio_array, -1.0, 1.0)
    pcm = (audio_array * 32767.0).astype(np.int16)
    with wave.open(audio_path, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return audio_path


def prepare_audio_conditioning(audio_path, audio_sample_rate, num_frames, fps, num_clips, max_audio_seconds=None):
    effective_clips = max(1, int(num_clips) if num_clips is not None else 1)
    requested_seconds = max(num_frames * effective_clips / float(fps), 1.0 / float(fps))
    clip_seconds = requested_seconds
    if max_audio_seconds is not None:
        clip_seconds = min(clip_seconds, float(max_audio_seconds))

    if audio_path is not None and str(audio_path).strip():
        if os.path.exists(audio_path):
            input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
            max_samples = max(1, int(round(clip_seconds * sample_rate)))
            if len(input_audio) > max_samples:
                input_audio = input_audio[:max_samples]
                temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                temp_audio.close()
                write_audio_wav(temp_audio.name, input_audio, sample_rate)
                print(
                    f"[audio] trimmed source audio to {clip_seconds:.2f}s "
                    f"for feature extraction and output muxing: {temp_audio.name}"
                )
                return input_audio, sample_rate, temp_audio.name, temp_audio.name
            return input_audio, sample_rate, audio_path, None
        print(f'[audio] missing audio path {audio_path}; falling back to silence')

    duration_seconds = clip_seconds
    input_audio = build_silent_audio(duration_seconds, audio_sample_rate)
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio.close()
    write_audio_wav(temp_audio.name, input_audio, audio_sample_rate)
    print(f'[audio] no audio provided; generated silent conditioning track: {temp_audio.name}')
    return input_audio, audio_sample_rate, temp_audio.name, temp_audio.name


def build_overlap_weights(overlap_frames: int, scheme: str) -> np.ndarray:
    if overlap_frames <= 0:
        return np.zeros((0,), dtype=np.float32)
    if overlap_frames == 1:
        return np.ones((1,), dtype=np.float32)
    if scheme == "uniform":
        return np.linspace(0.0, 1.0, overlap_frames, dtype=np.float32)
    if scheme == "log":
        init_weight = np.linspace(0.0, 1.0, overlap_frames, dtype=np.float32)
        log_weight = np.log1p(init_weight * (np.e - 1.0))
        denom = max(float(log_weight.max() - log_weight.min()), 1e-6)
        return ((log_weight - log_weight.min()) / denom).astype(np.float32)
    raise ValueError(f"Unsupported overlap weight scheme: {scheme}")


def fuse_clip_overlap(existing_video, current_clip, overlap_frames: int, scheme: str):
    if not existing_video:
        return list(current_clip), 0
    overlap_frames = min(int(overlap_frames), len(existing_video), len(current_clip))
    if overlap_frames <= 0:
        return list(existing_video) + list(current_clip), 0

    weights = build_overlap_weights(overlap_frames, scheme)
    fused_clip = list(current_clip)
    for idx in range(overlap_frames):
        prev_frame = np.asarray(existing_video[-overlap_frames + idx], dtype=np.float32)
        curr_frame = np.asarray(current_clip[idx], dtype=np.float32)
        blend_weight = float(weights[idx])
        blended = prev_frame * (1.0 - blend_weight) + curr_frame * blend_weight
        fused_clip[idx] = Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
    return list(existing_video[:-overlap_frames]) + fused_clip, overlap_frames


def concat_pose_latents_for_global_dwsw(pose_latents, num_repeat: int):
    if pose_latents is None:
        return None
    selected = [pose_latents[i] for i in range(min(num_repeat, len(pose_latents)))]
    if not selected:
        return None
    return torch.cat(selected, dim=2)


def speech_to_video_true_global_dwsw(pipe, prompts, person_image, audio_path,
                                    product_image=None, product_image_scale=1.0,
                                    negative_prompt="", num_clips=None,
                                    audio_sample_rate=16000, pose_video_path=None,
                                    num_frames=80, height=448, width=832,
                                    num_inference_steps=40, fps=25,
                                    save_path=None, sigma_shift=5.0, cfg_scale=5.0,
                                    cfg_scale_audio=4.0,
                                    max_audio_seconds=None,
                                    use_audio_native_guidance=False,
                                    latent_window_size=20,
                                    latent_overlap=10,
                                    scheduler_global_weight_scheme="uniform",
                                    use_static_reference_motion=False,
                                    base_seed=0,
                                    disable_global_dwsw=False):
    """StableAvatar-style global latent DWSW for CoInteract S2V/TIA2V.

    This differs from the legacy CoInteract loop: it builds one global latent
    timeline, tiles model calls inside each denoising step, and decodes once.
    The previous-frame motion memory is intentionally disabled in this first
    prototype so global pose/audio continuity is not mixed with chunk carryover.
    """
    input_audio, sample_rate, audio_path_for_save, temp_audio_path = prepare_audio_conditioning(
        audio_path=audio_path,
        audio_sample_rate=audio_sample_rate,
        num_frames=num_frames,
        fps=fps,
        num_clips=num_clips,
        max_audio_seconds=max_audio_seconds,
    )
    pose_video = (VideoData(pose_video_path, height=height, width=width)
                  if pose_video_path else None)

    # Precompute pose in normal clip-sized pieces to avoid a huge VAE encode,
    # then concatenate the target pose latents into one global condition.
    _, pose_latents, num_repeat = WanVideoUnit_S2V.pre_calculate_audio_pose(
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
    num_repeat = max(1, int(num_repeat))
    global_target_frames = int(num_repeat) * int(num_frames)
    global_num_frames = global_target_frames + 1
    s2v_pose_latents = concat_pose_latents_for_global_dwsw(pose_latents, num_repeat)

    if disable_global_dwsw:
        sliding_window_size = None
        sliding_window_stride = None
    else:
        sliding_window_size = max(2, int(latent_window_size))
        latent_overlap_value = max(0, min(int(latent_overlap), sliding_window_size - 1))
        sliding_window_stride = max(1, sliding_window_size - latent_overlap_value)
    prompt = prompts[0] if prompts else ""
    mode_name = "global-no-dwsw" if disable_global_dwsw else "global-dwsw"
    print(
        f"[{mode_name}] one-call latent trajectory "
        f"clips={num_repeat}, target_frames={global_target_frames}, "
        f"pipe_num_frames={global_num_frames}, window={sliding_window_size}, "
        f"stride={sliding_window_stride}"
    )
    if s2v_pose_latents is not None:
        print(f"[{mode_name}] pose latents: {tuple(s2v_pose_latents.shape)}")
    motion_video = None
    if use_static_reference_motion:
        motion_video = [person_image.copy() for _ in range(73)]
        print(f"[{mode_name}] static reference motion anchor enabled: {len(motion_video)} repeated frames")

    try:
        video = pipe(
            prompt=prompt,
            person_image=person_image,
            product_image=product_image,
            product_image_scale=product_image_scale,
            negative_prompt=negative_prompt,
            seed=int(base_seed),
            num_frames=global_num_frames,
            height=height,
            width=width,
            input_audio=input_audio,
            audio_sample_rate=sample_rate,
            s2v_pose_latents=s2v_pose_latents,
            motion_video=motion_video,
            num_inference_steps=num_inference_steps,
            sigma_shift=sigma_shift,
            cfg_scale=cfg_scale,
            cfg_scale_audio=cfg_scale_audio,
            use_audio_native_guidance=use_audio_native_guidance,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            scheduler_global_dwsw=(not disable_global_dwsw),
            scheduler_global_weight_scheme=scheduler_global_weight_scheme,
        )
        video = video[-global_target_frames:]
        save_video_with_audio(video, save_path, audio_path_for_save, fps=fps, quality=5)
        print(f"[{mode_name}] saved: {save_path}")
    finally:
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return video


def speech_to_video(pipe, prompts, person_image, audio_path,
                    product_image=None, product_image_scale=1.0,
                    negative_prompt="", num_clips=None,
                    audio_sample_rate=16000, pose_video_path=None,
                    num_frames=80, height=448, width=832,
                    num_inference_steps=40, fps=25, motion_frames=73,
                    save_path=None, sigma_shift=5.0, cfg_scale=5.0,
                    cfg_scale_audio=4.0,
                    max_audio_seconds=None,
                    use_audio_native_guidance=False,
                    enable_long_fusion=False,
                    overlap_frames=40,
                    overlap_weight_scheme="log",
                    clip_seed_mode="fixed",
                    base_seed=0,
                    enable_latent_dwsw=False,
                    latent_window_size=12,
                    latent_overlap=4):
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
        max_audio_seconds=max_audio_seconds,
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

    motion_videos = []
    video = []

    try:
        for r in range(num_repeat):
            current_prompt = prompts[min(r, len(prompts) - 1)]
            clip_seed = int(base_seed if clip_seed_mode == "fixed" else base_seed + r)

            s2v_pose_latents = pose_latents[r] if pose_latents is not None else None
            sliding_window_size = None
            sliding_window_stride = None
            if enable_latent_dwsw:
                sliding_window_size = max(2, int(latent_window_size))
                latent_overlap_value = max(0, min(int(latent_overlap), sliding_window_size - 1))
                sliding_window_stride = max(1, sliding_window_size - latent_overlap_value)
            current_clip = pipe(
                prompt=current_prompt,
                person_image=person_image,
                product_image=product_image,
                product_image_scale=product_image_scale,
                negative_prompt=negative_prompt,
                seed=clip_seed,
                num_frames=num_frames + 1,
                height=height,
                width=width,
                audio_embeds=audio_embeds[r],
                s2v_pose_latents=s2v_pose_latents,
                motion_video=motion_videos,
                num_inference_steps=num_inference_steps,
                sigma_shift=sigma_shift,
                cfg_scale=cfg_scale,
                cfg_scale_audio=cfg_scale_audio,
                use_audio_native_guidance=use_audio_native_guidance,
                sliding_window_size=sliding_window_size,
                sliding_window_stride=sliding_window_stride,
            )
            current_clip = current_clip[-num_frames:]

            if enable_long_fusion:
                video, fused_frames = fuse_clip_overlap(video, current_clip, overlap_frames, overlap_weight_scheme)
                overlap_frames_num = min(motion_frames, len(video))
                motion_videos = video[-overlap_frames_num:]
                print(
                  f"  StableAvatar-like fusion clip={r+1}/{num_repeat} "
                  f"seed={clip_seed} overlap={fused_frames}/{overlap_frames} "
                  f"motion_memory=fused"
                )
            else:
                video.extend(current_clip)
                overlap_frames_num = min(motion_frames, len(current_clip))
                motion_videos = current_clip[-overlap_frames_num:]
            if enable_latent_dwsw:
                print(
                    f"  Latent DWSW clip={r+1}/{num_repeat} "
                    f"window={sliding_window_size} stride={sliding_window_stride}"
                )
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
            prompt1 = str(row['prompt']).strip()
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

            sample_name = str(row.get("sample_id")).strip() if "sample_id" in df.columns and pd.notna(row.get("sample_id")) and str(row.get("sample_id")).strip() else Path(audio_rel_str or image_rel).stem
            save_path = os.path.join(args.output_dir, f"{sample_name}.mp4")

            if os.path.exists(save_path):
                print(f"[{idx+1}/{total}] Skip (exists): {sample_name}")
                continue

            print(f"\n[{idx+1}/{total}] Processing: {sample_name}")
            # OG CoInteract behavior: stretch the person image directly to the target canvas.
            person_image = Image.open(image_path).convert("RGB").resize((args.width, args.height))

            product_image = None
            if product_ref_path and os.path.exists(product_ref_path):
                product_image = resize_and_pad(
                    Image.open(product_ref_path).convert("RGB"),
                    target_height=args.height, target_width=args.width,
                    pad_color=(0, 0, 0)
                )

            if args.enable_true_global_dwsw:
                speech_to_video_true_global_dwsw(
                    pipe=pipe,
                    prompts=prompts,
                    person_image=person_image,
                    product_image=product_image,
                    product_image_scale=product_image_scale,
                    audio_path=audio_path,
                    pose_video_path=pose_video_path,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_clips=args.num_clips,
                    num_inference_steps=args.num_inference_steps,
                    save_path=save_path,
                    sigma_shift=args.sigma_shift,
                    cfg_scale=args.cfg_scale,
                    cfg_scale_audio=args.cfg_scale_audio,
                    max_audio_seconds=args.max_audio_seconds,
                    use_audio_native_guidance=args.enable_stableavatar_audio_guidance,
                    latent_window_size=args.latent_window_size,
                    latent_overlap=args.latent_overlap,
                    scheduler_global_weight_scheme=args.scheduler_global_weight_scheme,
                    use_static_reference_motion=args.use_static_reference_motion,
                    base_seed=args.base_seed,
                    disable_global_dwsw=args.disable_global_dwsw,
                )
            else:
                speech_to_video(
                    pipe=pipe,
                    prompts=prompts,
                    person_image=person_image,
                    product_image=product_image,
                    product_image_scale=product_image_scale,
                    audio_path=audio_path,
                    pose_video_path=pose_video_path,
                    negative_prompt=args.negative_prompt,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    num_clips=args.num_clips,
                    num_inference_steps=args.num_inference_steps,
                    save_path=save_path,
                    sigma_shift=args.sigma_shift,
                    cfg_scale=args.cfg_scale,
                    cfg_scale_audio=args.cfg_scale_audio,
                    max_audio_seconds=args.max_audio_seconds,
                    use_audio_native_guidance=args.enable_stableavatar_audio_guidance,
                    enable_long_fusion=args.enable_stableavatar_long_fusion,
                    overlap_frames=args.overlap_frames,
                    overlap_weight_scheme=args.overlap_weight_scheme,
                    clip_seed_mode=args.clip_seed_mode,
                    base_seed=args.base_seed,
                    enable_latent_dwsw=args.enable_latent_dwsw,
                    latent_window_size=args.latent_window_size,
                    latent_overlap=args.latent_overlap,
                )
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
