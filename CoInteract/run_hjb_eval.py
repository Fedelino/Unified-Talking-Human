"""
CoInteract Batch Inference Script

Generate speech-driven human-object interaction videos from a CSV file.
Each row in the CSV should contain: audio, person_image, prompt columns.
Optional columns: prompt2, prompt3, product_image, scale, pose_video,
person_scale, person_v_align.
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
import gc
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
from hjb_face_opt import DifferentiableArcFace, hjb_refine_latent
_ARC=None
def _get_arc():
    global _ARC
    if _ARC is None:
        _ARC=DifferentiableArcFace("models/arcface/w600k_r50.onnx", device=DEVICE)
    return _ARC
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
    parser.add_argument("--num_inference_steps", type=int, default=40)
    parser.add_argument("--cfg_scale", type=float, default=7.0,
                        help="Classifier-free guidance scale")
    parser.add_argument("--sigma_shift", type=float, default=7.0,
                        help="Noise schedule time shift parameter")
    parser.add_argument("--reference_compose_mode", type=str, default="pad", choices=["pad", "stretch"],
                        help="Map person reference to model canvas. 'pad' preserves aspect ratio with black padding; 'stretch' directly resizes with no black canvas.")
    parser.add_argument("--hjb_steps", type=int, default=8,
                        help="Number of HJB latent optimization steps")
    parser.add_argument("--hjb_decode_lat_frames", type=int, default=6,
                        help="Number of latent frames decoded inside HJB identity loss")
    parser.add_argument("--hjb_lr", type=float, default=0.06,
                        help="HJB latent optimization learning rate")
    parser.add_argument("--hjb_tiled", action="store_true",
                        help="Use tiled VAE decode inside HJB to reduce peak memory")
    parser.add_argument("--hjb_target_cosine", type=float, default=None,
                        help="Stop HJB early when sampled-frame mean ArcFace cosine reaches this value")
    parser.add_argument("--hjb_min_steps", type=int, default=0,
                        help="Minimum HJB steps to run before target-cosine early stopping is allowed")
    parser.add_argument("--hjb_score_frame_count", type=int, default=3,
                        help="Number of decoded frames sampled for per-step ArcFace scoring")
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


def resize_stretch(image: Image.Image, target_height: int, target_width: int) -> Image.Image:
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


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


def prepare_audio_conditioning(audio_path, audio_sample_rate, num_frames, fps, num_clips):
    if audio_path is not None and str(audio_path).strip():
        if os.path.exists(audio_path):
            input_audio, sample_rate = librosa.load(audio_path, sr=audio_sample_rate)
            return input_audio, sample_rate, audio_path, None
        print(f'[audio] missing audio path {audio_path}; falling back to silence')

    effective_clips = max(1, int(num_clips) if num_clips is not None else 1)
    duration_seconds = max(num_frames * effective_clips / float(fps), 1.0 / float(fps))
    input_audio = build_silent_audio(duration_seconds, audio_sample_rate)
    temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_audio.close()
    write_audio_wav(temp_audio.name, input_audio, audio_sample_rate)
    print(f'[audio] no audio provided; generated silent conditioning track: {temp_audio.name}')
    return input_audio, audio_sample_rate, temp_audio.name, temp_audio.name


def speech_to_video(pipe, prompts, person_image, audio_path,
                    product_image=None, product_image_scale=1.0,
                    negative_prompt="", num_clips=None,
                    audio_sample_rate=16000, pose_video_path=None,
                    num_frames=80, height=448, width=832,
                    num_inference_steps=40, fps=25, motion_frames=73,
                    save_path=None, sigma_shift=5.0, cfg_scale=5.0,
                    hjb_steps=8, hjb_decode_lat_frames=6, hjb_lr=0.06,
                    hjb_tiled=False, hjb_target_cosine=None, hjb_min_steps=0,
                    hjb_score_frame_count=3):
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

    import numpy as _np, torch as _t
    _arc=_get_arc()
    _r=_np.array(person_image.convert("RGB")).astype("float32")
    _rt=_t.from_numpy(_r).permute(2,0,1)[None].to(DEVICE)/127.5-1.0
    _REF_EMB=_arc.embed(_rt).detach()
    _HJB_BBOX=(int(width*0.36),int(height*0.14),int(width*0.57),int(height*0.34))
    try:
        for r in range(num_repeat):
            current_prompt = prompts[min(r, len(prompts) - 1)]

            s2v_pose_latents = pose_latents[r] if pose_latents is not None else None
            current_clip, _latent = pipe(
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
                return_latents=True,
                decode_video=False,
            )
            # HJB decodes only once after latent refinement.
            _latent = keep_only_vae_for_hjb(pipe, _latent)
            _ref2=hjb_refine_latent(
                pipe.vae,
                _latent,
                _REF_EMB,
                _arc,
                _HJB_BBOX,
                DEVICE,
                decode_lat_frames=hjb_decode_lat_frames,
                n_steps=hjb_steps,
                lr=hjb_lr,
                tiled=hjb_tiled,
                target_cosine=hjb_target_cosine,
                min_steps=hjb_min_steps,
                score_frame_count=hjb_score_frame_count,
            )
            clear_device_cache()
            _vid=pipe.vae.decode(_ref2,device=DEVICE,tiled=True)
            current_clip=pipe.vae_output_to_video(_vid)
            del _latent, _ref2, _vid
            clear_device_cache()
            current_clip = current_clip[-num_frames:]

            overlap_frames_num = min(motion_frames, len(current_clip))
            motion_videos = motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:]
            video.extend(current_clip)
            save_video_with_audio(video, save_path, audio_path_for_save, fps=25, quality=5)
            print(f"  Processed clip {r+1}/{num_repeat}")
    finally:
        if temp_audio_path is not None and os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    return video


def clear_device_cache():
    gc.collect()
    if DEVICE == "npu":
        torch.npu.empty_cache()
        torch.npu.synchronize()
    elif DEVICE == "cuda":
        torch.cuda.empty_cache()


def keep_only_vae_for_hjb(pipe, latent):
    """Free non-VAE components before HJB's VAE-backward refinement.

    The denoising path has already produced the final latent at this point.
    HJB only needs the detached latent, VAE decoder, and ArcFace. Dropping
    the DiT/text/audio/image modules avoids carrying the 14B generation stack
    into the VAE backward pass.
    """
    latent = latent.detach()
    vae = pipe.vae

    for name in (
        "dit", "dit2", "text_encoder", "image_encoder", "audio_encoder",
        "motion_controller", "vace", "vace2", "animate_adapter",
    ):
        if hasattr(pipe, name):
            setattr(pipe, name, None)

    pipe.in_iteration_models = ()
    pipe.in_iteration_models_2 = ()
    pipe.vae = vae
    clear_device_cache()
    pipe.load_models_to_device(["vae"])
    clear_device_cache()
    return latent


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
            person_scale = float(row.get('person_scale', 1.0)) if 'person_scale' in df.columns and pd.notna(row.get('person_scale')) else 1.0
            person_v_align = str(row.get('person_v_align', 'center')).strip().lower() if 'person_v_align' in df.columns and pd.notna(row.get('person_v_align')) else 'center'
            original_person_image = Image.open(image_path).convert("RGB")
            if args.reference_compose_mode == "stretch":
                person_image = resize_stretch(
                    original_person_image,
                    target_height=args.height, target_width=args.width,
                )
            else:
                person_image = resize_and_pad(
                    original_person_image,
                    target_height=args.height, target_width=args.width,
                    pad_color=(0, 0, 0),
                    extra_scale=person_scale,
                    v_align=person_v_align,
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
                hjb_steps=args.hjb_steps,
                hjb_decode_lat_frames=args.hjb_decode_lat_frames,
                hjb_lr=args.hjb_lr,
                hjb_tiled=args.hjb_tiled,
                hjb_target_cosine=args.hjb_target_cosine,
                hjb_min_steps=args.hjb_min_steps,
                hjb_score_frame_count=args.hjb_score_frame_count,
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
