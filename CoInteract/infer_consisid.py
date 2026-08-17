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
from diffsynth.pipelines.wan_video_new_consisid import WanVideoPipeline, ModelConfig, WanVideoUnit_S2V
from train_consisid import ArcFaceReferenceEncoder, load_wan_image_encoder_fallback
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
    parser.add_argument("--consisid_ckpt", type=str, required=True, help="Trained ConsisID checkpoint (safetensors)")
    parser.add_argument("--arcface_cache", type=str, default="models/arcface/personimg_arcface_cache.npz")
    parser.add_argument("--arcface_onnx", type=str, default="models/arcface/w600k_r50.onnx")
    parser.add_argument("--image_encoder_path", type=str, default="/data1/Wan-AI/wan21_14b_480p/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")
    parser.add_argument("--dry_run", action="store_true", help="Only build ID tokens + print stats, skip generation")
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
                    save_path=None, sigma_shift=5.0, cfg_scale=5.0):
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
            motion_videos = motion_videos[overlap_frames_num:] + current_clip[-overlap_frames_num:]
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
    # ===== ConsisID attach (runtime) =====
    from diffsynth.models.consisid_faithful import LocalFacialExtractor, PerceiverCrossAttention
    import torch.nn as _nn
    from safetensors.torch import load_file as _load_sft
    _dt = next(pipe.dit.parameters()).dtype
    _d = pipe.dit
    _d.consisid_interval     = int(os.environ.get("CONSISID_INTERVAL", "8"))
    _d.consisid_start_block  = int(os.environ.get("CONSISID_START_BLOCK", "32"))
    _edepth   = int(os.environ.get("CONSISID_EXTRACTOR_DEPTH", "1"))
    _nq       = int(os.environ.get("CONSISID_NUM_QUERIES", "4"))
    _idt_dim  = int(os.environ.get("CONSISID_ID_TOKEN_DIM", "512"))
    _ch       = int(os.environ.get("CONSISID_CROSS_HEADS", "4"))
    _cdh      = int(os.environ.get("CONSISID_CROSS_DIM_HEAD", "32"))
    _d.consisid_extractor = LocalFacialExtractor(id_dim=512, vit_dim=1280, depth=_edepth, num_scale=1, num_queries=_nq, output_dim=_idt_dim).to(device=DEVICE, dtype=_dt)
    _active = max(0, len(_d.blocks) - _d.consisid_start_block)
    _n = max(1, (_active + _d.consisid_interval - 1)//_d.consisid_interval)
    _d.consisid_cross_attn = _nn.ModuleList([PerceiverCrossAttention(dim=5120, kv_dim=_idt_dim, heads=_ch, dim_head=_cdh).to(device=DEVICE, dtype=_dt) for _ in range(_n)])
    _d._consisid_enabled = True
    # load trained weights
    _sd = _load_sft(args.consisid_ckpt)
    _ex = {k[len("consisid_extractor."):]: v for k,v in _sd.items() if k.startswith("consisid_extractor.")}
    _ca = {k[len("consisid_cross_attn."):]: v for k,v in _sd.items() if k.startswith("consisid_cross_attn.")}
    _m1,_u1 = _d.consisid_extractor.load_state_dict({kk: vv.to(_dt) for kk,vv in _ex.items()}, strict=False)
    _m2,_u2 = _d.consisid_cross_attn.load_state_dict({kk: vv.to(_dt) for kk,vv in _ca.items()}, strict=False)
    print(f"[ConsisID] loaded {args.consisid_ckpt}: extractor(miss={len(_m1)},unexp={len(_u1)}) cross_attn(miss={len(_m2)},unexp={len(_u2)}) blocks={_n}")
    # encoders for real ID tokens
    global _ARC, _IMG_ENC
    _ARC = ArcFaceReferenceEncoder(args.arcface_onnx, precomputed_npz=args.arcface_cache)
    _IMG_ENC = load_wan_image_encoder_fallback(args.image_encoder_path).to(DEVICE, _dt)
    print("[ConsisID] ready: real ArcFace(cache)+CLIP id tokens per reference")

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
            person_image = resize_and_pad(
                Image.open(image_path).convert("RGB"),
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

            # Ensure DiT (and its ConsisID submodules) are on the compute device BEFORE
            # building ID tokens; VRAM mgmt may have offloaded it to CPU after the prior sample.
            try:
                pipe.load_models_to_device(["dit"])
            except Exception:
                pass
            # ConsisID: build real ID tokens for THIS reference (aligned ArcFace + CLIP)
            _p0 = next(pipe.dit.parameters()); _dev2 = _p0.device; _dt2 = _p0.dtype
            _clipf = _IMG_ENC.encode_image([pipe.preprocess_image(person_image)]).to(_dev2, _dt2)
            _arc_np, _det = _ARC.encode(person_image, image_rel)
            _arc_t = torch.from_numpy(_arc_np).unsqueeze(0).to(_dev2, _dt2)
            pipe.dit._consisid_id_tokens = pipe.dit.consisid_extractor(_arc_t, [_clipf])
            pipe.dit._consisid_id_tokens = pipe.dit._consisid_id_tokens.to(next(pipe.dit.parameters()).device)
            print(f"  [ConsisID] id_tokens={tuple(pipe.dit._consisid_id_tokens.shape)} arc_norm={float(_arc_t.float().norm()):.3f} face_detected={_det}")
            if args.dry_run:
                print("  [ConsisID] dry_run -> skip generation"); continue
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
