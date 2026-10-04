#!/usr/bin/env python3
"""
RealEstate 10K Data Processor for TKCAM

Processes RealEstate 10K camera trajectory data with AI-assisted captioning.
Scenes without video frames are automatically skipped.

RealEstate 10K format:
- Line 1: YouTube video URL
- Following lines: 19 columns per frame:
  1. timestamp (microseconds)
  2-5. camera intrinsics (fx, fy, cx, cy)
  6-7: unused
  8-19. camera pose (3x4 T_wc matrix, row-major order)

Output formats:
- rotmat: [x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z] (12D)
- quat:   [x, y, z, dx, dy, dz, qw, qx, qy, qz] (10D)
"""

import gc
import glob
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .camera_geometry import (
    build_12d_features,
    compute_relative_pose,
    ensure_twc,
    validate_rotation,
)
from .unified_data_format import CameraDataFormat, UnifiedCameraData


# ============================================================================
# AI Captioner
# ============================================================================


class QwenVideoCaptioner:
    """Qwen VL-based video captioning for camera motion analysis."""

    NEW_PROMPT_TEMPLATE = (
        "You act as a professional cinematographer describing a camera shot."
        "Given the camera motion sequence reference and frames from the video,"
        "Your task is to merge the motion sequence with the key scene elements into 1 or 2 fluid sentences.\n\n"
        "Rules: \n"
            "1. ABSOLUTE MOTION: You MUST use the provided motion sequence exactly as written. DO NOT alter the directions or sequence."
            "2. FORMAT: Use 'then' or 'finally' to separate stages. Keep it cinematic but factual.\n\n"
            "3. For each stage include: movement type (pan/tilt/dolly/truck/arc/pedestal) direction and pace (slow/medium/fast)."
        "Examples: \n"
        "- The camera dollies forward with slight shaking, gradually closing the distance between the viewer and the fireplace."
        "- The camera pans left slowly and smoothly, revealing a stone fireplace. Then it moves along the left wall, maintaining shot angle."
        "- The camera dollies backward smoothly, revealing a basketball court in the foreground, then continues backward while panning slightly left"
        "Reference: {guidance}\n\n"
        "Motion description: "
    )

    # --- Prompt without scene description (pipeline verification) ---
    PROMPT_TEMPLATE_NO_SCENE = (
        "Analyze this camera trajectory ({n_frames} frames from {total_frames}, uniformly sampled).\n\n"
        "Reference: {guidance}\n\n"
        "Describe every motion stage in 1-2 sentences using 'then' or 'finally' to separate stages. "
        "If the visual motion clearly conflicts with the reference "
        "(e.g., forward vs backward), prioritize what you see and append [CONFLICT].\n\n"
        "For each stage include: movement type (pan/tilt/dolly/truck/arc/pedestal), "
        "direction, pace (slow/medium/fast), and quality (smooth/shaky). "
        "Always moving, never static.\n\n"
        "Examples:\n"
        '- "The camera pans left slowly, then reverses right, smoothly."\n'
        '- "The camera dollies forward while tilting up, then arcs left at medium pace."\n'
        '- "The camera trucks right steadily, then tilts down, finally pulls back slowly."\n\n'
        "Motion description:"
    )

    # --- Prompt with scene description (future feature) ---
    PROMPT_TEMPLATE_WITH_SCENE = (
        "You act as a professional cinematographer describing a camera shot."
        "Given the camera motion sequence reference and {n_frames} frames from {total_frames} uniformly sampled frames from the video,"
        "Your task is to merge the motion sequence with the key scene elements into 1 or 2 fluid sentences.\n\n"
        "Rules: \n"
            "1. ABSOLUTE MOTION: You MUST use the provided motion sequence exactly as written. DO NOT alter the directions or sequence."
            "2. FORMAT: Use 'then' or 'finally' to separate stages. Keep it cinematic but factual.\n\n"
            "3. For each stage include: movement type (pan/tilt/dolly/truck/arc/pedestal) direction and pace (slow/medium/fast)."
        "Examples: \n"
        "- The camera dollies forward with slight shaking, gradually closing the distance between the viewer and the fireplace."
        "- The camera pans left slowly and smoothly, revealing a stone fireplace. Then it moves along the left wall, maintaining shot angle."
        "- The camera dollies backward smoothly, revealing a basketball court in the foreground, then continues backward while panning slightly left"
        "Reference: {guidance}\n\n"
        "Motion description: "
    )

    # Default template alias (no-scene, used when with_scene=False)
    PROMPT_TEMPLATE = PROMPT_TEMPLATE_NO_SCENE

    GENERATE_KWARGS = dict(
        max_new_tokens=128,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        top_k=50,
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        max_frames: int = 32,
        device: Optional[str] = None,
        with_scene: bool = False,
    ):
        self.model_name = model_name
        self.max_frames = max_frames
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.with_scene = with_scene
        self.model = None
        self.processor = None
        self.model_loaded = False
        self.is_qwen3 = "qwen3" in model_name.lower()
        self.process_vision_info = None
        self._temp_dirs: List[str] = []

        mode = "with-scene" if with_scene else "no-scene"
        print(f"QwenVideoCaptioner: {model_name} (device={self.device}, qwen3={self.is_qwen3}, mode={mode})")

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self, use_flash_attention: bool = True):
        """Load the Qwen VL model (lazy, called on first use)."""
        if self.model_loaded:
            return

        try:
            import importlib

            transformers = importlib.import_module("transformers")
            cls_name = (
                "Qwen3VLForConditionalGeneration"
                if self.is_qwen3
                else "Qwen2_5_VLForConditionalGeneration"
            )
            ModelClass = getattr(transformers, cls_name)

            if use_flash_attention and self.device == "cuda":
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.bfloat16,
                    attn_implementation="flash_attention_2",
                    device_map="auto",
                )
                print("  Flash Attention 2 enabled")
            else:
                dtype = torch.float16 if self.device == "cuda" else torch.float32
                self.model = ModelClass.from_pretrained(
                    self.model_name,
                    torch_dtype=dtype,
                    device_map="auto" if self.device == "cuda" else None,
                )
                if self.device != "cuda":
                    self.model = self.model.to(self.device)

            AutoProcessor = getattr(transformers, "AutoProcessor")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model_loaded = True

            # Optional Qwen2.5 vision utils
            try:
                qwen_utils = importlib.import_module("qwen_vl_utils")
                self.process_vision_info = getattr(qwen_utils, "process_vision_info", None)
            except ImportError:
                pass

            print(f"Model loaded on {self.device}")

        except Exception as e:
            if use_flash_attention:
                print(f"Flash attention failed ({e}), retrying without...")
                self.load_model(use_flash_attention=False)
            else:
                raise

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------

    def sample_existing_frames(
        self,
        frames_dir: str,
        max_frames: Optional[int] = None,
        max_resolution: Tuple[int, int] = (640, 360),
    ) -> List[str]:
        """Sample and optionally resize frames from a directory."""
        max_frames = max_frames or self.max_frames

        frame_files: List[str] = []
        for pat in ("*.jpg", "*.jpeg", "*.png"):
            frame_files.extend(glob.glob(os.path.join(frames_dir, pat)))
        frame_files = sorted(frame_files)

        if not frame_files:
            return []

        # Uniform sampling
        if len(frame_files) <= max_frames:
            sampled = frame_files
        else:
            indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
            sampled = [frame_files[i] for i in indices]

        # Auto-resize oversized images
        processed: List[str] = []
        temp_dir: Optional[str] = None
        for img_path in sampled:
            try:
                img = Image.open(img_path)
                w, h = img.size
                if w > max_resolution[0] or h > max_resolution[1]:
                    if temp_dir is None:
                        temp_dir = tempfile.mkdtemp(prefix="qwen_resized_")
                    img.thumbnail(max_resolution, Image.Resampling.LANCZOS)
                    tmp = os.path.join(temp_dir, os.path.basename(img_path))
                    img.save(tmp, quality=90)
                    processed.append(tmp)
                else:
                    processed.append(img_path)
                img.close()
            except Exception:
                processed.append(img_path)

        if temp_dir:
            self._temp_dirs.append(temp_dir)
        return processed

    def cleanup_temp_images(self):
        """Remove temporary resized images."""
        import shutil

        for d in self._temp_dirs:
            try:
                if os.path.exists(d):
                    shutil.rmtree(d)
            except Exception:
                pass
        self._temp_dirs.clear()

    # ------------------------------------------------------------------
    # Prompt / message helpers
    # ------------------------------------------------------------------

    def _build_prompt(self, n_frames: int, total_frames: int, guidance: str) -> str:
        template = (
            self.PROMPT_TEMPLATE_WITH_SCENE if self.with_scene else self.PROMPT_TEMPLATE_NO_SCENE
        )
        return template.format(
            n_frames=n_frames,
            total_frames=total_frames,
            guidance=guidance,
        )

    @staticmethod
    def _build_messages(frame_paths: List[str], prompt: str) -> List[dict]:
        content = [{"type": "image", "image": p} for p in frame_paths]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Encoding helpers (Qwen3 vs Qwen2.5 API differences)
    # ------------------------------------------------------------------

    def _encode_single(self, messages: List[dict]) -> dict:
        """Encode a single message list into model inputs."""
        if self.is_qwen3:
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)
        else:
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            if self.process_vision_info is None:
                raise RuntimeError("qwen_vl_utils.process_vision_info required for Qwen2.5")
            img_in, vid_in = self.process_vision_info(messages)
            return self.processor(
                text=[text],
                images=img_in,
                videos=vid_in,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

    def _encode_batch(self, batch_messages: List[List[dict]]) -> dict:
        """Encode a batch of message lists into model inputs with padding."""
        if self.is_qwen3:
            return self.processor.apply_chat_template(
                batch_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            ).to(self.device)
        else:
            all_texts, all_imgs, all_vids = [], [], []
            for msgs in batch_messages:
                all_texts.append(
                    self.processor.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                )
                imgs, vids = self.process_vision_info(msgs)
                all_imgs.extend(imgs or [])
                all_vids.extend(vids or [])
            return self.processor(
                text=all_texts,
                images=all_imgs or None,
                videos=all_vids or None,
                padding=True,
                return_tensors="pt",
            ).to(self.device)

    # ------------------------------------------------------------------
    # Single-scene captioning
    # ------------------------------------------------------------------

    def generate_caption_with_context(
        self,
        frame_paths: List[str],
        guidance: str,
        scene_id: str = "",
        total_frames: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate an AI caption for one scene using frames + geometric guidance."""
        if not self.model_loaded:
            self.load_model()
        if self.model is None or self.processor is None:
            return {"success": False, "caption": guidance, "error": "model unavailable"}
        if not frame_paths:
            return {"success": False, "caption": guidance, "error": "no frames"}

        prompt = self._build_prompt(len(frame_paths), total_frames or len(frame_paths), guidance)
        messages = self._build_messages(frame_paths, prompt)

        try:
            inputs = self._encode_single(messages)

            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, **self.GENERATE_KWARGS)

            in_len = len(inputs.input_ids[0])
            del inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            caption = self.processor.batch_decode(
                [gen_ids[0][in_len:]], skip_special_tokens=True
            )[0].strip()
            del gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return {"success": True, "caption": caption, "scene_id": scene_id}

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {"success": False, "caption": guidance, "error": str(e)}

    # ------------------------------------------------------------------
    # Batch captioning
    # ------------------------------------------------------------------

    def generate_captions_batch(
        self, batch_data: List[Tuple[List[str], str, str, int]]
    ) -> List[Dict[str, Any]]:
        """Generate captions for multiple scenes in one forward pass.

        Args:
            batch_data: List of (frame_paths, guidance, scene_id, total_frames).

        Returns:
            One result dict per scene.
        """
        if not self.model_loaded:
            self.load_model()
        if self.model is None or self.processor is None:
            return [
                {"success": False, "caption": g, "scene_id": sid}
                for _, g, sid, _ in batch_data
            ]
        if not batch_data:
            return []
        if len(batch_data) == 1:
            fp, g, sid, tf = batch_data[0]
            return [self.generate_caption_with_context(fp, g, sid, tf)]

        # Left padding for batch generation
        orig_pad = None
        if hasattr(self.processor, "tokenizer"):
            orig_pad = self.processor.tokenizer.padding_side
            self.processor.tokenizer.padding_side = "left"

        try:
            batch_messages: List[List[dict]] = []
            metadata: List[dict] = []

            for fp, guidance, sid, tf in batch_data:
                if not fp:
                    continue
                # Pad/truncate to uniform frame count
                padded = fp[: self.max_frames]
                if len(padded) < self.max_frames:
                    padded += [padded[-1]] * (self.max_frames - len(padded))

                prompt = self._build_prompt(len(fp), tf, guidance)
                batch_messages.append(self._build_messages(padded, prompt))
                metadata.append({"scene_id": sid, "guidance": guidance})

            if not batch_messages:
                return []

            inputs = self._encode_batch(batch_messages)

            with torch.no_grad():
                gen_ids = self.model.generate(**inputs, **self.GENERATE_KWARGS)

            in_lens = [len(ids) for ids in inputs.input_ids]
            del inputs
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            trimmed = [gen_ids[i][in_lens[i]:] for i in range(len(in_lens))]
            del gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            texts = self.processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            return [
                {"success": True, "caption": texts[i].strip(), "scene_id": m["scene_id"]}
                for i, m in enumerate(metadata)
            ]

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return [
                {"success": False, "caption": g, "scene_id": sid, "error": str(e)}
                for _, g, sid, _ in batch_data
            ]

        finally:
            if orig_pad is not None and hasattr(self.processor, "tokenizer"):
                self.processor.tokenizer.padding_side = orig_pad


# ============================================================================
# Dataset Processor
# ============================================================================


class RealEstate10KProcessor:
    """Process Real Estate 10K camera trajectories with AI captioning.

    Pipeline:
        1. Parse raw RE10K camera files (T_wc, always treated as camera-to-world).
        2. Relativize to first frame (if transform='relative').
        3. Build 12-D features [pos, vel, rot6d].
        4. Convert to target format (rotmat or quat).
        5. Generate deterministic motion guidance.
        6. Batch AI captioning with Qwen VL using frames + guidance.
        7. Save trajectories, captions, metadata, splits, and statistics.

    Scenes without video frames are automatically skipped.
    """

    def __init__(
        self,
        output_format: str = "rotmat",
        min_sequence_length: int = 30,
        max_sequence_length: int = 300,
        transform: str = "relative",
        ai_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        ai_max_frames: int = 32,
        video_source_dir: Optional[str] = None,
        ai_batch_size: int = 2,
        resume: bool = False,
        filter_min_frames: int = 0,
        ai_with_scene: bool = False,
    ):
        self.output_format = output_format
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.transform = transform
        self.filter_min_frames = filter_min_frames
        self.video_source_dir = video_source_dir
        self.ai_batch_size = ai_batch_size
        self.resume = resume

        # Scene ID bookkeeping
        self.scene_counter: int = 0
        self.scene_id_mapping: Dict[str, str] = {}

        # Relative motion data (set per-scene during parse_camera_file)
        self.relative_motion_data: Optional[List[Dict]] = None

        # Target format
        if output_format == "rotmat":
            self.target_format = CameraDataFormat.FULL_12_ROTMAT
        elif output_format == "quat":
            self.target_format = CameraDataFormat.QUATERNION_10
        else:
            raise ValueError(
                f"Unsupported output format: {output_format!r}. Use 'rotmat' or 'quat'."
            )

        # AI captioner (single-GPU batched)
        self.ai_captioner = QwenVideoCaptioner(
            model_name=ai_model_name, max_frames=ai_max_frames, with_scene=ai_with_scene
        )

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def parse_camera_file(self, file_path: str) -> Tuple[str, List[Dict]]:
        """Parse a Real Estate 10K camera parameter file.

        Poses are always interpreted as T_wc (camera-to-world).
        When transform='relative', poses are relativized so frame 0 = Identity.

        Returns:
            (video_url, list_of_camera_data_dicts)
        """
        with open(file_path, "r") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        if not lines:
            raise ValueError(f"Empty camera file: {file_path}")

        video_url = lines[0]
        camera_data: List[Dict] = []
        relative_motions: List[Dict] = []

        # First-frame reference for relative transform
        R0_wc = t0_wc = None
        if self.transform == "relative":
            vals = lines[1].split()
            raw = np.array([float(v) for v in vals[7:]]).reshape(3, 4)
            R0_wc, t0_wc = ensure_twc(raw, "twc")
            R0_wc = validate_rotation(R0_wc)

            camera_data.append(
                {
                    "timestamp_us": int(float(vals[0])),
                    "timestamp_s": int(float(vals[0])) / 1_000_000,
                    "intrinsics": {
                        "fx": float(vals[1]),
                        "fy": float(vals[2]),
                        "cx": float(vals[3]),
                        "cy": float(vals[4]),
                    },
                    "pose_matrix": np.concatenate(
                        [np.eye(3), np.zeros((3, 1))], axis=1
                    ),
                }
            )
            relative_motions.append(
                {
                    "relative_rotation": np.eye(3),
                    "relative_translation": np.zeros(3),
                    "timestamp_s": camera_data[0]["timestamp_s"],
                }
            )

        start_idx = 2 if self.transform == "relative" else 1
        for i, line in enumerate(lines[1:], start=start_idx):
            try:
                vals = line.split()
                if len(vals) < 19:
                    continue

                ts_us = int(float(vals[0]))
                raw = np.array([float(v) for v in vals[7:]]).reshape(3, 4)
                R_wc, t_wc = ensure_twc(raw, "twc")
                R_wc = validate_rotation(R_wc)

                if self.transform == "relative":
                    R_rel, t_rel = compute_relative_pose(R0_wc, t0_wc, R_wc, t_wc)
                    pose = np.concatenate([R_rel, t_rel.reshape(-1, 1)], axis=1)
                    relative_motions.append(
                        {
                            "relative_rotation": R_rel,
                            "relative_translation": t_rel,
                            "timestamp_s": ts_us / 1e6,
                        }
                    )
                else:
                    pose = np.concatenate([R_wc, t_wc.reshape(-1, 1)], axis=1)

                camera_data.append(
                    {
                        "timestamp_us": ts_us,
                        "timestamp_s": ts_us / 1_000_000,
                        "intrinsics": {
                            "fx": float(vals[1]),
                            "fy": float(vals[2]),
                            "cx": float(vals[3]),
                            "cy": float(vals[4]),
                        },
                        "pose_matrix": pose,
                    }
                )
            except (ValueError, IndexError):
                continue

        # Store for guidance generation
        if self.transform == "relative":
            self.relative_motion_data = relative_motions

        return video_url, camera_data

    # ------------------------------------------------------------------
    # Trajectory extraction
    # ------------------------------------------------------------------

    def extract_camera_trajectory(self, camera_data: List[Dict]) -> np.ndarray:
        """Build canonical 12-D features [pos(3), vel(3), rot6d(6)] from camera data."""
        if not camera_data:
            raise ValueError("No camera data")

        if self.filter_min_frames > 0 and len(camera_data) < self.filter_min_frames:
            return np.empty((0, 12))

        positions = []
        rotations = []
        for fd in camera_data:
            pm = fd["pose_matrix"]
            positions.append(pm[:, 3])
            rotations.append(validate_rotation(pm[:, :3]))

        return build_12d_features(np.array(positions), np.array(rotations))

    @staticmethod
    def smooth_trajectory(trajectory: np.ndarray, sigma: float = 1.0) -> np.ndarray:
        """Apply Gaussian smoothing to a trajectory."""
        from scipy.ndimage import gaussian_filter1d

        if trajectory.ndim != 2 or trajectory.shape[0] == 0:
            return trajectory
        smoothed = np.zeros_like(trajectory)
        for i in range(trajectory.shape[1]):
            smoothed[:, i] = gaussian_filter1d(trajectory[:, i], sigma=sigma)
        return smoothed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _convert_to_numeric_id(self, scene_id: str) -> str:
        """Assign a sequential 6-digit numeric ID to a scene."""
        nid = f"{self.scene_counter:06d}"
        self.scene_id_mapping[nid] = scene_id
        self.scene_counter += 1
        return nid

    def _check_frames_exist(self, scene_id: str) -> bool:
        """Check if video frames directory exists for a scene."""
        if not self.video_source_dir:
            return False
        d = Path(self.video_source_dir) / scene_id
        if d.is_dir():
            for pat in ("*.jpg", "*.jpeg", "*.png"):
                if list(d.glob(pat)):
                    return True
        return False

    # ------------------------------------------------------------------
    # Deterministic motion guidance (no Euler angles)
    # ------------------------------------------------------------------

    def _generate_guidance(self) -> str:
        """Build a deterministic motion description from relative motion data.

        Uses integrated local velocity and forward-vector analysis.
        This text is passed to the AI captioner as geometric reference.
        """
        if not self.relative_motion_data or len(self.relative_motion_data) < 2:
            return "camera remains static"

        translations = np.array(
            [m["relative_translation"] for m in self.relative_motion_data]
        )
        rotations = np.array(
            [m["relative_rotation"] for m in self.relative_motion_data]
        )

        parts: List[str] = []

        # --- Translation via integrated local velocity ---
        # OpenGL camera local axes: +X right, +Y up, -Z forward.
        v_world = np.diff(translations, axis=0)
        r_c2w = rotations[:-1]
        r_w2c = np.transpose(r_c2w, (0, 2, 1))
        v_local = np.einsum("nij,nj->ni", r_w2c, v_world)

        # Filter tiny COLMAP jitter before integration.
        v_local[np.abs(v_local) < 0.005] = 0.0
        accumulated_local_t = np.sum(v_local, axis=0)

        # --- Rotation (from forward vector change, no Euler conversion) ---
        # OpenGL: forward = -col2 of R
        fwd0 = -rotations[0][:, 2]
        fwd1 = -rotations[-1][:, 2]

        yaw0 = np.arctan2(fwd0[0], -fwd0[2])
        yaw1 = np.arctan2(fwd1[0], -fwd1[2])
        dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))  # wrap to [-π, π]

        pitch0 = np.arcsin(np.clip(fwd0[1], -1, 1))
        pitch1 = np.arcsin(np.clip(fwd1[1], -1, 1))
        dpitch = pitch1 - pitch0

        t_thresh = 0.10
        yaw_thresh = 0.12
        pitch_thresh = 0.10

        is_panning = abs(dyaw) > yaw_thresh
        is_trucking = abs(accumulated_local_t[0]) > t_thresh

        # Joint kinematic branch: panning + trucking => cinematic arc/orbit.
        if is_panning and is_trucking:
            parts.append("arcs right" if dyaw > 0 else "arcs left")
        else:
            if is_trucking:
                parts.append(
                    "tracks right"
                    if accumulated_local_t[0] > 0
                    else "tracks left"
                )
            if is_panning:
                parts.append("pans right" if dyaw > 0 else "pans left")

        if abs(accumulated_local_t[1]) > t_thresh:
            parts.append(
                "moves up" if accumulated_local_t[1] > 0 else "moves down"
            )
        if abs(accumulated_local_t[2]) > t_thresh:
            parts.append(
                "dollies forward"
                if accumulated_local_t[2] < 0
                else "dollies backward"
            )
        if abs(dpitch) > pitch_thresh:
            parts.append("tilts up" if dpitch > 0 else "tilts down")

        if not parts:
            return "camera remains relatively static"
        if len(parts) == 1:
            return f"camera {parts[0]}"
        return f"camera {parts[0]} while it {' and '.join(parts[1:])}"

    # ------------------------------------------------------------------
    # Main Processing Pipeline
    # ------------------------------------------------------------------

    def process_dataset(
        self,
        input_dir: str,
        output_dir: str,
        scene_list_file: Optional[str] = None,
        max_scenes: Optional[int] = None,
        apply_smoothing: bool = True,
    ) -> Dict[str, int]:
        """Process the entire RE10K dataset.

        Phase 1: Parse trajectories, save .npy, generate deterministic guidance.
        Phase 2: Batch AI captioning for all scenes with video frames.
        Phase 3: Create dataset splits and statistics.
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # --- Find camera files ---
        if scene_list_file and os.path.exists(scene_list_file):
            with open(scene_list_file) as f:
                sids = [ln.strip() for ln in f if ln.strip()]
            camera_files = [
                p for sid in sids if (p := input_path / f"{sid}.txt").exists()
            ]
        else:
            camera_files = sorted(input_path.glob("*.txt"), key=lambda p: p.stem)

        if max_scenes:
            camera_files = camera_files[:max_scenes]

        print(f"Found {len(camera_files)} camera files")

        # --- Resume handling ---
        existing_captions: set = set()
        mapping_file = output_path / "scene_id_mapping.json"
        if self.resume and mapping_file.exists():
            with open(mapping_file) as f:
                self.scene_id_mapping = json.load(f)
            self.scene_counter = len(self.scene_id_mapping)
            utd = output_path / "untagged_text"
            if utd.exists():
                existing_captions = {
                    tf.stem for tf in utd.glob("*.txt") if tf.stat().st_size > 0
                }
            print(
                f"Resuming: {self.scene_counter} scenes mapped, "
                f"{len(existing_captions)} captions exist"
            )
        else:
            self.scene_counter = 0
            self.scene_id_mapping = {}

        # --- Create output dirs ---
        for subdir in ("new_joint_vecs", "untagged_text", "metadata"):
            (output_path / subdir).mkdir(exist_ok=True)

        stats = {"processed": 0, "failed": 0, "skipped": 0}
        scene_ids: List[str] = []
        scenes_for_ai: List[Tuple[str, str, str, int]] = []

        # ===== Phase 1: Trajectory processing =====
        print("\nPhase 1: Processing trajectories...")
        for camera_file in tqdm(camera_files, desc="Trajectories"):
            scene_id = camera_file.stem

            # Must have video frames
            if not self._check_frames_exist(scene_id):
                stats["failed"] += 1
                continue

            # Check for resume (peek at what numeric ID this scene would get)
            if scene_id in self.scene_id_mapping.values():
                nid = next(
                    k for k, v in self.scene_id_mapping.items() if v == scene_id
                )
            else:
                nid = f"{self.scene_counter:06d}"

            if self.resume and nid in existing_captions:
                stats["skipped"] += 1
                scene_ids.append(nid)
                continue

            # Assign numeric ID
            nid = self._convert_to_numeric_id(scene_id)

            try:
                video_url, camera_data = self.parse_camera_file(str(camera_file))

                if len(camera_data) < self.min_sequence_length:
                    stats["failed"] += 1
                    continue
                if len(camera_data) > self.max_sequence_length:
                    camera_data = camera_data[: self.max_sequence_length]

                traj = self.extract_camera_trajectory(camera_data)
                if traj.shape[0] == 0:
                    stats["failed"] += 1
                    continue

                if apply_smoothing:
                    traj = self.smooth_trajectory(traj)

                # Convert and save
                unified = UnifiedCameraData(traj, CameraDataFormat.FULL_12_ROTMAT)
                final = unified.to_format(self.target_format)
                np.save(
                    output_path / "new_joint_vecs" / f"{nid}.npy",
                    final.get_momask_compatible_data().numpy(),
                )

                # Metadata
                meta = {
                    "scene_id": nid,
                    "original_scene_id": scene_id,
                    "video_url": video_url,
                    "num_frames": len(camera_data),
                    "format": self.target_format.name,
                }
                with open(output_path / "metadata" / f"{nid}.json", "w") as f:
                    json.dump(meta, f, indent=2)

                # Deterministic guidance
                guidance = self._generate_guidance()
                scenes_for_ai.append((scene_id, nid, guidance, len(camera_data)))
                scene_ids.append(nid)
                stats["processed"] += 1

            except Exception as e:
                print(f"Failed {scene_id}: {e}")
                stats["failed"] += 1

        # Save mapping after Phase 1
        with open(mapping_file, "w") as f:
            json.dump(self.scene_id_mapping, f, indent=2)

        # ===== Phase 2: AI captioning =====
        if scenes_for_ai:
            print(
                f"\nPhase 2: AI captioning for {len(scenes_for_ai)} scenes "
                f"(batch_size={self.ai_batch_size})..."
            )
            self._run_ai_captioning(scenes_for_ai, output_path)

        # ===== Phase 3: Splits & statistics =====
        self._create_dataset_splits(output_path, scene_ids)
        self._calculate_dataset_statistics(output_path, scene_ids)

        print(f"\nDone: {stats}")
        return stats

    # ------------------------------------------------------------------
    # AI Captioning
    # ------------------------------------------------------------------

    def _run_ai_captioning(
        self,
        scenes_for_ai: List[Tuple[str, str, str, int]],
        output_path: Path,
    ):
        """Batch AI captioning for all scenes (all guaranteed to have frames)."""
        captioner = self.ai_captioner

        try:
            captioner.load_model()
        except Exception as e:
            print(f"Failed to load AI model: {e}. Writing deterministic captions.")
            for _, nid, guidance, _ in scenes_for_ai:
                with open(output_path / "untagged_text" / f"{nid}.txt", "w") as f:
                    f.write(guidance)
            return

        pbar = tqdm(total=len(scenes_for_ai), desc="AI Captioning")
        idx = 0
        batch_count = 0

        while idx < len(scenes_for_ai):
            end = min(idx + self.ai_batch_size, len(scenes_for_ai))
            batch: List[Tuple[List[str], str, str, int]] = []
            fallback_nids: List[Tuple[str, str]] = []

            for j in range(idx, end):
                scene_id, nid, guidance, total_frames = scenes_for_ai[j]
                fp = captioner.sample_existing_frames(
                    str(Path(self.video_source_dir) / scene_id)
                )
                if fp:
                    batch.append((fp, guidance, nid, total_frames))
                else:
                    # Edge case: directory exists but no usable frames
                    fallback_nids.append((nid, guidance))

            # Write fallback guidance for scenes without usable frames
            for nid, guidance in fallback_nids:
                with open(output_path / "untagged_text" / f"{nid}.txt", "w") as f:
                    f.write(guidance)

            # Run batch inference
            if batch:
                batch_count += 1
                results = captioner.generate_captions_batch(batch)
                for i, (_, _, nid, _) in enumerate(batch):
                    with open(output_path / "untagged_text" / f"{nid}.txt", "w") as f:
                        f.write(results[i]["caption"])
                    status = "OK" if results[i].get("success") else "FAIL"
                    pbar.set_postfix(
                        {"batch": batch_count, "last": f"{nid} {status}"}
                    )

            pbar.update(end - idx)
            idx = end

            # Memory cleanup
            captioner.cleanup_temp_images()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pbar.close()

    # ------------------------------------------------------------------
    # Dataset finalization
    # ------------------------------------------------------------------

    def _create_dataset_splits(self, output_path: Path, scene_ids: List[str]):
        """Create reproducible train/val/test splits (70/15/15)."""
        rng = np.random.RandomState(42)
        ids = list(scene_ids)
        rng.shuffle(ids)

        n = len(ids)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        splits = {
            "train": ids[:n_train],
            "val": ids[n_train : n_train + n_val],
            "test": ids[n_train + n_val :],
        }
        for name, subset in splits.items():
            with open(output_path / f"{name}.txt", "w") as f:
                f.write("\n".join(subset) + "\n")

        print(
            f"Splits: {len(splits['train'])} train, "
            f"{len(splits['val'])} val, {len(splits['test'])} test"
        )

    def _calculate_dataset_statistics(
        self, output_path: Path, scene_ids: List[str]
    ):
        """Calculate and save dataset mean/std statistics."""
        mdir = output_path / "new_joint_vecs"
        all_data = [
            np.load(mdir / f"{sid}.npy")
            for sid in scene_ids
            if (mdir / f"{sid}.npy").exists()
        ]
        if all_data:
            combined = np.concatenate(all_data)
            np.save(output_path / "Mean.npy", np.mean(combined, axis=0))
            np.save(output_path / "Std.npy", np.std(combined, axis=0))
            print("Dataset statistics saved (Mean.npy, Std.npy)")

        # Also persist the scene ID mapping
        if self.scene_id_mapping:
            with open(output_path / "scene_id_mapping.json", "w") as f:
                json.dump(self.scene_id_mapping, f, indent=2)
