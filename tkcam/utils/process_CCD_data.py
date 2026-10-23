"""
CCD (Jiang et al.) Dataset Processing → 12-D rotmat format

Converts the 5-D character-centric camera representation
    {x, y, z, px, py}
into our canonical 12-D feature vector
    [pos(3), vel(3), rot6d(6)]
following strict OpenGL convention (see camera_geometry.py).

Coordinate Mapping (Jiang character-local → OpenGL world):
    x_jiang  →  x   (Right)
    y_jiang  →  y   (Up)
    z_jiang  → -z   (Jiang +z = character forward = our -Z forward)

Screen-offset sign convention (empirically verified over 27k trajectories):
    +px = screen-Left   →  camera-local -X
    +py = screen-Down   →  camera-local -Y
    ⇒  d_local = [-px, -py, -f]

Focal length:
    f = 1.0.  The screen coordinates are already in camera-normalised form
    (offset / focal_length).  Median |offset| ≈ 0.55 matches tan(30°) for
    Unity's default ~60° vertical FOV.

Pipeline:
    1. Convert (x,y,z) → OpenGL world position.
    2. Compute R_wc via look-at-with-offset using (px, py).
    3. Relativize trajectory so frame-0 = (origin, Identity).
    4. Build 12-D features via build_12d_features().
    5. Generate deterministic motion guidance via trajectory analysis.
    6. (Optional) LLM re-captioning: feed guidance + original template text
       to a text-only Qwen model (4B/8B) for natural language rewrite.
    7. Write untagged text to untagged_text/, then POS-tag → texts/.
    8. Compute Mean/Std from training split (rotation dims excluded from
       normalisation downstream, same as realestate10k_rotmat).
"""

import argparse
import gc
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

# Add current directory to path to import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tkcam.utils.dataset_layout import TRAJ_DIR
from tkcam.utils.camera_geometry import (
    build_12d_features,
    relativize_trajectory,
    validate_rotation,
)


# ============================================================================
# Look-At with Screen Offset
# ============================================================================

def look_at_with_offset(camera_pos, target_pos, screen_pos, f=1.0):
    """Compute R_wc such that *target_pos* projects to *screen_pos* on the image.

    Args:
        camera_pos: (3,) camera position in world.
        target_pos: (3,) look-at target in world (character head = origin).
        screen_pos: (2,) [px, py] normalised screen offset (Jiang convention).
        f:          virtual focal length.  1.0 for camera-normalised coords.

    Returns:
        R_wc: (3, 3) camera-to-world rotation matrix.

    Sign convention (empirically verified):
        +px = Left on screen   →  camera-local -X  (OpenGL +X = Right)
        +py = Down on screen   →  camera-local -Y  (OpenGL +Y = Up)
        ⇒  d_local = [-px, -py, -f]
    """
    v_world = target_pos - camera_pos
    dist = np.linalg.norm(v_world)
    if dist < 1e-6:
        return np.eye(3)
    v_world_unit = v_world / dist

    px, py = screen_pos
    d_local = np.array([-px, -py, -f])
    d_local_unit = d_local / np.linalg.norm(d_local)

    # --- Step 1: base rotation that maps d_local → v_world ----------------
    R_base = _rotation_between(d_local_unit, v_world_unit)

    # --- Step 2: roll correction to align camera-up with world-up ---------
    cam_up_current = R_base @ np.array([0.0, 1.0, 0.0])
    world_up = np.array([0.0, 1.0, 0.0])

    # Project world-up onto the plane perpendicular to the forward axis
    target_up = world_up - np.dot(world_up, v_world_unit) * v_world_unit
    if np.linalg.norm(target_up) < 1e-6:
        # Camera looking straight up/down – use arbitrary fallback
        target_up = (np.array([0.0, 0.0, 1.0])
                     - np.dot(np.array([0.0, 0.0, 1.0]), v_world_unit) * v_world_unit)
    target_up = target_up / np.linalg.norm(target_up)

    current_up_proj = cam_up_current - np.dot(cam_up_current, v_world_unit) * v_world_unit
    norm_cup = np.linalg.norm(current_up_proj)
    if norm_cup < 1e-6:
        return R_base
    current_up_proj = current_up_proj / norm_cup

    dot = np.clip(np.dot(current_up_proj, target_up), -1.0, 1.0)
    angle = np.arccos(dot)
    cross = np.cross(current_up_proj, target_up)
    if np.dot(cross, v_world_unit) < 0:
        angle = -angle
    R_roll = _rotation_around_axis(v_world_unit, angle)

    R_wc = R_roll @ R_base
    return R_wc


def _rotation_between(a, b):
    """Rodrigues rotation that maps unit vector *a* to unit vector *b*."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s < 1e-8:
        if c > 0:
            return np.eye(3)
        # 180-degree rotation
        ortho = np.array([1, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1, 0])
        v = np.cross(a, ortho)
        v = v / np.linalg.norm(v)
        return 2.0 * np.outer(v, v) - np.eye(3)
    vx = np.array([[0, -v[2], v[1]],
                    [v[2], 0, -v[0]],
                    [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s))


def _rotation_around_axis(axis, angle):
    """Rodrigues rotation by *angle* radians around *axis*."""
    axis = axis / np.linalg.norm(axis)
    ux, uy, uz = axis
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [c + ux*ux*(1-c),    ux*uy*(1-c) - uz*s, ux*uz*(1-c) + uy*s],
        [uy*ux*(1-c) + uz*s, c + uy*uy*(1-c),     uy*uz*(1-c) - ux*s],
        [uz*ux*(1-c) - uy*s, uz*uy*(1-c) + ux*s,  c + uz*uz*(1-c)],
    ])


# ============================================================================
# Trajectory Motion Analysis
# ============================================================================

def analyze_trajectory_motion(
    positions: np.ndarray,
    rotations: np.ndarray,
    screen_positions: Optional[np.ndarray] = None,
    fps: int = 30,
) -> str:
    """Deterministic motion analysis from raw (pre-relativized) trajectory.

    Produces a structured text that the LLM captioner (or POS tagger) can use.
    Captures both camera-centric motion (pan/tilt/track/dolly in the camera's
    reference frame at frame 0) and character-centric context (distance to
    subject, camera angle, screen position).

    OpenGL convention (camera_geometry.py):
        +X = Right,  +Y = Up,  -Z = Forward
        Forward vector = -R_wc[:, 2]

    Translation mapping (camera-local at frame 0):
        +x_local → "tracks right"      -x_local → "tracks left"
        +y_local → "cranes up"         -y_local → "cranes down"
        -z_local → "dollies forward"   +z_local → "dollies backward"

    Yaw (pan):
        yaw = atan2(fwd_x, -fwd_z)
        Positive Δyaw → "pans right"   (forward rotates from -Z toward +X)

    Pitch (tilt):
        pitch = asin(fwd_y)
        Positive Δpitch → "tilts up"   (forward rotates from -Z toward +Y)

    Args:
        positions:       (N, 3) world-space camera positions (OpenGL).
        rotations:       (N, 3, 3) R_wc rotation matrices.
        screen_positions: (N, 2) optional CCD [px, py] screen offset.
                          +px = Left on screen, +py = Down (verified).
        fps:             frame rate (for duration estimate).

    Returns:
        Structured guidance string.
    """
    N = len(positions)
    if N < 2:
        return "static camera, single frame"

    duration = N / fps
    motion_parts: List[str] = []
    context_parts: List[str] = []

    # ------------------------------------------------------------------
    # 1. Translation in camera-local frame of frame 0
    #    rel_disp = R0^T @ (p_end - p_start)  →  camera-local displacement
    # ------------------------------------------------------------------
    R0, t0 = rotations[0], positions[0]
    total_t = R0.T @ (positions[-1] - t0)          # (3,) camera-local
    abs_t = np.abs(total_t)
    max_t = float(np.max(abs_t))

    if max_t > 0.05:
        cands: List[Tuple[float, str]] = []
        # Camera-local +X → right
        if abs_t[0] > 0.05 and abs_t[0] >= 0.4 * max_t:
            cands.append(
                (abs_t[0], "tracks right" if total_t[0] > 0 else "tracks left")
            )
        # Camera-local +Y → up
        if abs_t[1] > 0.05 and abs_t[1] >= 0.4 * max_t:
            cands.append(
                (abs_t[1], "cranes up" if total_t[1] > 0 else "cranes down")
            )
        # Camera-local -Z → forward (OpenGL)
        if abs_t[2] > 0.05 and abs_t[2] >= 0.4 * max_t:
            cands.append(
                (abs_t[2],
                 "dollies forward" if total_t[2] < 0 else "dollies backward")
            )
        cands.sort(key=lambda x: x[0], reverse=True)
        motion_parts.extend(c[1] for c in cands[:2])

    # ------------------------------------------------------------------
    # 2. Rotation (pan / tilt from forward-vector change)
    #    Forward = -col2 of R_wc
    #    yaw   = atan2(fwd_x, -fwd_z)   positive → pans right
    #    pitch = asin(fwd_y)             positive → tilts up
    # ------------------------------------------------------------------
    fwd0 = -rotations[0][:, 2]
    fwd1 = -rotations[-1][:, 2]

    yaw0 = np.arctan2(fwd0[0], -fwd0[2])
    yaw1 = np.arctan2(fwd1[0], -fwd1[2])
    dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))

    pitch0 = np.arcsin(np.clip(fwd0[1], -1, 1))
    pitch1 = np.arcsin(np.clip(fwd1[1], -1, 1))
    dpitch = pitch1 - pitch0

    if abs(dyaw) > 0.10:        # ~6°
        motion_parts.append("pans right" if dyaw > 0 else "pans left")
    if abs(dpitch) > 0.08:      # ~5°
        motion_parts.append("tilts up" if dpitch > 0 else "tilts down")

    # ------------------------------------------------------------------
    # 3. Character-centric context (CCD-specific)
    # ------------------------------------------------------------------
    # 3a. Distance to subject (character at origin)
    distances = np.linalg.norm(positions, axis=1)
    d_start, d_end = float(distances[0]), float(distances[-1])
    d_mean = float(np.mean(distances))
    delta_d_pct = (d_end - d_start) / (d_start + 1e-6) * 100

    if abs(delta_d_pct) > 15:
        verb = "approaches" if delta_d_pct < 0 else "retreats from"
        context_parts.append(f"{verb} subject ({abs(delta_d_pct):.0f}%)")

    # 3b. Camera elevation angle relative to character
    xz_start = np.sqrt(positions[0, 0] ** 2 + positions[0, 2] ** 2)
    el_deg = float(np.degrees(np.arctan2(positions[0, 1], xz_start + 1e-8)))
    if el_deg > 30:
        context_parts.append("high angle")
    elif el_deg > 10:
        context_parts.append("slightly elevated")
    elif el_deg > -10:
        context_parts.append("eye-level")
    elif el_deg > -30:
        context_parts.append("low angle")
    else:
        context_parts.append("very low angle")

    # ------------------------------------------------------------------
    # 4. Speed & smoothness
    # ------------------------------------------------------------------
    step_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    total_path = float(np.sum(step_lengths))
    speed = total_path / duration if duration > 0 else 0

    if speed < 0.02:
        context_parts.append("very slow")
    elif speed < 0.05:
        context_parts.append("slow")
    elif speed < 0.12:
        context_parts.append("moderate pace")
    else:
        context_parts.append("fast")

    mean_step = float(np.mean(step_lengths)) if len(step_lengths) > 0 else 0
    if mean_step > 0:
        cv = float(np.std(step_lengths) / mean_step)
        if cv < 0.3:
            context_parts.append("steady")
        elif cv > 0.8:
            context_parts.append("variable speed")

    # ------------------------------------------------------------------
    # 5. Screen position of character (CCD-specific)
    #    +px = Left on screen,  +py = Down on screen  (verified)
    # ------------------------------------------------------------------
    if screen_positions is not None and len(screen_positions) > 0:
        px_mean = float(np.mean(screen_positions[:, 0]))
        py_mean = float(np.mean(screen_positions[:, 1]))
        h = ("screen-left" if px_mean > 0.2
             else ("screen-right" if px_mean < -0.2 else "screen-center"))
        v = ("upper" if py_mean < -0.2
             else ("lower" if py_mean > 0.2 else "middle"))
        context_parts.append(f"subject in {v} {h}")

    # ------------------------------------------------------------------
    # 6. Direction reversal detection (midpoint heuristic)
    # ------------------------------------------------------------------
    if N > 20:
        mid = N // 2
        t_mid = R0.T @ (positions[mid] - t0)
        t_end = R0.T @ (positions[-1] - t0)
        t_second_half = t_end - t_mid
        for axis_idx, axis_name in [(0, "lateral"), (2, "depth")]:
            if abs(t_mid[axis_idx]) > 0.08 and abs(t_second_half[axis_idx]) > 0.08:
                if t_mid[axis_idx] * t_second_half[axis_idx] < 0:
                    motion_parts.append(f"reverses {axis_name} direction")
                    break

    # ------------------------------------------------------------------
    # 7. Compose final guidance
    # ------------------------------------------------------------------
    if not motion_parts:
        motion_str = "camera remains relatively static"
    elif len(motion_parts) == 1:
        motion_str = f"camera {motion_parts[0]}"
    else:
        motion_str = (
            f"camera {motion_parts[0]} while it "
            f"{' and '.join(motion_parts[1:])}"
        )

    ctx = "; ".join(context_parts) if context_parts else ""
    return f"{motion_str}. Duration: {duration:.1f}s. {ctx}."


# ============================================================================
# Text-Only LLM Captioner (Qwen 4B / 8B)
# ============================================================================

class QwenTextCaptioner:
    """Text-only LLM for rewriting CCD camera motion descriptions.

    Since CCD has no video frames, we use a text-only Qwen model and feed it
    the deterministic motion analysis + original template text.  The LLM
    produces a natural one-sentence camera motion description.
    """

    SYSTEM_PROMPT = (
        "You are a professional cinematographer writing camera motion "
        "descriptions for a character animation dataset. The camera is always "
        "filming a character. Given a structured motion analysis and an "
        "original template description, write ONE natural sentence describing "
        "the camera movement.\n\n"
        "Rules:\n"
        "1. Start with 'The camera'\n"
        "2. Use professional terms: dolly, track, pan, tilt, crane, arc, "
        "push in, pull out\n"
        "3. Include direction, pace, and shot character "
        "(smooth, steady, dynamic)\n"
        "4. If the original mentions shot type (close-up, medium, wide), "
        "preserve it\n"
        "5. One sentence only, under 30 words\n"
        "6. You may briefly reference the character's position in frame "
        "(e.g. 'keeping the character in the left of frame') but keep the "
        "focus on camera motion\n\n"
        "Examples:\n"
        '- "The camera slowly dollies forward toward the character while '
        'craning up, framing a low-angle medium shot."\n'
        '- "The camera smoothly tracks left around the character at eye '
        'level, keeping them screen-right in a steady arc."'
    )

    USER_TEMPLATE = (
        "Motion analysis: {guidance}\n"
        "Original description: {original_text}\n\n"
        "Natural one-sentence camera motion description:"
    )

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        device: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None
        self.tokenizer = None

    def load_model(self):
        """Lazy-load the text-only Qwen model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading {self.model_name} on {self.device} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, padding_side="left"
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
        )
        if self.device != "cuda":
            self.model = self.model.to(self.device)
        print(f"Model loaded: {self.model_name}")

    def _build_messages(
        self, guidance: str, original_text: str,
    ) -> List[Dict]:
        return [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": self.USER_TEMPLATE.format(
                    guidance=guidance, original_text=original_text,
                ),
            },
        ]

    @staticmethod
    def _clean_caption(raw: str) -> str:
        """Post-process LLM output: strip think-tags, take first sentence."""
        import re
        # Strip Qwen3 <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Take only first sentence
        for sep in [". ", ".\n", "\n"]:
            if sep in text:
                text = text[: text.index(sep) + 1]
                break
        # Strip stray quotes
        text = text.strip().strip('"').strip("'").strip()
        return text

    def generate_captions(
        self, items: List[Tuple[str, str, str]],
    ) -> List[str]:
        """Batch-generate natural captions.

        Args:
            items: list of (guidance, original_text, scene_id) tuples.

        Returns:
            list of caption strings (same length as *items*).
        """
        import torch

        if self.model is None:
            self.load_model()

        results: List[str] = [""] * len(items)
        pbar = tqdm(total=len(items), desc="LLM captioning")

        for start in range(0, len(items), self.batch_size):
            batch = items[start : start + self.batch_size]

            # Build chat-template prompts
            texts: List[str] = []
            for guidance, original_text, _ in batch:
                msgs = self._build_messages(guidance, original_text)
                texts.append(
                    self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True,
                    )
                )

            inputs = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
            ).to(self.device)

            with torch.no_grad():
                gen_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            for i, input_ids in enumerate(inputs.input_ids):
                new_tokens = gen_ids[i][len(input_ids) :]
                raw = self.tokenizer.decode(
                    new_tokens, skip_special_tokens=True,
                ).strip()
                caption = self._clean_caption(raw)
                # Fallback: if caption is empty or too short, keep guidance
                if len(caption) < 10:
                    caption = ""
                results[start + i] = caption

            del inputs, gen_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pbar.update(len(batch))

        pbar.close()
        return results

    def unload_model(self):
        """Free GPU memory."""
        import torch

        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================================
# POS Tagging  (spaCy)
# ============================================================================

# incorporate tagging.py

def pos_tag_texts(untagged_dir, tagged_dir):
    """Run spaCy POS tagging on every file in *untagged_dir* → *tagged_dir*.

    Output format per file (one line):
        <caption>#<word1/POS1 word2/POS2 ...>
    This is the format expected by the dataset loader (t2m_dataset.py).
    """
    try:
        import spacy
    except ImportError:
        print("ERROR: spacy is required for POS tagging.  pip install spacy && "
              "python -m spacy download en_core_web_sm")
        raise

    nlp = spacy.load("en_core_web_sm")
    os.makedirs(tagged_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(untagged_dir) if f.endswith('.txt'))
    print(f"POS-tagging {len(files)} text files ...")
    for fname in files:
        with open(os.path.join(untagged_dir, fname), 'r') as f:
            text = f.read().strip()
        doc = nlp(text)
        tokens = [f"{tok.text}/{tok.pos_}" for tok in doc if not tok.is_punct]
        with open(os.path.join(tagged_dir, fname), 'w') as f:
            f.write(f"{text}#{' '.join(tokens)}\n")
    print("POS tagging complete.")


# ============================================================================
# Main Processing
# ============================================================================

def process_data(
    input_path: str,
    output_dir: str,
    skip_tagging: bool = False,
    min_frames: int = 0,
    max_frames: int = 300,
    caption_model: Optional[str] = None,
    caption_batch_size: int = 16,
    fps: int = 30,
):
    """Process CCD dataset: 5-D → 12-D conversion with optional LLM captioning.

    Phases:
        0. Length filtering → kept_indices; deterministic train/val/test split.
        1. Trajectory processing (5D→12D) + deterministic motion analysis.
        2. (Optional) LLM re-captioning via text-only Qwen.
        3. Write untagged_text/, POS tagging → texts/.
        4. Statistics (Mean/Std from training split).
    """
    print(f"Loading data from {input_path} ...")
    data = np.load(input_path, allow_pickle=True).item()
    cams = data['cam']
    infos = data['info']
    num_total = len(cams)
    print(f"Total trajectories: {num_total}")
    if min_frames > 0:
        print(f"  Length filter: min_frames={min_frames} "
              f"(≥{min_frames / fps:.1f}s at {fps}fps)")
    if max_frames < 9999:
        print(f"  Length filter: max_frames={max_frames} "
              f"(≤{max_frames / fps:.1f}s at {fps}fps)")

    # Create output directories
    motion_dir = os.path.join(output_dir, TRAJ_DIR)
    untagged_dir = os.path.join(output_dir, 'untagged_text')
    tagged_dir = os.path.join(output_dir, 'texts')
    os.makedirs(motion_dir, exist_ok=True)
    os.makedirs(untagged_dir, exist_ok=True)
    os.makedirs(tagged_dir, exist_ok=True)

    # ================================================================
    # Phase 0:  Length filtering + deterministic split
    # ================================================================
    kept_indices: List[int] = []
    skipped_short = 0
    for i in range(num_total):
        N = len(cams[i])
        if min_frames > 0 and N < min_frames:
            skipped_short += 1
            continue
        kept_indices.append(i)

    num_kept = len(kept_indices)
    print(f"After length filter: {num_kept} / {num_total} kept "
          f"({skipped_short} too short)")

    # Deterministic split (80 / 10 / 10) on *kept* indices
    rng = random.Random(42)
    shuffled = list(range(num_kept))
    rng.shuffle(shuffled)

    train_end = int(0.8 * num_kept)
    val_end = int(0.9 * num_kept)
    train_positions = set(shuffled[:train_end])       # positions in kept_indices

    def save_split(positions, filename):
        ids = sorted(kept_indices[j] for j in positions)
        with open(os.path.join(output_dir, filename), 'w') as f:
            for idx in ids:
                f.write(f"{idx:06d}\n")

    save_split(set(shuffled[:train_end]), 'train.txt')
    save_split(set(shuffled[train_end:val_end]), 'val.txt')
    save_split(set(shuffled[val_end:]), 'test.txt')
    print(f"  Splits: {train_end} train, "
          f"{val_end - train_end} val, {num_kept - val_end} test")

    # ================================================================
    # Phase 1:  Trajectory processing + motion analysis
    # ================================================================
    all_guidances: List[str] = []
    all_originals: List[str] = []
    train_features: List[np.ndarray] = []

    print("\nPhase 1: Processing trajectories + motion analysis ...")
    for j, orig_idx in enumerate(tqdm(kept_indices, desc="Trajectories")):
        cam_traj = np.array(cams[orig_idx])  # (N, 5)
        N = len(cam_traj)
        if max_frames > 0 and N > max_frames:
            cam_traj = cam_traj[:max_frames]
            N = max_frames

        positions = np.empty((N, 3))
        rotations = np.empty((N, 3, 3))
        screen_pos_arr = np.empty((N, 2))

        for n in range(N):
            x, y, z, px, py = cam_traj[n]
            # Jiang character-local → OpenGL world
            pos_world = np.array([x, y, -z])
            target_world = np.array([0.0, 0.0, 0.0])
            screen_pos = np.array([px, py])

            R_wc = look_at_with_offset(pos_world, target_world, screen_pos, f=1.0)
            R_wc = validate_rotation(R_wc)

            positions[n] = pos_world
            rotations[n] = R_wc
            screen_pos_arr[n] = screen_pos

        # Deterministic motion analysis (on PRE-relativized data)
        guidance = analyze_trajectory_motion(
            positions, rotations, screen_positions=screen_pos_arr, fps=fps,
        )

        # Relativize so frame 0 = (origin, Identity)
        positions, rotations = relativize_trajectory(positions, rotations)

        # Build 12-D features [pos(3), vel(3), rot6d(6)]
        features = build_12d_features(positions, rotations)

        filename = f"{orig_idx:06d}"
        np.save(os.path.join(motion_dir, f"{filename}.npy"), features)

        if j in train_positions:
            train_features.append(features)

        all_guidances.append(guidance)
        original_text = " ".join(s.strip() for s in infos[orig_idx] if s.strip())
        all_originals.append(original_text)

    # ================================================================
    # Phase 2:  Captioning (LLM or deterministic fallback)
    # ================================================================
    captions: List[str] = list(all_guidances)       # default = guidance

    if caption_model:
        print(f"\nPhase 2: LLM re-captioning with {caption_model} ...")
        try:
            captioner = QwenTextCaptioner(
                model_name=caption_model, batch_size=caption_batch_size,
            )
            items = [
                (all_guidances[j], all_originals[j], f"{kept_indices[j]:06d}")
                for j in range(num_kept)
            ]
            llm_captions = captioner.generate_captions(items)
            # Merge: use LLM output when non-empty, else keep guidance
            for j in range(num_kept):
                if llm_captions[j]:
                    captions[j] = llm_captions[j]
            captioner.unload_model()
        except Exception as e:
            print(f"LLM captioning failed: {e}")
            print("Falling back to deterministic guidance for all captions.")
    else:
        print("\nPhase 2: Skipped (no --caption-model). "
              "Using deterministic guidance as captions.")

    # Write untagged text
    for j in range(num_kept):
        filename = f"{kept_indices[j]:06d}"
        with open(os.path.join(untagged_dir, f"{filename}.txt"), 'w') as f:
            f.write(captions[j])

    # ================================================================
    # Phase 3:  POS tagging
    # ================================================================
    if not skip_tagging:
        print("\nPhase 3: POS tagging ...")
        pos_tag_texts(untagged_dir, tagged_dir)
    else:
        print("\nPhase 3: Skipped (--skip-tagging). "
              "Run tagging separately before training.")

    # ================================================================
    # Phase 4:  Statistics (training split only)
    # ================================================================
    print("\nPhase 4: Computing statistics from training set ...")
    if train_features:
        all_train = np.concatenate(train_features, axis=0)  # (M, 12)
        mean = np.mean(all_train, axis=0)
        std = np.std(all_train, axis=0)
        np.save(os.path.join(output_dir, 'Mean.npy'), mean)
        np.save(os.path.join(output_dir, 'Std.npy'), std)
        print(f"  Mean shape: {mean.shape}  Std shape: {std.shape}")
        print(f"  Mean: {mean}")
        print(f"  Std:  {std}")
    else:
        print("  WARNING: no training features — statistics not saved.")

    print(f"\nDone. {num_kept} trajectories → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process CCD (Jiang et al.) data into 12-D rotmat format "
                    "with trajectory-based motion analysis and optional "
                    "LLM captioning."
    )
    parser.add_argument(
        "--input", default="/data5/haozhe/CamTraj/data.npy",
        help="Path to the raw CCD data.npy file",
    )
    parser.add_argument(
        "--output", default="./dataset/CCD_rotmat",
        help="Output directory for the processed dataset",
    )
    parser.add_argument(
        "--skip-tagging", action="store_true",
        help="Skip POS tagging step (run tagging.py separately later)",
    )
    parser.add_argument(
        "--min-frames", type=int, default=0,
        help="Minimum trajectory length in frames (0 = no filter). "
             "E.g. 120 for ≥4s at 30fps.",
    )
    parser.add_argument(
        "--max-frames", type=int, default=300,
        help="Maximum trajectory length in frames (truncate if longer).",
    )
    parser.add_argument(
        "--caption-model", type=str, default=None,
        help="HuggingFace model for text-only LLM captioning, e.g. "
             "'Qwen/Qwen2.5-7B-Instruct'. If omitted, uses deterministic "
             "guidance as captions.",
    )
    parser.add_argument(
        "--caption-batch-size", type=int, default=16,
        help="Batch size for LLM captioning inference.",
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Frame rate of the CCD data (for duration estimates).",
    )
    args = parser.parse_args()
    process_data(
        input_path=args.input,
        output_dir=args.output,
        skip_tagging=args.skip_tagging,
        min_frames=args.min_frames,
        max_frames=args.max_frames,
        caption_model=args.caption_model,
        caption_batch_size=args.caption_batch_size,
        fps=args.fps,
    )
