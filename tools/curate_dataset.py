#!/usr/bin/env python3
"""
Balanced Dataset Curation for RealEstate10K

Selects N scenes from a fully-processed RE10K dataset with:
  1. Motion-type balancing  — equal distribution across motion buckets,
     with dolly_forward allowed up to 2× any other bucket.
  2. Caption–trajectory alignment checking  — both CLIP-based semantic
     similarity and rule-based direction consistency.

Pipeline:
  Phase 1: Load all scenes, classify motion, load captions.
  Phase 2: Caption quality filter (rule-based alignment + CLIP).
  Phase 3: Motion-balanced selection with dolly-forward cap.
  Phase 4: Assemble output dataset (copy files, recompute stats, splits).

The script operates on an ALREADY-PROCESSED dataset directory that has:
    trajectories/    *.npy   (12-D rotmat features; legacy name: new_joint_vecs/)
    texts/           *.txt   (caption#POS-tagged tokens)
    untagged_text/   *.txt   (plain captions)
    metadata/        *.json  (optional)
    scene_id_mapping.json    (optional)
    train.txt / val.txt / test.txt

Usage:
    # Dry-run: show plan only
    python curate_dataset.py \\
        --source ./dataset/RealEstate10K_rotmat_full \\
        -n 3000 --dry-run

    # Full run with CLIP alignment filtering
    python curate_dataset.py \\
        --source ./dataset/RealEstate10K_rotmat_full \\
        --output ./dataset/RealEstate10K_rotmat_3k \\
        -n 3000 --clip-filter --clip-threshold 0.18

    # Skip CLIP (rule-based only, faster)
    python curate_dataset.py \\
        --source ./dataset/RealEstate10K_rotmat_full \\
        -n 3000

    # Custom dolly-forward multiplier
    python curate_dataset.py \\
        --source ./dataset/RealEstate10K_rotmat_full \\
        -n 3000 --dolly-multiplier 2.5
"""

import argparse
import gc
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, *args, **kwargs):
        return iterable

sys.path.insert(0, str(Path(__file__).parent))

from tkcam.utils.dataset_layout import TRAJ_DIR, traj_dir
from tkcam.compute_dataset_statistics import (
    CameraDataFormat,
    classify_motion,
    detect_format_from_dataset_name,
)


# ============================================================================
# Constants
# ============================================================================

# Motion buckets ordered by expected rarity (rarest first).
# Same order as curate_overfit_dataset.py for consistency.
MOTION_BUCKETS = [
    "static",
    "tilts_up",
    "moves_down",
    "moves_up",
    "tilts_down",
    "arcs_right",
    "arcs_left",
    "dollies_backward",
    "tracks_right",
    "tracks_left",
    "pans_right",
    "pans_left",
    "dollies_forward",
]

# Keyword mapping: motion flag → expected caption keywords.
# The caption should mention at least one keyword for each active motion flag.
MOTION_KEYWORDS = {
    "arcs_left":        ["arc", "arcs", "arcing", "orbit", "orbits", "orbiting",
                         "left", "curves", "circles"],
    "arcs_right":       ["arc", "arcs", "arcing", "orbit", "orbits", "orbiting",
                         "right", "curves", "circles"],
    "dollies_forward":  ["forward", "dolly", "dollies", "push", "pushing", "closer",
                         "approach", "advancing", "towards", "into", "toward"],
    "dollies_backward": ["backward", "back", "pull", "pulling", "away", "retreat",
                         "receding", "reverse", "out"],
    "tracks_left":      ["left", "track", "tracks", "tracking", "truck", "trucks",
                         "trucking", "lateral"],
    "tracks_right":     ["right", "track", "tracks", "tracking", "truck", "trucks",
                         "trucking", "lateral"],
    "pans_left":        ["pan", "pans", "panning", "left", "swivel", "arc", "arcs",
                         "arcing"],
    "pans_right":       ["pan", "pans", "panning", "right", "swivel", "arc", "arcs",
                         "arcing"],
    "moves_up":         ["up", "rise", "rising", "ascend", "crane", "boom", "elevat",
                         "pedestal"],
    "moves_down":       ["down", "descend", "lower", "drop", "crane", "boom", "sink",
                         "pedestal"],
    "tilts_up":         ["tilt", "tilts", "tilting", "up", "upward"],
    "tilts_down":       ["tilt", "tilts", "tilting", "down", "downward"],
    "static":           ["static", "still", "stationary", "remain", "fixed", "hold",
                         "steady", "stable", "slight"],
}

# Direction conflict pairs: (flag, keyword_that_contradicts_it)
DIRECTION_CONFLICTS = [
    ("arcs_left",        []),
    ("arcs_right",       []),
    ("dollies_forward",  ["backward", "back", "pull", "pulling", "retreat", "away"]),
    ("dollies_backward", ["forward", "push", "pushing", "closer", "approach", "toward"]),
    ("tracks_left",      []),   # "left" is shared with pans_left, hard to conflict
    ("tracks_right",     []),
    ("pans_left",        []),   # not enough signal to conflict-check pans vs tracks
    ("pans_right",       []),
    ("moves_up",         ["down", "descend", "lower", "drop", "sink"]),
    ("moves_down",       ["up", "rise", "rising", "ascend", "elevat"]),
    ("tilts_up",         ["down", "downward"]),
    ("tilts_down",       ["up", "upward"]),
]


# ============================================================================
# Phase 1: Scene Loading & Classification
# ============================================================================

def load_all_scenes(
    source_dir: Path,
    fmt: CameraDataFormat,
) -> List[Dict]:
    """Load all scenes: trajectory classification + caption text.

    Returns list of dicts with keys:
        scene_id, flags, active_motions, caption, npy_path, num_frames
    """
    motion_dir = Path(traj_dir(source_dir))
    npy_files = sorted(motion_dir.glob("*.npy"))
    print(f"Found {len(npy_files)} trajectory files")

    scenes = []
    for npy_file in tqdm(npy_files, desc="Classifying"):
        sid = npy_file.stem

        # Load trajectory & classify
        try:
            traj = np.load(npy_file)
        except Exception as e:
            print(f"  WARNING: failed to load {sid}: {e}")
            continue

        try:
            flags = classify_motion(traj, fmt)
        except Exception as e:
            print(f"  WARNING: classify failed {sid}: {e}")
            continue

        active = sorted([k for k, v in flags.items() if v])

        # Load caption
        caption = ""
        for subdir in ("texts", "untagged_text"):
            txt = source_dir / subdir / f"{sid}.txt"
            if txt.exists():
                raw = txt.read_text().strip()
                caption = raw.split("#")[0].strip() if "#" in raw else raw
                # Strip [CONFLICT] tag emitted by the AI captioner so it
                # doesn't contaminate CLIP embeddings or keyword matching.
                caption, _ = _strip_conflict_tag(caption)
                break

        scenes.append({
            "scene_id": sid,
            "flags": flags,
            "active_motions": active,
            "caption": caption,
            "npy_path": str(npy_file),
            "num_frames": len(traj),
        })

    return scenes


# ============================================================================
# Phase 2: Caption–Trajectory Alignment
# ============================================================================

# Regex for multi-stage caption separators (matches 'then', 'finally', 'before', 'after')
_STAGE_SPLIT_RE = re.compile(r'\b(?:then|finally|before|after(?:\s+that)?|next)\b', re.IGNORECASE)
# Regex to strip the model-emitted [CONFLICT] tag
_CONFLICT_TAG_RE = re.compile(r'\[CONFLICT\]', re.IGNORECASE)


def _strip_conflict_tag(caption: str) -> Tuple[str, bool]:
    """Remove [CONFLICT] from caption text and return (clean_caption, tag_was_present)."""
    had_tag = bool(_CONFLICT_TAG_RE.search(caption))
    clean = _CONFLICT_TAG_RE.sub("", caption).strip()
    return clean, had_tag


def rule_based_alignment(flags: Dict[str, bool], caption: str) -> Dict:
    """Check if caption text is consistent with detected motion.

    Handles multi-stage captions (joined by 'then'/'finally') by checking
    direction conflicts *per stage* rather than across the whole caption —
    e.g. "dollies forward, then pulls back" is valid for a dollies_forward
    scene and should not be flagged as a conflict.

    Returns:
        dict with keys: score (0-1), has_conflict (bool), details (str)
    """
    if not caption:
        return {"score": 0.0, "has_conflict": False, "details": "empty caption"}

    # Strip [CONFLICT] tag; if present the AI itself flagged the caption.
    clean_caption, ai_flagged = _strip_conflict_tag(caption)

    words = set(re.findall(r"[a-z]+", clean_caption.lower()))
    active = [k for k, v in flags.items() if v]

    if not active:
        # Static scene — any caption is OK
        return {"score": 0.5, "has_conflict": ai_flagged, "details": "static/no motion"}

    # 1. Keyword coverage: for each active motion, does the caption mention it?
    covered = 0
    for motion in active:
        kw_list = MOTION_KEYWORDS.get(motion, [])
        if any(kw in words for kw in kw_list):
            covered += 1
    coverage = covered / len(active) if active else 0

    # 2. Direction conflict check — operate per stage to avoid false positives
    #    in multi-stage captions like "dollies forward, then pulls back".
    stages = [s.strip() for s in _STAGE_SPLIT_RE.split(clean_caption) if s.strip()]
    # Use only the *first* stage for conflict checking (primary motion of the scene).
    first_stage_words = set(re.findall(r"[a-z]+", stages[0].lower())) if stages else words

    rule_conflict = False
    conflict_details = []
    for motion, conflict_words in DIRECTION_CONFLICTS:
        if flags.get(motion, False):
            for cw in conflict_words:
                if cw in first_stage_words:
                    rule_conflict = True
                    conflict_details.append(f"{motion}↔'{cw}'")

    has_conflict = ai_flagged or rule_conflict

    # Score: coverage weighted, penalize conflicts
    score = coverage
    if has_conflict:
        score *= 0.3  # heavy penalty

    details = f"coverage={coverage:.2f}"
    if ai_flagged:
        details += ", ai_flagged=[CONFLICT]"
    if conflict_details:
        details += f", rule_conflicts=[{', '.join(conflict_details)}]"

    return {"score": score, "has_conflict": has_conflict, "details": details}


def clip_alignment_scores(
    scenes: List[Dict],
    guidance_texts: Dict[str, str],
    device: str = "cuda",
    batch_size: int = 256,
) -> Dict[str, float]:
    """Compute CLIP text–text similarity between caption and guidance.

    We encode both the AI caption and the deterministic guidance as text
    embeddings with CLIP ViT-B/32, then compute cosine similarity.
    High similarity = caption is consistent with trajectory-derived guidance.

    Returns:
        dict mapping scene_id → cosine similarity score.
    """
    import torch
    try:
        import clip
    except ImportError:
        print("  CLIP not available (pip install git+https://github.com/openai/CLIP.git)")
        print("  Falling back to rule-based alignment only.")
        return {}

    print(f"Loading CLIP ViT-B/32 on {device}...")
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # Collect pairs: (scene_id, caption, guidance)
    pairs = []
    for sc in scenes:
        sid = sc["scene_id"]
        caption = sc["caption"]
        guidance = guidance_texts.get(sid, "")
        if caption and guidance:
            pairs.append((sid, caption, guidance))

    if not pairs:
        return {}

    print(f"Computing CLIP alignment for {len(pairs)} scenes...")
    scores: Dict[str, float] = {}

    for start in tqdm(range(0, len(pairs), batch_size), desc="CLIP"):
        batch = pairs[start:start + batch_size]
        sids = [p[0] for p in batch]
        captions = [p[1] for p in batch]
        guidances = [p[2] for p in batch]

        with torch.no_grad():
            cap_tokens = clip.tokenize(captions, truncate=True).to(device)
            gui_tokens = clip.tokenize(guidances, truncate=True).to(device)

            cap_feats = model.encode_text(cap_tokens)
            gui_feats = model.encode_text(gui_tokens)

            # L2 normalize
            cap_feats = cap_feats / cap_feats.norm(dim=-1, keepdim=True)
            gui_feats = gui_feats / gui_feats.norm(dim=-1, keepdim=True)

            # Cosine similarity per pair
            cos_sim = (cap_feats * gui_feats).sum(dim=-1).cpu().numpy()

        for sid, sim in zip(sids, cos_sim):
            scores[sid] = float(sim)

        # Memory cleanup
        del cap_tokens, gui_tokens, cap_feats, gui_feats, cos_sim
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scores


def generate_guidance_from_trajectory(
    trajectory: np.ndarray,
    fmt: CameraDataFormat,
) -> str:
    """Re-derive deterministic guidance from a saved 12D trajectory.

    Mirrors _generate_guidance() from realestate10k_processor but works
    on the saved 12D features (already relativized) rather than raw
    relative_motion_data.
    """
    from tkcam.utils.camera_geometry import forward_from_sixd, sixd_to_matrix

    if len(trajectory) < 2:
        return "camera remains static"

    positions = trajectory[:, :3]

    if fmt == CameraDataFormat.FULL_12_ROTMAT:
        rotations = sixd_to_matrix(trajectory[:, 6:12])
        fwd0 = forward_from_sixd(trajectory[0, 6:12])
        fwd1 = forward_from_sixd(trajectory[-1, 6:12])
    else:
        raise ValueError(f"Guidance generation only supports rotmat, got {fmt.name}")

    parts: List[str] = []

    # Translation via integrated local velocity, matching the processor.
    v_world = np.diff(positions, axis=0)
    r_w2c = np.transpose(rotations[:-1], (0, 2, 1))
    v_local = np.einsum("nij,nj->ni", r_w2c, v_world)
    v_local[np.abs(v_local) < 0.005] = 0.0
    accumulated_local_t = np.sum(v_local, axis=0)

    # Rotation
    yaw0 = np.arctan2(fwd0[0], -fwd0[2])
    yaw1 = np.arctan2(fwd1[0], -fwd1[2])
    dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))

    pitch0 = np.arcsin(np.clip(fwd0[1], -1, 1))
    pitch1 = np.arcsin(np.clip(fwd1[1], -1, 1))
    dpitch = pitch1 - pitch0

    is_panning = abs(dyaw) > 0.12
    is_trucking = abs(accumulated_local_t[0]) > 0.10

    if is_panning and is_trucking:
        parts.append("arcs right" if dyaw > 0 else "arcs left")
    else:
        if is_trucking:
            parts.append("tracks right" if accumulated_local_t[0] > 0 else "tracks left")
        if is_panning:
            parts.append("pans right" if dyaw > 0 else "pans left")

    if abs(accumulated_local_t[1]) > 0.10:
        parts.append("moves up" if accumulated_local_t[1] > 0 else "moves down")
    if abs(accumulated_local_t[2]) > 0.10:
        parts.append(
            "dollies forward"
            if accumulated_local_t[2] < 0
            else "dollies backward"
        )
    if abs(dpitch) > 0.10:
        parts.append("tilts up" if dpitch > 0 else "tilts down")

    if not parts:
        return "camera remains relatively static"
    if len(parts) == 1:
        return f"camera {parts[0]}"
    return f"camera {parts[0]} while it {' and '.join(parts[1:])}"


# ============================================================================
# Phase 3: Motion-Balanced Selection
# ============================================================================

def assign_bucket(flags: Dict[str, bool]) -> str:
    """Assign scene to primary motion bucket (rarest-wins)."""
    for bucket in MOTION_BUCKETS:
        if flags.get(bucket, False):
            return bucket
    return "static"


def balanced_select(
    scenes: List[Dict],
    n: int,
    dolly_multiplier: float = 2.0,
    prefer_composed_dolly: bool = True,
    seed: int = 42,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Select n scenes with motion-balanced distribution.

    Allocation strategy:
      - Let B = number of non-empty, non-dolly-forward buckets.
      - Base quota per bucket = n / (B + dolly_multiplier).
      - dolly_forward quota  = base * dolly_multiplier.
      - Within dolly_forward, prefer composed motions (dolly+pan, dolly+track)
        over pure dolly-forward-only scenes.
      - If any bucket has fewer scenes than its quota, redistribute surplus
        to other buckets (rarest first).

    Args:
        scenes:              list of scene dicts (from load_all_scenes, filtered).
        n:                   target number of scenes.
        dolly_multiplier:    dolly_forward gets this × the base quota.
        prefer_composed_dolly: if True, for dolly_forward bucket, sort so that
                               scenes with additional motions are selected first.
        seed:                random seed.

    Returns:
        (selected_scenes, allocation_dict)
    """
    rng = random.Random(seed)

    # Bucket all scenes
    bucket_scenes: Dict[str, List[Dict]] = defaultdict(list)
    for sc in scenes:
        bucket = assign_bucket(sc["flags"])
        bucket_scenes[bucket].append(sc)

    non_empty = {b: sc_list for b, sc_list in bucket_scenes.items() if sc_list}
    num_buckets = len(non_empty)

    if num_buckets == 0:
        return [], {}

    # Compute quotas
    non_dolly_buckets = [b for b in non_empty if b != "dollies_forward"]
    B = len(non_dolly_buckets)

    if "dollies_forward" in non_empty:
        # base = n / (B + multiplier)
        base_quota = n / (B + dolly_multiplier) if (B + dolly_multiplier) > 0 else n
        dolly_quota = int(round(base_quota * dolly_multiplier))
        other_quota = int(round(base_quota))
    else:
        other_quota = n // num_buckets if num_buckets > 0 else n
        dolly_quota = 0

    allocation: Dict[str, int] = {}
    for bucket in non_empty:
        if bucket == "dollies_forward":
            allocation[bucket] = min(dolly_quota, len(non_empty[bucket]))
        else:
            allocation[bucket] = min(other_quota, len(non_empty[bucket]))

    # Redistribute shortfall (when a bucket has fewer scenes than quota)
    total_allocated = sum(allocation.values())
    shortfall = n - total_allocated
    if shortfall > 0:
        # Give surplus to buckets with remaining capacity, rarest first
        for bucket in MOTION_BUCKETS:
            if shortfall <= 0:
                break
            if bucket not in non_empty:
                continue
            pool_size = len(non_empty[bucket])
            current = allocation[bucket]
            can_add = pool_size - current
            add = min(can_add, shortfall)
            allocation[bucket] += add
            shortfall -= add

    # Select within each bucket
    selected: List[Dict] = []
    for bucket, alloc in allocation.items():
        pool = non_empty[bucket]

        if bucket == "dollies_forward" and prefer_composed_dolly:
            # Sort: composed motions first (more active flags = better)
            pool_sorted = sorted(
                pool,
                key=lambda sc: len(sc["active_motions"]),
                reverse=True,
            )
            # Take top by composition, with some randomness in ties
            # Group by motion count, shuffle within each group
            by_count = defaultdict(list)
            for sc in pool_sorted:
                by_count[len(sc["active_motions"])].append(sc)
            ordered = []
            for count in sorted(by_count.keys(), reverse=True):
                group = by_count[count]
                rng.shuffle(group)
                ordered.extend(group)
            selected.extend(ordered[:alloc])
        else:
            rng.shuffle(pool)
            selected.extend(pool[:alloc])

    return selected, allocation


# ============================================================================
# Phase 4: Dataset Assembly
# ============================================================================

def assemble_dataset(
    source_dir: Path,
    output_dir: Path,
    selected: List[Dict],
    fmt: CameraDataFormat,
    train_ratio: float = 0.85,
    val_ratio: float = 0.10,
    seed: int = 42,
):
    """Create a self-contained dataset directory from selected scenes.

    Copies:  trajectories/, texts/, untagged_text/, metadata/
    Recomputes: Mean.npy, Std.npy, train.txt, val.txt, test.txt
    Saves: curation_report.json with full provenance.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in (TRAJ_DIR, "texts", "untagged_text", "metadata"):
        (output_dir / sub).mkdir(exist_ok=True)

    # Load source mapping
    mapping_file = source_dir / "scene_id_mapping.json"
    full_mapping = {}
    if mapping_file.exists():
        with open(mapping_file) as f:
            full_mapping = json.load(f)

    subset_mapping = {}
    all_trajectories = []
    sids_sorted = []

    for sc in tqdm(sorted(selected, key=lambda s: s["scene_id"]), desc="Copying"):
        sid = sc["scene_id"]

        # NPY (required)
        npy_src = Path(traj_dir(source_dir)) / f"{sid}.npy"
        if not npy_src.exists():
            continue
        shutil.copy2(npy_src, output_dir / TRAJ_DIR / f"{sid}.npy")
        all_trajectories.append(np.load(npy_src))

        # Text files
        for sub in ("texts", "untagged_text"):
            txt_src = source_dir / sub / f"{sid}.txt"
            if txt_src.exists():
                shutil.copy2(txt_src, output_dir / sub / f"{sid}.txt")

        # Metadata
        meta_src = source_dir / "metadata" / f"{sid}.json"
        if meta_src.exists():
            shutil.copy2(meta_src, output_dir / "metadata" / f"{sid}.json")

        if sid in full_mapping:
            subset_mapping[sid] = full_mapping[sid]

        sids_sorted.append(sid)

    # Scene ID mapping
    with open(output_dir / "scene_id_mapping.json", "w") as f:
        json.dump(subset_mapping, f, indent=2)

    # Mean / Std
    if all_trajectories:
        combined = np.concatenate(all_trajectories, axis=0)
        np.save(output_dir / "Mean.npy", np.mean(combined, axis=0))
        np.save(output_dir / "Std.npy", np.std(combined, axis=0))

    # Splits (deterministic shuffle)
    rng = random.Random(seed)
    shuffled = list(sids_sorted)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(train_ratio * n)
    n_val = int(val_ratio * n)

    splits = {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train:n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val:]),
    }
    for name, ids in splits.items():
        with open(output_dir / f"{name}.txt", "w") as f:
            f.write("\n".join(ids) + "\n")

    print(f"  Splits: {len(splits['train'])} train, "
          f"{len(splits['val'])} val, {len(splits['test'])} test")

    return len(sids_sorted)


# ============================================================================
# Reporting
# ============================================================================

def print_distribution(
    label: str,
    bucket_counts: Dict[str, int],
    total: int,
):
    """Pretty-print a motion distribution table."""
    print(f"\n{'─' * 70}")
    print(f"  {label}")
    print(f"{'─' * 70}")
    for bucket in MOTION_BUCKETS:
        count = bucket_counts.get(bucket, 0)
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:22s} {count:6,}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':22s} {total:6,}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curate a motion-balanced RE10K subset with caption alignment filtering",
    )
    parser.add_argument(
        "--source", required=True,
        help="Source dataset directory (must have trajectories/ or new_joint_vecs/, texts/)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output dataset directory (default: <source>_curated<N>)",
    )
    parser.add_argument(
        "-n", "--num-scenes", type=int, default=3000,
        help="Target number of scenes to select (default: 3000)",
    )
    parser.add_argument(
        "--dolly-multiplier", type=float, default=2.0,
        help="dolly_forward gets this × the base quota per bucket (default: 2.0)",
    )
    parser.add_argument(
        "--clip-filter", action="store_true",
        help="Enable CLIP-based caption–guidance alignment filtering",
    )
    parser.add_argument(
        "--clip-threshold", type=float, default=0.18,
        help="Minimum CLIP cosine similarity to keep a scene (default: 0.18)",
    )
    parser.add_argument(
        "--rule-threshold", type=float, default=0.0,
        help="Minimum rule-based alignment score to keep a scene, "
             "0 = only reject conflicts (default: 0.0)",
    )
    parser.add_argument(
        "--reject-conflicts", action="store_true", default=True,
        help="Reject scenes where caption contradicts trajectory direction "
             "(default: True)",
    )
    parser.add_argument(
        "--no-reject-conflicts", action="store_false", dest="reject_conflicts",
        help="Disable direction-conflict rejection",
    )
    parser.add_argument(
        "--format", choices=["rotmat", "auto"], default="auto",
        help="Data format (default: auto-detect rotmat)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print plan without creating files",
    )
    parser.add_argument(
        "--save-report", default=None,
        help="Save curation report JSON to this path",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist")
        return 1

    output_dir = (Path(args.output) if args.output
                  else Path(f"{str(source_dir).rstrip('/')}_curated{args.num_scenes}"))

    # Resolve format
    if args.format == "auto":
        fmt = detect_format_from_dataset_name(source_dir.name)
        if fmt is None:
            print("Cannot auto-detect format, defaulting to rotmat")
            fmt = CameraDataFormat.FULL_12_ROTMAT
    else:
        fmt = CameraDataFormat.FULL_12_ROTMAT

    if fmt != CameraDataFormat.FULL_12_ROTMAT:
        print(f"Error: only rotmat datasets are supported, detected {fmt.name}")
        return 1

    print(f"Source:           {source_dir}")
    print(f"Format:           {fmt.name}")
    print(f"Target:           {args.num_scenes} scenes")
    print(f"Dolly multiplier: {args.dolly_multiplier}×")
    print(f"CLIP filter:      {args.clip_filter}")
    print(f"Reject conflicts: {args.reject_conflicts}")

    # =================================================================
    # Phase 1: Load & classify all scenes
    # =================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 1: Loading & classifying scenes")
    print(f"{'=' * 70}")
    all_scenes = load_all_scenes(source_dir, fmt)
    print(f"Loaded {len(all_scenes)} scenes")

    # Source distribution
    source_buckets = Counter()
    for sc in all_scenes:
        source_buckets[assign_bucket(sc["flags"])] += 1
    print_distribution("SOURCE DISTRIBUTION (primary bucket)", source_buckets, len(all_scenes))

    # =================================================================
    # Phase 2: Caption alignment filtering
    # =================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 2: Caption–trajectory alignment filtering")
    print(f"{'=' * 70}")

    # 2a. Rule-based alignment
    rule_scores: Dict[str, Dict] = {}
    for sc in all_scenes:
        rule_scores[sc["scene_id"]] = rule_based_alignment(sc["flags"], sc["caption"])

    conflict_count = sum(1 for v in rule_scores.values() if v["has_conflict"])
    no_caption = sum(1 for sc in all_scenes if not sc["caption"])
    print(f"  Scenes without caption:     {no_caption}")
    print(f"  Scenes with direction conflict: {conflict_count}")

    # Filter pass 1: rule-based
    filtered = []
    rejected_rule = 0
    for sc in all_scenes:
        sid = sc["scene_id"]
        rule = rule_scores[sid]

        # Must have a caption
        if not sc["caption"]:
            rejected_rule += 1
            continue

        # Reject direction conflicts
        if args.reject_conflicts and rule["has_conflict"]:
            rejected_rule += 1
            continue

        # Minimum rule score threshold
        if rule["score"] < args.rule_threshold:
            rejected_rule += 1
            continue

        sc["rule_score"] = rule["score"]
        filtered.append(sc)

    print(f"  After rule-based filter: {len(filtered)} "
          f"(rejected {rejected_rule})")

    # 2b. CLIP alignment (optional)
    rejected_clip = 0
    if args.clip_filter and filtered:
        # Generate guidance for each scene trajectory
        print("  Generating trajectory guidance for CLIP comparison...")
        guidance_texts: Dict[str, str] = {}
        for sc in tqdm(filtered, desc="  Guidance"):
            traj = np.load(sc["npy_path"])
            guidance_texts[sc["scene_id"]] = generate_guidance_from_trajectory(traj, fmt)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_scores = clip_alignment_scores(filtered, guidance_texts, device=device)

        if clip_scores:
            # Print CLIP score distribution
            score_arr = np.array(list(clip_scores.values()))
            print(f"  CLIP scores: mean={score_arr.mean():.3f}, "
                  f"median={np.median(score_arr):.3f}, "
                  f"min={score_arr.min():.3f}, max={score_arr.max():.3f}")

            # Filter
            filtered_clip = []
            for sc in filtered:
                sid = sc["scene_id"]
                cs = clip_scores.get(sid, 1.0)  # keep if no CLIP score
                if cs >= args.clip_threshold:
                    sc["clip_score"] = cs
                    filtered_clip.append(sc)
                else:
                    rejected_clip += 1
            filtered = filtered_clip
            print(f"  After CLIP filter (threshold={args.clip_threshold}): "
                  f"{len(filtered)} (rejected {rejected_clip})")
        else:
            print("  CLIP scoring unavailable, skipping.")

    print(f"\n  Total after all filters: {len(filtered)}")

    # Post-filter distribution
    filtered_buckets = Counter()
    for sc in filtered:
        filtered_buckets[assign_bucket(sc["flags"])] += 1
    print_distribution("FILTERED POOL DISTRIBUTION", filtered_buckets, len(filtered))

    if len(filtered) < args.num_scenes:
        print(f"\n  WARNING: only {len(filtered)} scenes pass filters, "
              f"less than target {args.num_scenes}")
        print(f"  Will select all {len(filtered)} scenes.")

    # =================================================================
    # Phase 3: Motion-balanced selection
    # =================================================================
    print(f"\n{'=' * 70}")
    print("PHASE 3: Motion-balanced selection")
    print(f"{'=' * 70}")

    target_n = min(args.num_scenes, len(filtered))
    selected, allocation = balanced_select(
        filtered,
        n=target_n,
        dolly_multiplier=args.dolly_multiplier,
        prefer_composed_dolly=True,
        seed=args.seed,
    )

    print(f"\n  Allocation plan ({target_n} scenes):")
    for bucket in MOTION_BUCKETS:
        pool = filtered_buckets.get(bucket, 0)
        alloc = allocation.get(bucket, 0)
        if pool > 0 or alloc > 0:
            print(f"    {bucket:22s}  {alloc:4d} / {pool:6d} pool")
    print(f"    {'TOTAL':22s}  {len(selected):4d}")

    # Final distribution check
    final_buckets = Counter()
    final_motion_counts = Counter()
    composed_dolly = 0
    pure_dolly = 0
    for sc in selected:
        bucket = assign_bucket(sc["flags"])
        final_buckets[bucket] += 1
        for m in sc["active_motions"]:
            final_motion_counts[m] += 1
        if "dollies_forward" in sc["active_motions"]:
            if len(sc["active_motions"]) > 1:
                composed_dolly += 1
            else:
                pure_dolly += 1

    print_distribution("FINAL SELECTION (primary bucket)", final_buckets, len(selected))

    print(f"\n  dolly_forward detail: {composed_dolly} composed, {pure_dolly} pure "
          f"(total {composed_dolly + pure_dolly})")

    print(f"\n  All motion flags in selected set:")
    for m, c in final_motion_counts.most_common():
        pct = c / len(selected) * 100
        print(f"    {m:22s}  {c:4d}  ({pct:5.1f}%)")

    # =================================================================
    # Report
    # =================================================================
    report = {
        "source": str(source_dir),
        "format": fmt.name,
        "total_source_scenes": len(all_scenes),
        "rejected_rule": rejected_rule,
        "rejected_clip": rejected_clip,
        "filtered_pool": len(filtered),
        "target": args.num_scenes,
        "selected": len(selected),
        "dolly_multiplier": args.dolly_multiplier,
        "allocation": allocation,
        "source_distribution": dict(source_buckets),
        "final_distribution": dict(final_buckets),
        "composed_dolly": composed_dolly,
        "pure_dolly": pure_dolly,
        "scene_ids": sorted(sc["scene_id"] for sc in selected),
    }

    if args.save_report:
        with open(args.save_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\n  Report saved: {args.save_report}")

    # =================================================================
    # Phase 4: Assemble dataset
    # =================================================================
    if args.dry_run:
        print(f"\n[DRY RUN] Would create {output_dir} with {len(selected)} scenes")
        return 0

    print(f"\n{'=' * 70}")
    print("PHASE 4: Assembling output dataset")
    print(f"{'=' * 70}")

    copied = assemble_dataset(
        source_dir, output_dir, selected, fmt, seed=args.seed,
    )

    # Save report inside dataset
    report["output"] = str(output_dir)
    with open(output_dir / "curation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Dataset created: {output_dir}")
    print(f"  Scenes:  {copied}")
    print(f"  Format:  {fmt.name}")
    print(f"  Stats:   Mean.npy, Std.npy recomputed from curated subset")
    print(f"{'=' * 70}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
