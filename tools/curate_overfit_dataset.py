#!/usr/bin/env python3
"""
Stratified Sampling for Overfitting Dataset Curation

Selects N scenes from a processed RealEstate10K dataset using stratified
sampling to ensure diverse motion representation — prevents model collapse
into forward-only mode.

Strategy:
  1. Classify every scene using the same forward-vector motion analysis
     as the processor pipeline (via classify_motion from compute_dataset_statistics).
  2. Assign each scene a "primary motion bucket" based on its dominant motion.
  3. Set minimum quotas per bucket to guarantee rare-motion coverage.
  4. Fill remaining slots proportionally.
  5. Copy selected scenes into a self-contained dataset directory.

The output dataset has the same structure as the source and can be used
directly with the training pipeline (train.txt, Mean.npy, Std.npy, etc.).

Usage:
    # Quick 50-scene overfit set (default)
    python curate_overfit_dataset.py \
        --source ./dataset/RealEstate10K_rotmat1 \
        --output ./dataset/RealEstate10K_rotmat1_overfit50

    # Custom size and seed
    python curate_overfit_dataset.py \\
        --source ./dataset/RealEstate10K_rotmat1 \\
        --output ./dataset/RealEstate10K_rotmat1_overfit100 \\
        -n 100 --seed 42

    # Dry-run (print plan without copying)
    python curate_overfit_dataset.py \\
        --source ./dataset/RealEstate10K_rotmat1 \\
        -n 50 --dry-run
"""

import sys
import os
import json
import shutil
import argparse
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tkcam.utils.unified_data_format import CameraDataFormat, detect_format_from_dataset_name
from tkcam.compute_dataset_statistics import classify_motion


# ============================================================================
# Motion Buckets
# ============================================================================

# Buckets ordered by expected rarity (rarest first for quota priority).
# Each scene is assigned to exactly ONE bucket based on its primary motion.
MOTION_BUCKETS = [
    "static",
    "tilts_up",
    "moves_down",
    "moves_up",
    "tilts_down",
    "dollies_backward",
    "tracks_right",
    "tracks_left",
    "pans_right",
    "pans_left",
    "dollies_forward",
]


def assign_bucket(flags: Dict[str, bool]) -> str:
    """Assign a scene to a single motion bucket.

    Priority: rarest motion type wins.  If a scene has both
    dollies_forward + tilts_up, it goes into 'tilts_up' because
    that's rarer and we need it more.
    """
    for bucket in MOTION_BUCKETS:
        if flags.get(bucket, False):
            return bucket
    return "static"  # fallback


# ============================================================================
# Stratified Sampling
# ============================================================================

def stratified_sample(
    bucket_scenes: Dict[str, List[str]],
    n: int,
    seed: int = 42,
) -> Tuple[List[str], Dict[str, int]]:
    """Select n scenes with stratified sampling that favors rare motions.

    Strategy:
      1. Guarantee a minimum quota per non-empty bucket (≥ 2).
      2. Distribute remaining slots *equally* across all non-empty buckets
         (NOT proportionally to pool size).  This ensures rare buckets
         like tilts_up (156 pool) get the same extra slots as large
         buckets like tracks_left (775 pool).
      3. If a bucket's pool is exhausted, redistribute its surplus
         to the remaining buckets (round-robin by rarity order).
      4. Within each bucket, select randomly.

    Returns:
        (selected_ids, allocation_dict)
    """
    rng = random.Random(seed)

    non_empty = {b: ids for b, ids in bucket_scenes.items() if ids}
    num_buckets = len(non_empty)

    if num_buckets == 0:
        return [], {}

    # Phase 1: minimum quotas — guarantee at least 2 per bucket
    min_per_bucket = max(2, n // (num_buckets * 2))
    min_per_bucket = min(min_per_bucket, n // num_buckets)

    allocation: Dict[str, int] = {}
    for bucket, ids in non_empty.items():
        alloc = min(min_per_bucket, len(ids))
        allocation[bucket] = alloc

    # Phase 2: distribute remaining slots EQUALLY (not proportionally)
    remaining = n - sum(allocation.values())
    if remaining > 0:
        # Equal share per bucket, capped by what the pool can still provide
        equal_share = remaining // num_buckets
        for bucket in non_empty:
            can_add = len(non_empty[bucket]) - allocation[bucket]
            add = min(equal_share, can_add)
            allocation[bucket] += add

        # Phase 3: redistribute any leftover (from rounding or exhausted pools)
        # Give surplus to buckets in rarity order (MOTION_BUCKETS order)
        leftover = n - sum(allocation.values())
        if leftover > 0:
            # Iterate in rarity order so rare buckets get first dibs
            for bucket in MOTION_BUCKETS:
                if leftover <= 0:
                    break
                if bucket not in non_empty:
                    continue
                can_add = len(non_empty[bucket]) - allocation[bucket]
                add = min(can_add, leftover)
                allocation[bucket] += add
                leftover -= add

    # Phase 4: random selection within each bucket
    selected: List[str] = []
    for bucket, alloc in allocation.items():
        ids = non_empty[bucket]
        chosen = rng.sample(ids, min(alloc, len(ids)))
        selected.extend(chosen)

    return selected, allocation


# ============================================================================
# Dataset Assembly
# ============================================================================

def assemble_dataset(
    source_dir: Path,
    output_dir: Path,
    selected_ids: List[str],
    fmt: CameraDataFormat,
):
    """Create a self-contained dataset directory with selected scenes.

    Copies: new_joint_vecs/, texts/, metadata/, scene_id_mapping.json
    Recomputes: Mean.npy, Std.npy, train.txt (all scenes in train split)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("new_joint_vecs", "texts", "metadata"):
        (output_dir / sub).mkdir(exist_ok=True)

    # Load source mapping
    mapping_file = source_dir / "scene_id_mapping.json"
    if mapping_file.exists():
        with open(mapping_file) as f:
            full_mapping = json.load(f)
    else:
        full_mapping = {}

    # Copy files + build subset mapping
    subset_mapping = {}
    all_trajectories = []
    copied = 0

    for sid in sorted(selected_ids):
        # NPY
        npy_src = source_dir / "new_joint_vecs" / f"{sid}.npy"
        if not npy_src.exists():
            print(f"  Warning: missing {npy_src}, skipping")
            continue
        shutil.copy2(npy_src, output_dir / "new_joint_vecs" / f"{sid}.npy")

        # Load trajectory for stats
        traj = np.load(npy_src)
        all_trajectories.append(traj)

        # Text
        txt_src = source_dir / "texts" / f"{sid}.txt"
        if txt_src.exists():
            shutil.copy2(txt_src, output_dir / "texts" / f"{sid}.txt")

        # Metadata
        meta_src = source_dir / "metadata" / f"{sid}.json"
        if meta_src.exists():
            shutil.copy2(meta_src, output_dir / "metadata" / f"{sid}.json")

        # Also copy untagged text if it exists
        untagged_src = source_dir / "untagged_text" / f"{sid}.txt"
        if untagged_src.exists():
            (output_dir / "untagged_text").mkdir(exist_ok=True)
            shutil.copy2(untagged_src, output_dir / "untagged_text" / f"{sid}.txt")

        if sid in full_mapping:
            subset_mapping[sid] = full_mapping[sid]

        copied += 1

    # Write scene_id_mapping.json
    with open(output_dir / "scene_id_mapping.json", "w") as f:
        json.dump(subset_mapping, f, indent=2)

    # Compute Mean.npy, Std.npy from subset
    if all_trajectories:
        combined = np.concatenate(all_trajectories, axis=0)
        np.save(output_dir / "Mean.npy", np.mean(combined, axis=0))
        np.save(output_dir / "Std.npy", np.std(combined, axis=0))

    # All scenes go into train split (overfitting = train on everything)
    sorted_ids = sorted(selected_ids)
    with open(output_dir / "train.txt", "w") as f:
        f.write("\n".join(sorted_ids) + "\n")
    # Also create val/test with same content (eval scripts need them)
    with open(output_dir / "val.txt", "w") as f:
        f.write("\n".join(sorted_ids) + "\n")
    with open(output_dir / "test.txt", "w") as f:
        f.write("\n".join(sorted_ids) + "\n")

    # Save curation metadata
    curation_info = {
        "source_dataset": str(source_dir),
        "format": fmt.name,
        "num_scenes": copied,
        "scene_ids": sorted_ids,
    }
    with open(output_dir / "curation_info.json", "w") as f:
        json.dump(curation_info, f, indent=2)

    return copied


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Curate a stratified overfitting dataset from RealEstate10K",
    )
    parser.add_argument(
        "--source", required=True,
        help="Source dataset directory (e.g. ./dataset/RealEstate10K_rotmat1)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output dataset directory (default: <source>_overfit<N>)",
    )
    parser.add_argument(
        "-n", "--num-scenes", type=int, default=50,
        help="Number of scenes to select (default: 50)",
    )
    parser.add_argument(
        "--format", choices=["rotmat", "quat", "auto"], default="auto",
        help="Data format (default: auto-detect from directory name)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print sampling plan without creating files",
    )
    args = parser.parse_args()

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"Error: {source_dir} does not exist")
        return 1

    output_dir = Path(args.output) if args.output else Path(f"{str(source_dir).rstrip('/')}_overfit{args.num_scenes}")

    # Resolve format
    if args.format == "auto":
        fmt = detect_format_from_dataset_name(source_dir.name)
        if fmt is None:
            print("Cannot auto-detect format, defaulting to rotmat")
            fmt = CameraDataFormat.FULL_12_ROTMAT
    else:
        fmt = (CameraDataFormat.FULL_12_ROTMAT if args.format == "rotmat"
               else CameraDataFormat.QUATERNION_10)

    print(f"Source:  {source_dir}")
    print(f"Format:  {fmt.name}")
    print(f"Target:  {args.num_scenes} scenes")
    print(f"Seed:    {args.seed}")

    # --- Phase 1: Classify all scenes ---
    motion_dir = source_dir / "new_joint_vecs"
    npy_files = sorted(motion_dir.glob("*.npy"))
    print(f"\nClassifying {len(npy_files)} scenes...")

    bucket_scenes: Dict[str, List[str]] = defaultdict(list)
    scene_flags: Dict[str, Dict[str, bool]] = {}

    for i, npy_file in enumerate(npy_files):
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(npy_files)}...")
        sid = npy_file.stem
        try:
            traj = np.load(npy_file)
            flags = classify_motion(traj, fmt)
            bucket = assign_bucket(flags)
            bucket_scenes[bucket].append(sid)
            scene_flags[sid] = flags
        except Exception as e:
            print(f"  Warning: {sid}: {e}")

    # --- Phase 2: Print bucket distribution ---
    print(f"\n{'BUCKET DISTRIBUTION':^60}")
    print("-" * 60)
    total = sum(len(v) for v in bucket_scenes.values())
    for bucket in MOTION_BUCKETS:
        count = len(bucket_scenes.get(bucket, []))
        pct = count / total * 100 if total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {bucket:22s} {count:5,}  ({pct:5.1f}%)  {bar}")
    print(f"  {'TOTAL':22s} {total:5,}")

    # --- Phase 3: Stratified sampling ---
    selected, allocation = stratified_sample(bucket_scenes, args.num_scenes, args.seed)

    print(f"\n{'SAMPLING ALLOCATION':^60}")
    print("-" * 60)
    for bucket in MOTION_BUCKETS:
        pool = len(bucket_scenes.get(bucket, []))
        alloc = allocation.get(bucket, 0)
        if pool > 0 or alloc > 0:
            print(f"  {bucket:22s}  {alloc:3d} / {pool:5d}  selected")
    print(f"  {'TOTAL':22s}  {len(selected):3d}")

    # --- Phase 4: Verify diversity of selected set ---
    print(f"\n{'SELECTED SET MOTION COVERAGE':^60}")
    print("-" * 60)
    motion_coverage = defaultdict(int)
    for sid in selected:
        flags = scene_flags[sid]
        for k, v in flags.items():
            if v:
                motion_coverage[k] += 1
    for motion in MOTION_BUCKETS:
        cnt = motion_coverage.get(motion, 0)
        pct = cnt / len(selected) * 100 if selected else 0
        print(f"  {motion:22s}  {cnt:3d}  ({pct:5.1f}%)")

    if args.dry_run:
        print(f"\n[DRY RUN] Would create {output_dir} with {len(selected)} scenes")
        print("Selected IDs:", ", ".join(sorted(selected)[:10]), "...")
        return 0

    # --- Phase 5: Assemble dataset ---
    print(f"\nAssembling dataset at {output_dir}...")
    copied = assemble_dataset(source_dir, output_dir, selected, fmt)

    print(f"\n{'=' * 60}")
    print(f"Overfitting dataset created: {output_dir}")
    print(f"  Scenes:  {copied}")
    print(f"  Format:  {fmt.name}")
    print(f"  Splits:  train.txt = val.txt = test.txt = all {copied} scenes")
    print(f"  Stats:   Mean.npy, Std.npy recomputed from subset")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
