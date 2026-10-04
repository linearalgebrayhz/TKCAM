#!/usr/bin/env python3
"""
Convert GenDoP DataDop dataset into universal evaluator format.

Input (GenDoP DataDop):
  - **/*_transforms_cleaning.json   (c2w per frame)
  - **/*_caption.json               (caption fields)

Output (universal root):
  - new_joint_vecs/gendop_<group>_<shot>.npy   (T, 9): [rot6d(6), tx, ty, tz]
  - texts/gendop_<group>_<shot>.txt
  - untagged_text/gendop_<group>_<shot>.txt
  - metadata/gendop_<group>_<shot>.json
  - source_splits/gendop_{train,val,test}.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from tkcam.utils.camera_geometry import matrix_to_sixd, validate_rotation, forward_from_sixd


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def ensure_output_dirs(out_root: Path):
    for sub in ("new_joint_vecs", "texts", "untagged_text", "metadata", "source_splits"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)


def write_caption_files(out_root: Path, sid: str, caption: str):
    for sub in ("texts", "untagged_text"):
        with open(out_root / sub / f"{sid}.txt", "w") as f:
            f.write(caption + "\n")


def upsert_lines(path: Path, new_items: List[str]):
    existing = read_lines(path)
    seen = set(existing)
    merged = existing[:]
    for item in new_items:
        if item not in seen:
            merged.append(item)
            seen.add(item)
    with open(path, "w") as f:
        for item in merged:
            f.write(item + "\n")


def parse_gendop_traj9(transforms_json_path: Path) -> np.ndarray:
    """Parse c2w frames and produce [rot6d, position] trajectory."""
    with open(transforms_json_path, "r") as f:
        payload = json.load(f)

    frames = payload.get("frames", [])
    rows = []
    for fr in frames:
        mat = np.array(fr.get("transform_matrix", []), dtype=np.float64)
        if mat.shape == (4, 4):
            mat = mat[:3, :]
        if mat.shape != (3, 4):
            continue

        r_wc = validate_rotation(mat[:, :3])
        t_wc = mat[:, 3]
        rot6d = matrix_to_sixd(r_wc[None, ...]).reshape(-1)
        rows.append(np.concatenate([rot6d, t_wc], axis=0))

    if not rows:
        return np.empty((0, 9), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def parse_gendop_caption(caption_json_path: Path) -> str:
    with open(caption_json_path, "r") as f:
        payload = json.load(f)

    for key in ("Concise Interaction", "Detailed Interaction", "Movement"):
        text = payload.get(key, "")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def discover_transforms(gendop_root: Path) -> List[Path]:
    return sorted(gendop_root.rglob("*_transforms_cleaning.json"))


def assign_split(indices: np.ndarray, train_ratio: float, val_ratio: float):
    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]
    return set(train_idx.tolist()), set(val_idx.tolist()), set(test_idx.tolist())


def _motion_profile_from_traj9(
    traj9: np.ndarray,
    translation_thresh: float = 0.10,
    yaw_thresh: float = 0.12,
    pitch_thresh: float = 0.10,
) -> Dict[str, float]:
    """Compute simple motion profile from 9D [rot6d, pos]."""
    pos = traj9[:, 6:9]
    total_disp = float(np.linalg.norm(pos[-1] - pos[0]))

    fwd0 = forward_from_sixd(traj9[0, :6])
    fwd1 = forward_from_sixd(traj9[-1, :6])
    yaw0 = np.arctan2(fwd0[0], -fwd0[2])
    yaw1 = np.arctan2(fwd1[0], -fwd1[2])
    dyaw = float(np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0)))

    pitch0 = np.arcsin(np.clip(fwd0[1], -1, 1))
    pitch1 = np.arcsin(np.clip(fwd1[1], -1, 1))
    dpitch = float(pitch1 - pitch0)

    is_static = (
        total_disp <= translation_thresh
        and abs(dyaw) <= yaw_thresh
        and abs(dpitch) <= pitch_thresh
    )
    return {
        "total_disp": total_disp,
        "dyaw": dyaw,
        "dpitch": dpitch,
        "is_static": is_static,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert GenDoP to universal 9D+text format")
    parser.add_argument("--gendop_root", type=str, default="/data4/haozhe/CamTraj/data/GenDop/DataDop")
    parser.add_argument("--out_root", type=str, default="./dataset/RE10K_ET_GenDoP")
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--prefix", type=str, default="gendop")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--min_duration_sec", type=float, default=3.0)
    parser.add_argument("--enable_motion_filter", action="store_true",
                        help="Enable duration/static-scene filtering.")
    parser.add_argument("--max_static_ratio", type=float, default=0.35,
                        help="Max accepted static ratio among kept scenes when filtering.")
    parser.add_argument("--translation_thresh", type=float, default=0.10)
    parser.add_argument("--yaw_thresh", type=float, default=0.12)
    parser.add_argument("--pitch_thresh", type=float, default=0.10)
    parser.add_argument("--append_global_splits", action="store_true",
                        help="Append converted IDs into out_root/{train,val,test}.txt")
    args = parser.parse_args()

    gendop_root = Path(args.gendop_root)
    out_root = Path(args.out_root)
    if not gendop_root.exists():
        raise FileNotFoundError(f"GenDoP root not found: {gendop_root}")
    ensure_output_dirs(out_root)

    transforms_files = discover_transforms(gendop_root)
    if not transforms_files:
        raise FileNotFoundError(f"No *_transforms_cleaning.json found under {gendop_root}")

    processed_ids: List[str] = []
    skipped = 0
    skipped_short = 0
    skipped_static = 0
    accepted_static = 0
    accepted_total = 0

    for tf in tqdm(transforms_files, desc="Converting GenDoP"):
        if len(processed_ids) >= args.max_samples:
            break

        # Example: /.../0_0002/shot_0003_transforms_cleaning.json
        group = tf.parent.name
        stem = tf.stem.replace("_transforms_cleaning", "")
        sid = f"{args.prefix}_{group}_{stem}"

        out_npy = out_root / "new_joint_vecs" / f"{sid}.npy"
        if out_npy.exists() and not args.overwrite:
            processed_ids.append(sid)
            continue

        caption_path = tf.with_name(f"{stem}_caption.json")
        if not caption_path.exists():
            skipped += 1
            continue

        traj9 = parse_gendop_traj9(tf)
        if len(traj9) < args.min_frames:
            skipped += 1
            continue

        duration_sec = float(len(traj9) / max(args.fps, 1e-8))
        if args.enable_motion_filter and duration_sec < args.min_duration_sec:
            skipped_short += 1
            continue

        motion = _motion_profile_from_traj9(
            traj9,
            translation_thresh=args.translation_thresh,
            yaw_thresh=args.yaw_thresh,
            pitch_thresh=args.pitch_thresh,
        )
        if args.enable_motion_filter and motion["is_static"]:
            projected_static = accepted_static + 1
            projected_total = accepted_total + 1
            projected_ratio = projected_static / max(projected_total, 1)
            if projected_ratio > args.max_static_ratio:
                skipped_static += 1
                continue

        caption = parse_gendop_caption(caption_path)
        if not caption:
            skipped += 1
            continue

        np.save(out_npy, traj9)
        write_caption_files(out_root, sid, caption)

        meta = {
            "id": sid,
            "source": "GenDoP",
            "original_group": group,
            "original_stem": stem,
            "transforms_file": str(tf),
            "caption_file": str(caption_path),
            "num_frames": int(traj9.shape[0]),
            "duration_sec": duration_sec,
            "feature_dim": 9,
            "pose_semantics": "T_wc (from GenDoP c2w)",
            "motion_profile": motion,
        }
        with open(out_root / "metadata" / f"{sid}.json", "w") as f:
            json.dump(meta, f, indent=2)

        processed_ids.append(sid)
        accepted_total += 1
        if motion["is_static"]:
            accepted_static += 1

    # Deterministic train/val/test split for converted GenDoP IDs.
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(processed_ids))
    rng.shuffle(idx)
    train_idx, val_idx, test_idx = assign_split(idx, args.train_ratio, args.val_ratio)

    split_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    for i, sid in enumerate(processed_ids):
        if i in train_idx:
            split_ids["train"].append(sid)
        elif i in val_idx:
            split_ids["val"].append(sid)
        else:
            split_ids["test"].append(sid)

    for split_name in ("train", "val", "test"):
        upsert_lines(out_root / "source_splits" / f"gendop_{split_name}.txt", split_ids[split_name])

    if args.append_global_splits:
        for split_name in ("train", "val", "test"):
            upsert_lines(out_root / f"{split_name}.txt", split_ids[split_name])

    print("\nGenDoP conversion complete")
    print(f"  processed: {len(processed_ids)}")
    print(f"  skipped:   {skipped}")
    if args.enable_motion_filter:
        print(f"  skipped_short(<{args.min_duration_sec:.2f}s): {skipped_short}")
        print(f"  skipped_static(ratio>{args.max_static_ratio:.2f}): {skipped_static}")
        ratio = accepted_static / max(accepted_total, 1)
        print(f"  kept_static_ratio: {ratio:.3f} ({accepted_static}/{accepted_total})")
    print(f"  out_root:  {out_root}")


if __name__ == "__main__":
    main()

