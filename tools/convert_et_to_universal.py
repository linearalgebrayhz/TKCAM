#!/usr/bin/env python3
"""
Convert Exceptional Trajectories (ET) dataset into universal evaluator format.

Input (ET):
  - traj/<scene>.txt       (KITTI-like 12 floats per frame, w2c)
  - caption/<scene>.txt    (plain caption text)
  - optional split files:
      clatr_train_split.txt / clatr_val_split.txt / clatr_test_split.txt

Output (universal root):
  - trajectories/et_<scene>.npy     (T, 9): [rot6d(6), tx, ty, tz]
  - texts/et_<scene>.txt
  - untagged_text/et_<scene>.txt
  - metadata/et_<scene>.json
  - source_splits/et_{train,val,test}.txt
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from tkcam.utils.camera_geometry import matrix_to_sixd, validate_rotation, forward_from_sixd
from tkcam.utils.dataset_layout import TRAJ_DIR


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def parse_et_pose_file(traj_path: Path) -> np.ndarray:
    """
    Parse ET trajectory text file and return 9D [rot6d, position] in T_wc frame.

    ET line format (w2c):
      R11 R12 R13 tx R21 R22 R23 ty R31 R32 R33 tz
    """
    rows = []
    with open(traj_path, "r") as f:
        for ln in f:
            vals = ln.strip().split()
            if len(vals) != 12:
                continue
            row = np.array([float(v) for v in vals], dtype=np.float64)
            block = row.reshape(3, 4)  # w2c

            r_w2c = block[:, :3]
            t_w2c = block[:, 3]

            # Convert w2c -> c2w (T_wc), consistent with OpenGL/T_wc pipeline.
            r_wc = r_w2c.T
            t_wc = -r_w2c.T @ t_w2c
            r_wc = validate_rotation(r_wc)

            rot6d = matrix_to_sixd(r_wc[None, ...]).reshape(-1)  # (6,)
            rows.append(np.concatenate([rot6d, t_wc], axis=0))

    if not rows:
        return np.empty((0, 9), dtype=np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def load_et_caption_multi(caption_paths: List[Path]) -> str:
    """Concatenate unique first lines from caption sources (e.g., caption + caption_cam)."""
    pieces: List[str] = []
    seen = set()
    for cp in caption_paths:
        lines = read_lines(cp)
        if not lines:
            continue
        text = lines[0].strip()
        if not text:
            continue
        norm = " ".join(text.lower().split())
        if norm not in seen:
            pieces.append(text)
            seen.add(norm)
    return " ".join(pieces).strip()


def load_et_split_stems(et_root: Path) -> Dict[str, List[str]]:
    return {
        "train": read_lines(et_root / "clatr_train_split.txt"),
        "val": read_lines(et_root / "clatr_val_split.txt"),
        "test": read_lines(et_root / "clatr_test_split.txt"),
    }


def ensure_output_dirs(out_root: Path):
    for sub in (TRAJ_DIR, "texts", "untagged_text", "metadata", "source_splits"):
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


def _motion_profile_from_traj9(
    traj9: np.ndarray,
    translation_thresh: float = 0.10,
    yaw_thresh: float = 0.12,
    pitch_thresh: float = 0.10,
) -> Dict[str, float]:
    """
    Compute simple motion profile from 9D [rot6d, pos].
    Mirrors static detection style from classify_motion.
    """
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
    parser = argparse.ArgumentParser(description="Convert ET data to universal 9D+text format")
    parser.add_argument("--et_root", type=str, default="/data4/haozhe/CamTraj/data/ET/et-data")
    parser.add_argument("--out_root", type=str, default="./dataset/RE10K_ET_GenDoP")
    parser.add_argument(
        "--caption_subdirs",
        type=str,
        default="caption,caption_cam",
        help="Comma-separated ET caption folders to concatenate (order preserved)",
    )
    parser.add_argument("--min_frames", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=10000)
    parser.add_argument("--prefix", type=str, default="et")
    parser.add_argument("--overwrite", action="store_true")
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

    et_root = Path(args.et_root)
    out_root = Path(args.out_root)
    traj_dir = et_root / "traj"
    caption_subdirs = [s.strip() for s in args.caption_subdirs.split(",") if s.strip()]
    caption_dirs = [et_root / sub for sub in caption_subdirs]

    if not traj_dir.exists():
        raise FileNotFoundError(f"ET traj dir not found: {traj_dir}")
    if not any(cd.exists() for cd in caption_dirs):
        raise FileNotFoundError(
            f"No ET caption dirs found among: {[str(cd) for cd in caption_dirs]}"
        )

    ensure_output_dirs(out_root)

    split_stems = load_et_split_stems(et_root)
    ordered_stems = split_stems["train"] + split_stems["val"] + split_stems["test"]

    if not ordered_stems:
        ordered_stems = sorted([p.stem for p in traj_dir.glob("*.txt")])

    converted_ids: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
    processed = 0
    skipped = 0
    skipped_short = 0
    skipped_static = 0
    accepted_static = 0
    accepted_total = 0

    train_set = set(split_stems["train"])
    val_set = set(split_stems["val"])
    test_set = set(split_stems["test"])

    for stem in tqdm(ordered_stems, desc="Converting ET"):
        if processed >= args.max_samples:
            break

        traj_path = traj_dir / f"{stem}.txt"
        cap_paths = [cd / f"{stem}.txt" for cd in caption_dirs if cd.exists()]
        if not traj_path.exists() or not any(cp.exists() for cp in cap_paths):
            skipped += 1
            continue

        sid = f"{args.prefix}_{stem}"
        out_npy = out_root / TRAJ_DIR / f"{sid}.npy"
        if out_npy.exists() and not args.overwrite:
            # Still register split for bookkeeping.
            if stem in train_set:
                converted_ids["train"].append(sid)
            elif stem in val_set:
                converted_ids["val"].append(sid)
            elif stem in test_set:
                converted_ids["test"].append(sid)
            skipped += 1
            continue

        traj9 = parse_et_pose_file(traj_path)
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

        caption = load_et_caption_multi(cap_paths)
        if not caption:
            skipped += 1
            continue

        np.save(out_npy, traj9)
        write_caption_files(out_root, sid, caption)

        meta = {
            "id": sid,
            "source": "ET",
            "original_id": stem,
            "traj_file": str(traj_path),
            "caption_files": [str(cp) for cp in cap_paths if cp.exists()],
            "num_frames": int(traj9.shape[0]),
            "duration_sec": duration_sec,
            "feature_dim": 9,
            "pose_semantics": "T_wc (converted from ET w2c)",
            "motion_profile": motion,
        }
        with open(out_root / "metadata" / f"{sid}.json", "w") as f:
            json.dump(meta, f, indent=2)

        if stem in train_set:
            converted_ids["train"].append(sid)
        elif stem in val_set:
            converted_ids["val"].append(sid)
        elif stem in test_set:
            converted_ids["test"].append(sid)
        else:
            # If split files are unavailable/incomplete, place in train by default.
            converted_ids["train"].append(sid)

        accepted_total += 1
        if motion["is_static"]:
            accepted_static += 1
        processed += 1

    # Write ET source split files
    for split_name in ("train", "val", "test"):
        upsert_lines(out_root / "source_splits" / f"et_{split_name}.txt", converted_ids[split_name])

    # Optionally merge into global train/val/test files.
    if args.append_global_splits:
        for split_name in ("train", "val", "test"):
            upsert_lines(out_root / f"{split_name}.txt", converted_ids[split_name])

    print("\nET conversion complete")
    print(f"  processed: {processed}")
    print(f"  skipped:   {skipped}")
    if args.enable_motion_filter:
        print(f"  skipped_short(<{args.min_duration_sec:.2f}s): {skipped_short}")
        print(f"  skipped_static(ratio>{args.max_static_ratio:.2f}): {skipped_static}")
        ratio = accepted_static / max(accepted_total, 1)
        print(f"  kept_static_ratio: {ratio:.3f} ({accepted_static}/{accepted_total})")
    print(f"  out_root:  {out_root}")


if __name__ == "__main__":
    main()

