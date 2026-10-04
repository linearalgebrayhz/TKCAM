#!/usr/bin/env python3
"""
Convert RE10K_eval 12D trajectories into universal 9D format.

Input format (12D):
  [x, y, z, dx, dy, dz, r11, r21, r31, r12, r22, r32]

Output format (9D):
  [r11, r21, r31, r12, r22, r32, x, y, z]

The script keeps scene IDs unchanged and copies side files so the output can be
merged directly by merge_universal_datasets.py.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import List

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


def read_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    with open(path, "r") as f:
        return [ln.strip() for ln in f.readlines() if ln.strip()]


def write_lines(path: Path, items: List[str]):
    with open(path, "w") as f:
        for item in items:
            f.write(item + "\n")


def ensure_dirs(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for sub in ("new_joint_vecs", "texts", "untagged_text", "metadata"):
        (root / sub).mkdir(exist_ok=True)


def convert_12d_to_9d(arr: np.ndarray) -> np.ndarray:
    """
    12D -> 9D conversion:
      drop dx/dy/dz at [3:6]
      move rot6d [6:12] to front
      keep position [0:3] at end
    """
    if arr.ndim != 2 or arr.shape[1] != 12:
        raise ValueError(f"Expected (T,12), got {arr.shape}")
    pos = arr[:, 0:3]
    rot6d = arr[:, 6:12]
    out = np.concatenate([rot6d, pos], axis=1)
    return out.astype(np.float32)


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(description="Convert RE10K_eval 12D -> 9D")
    parser.add_argument("--input_root", type=str, default="./dataset/RE10K_eval")
    parser.add_argument("--output_root", type=str, default="./dataset/RE10K_eval_9d")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy_splits", action="store_true", default=True)
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    motion_in = input_root / "new_joint_vecs"
    if not motion_in.exists():
        raise FileNotFoundError(f"Missing input motion dir: {motion_in}")

    ensure_dirs(output_root)
    npy_files = sorted(motion_in.glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found under: {motion_in}")

    converted = 0
    skipped = 0
    train_ids = set(read_lines(input_root / "train.txt"))
    val_ids = set(read_lines(input_root / "val.txt"))
    test_ids = set(read_lines(input_root / "test.txt"))

    out_train, out_val, out_test = [], [], []
    train_frames = []

    for npy in tqdm(npy_files, desc="Converting RE10K_eval"):
        sid = npy.stem
        out_npy = output_root / "new_joint_vecs" / f"{sid}.npy"

        if out_npy.exists() and not args.overwrite:
            skipped += 1
            # still track splits
            if sid in train_ids:
                out_train.append(sid)
            elif sid in val_ids:
                out_val.append(sid)
            elif sid in test_ids:
                out_test.append(sid)
            else:
                out_train.append(sid)
            continue

        arr = np.load(npy)
        arr9 = convert_12d_to_9d(arr)
        np.save(out_npy, arr9)
        converted += 1

        # Copy text (prefer texts then untagged_text)
        txt = input_root / "texts" / f"{sid}.txt"
        utxt = input_root / "untagged_text" / f"{sid}.txt"
        if txt.exists():
            copy_if_exists(txt, output_root / "texts" / f"{sid}.txt")
            copy_if_exists(txt, output_root / "untagged_text" / f"{sid}.txt")
        elif utxt.exists():
            copy_if_exists(utxt, output_root / "texts" / f"{sid}.txt")
            copy_if_exists(utxt, output_root / "untagged_text" / f"{sid}.txt")

        # Copy metadata
        meta_src = input_root / "metadata" / f"{sid}.json"
        meta_dst = output_root / "metadata" / f"{sid}.json"
        if meta_src.exists():
            try:
                with open(meta_src, "r") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            meta["converted_to"] = "9D [rot6d, x, y, z]"
            with open(meta_dst, "w") as f:
                json.dump(meta, f, indent=2)

        if sid in train_ids:
            out_train.append(sid)
            train_frames.append(arr9)
        elif sid in val_ids:
            out_val.append(sid)
        elif sid in test_ids:
            out_test.append(sid)
        else:
            out_train.append(sid)
            train_frames.append(arr9)

    # Copy optional files
    copy_if_exists(input_root / "scene_id_mapping.json", output_root / "scene_id_mapping.json")
    copy_if_exists(input_root / "curation_report.json", output_root / "curation_report.json")

    # Save splits
    if args.copy_splits:
        write_lines(output_root / "train.txt", sorted(set(out_train)))
        write_lines(output_root / "val.txt", sorted(set(out_val)))
        write_lines(output_root / "test.txt", sorted(set(out_test)))

    # Recompute Mean/Std from train split (if possible)
    if train_frames:
        train_concat = np.concatenate(train_frames, axis=0)
    else:
        # fallback: use all converted frames
        all_frames = [np.load(p) for p in sorted((output_root / "new_joint_vecs").glob("*.npy"))]
        train_concat = np.concatenate(all_frames, axis=0)
    np.save(output_root / "Mean.npy", np.mean(train_concat, axis=0).astype(np.float32))
    np.save(output_root / "Std.npy", np.std(train_concat, axis=0).astype(np.float32))

    print("Conversion complete")
    print(f"  input_root : {input_root}")
    print(f"  output_root: {output_root}")
    print(f"  converted  : {converted}")
    print(f"  skipped    : {skipped}")


if __name__ == "__main__":
    main()

