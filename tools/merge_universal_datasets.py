#!/usr/bin/env python3
"""
Merge multiple processed camera datasets into one universal evaluator dataset.

The output is directly compatible with train_universal_evaluator.py:
  output_root/
    new_joint_vecs/*.npy
    texts/*.txt
    untagged_text/*.txt
    metadata/*.json
    train.txt
    val.txt
    test.txt
    Mean.npy
    Std.npy

Each input source should already be processed into trajectory+text files.
"""

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

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


def ensure_dirs(output_root: Path):
    output_root.mkdir(parents=True, exist_ok=True)
    for sub in ("new_joint_vecs", "texts", "untagged_text", "metadata"):
        (output_root / sub).mkdir(exist_ok=True)


def parse_sources(source_args: List[str]) -> List[Tuple[str, Path]]:
    """
    Parse --source items.
    Supports:
      --source /path/to/dataset
      --source alias=/path/to/dataset
    """
    parsed: List[Tuple[str, Path]] = []
    for i, raw in enumerate(source_args):
        if "=" in raw:
            alias, p = raw.split("=", 1)
            alias = alias.strip()
            path = Path(p).expanduser()
        else:
            path = Path(raw).expanduser()
            alias = path.name
        if not alias:
            alias = f"src{i:02d}"
        parsed.append((alias, path))
    return parsed


def list_scene_ids(source_root: Path) -> List[str]:
    motion_dir = source_root / "new_joint_vecs"
    if not motion_dir.exists():
        return []
    return sorted([p.stem for p in motion_dir.glob("*.npy")])


def load_source_ids_for_split(source_root: Path, split_name: str, fallback_ids: List[str]) -> List[str]:
    split_file = source_root / f"{split_name}.txt"
    ids = read_lines(split_file)
    if ids:
        return ids
    return fallback_ids


def first_existing_text_path(source_root: Path, sid: str) -> Path:
    for sub in ("texts", "untagged_text"):
        p = source_root / sub / f"{sid}.txt"
        if p.exists():
            return p
    return Path("")


def main():
    parser = argparse.ArgumentParser(description="Merge datasets for universal evaluator training")
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        help="Input dataset root, optionally alias=/abs/path. Repeat for multiple sources.",
    )
    parser.add_argument("--output_root", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_dim", type=int, default=9, help="Expected trajectory dim")
    parser.add_argument("--prefix_mode", choices=["auto", "always", "never"], default="auto")
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--split_mode", choices=["resample", "source"], default="resample",
                        help="resample: make new split by ratios; source: merge source train/val/test if available.")
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")

    rng = random.Random(args.seed)
    output_root = Path(args.output_root).expanduser()
    ensure_dirs(output_root)

    sources = parse_sources(args.source)
    for alias, path in sources:
        if not path.exists():
            raise FileNotFoundError(f"Source not found: {alias}={path}")
        if not (path / "new_joint_vecs").exists():
            raise FileNotFoundError(f"Missing new_joint_vecs in source: {alias}={path}")

    # Pass 1: gather candidate IDs and detect collisions for auto-prefix mode.
    all_candidates: List[Tuple[str, Path, str]] = []
    stem_counts: Dict[str, int] = {}
    for alias, source_root in sources:
        ids = list_scene_ids(source_root)
        for sid in ids:
            all_candidates.append((alias, source_root, sid))
            stem_counts[sid] = stem_counts.get(sid, 0) + 1

    def map_out_id(alias: str, sid: str) -> str:
        if args.prefix_mode == "always":
            return f"{alias}__{sid}"
        if args.prefix_mode == "never":
            return sid
        # auto: prefix only when collision exists
        if stem_counts.get(sid, 0) > 1:
            return f"{alias}__{sid}"
        return sid

    manifest = {
        "sources": [{"alias": a, "path": str(p)} for a, p in sources],
        "seed": args.seed,
        "input_dim": args.input_dim,
        "prefix_mode": args.prefix_mode,
        "split_mode": args.split_mode,
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "items": [],
    }

    out_train: List[str] = []
    out_val: List[str] = []
    out_test: List[str] = []
    all_out_ids: List[str] = []
    all_train_frames: List[np.ndarray] = []

    # Cache source splits if needed.
    source_splits: Dict[Tuple[str, str], set] = {}
    if args.split_mode == "source":
        for alias, source_root in sources:
            all_ids = list_scene_ids(source_root)
            source_splits[(alias, "train")] = set(load_source_ids_for_split(source_root, "train", all_ids))
            source_splits[(alias, "val")] = set(read_lines(source_root / "val.txt"))
            source_splits[(alias, "test")] = set(read_lines(source_root / "test.txt"))

    # Pass 2: copy data and build mapping.
    for alias, source_root, sid in tqdm(all_candidates, desc="Merging scenes"):
        motion_path = source_root / "new_joint_vecs" / f"{sid}.npy"
        text_path = first_existing_text_path(source_root, sid)
        if not motion_path.exists() or not text_path.exists():
            continue

        motion = np.load(motion_path)
        if motion.ndim != 2 or motion.shape[1] != args.input_dim:
            continue

        out_id = map_out_id(alias, sid)
        out_motion = output_root / "new_joint_vecs" / f"{out_id}.npy"
        out_text = output_root / "texts" / f"{out_id}.txt"
        out_untagged = output_root / "untagged_text" / f"{out_id}.txt"
        out_meta = output_root / "metadata" / f"{out_id}.json"

        if out_motion.exists():
            # Safety for unexpected collisions.
            raise RuntimeError(f"Output ID collision after mapping: {out_id}")

        shutil.copy2(motion_path, out_motion)
        shutil.copy2(text_path, out_text)
        shutil.copy2(text_path, out_untagged)

        # Preserve metadata if exists, and augment.
        src_meta = source_root / "metadata" / f"{sid}.json"
        meta_payload = {}
        if src_meta.exists():
            with open(src_meta, "r") as f:
                meta_payload = json.load(f)
        meta_payload["merged_id"] = out_id
        meta_payload["source_alias"] = alias
        meta_payload["source_id"] = sid
        meta_payload["source_root"] = str(source_root)
        with open(out_meta, "w") as f:
            json.dump(meta_payload, f, indent=2)

        all_out_ids.append(out_id)
        manifest["items"].append({"merged_id": out_id, "source_alias": alias, "source_id": sid})

        if args.split_mode == "source":
            if sid in source_splits.get((alias, "train"), set()):
                out_train.append(out_id)
                all_train_frames.append(motion)
            elif sid in source_splits.get((alias, "val"), set()):
                out_val.append(out_id)
            elif sid in source_splits.get((alias, "test"), set()):
                out_test.append(out_id)
            else:
                # Missing explicit source split -> place in train.
                out_train.append(out_id)
                all_train_frames.append(motion)

    if not all_out_ids:
        raise RuntimeError("No valid scenes were merged.")

    # Create splits.
    if args.split_mode == "resample":
        shuffled = all_out_ids[:]
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        out_train = sorted(shuffled[:n_train])
        out_val = sorted(shuffled[n_train:n_train + n_val])
        out_test = sorted(shuffled[n_train + n_val:])

        # Gather train frames for Mean/Std.
        for sid in out_train:
            arr = np.load(output_root / "new_joint_vecs" / f"{sid}.npy")
            all_train_frames.append(arr)
    else:
        out_train = sorted(set(out_train))
        out_val = sorted(set(out_val))
        out_test = sorted(set(out_test))
        # ensure disjointness
        val_set = set(out_val) - set(out_train)
        test_set = set(out_test) - set(out_train) - val_set
        out_val = sorted(val_set)
        out_test = sorted(test_set)

        # Any leftovers go to train.
        covered = set(out_train) | set(out_val) | set(out_test)
        leftovers = [sid for sid in all_out_ids if sid not in covered]
        if leftovers:
            out_train.extend(sorted(leftovers))
            for sid in leftovers:
                arr = np.load(output_root / "new_joint_vecs" / f"{sid}.npy")
                all_train_frames.append(arr)

    write_lines(output_root / "train.txt", out_train)
    write_lines(output_root / "val.txt", out_val)
    write_lines(output_root / "test.txt", out_test)

    # Compute Mean/Std from train split frames.
    if not all_train_frames:
        for sid in out_train:
            all_train_frames.append(np.load(output_root / "new_joint_vecs" / f"{sid}.npy"))
    train_concat = np.concatenate(all_train_frames, axis=0)
    mean = np.mean(train_concat, axis=0).astype(np.float32)
    std = np.std(train_concat, axis=0).astype(np.float32)
    np.save(output_root / "Mean.npy", mean)
    np.save(output_root / "Std.npy", std)

    with open(output_root / "merge_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("Merge complete")
    print(f"  output_root: {output_root}")
    print(f"  total scenes: {len(all_out_ids)}")
    print(f"  train/val/test: {len(out_train)}/{len(out_val)}/{len(out_test)}")


if __name__ == "__main__":
    main()

