#!/usr/bin/env python3
"""Convert TKCAM 12D rotmat datasets to velocity-free 9D rotmat datasets.

Input motion layout:
    [x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]

Output motion layout:
    [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    elif src.is_file():
        shutil.copy2(src, dst)


def _convert_stats(source: Path, output: Path) -> None:
    mean_path = source / "Mean.npy"
    std_path = source / "Std.npy"
    if not mean_path.is_file() or not std_path.is_file():
        return

    mean12 = np.load(mean_path)
    std12 = np.load(std_path)
    if mean12.shape[-1] != 12 or std12.shape[-1] != 12:
        raise ValueError(f"Expected 12D Mean/Std, got {mean12.shape} and {std12.shape}")

    mean9 = np.concatenate([mean12[:3], np.zeros(6, dtype=mean12.dtype)]).astype(mean12.dtype)
    std9 = np.concatenate([std12[:3], np.ones(6, dtype=std12.dtype)]).astype(std12.dtype)
    np.save(output / "Mean.npy", mean9)
    np.save(output / "Std.npy", std9)


def convert_dataset(source: Path, output: Path, overwrite: bool = False) -> None:
    motion_src = source / "new_joint_vecs"
    if not motion_src.is_dir():
        raise FileNotFoundError(f"Missing motion directory: {motion_src}")
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    motion_dst = output / "new_joint_vecs"
    motion_dst.mkdir(parents=True, exist_ok=True)

    for npy_path in tqdm(sorted(motion_src.glob("*.npy")), desc="Converting 12D to 9D"):
        arr12 = np.load(npy_path)
        if arr12.ndim != 2 or arr12.shape[1] != 12:
            raise ValueError(f"Expected (T,12) motion at {npy_path}, got {arr12.shape}")
        arr9 = np.concatenate([arr12[:, :3], arr12[:, 6:12]], axis=1).astype(np.float32)
        np.save(motion_dst / npy_path.name, arr9)

    for name in ("texts", "untagged_text", "train.txt", "val.txt", "test.txt", "scene_id_mapping.json"):
        _copy_if_exists(source / name, output / name)
    _convert_stats(source, output)

    with open(output / "format_info.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "format_type": "POSITION_ROTMAT_9",
                "layout": "[x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]",
                "converted_from": str(source),
                "source_layout": "[x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]",
            },
            f,
            indent=2,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="Source 12D rotmat dataset root.")
    parser.add_argument("--output", required=True, type=Path, help="Output 9D rotmat dataset root.")
    parser.add_argument("--overwrite", action="store_true", help="Replace output directory if it exists.")
    args = parser.parse_args()

    convert_dataset(args.source.expanduser().resolve(), args.output.expanduser().resolve(), args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
