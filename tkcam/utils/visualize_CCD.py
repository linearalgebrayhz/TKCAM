#!/usr/bin/env python3
"""
Visualize processed CCD (Jiang et al.) camera trajectories.

Generates debug MP4s showing 4-panel view (3D trajectory, velocity plot,
top-down view, statistics + text caption) for random or specified scenes.

Works directly with the CCD output directory structure:
    new_joint_vecs/     *.npy  (12-D rotmat features)
    texts/              *.txt  (caption#POS-tagged tokens)
    untagged_text/      *.txt  (plain captions, fallback)
    train.txt / val.txt / test.txt

Usage:
    # Quick check: 5 random scenes from val split
    python visualize_CCD.py --dataset-dir ./dataset/CCD_rotmat -n 5 --split val

    # Specific scenes by ID
    python visualize_CCD.py --dataset-dir ./dataset/CCD_rotmat --scene-ids 000000,000100,001000

    # Larger batch with parallel rendering
    python visualize_CCD.py --dataset-dir ./dataset/CCD_rotmat -n 30 --workers 4 --stride 3

    # All splits, 10 each
    python visualize_CCD.py --dataset-dir ./dataset/CCD_rotmat -n 10 --split all

    # GIF output instead of MP4
    python visualize_CCD.py --dataset-dir ./dataset/CCD_rotmat -n 5 --gif
"""

import argparse
import gc
import os
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from tkcam.gen_camera import plot_camera_trajectory_debug, plot_camera_trajectory_animation
from tkcam.utils.unified_data_format import CameraDataFormat


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scene_ids(dataset_dir: str, split: str) -> list:
    """Load scene IDs from split file(s)."""
    if split == "all":
        splits = ["train", "val", "test"]
    else:
        splits = [split]

    ids = []
    for s in splits:
        path = os.path.join(dataset_dir, f"{s}.txt")
        if os.path.exists(path):
            with open(path) as f:
                ids.extend(line.strip() for line in f if line.strip())
    return ids


def load_caption(dataset_dir: str, scene_id: str) -> str:
    """Load caption, preferring texts/ (POS-tagged) over untagged_text/."""
    for subdir in ("texts", "untagged_text"):
        path = os.path.join(dataset_dir, subdir, f"{scene_id}.txt")
        if os.path.exists(path):
            with open(path) as f:
                text = f.read().strip()
            # texts/ format: "caption#word/POS word/POS ..."
            if "#" in text:
                text = text.split("#")[0].strip()
            return text
    return ""


# ---------------------------------------------------------------------------
# Render worker
# ---------------------------------------------------------------------------

def _render_one(args_tuple):
    """Render a single scene (worker-safe for multiprocessing)."""
    scene_id, npy_path, caption, out_path, mode, stride, fps, ext = args_tuple

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        data = np.load(npy_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if mode == "debug":
            plot_camera_trajectory_debug(
                data, out_path,
                title=f"CCD #{scene_id}  ({data.shape[0]} frames)",
                fps=fps,
                text_prompt=caption,
                format_type=CameraDataFormat.FULL_12_ROTMAT,
                stride=stride,
            )
        else:  # animation
            plot_camera_trajectory_animation(
                data, out_path,
                title=caption or f"CCD #{scene_id}",
                fps=fps,
                show_trail=True,
                trail_length=20,
                format_type=CameraDataFormat.FULL_12_ROTMAT,
                stride=stride,
                rotate_view=True,
            )

        plt.close("all")
        gc.collect()
        return scene_id, data.shape, True
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        gc.collect()
        return scene_id, None, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize processed CCD camera trajectories",
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="Root of processed CCD dataset (e.g. ./dataset/CCD_rotmat)",
    )
    parser.add_argument(
        "-n", "--num-scenes", type=int, default=10,
        help="Number of scenes to sample (default: 10)",
    )
    parser.add_argument(
        "--scene-ids", type=str, default=None,
        help="Comma-separated scene IDs to visualize (overrides -n and --split)",
    )
    parser.add_argument(
        "--split", choices=["train", "val", "test", "all"], default="val",
        help="Which split to sample from (default: val)",
    )
    parser.add_argument(
        "-o", "--output-dir", default=None,
        help="Output directory (default: <dataset-dir>_vis)",
    )
    parser.add_argument(
        "--mode", choices=["debug", "animation"], default="debug",
        help="Visualization mode (default: debug — 4-panel view with text)",
    )
    parser.add_argument(
        "--stride", type=int, default=2,
        help="Render every Nth frame (default: 2)",
    )
    parser.add_argument(
        "--fps", type=int, default=20,
        help="Output video FPS (default: 20)",
    )
    parser.add_argument(
        "--gif", action="store_true",
        help="Output GIF instead of MP4",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Parallel rendering workers (default: 1)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir or f"{dataset_dir.rstrip('/')}_vis"
    ext = ".gif" if args.gif else ".mp4"

    # --- Discover scenes ---
    if args.scene_ids:
        scene_ids = [s.strip() for s in args.scene_ids.split(",")]
    else:
        scene_ids = load_scene_ids(dataset_dir, args.split)
        if not scene_ids:
            print(f"No scene IDs found in {dataset_dir}/{args.split}.txt")
            sys.exit(1)
        random.seed(args.seed)
        n = min(args.num_scenes, len(scene_ids))
        scene_ids = random.sample(scene_ids, n)

    # --- Filter to those with .npy files ---
    valid = []
    for sid in scene_ids:
        npy = os.path.join(dataset_dir, "new_joint_vecs", f"{sid}.npy")
        if os.path.exists(npy):
            valid.append((sid, npy))
        else:
            print(f" skip {sid}: no .npy")

    if not valid:
        print("No valid scenes found.")
        sys.exit(1)

    print(f"Visualizing {len(valid)} scenes from {dataset_dir}")
    print(f"  Mode: {args.mode}  |  Stride: {args.stride}  |  "
          f"FPS: {args.fps}  |  Workers: {args.workers}")

    # --- Print scene summary ---
    for sid, npy in valid:
        caption = load_caption(dataset_dir, sid)
        data = np.load(npy)
        print(f"  {sid}  ({data.shape[0]:3d} frames)  {caption[:80]}")

    # --- Build work items ---
    work_items = []
    for sid, npy in valid:
        caption = load_caption(dataset_dir, sid)
        out_path = os.path.join(output_dir, f"{sid}_{args.mode}{ext}")
        work_items.append((sid, npy, caption, out_path, args.mode, args.stride, args.fps, ext))

    # --- Render ---
    t0 = time.time()
    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = list(pool.imap_unordered(_render_one, work_items))
    else:
        results = [_render_one(item) for item in work_items]
    elapsed = time.time() - t0

    ok = [(sid, shape) for sid, shape, success in results if success]
    fail = [sid for sid, _, success in results if not success]

    print(f"\n{'=' * 60}")
    print(f"Done in {elapsed:.1f}s  ({len(ok)} ok, {len(fail)} failed)")
    for sid, shape in ok:
        print(f"  OK  {sid}  {shape}")
    for sid in fail:
        print(f"  FAIL  {sid}")
    print(f"Output: {os.path.abspath(output_dir)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
