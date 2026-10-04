"""
Camera Trajectory Scene Sampling and Visualization Tool

Randomly samples scenes from a processed RealEstate10K dataset, retrieves
metadata + captions, and generates trajectory visualizations.

Performance notes:
  - Default mode is 'debug' (most informative single animation).
  - Use --stride N to render every Nth frame (stride=3 → 3x faster).
  - Use --workers N to render scenes in parallel.
  - Use --no-rotate to disable the rotating 3D camera (faster render per frame).
  - Use --mode animation for the simpler single-panel animation.

Usage:
    # Quick 5-scene check (fast)
    python sample_scenes.py --dataset-dir ./dataset/RealEstate10K_rotmat -n 5 --stride 3

    # Full 100-scene batch with parallelism
    python sample_scenes.py --dataset-dir ./dataset/RealEstate10K_rotmat -n 100 --workers 4 --stride 2

    # Both animation types (slower, thorough)
    python sample_scenes.py --dataset-dir ./dataset/RealEstate10K_rotmat --mode both
"""

import os
import sys
import json
import random
import csv
import argparse
import gc
import time
import numpy as np
from pathlib import Path
from multiprocessing import Pool

sys.path.insert(0, str(Path(__file__).parent))

from tkcam.utils.unified_data_format import (
    UnifiedCameraData,
    CameraDataFormat,
    detect_format_from_dataset_name,
)
from tkcam.utils.camera_geometry import integrate_velocity_to_positions

# Import visualization functions
try:
    from tkcam.gen_camera import (
        plot_camera_trajectory,
        plot_camera_trajectory_animation,
        plot_camera_trajectory_debug,
    )

    VIS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: visualization functions unavailable: {e}")
    VIS_AVAILABLE = False

# RE10K raw data roots (for timestamp lookup)
RE10K_RAW_ROOTS = [
    "/data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train",
    "/data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/test",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def us_to_time(us):
    """Microseconds → 'Mm SSs' string."""
    if us is None:
        return None
    s = int(us / 1_000_000)
    return f"{s // 60}m{s % 60:02d}s"


def get_time_interval(scene_id):
    """Look up start/end timestamps from RE10K raw txt (microseconds)."""
    for root in RE10K_RAW_ROOTS:
        path = os.path.join(root, f"{scene_id}.txt")
        if os.path.exists(path):
            with open(path) as f:
                lines = f.readlines()
            if len(lines) < 2:
                return None, None
            return int(lines[1].split()[0]), int(lines[-1].split()[0])
    return None, None


def load_scene_info(scene_id, dataset_dir):
    """Load metadata + caption for one scene. Returns dict or None."""
    meta_path = os.path.join(dataset_dir, "metadata", f"{scene_id}.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        meta = json.load(f)

    untagged_path = os.path.join(dataset_dir, "untagged_text", f"{scene_id}.txt")
    text_path = os.path.join(dataset_dir, "texts", f"{scene_id}.txt")
    caption = ""
    if os.path.exists(untagged_path):
        with open(untagged_path, encoding="utf-8") as f:
            caption = f.read().strip()
    else:
        try:
            with open(text_path, encoding="utf-8") as f:
                caption = f.read().strip().split("#")[0].strip()
        except FileNotFoundError:
            caption = ""

    npy_path = os.path.join(dataset_dir, "new_joint_vecs", f"{scene_id}.npy")
    npy_exists = os.path.exists(npy_path)

    orig_id = meta.get("original_scene_id")
    start, end = get_time_interval(orig_id) if orig_id else (None, None)
    dur = (end - start) if (start is not None and end is not None) else None

    return {
        "scene_id": scene_id,
        "original_scene_id": orig_id,
        "video_url": meta.get("video_url"),
        "start_time": us_to_time(start),
        "end_time": us_to_time(end),
        "duration": us_to_time(dur),
        "caption": caption,
        "npy_path": npy_path if npy_exists else None,
        "npy_exists": npy_exists,
    }


# ---------------------------------------------------------------------------
# Visualization worker
# ---------------------------------------------------------------------------

def _render_scene(args_tuple):
    """Worker function for multiprocessing Pool."""
    (
        scene_id,
        caption,
        npy_path,
        out_dir,
        format_type,
        mode,
        stride,
        rotate,
        position_source,
        smooth_velocity,
        smooth_sigma,
    ) = args_tuple

    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (essential in workers)
    import matplotlib.pyplot as plt

    results = {}
    try:
        data = np.load(npy_path)
        scene_dir = os.path.join(out_dir, scene_id)

        use_vel_integration = (
            position_source == "velint"
            or (position_source == "auto" and format_type == CameraDataFormat.FULL_12_ROTMAT)
        )
        vis_data = data
        source_used = "raw"
        if use_vel_integration and data.ndim == 2 and data.shape[1] >= 6:
            try:
                vis_data = data.copy()
                vis_data[:, :3] = integrate_velocity_to_positions(
                    data,
                    smooth=smooth_velocity,
                    smooth_sigma=smooth_sigma,
                )
                source_used = "velint"
            except Exception as e:
                print(f"[{scene_id}] Warning: velocity integration failed, fallback to raw: {e}")

        if mode in ("animation", "both"):
            anim_path = os.path.join(scene_dir, f"{scene_id}_animation.mp4")
            os.makedirs(scene_dir, exist_ok=True)
            plot_camera_trajectory_animation(
                vis_data,
                anim_path,
                title=caption,
                fps=20,
                show_trail=True,
                trail_length=20,
                format_type=format_type,
                stride=stride,
                rotate_view=rotate,
            )
            results["animation"] = anim_path

        if mode in ("debug", "both"):
            dbg_path = os.path.join(scene_dir, f"{scene_id}_debug.mp4")
            os.makedirs(scene_dir, exist_ok=True)
            plot_camera_trajectory_debug(
                vis_data,
                dbg_path,
                title=f"Debug: {caption}",
                fps=20,
                show_velocity=True,
                show_topdown=True,
                text_prompt=caption,
                format_type=format_type,
                stride=stride,
            )
            results["debug"] = dbg_path

        results["data_shape"] = data.shape
        results["position_source_used"] = source_used
        plt.close("all")
        gc.collect()
        return scene_id, results

    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            import matplotlib.pyplot as plt
            plt.close("all")
        except Exception:
            pass
        gc.collect()
        return scene_id, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sample & visualize scenes from a processed RE10K dataset",
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Root of processed dataset (e.g. ./dataset/RealEstate10K_rotmat)",
    )
    parser.add_argument(
        "-n", "--num-scenes",
        type=int,
        default=10,
        help="Number of scenes to sample (default: 10)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Visualization output directory (default: <dataset-dir>_vis)",
    )
    parser.add_argument(
        "--format",
        choices=["rotmat", "quat", "auto"],
        default="auto",
        help="Data format, or 'auto' to detect from dataset name (default: auto)",
    )
    parser.add_argument(
        "--mode",
        choices=["debug", "animation", "both"],
        default="debug",
        help="Which animation(s) to generate (default: debug)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=2,
        help="Render every Nth frame (default: 2, i.e. 2x faster)",
    )
    parser.add_argument(
        "--no-rotate",
        action="store_true",
        help="Disable rotating 3D camera in animation mode (faster)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel rendering workers (default: 1)",
    )
    parser.add_argument(
        "--position-source",
        choices=["auto", "raw", "velint"],
        default="auto",
        help=(
            "Position channels used for visualization: "
            "'raw' uses xyz directly, "
            "'velint' reconstructs xyz from velocity (dx,dy,dz), "
            "'auto' uses velint for 12D rotmat datasets (default: auto)"
        ),
    )
    parser.add_argument(
        "--no-vel-smooth",
        action="store_true",
        help="Disable Gaussian smoothing when --position-source uses velocity integration",
    )
    parser.add_argument(
        "--vel-smooth-sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma for velocity-integrated positions (default: 1.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible sampling",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir
    output_dir = args.output_dir or f"{dataset_dir.rstrip('/')}_vis"

    if args.seed is not None:
        random.seed(args.seed)

    # --- Auto-detect format ---------------------------------------------------
    if args.format == "auto":
        fmt = detect_format_from_dataset_name(os.path.basename(dataset_dir))
        if fmt is None:
            print("Cannot auto-detect format from dataset name, defaulting to rotmat")
            fmt = CameraDataFormat.FULL_12_ROTMAT
    else:
        fmt = (CameraDataFormat.FULL_12_ROTMAT if args.format == "rotmat"
               else CameraDataFormat.QUATERNION_10)
    print(f"Format: {fmt.name}")

    # --- Discover & sample scenes --------------------------------------------
    meta_dir = os.path.join(dataset_dir, "metadata")
    if not os.path.isdir(meta_dir):
        print(f"Error: metadata dir not found: {meta_dir}")
        sys.exit(1)
    all_ids = [f.replace(".json", "") for f in os.listdir(meta_dir) if f.endswith(".json")]
    n = min(args.num_scenes, len(all_ids))
    sampled_ids = random.sample(all_ids, n)
    print(f"Sampled {n} / {len(all_ids)} scenes")

    # --- Load metadata --------------------------------------------------------
    results = []
    for sid in sampled_ids:
        info = load_scene_info(sid, dataset_dir)
        if info:
            results.append(info)
    print(f"Loaded metadata for {len(results)} scenes")

    # Print summary
    for i, r in enumerate(results, 1):
        npy_ok = "✓" if r["npy_exists"] else "✗"
        time_str = (f"{r['start_time']} - {r['end_time']}"
                    if r["start_time"] else "N/A")
        print(f"  [{i:3d}] {r['scene_id']}  npy:{npy_ok}  time:{time_str}")
        print(f"        {r['caption'][:100]}")

    # --- Save CSV -------------------------------------------------------------
    csv_path = os.path.join(output_dir, "sampled_scenes.csv")
    txt_path = os.path.join(output_dir, "sampled_scenes.txt")
    os.makedirs(output_dir, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["scene_id", "original_scene_id", "video_url",
                        "start_time", "end_time", "duration", "caption", "npy_exists"],
            extrasaction="ignore",
        )
        w.writeheader()
        w.writerows(results)
    print(f"CSV saved: {csv_path}")

    # --- Generate visualizations ----------------------------------------------
    # vis_paths[scene_id] = {"animation": path, "debug": path}
    vis_paths = {}

    if not VIS_AVAILABLE:
        print("Skipping visualizations (import failed)")
    else:
        scenes_with_npy = [r for r in results if r["npy_exists"]]
        if not scenes_with_npy:
            print("No scenes with NPY files found, skipping visualization")
        else:
            print(f"\nGenerating {args.mode} visualization for {len(scenes_with_npy)} scenes "
                  f"(stride={args.stride}, workers={args.workers}, rotate={not args.no_rotate}, "
                  f"position_source={args.position_source})")

            work_items = [
                (r["scene_id"], r["caption"], r["npy_path"], output_dir,
                 fmt, args.mode, args.stride, not args.no_rotate,
                 args.position_source, not args.no_vel_smooth, args.vel_smooth_sigma)
                for r in scenes_with_npy
            ]

            t0 = time.time()

            if args.workers > 1:
                with Pool(args.workers) as pool:
                    vis_results = list(pool.imap_unordered(_render_scene, work_items))
            else:
                vis_results = [_render_scene(item) for item in work_items]

            elapsed = time.time() - t0

            # Collect paths
            ok = [(sid, vr) for sid, vr in vis_results if vr is not None]
            fail = [(sid, vr) for sid, vr in vis_results if vr is None]
            for sid, vr in ok:
                vis_paths[sid] = vr

            # Report
            print(f"\n{'=' * 80}")
            print(f"Visualization complete in {elapsed:.1f}s")
            print(f"  Succeeded: {len(ok)}")
            print(f"  Failed:    {len(fail)}")
            if ok:
                per_scene = elapsed / len(ok)
                print(f"  Avg time/scene: {per_scene:.1f}s")
            for sid, vr in ok:
                for k in ("animation", "debug"):
                    if k in vr:
                        print(f"  ✓ {sid}/{k}")
                src = vr.get("position_source_used")
                if src:
                    print(f"    source: {src}")
            for sid, _ in fail:
                print(f"  ✗ {sid}")
            print(f"{'=' * 80}")

    # --- Save TXT (after visualization so we can include paths) ---------------
    abs_output_dir = os.path.abspath(output_dir)
    with open(txt_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(results, 1):
            sid = r["scene_id"]
            f.write(f"{'=' * 80}\n")
            f.write(f"Scene {i}: {sid}\n")
            f.write(f"{'=' * 80}\n")
            f.write(f"Original Scene ID: {r['original_scene_id']}\n")
            url = r["video_url"] or ""
            if r["start_time"]:
                url += f"&t={r['start_time']}"
            f.write(f"YouTube URL: {url}\n")
            f.write(f"Time Interval: {r['start_time'] or 'N/A'} - {r['end_time'] or 'N/A'}\n")
            f.write(f"Duration: {r['duration'] or 'N/A'}\n")
            f.write(f"Caption: {r['caption']}\n")
            f.write(f"NPY File: {'Available' if r['npy_exists'] else 'Not Found'}\n")

            # Include visualization paths if they were generated
            vr = vis_paths.get(sid, {})
            if "animation" in vr:
                f.write(f"Animations: {os.path.abspath(vr['animation'])}\n")
            if "debug" in vr:
                f.write(f"Debug: {os.path.abspath(vr['debug'])}\n")

            f.write("\n")

    print(f"TXT saved: {txt_path}")
    print(f"Output: {abs_output_dir}")


if __name__ == "__main__":
    main()
