#!/usr/bin/env python3
"""
Plot one benchmark sample from each result directory for quick visual QA.

Expected trajectory format in result folders: 9D CLaTr layout
    [r11, r21, r31, r12, r22, r32, x, y, z]

This script converts 9D -> 12D [x,y,z,dx,dy,dz,rot6d] so existing plotting
utilities in gen_camera.py can be reused directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from tkcam.gen_camera import plot_camera_trajectory, plot_camera_trajectory_animation
from tkcam.utils.unified_data_format import CameraDataFormat


_ROOT = Path(__file__).resolve().parent


def _resolve_benchmark_root(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    return (_ROOT.parent / "Benchmark_hub").resolve()


def _discover_result_dirs(benchmark_root: Path, prefix: str) -> List[Path]:
    dirs = sorted([p for p in benchmark_root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not dirs:
        raise FileNotFoundError(
            f"No result directories found under {benchmark_root} with prefix '{prefix}'"
        )
    return dirs


def _list_ids(result_dir: Path) -> Set[str]:
    return {p.stem for p in result_dir.glob("*.npy")}


def _choose_sample_id(result_dirs: List[Path], explicit_id: Optional[str]) -> Optional[str]:
    if explicit_id:
        return explicit_id
    id_sets = [_list_ids(d) for d in result_dirs]
    if not id_sets:
        return None
    common = set.intersection(*id_sets)
    if not common:
        return None
    return sorted(common)[0]


def _load_9d(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"Expected 2D 9D array at {path}, got shape {arr.shape}")
    return arr.astype(np.float32)


def _clatr9d_to_tkcam12d(arr9: np.ndarray) -> np.ndarray:
    rot6d = arr9[:, :6]
    pos = arr9[:, 6:9]
    vel = np.zeros_like(pos, dtype=np.float32)
    if pos.shape[0] > 1:
        vel[1:] = pos[1:] - pos[:-1]
        vel[0] = vel[1]
    out = np.concatenate([pos, vel, rot6d], axis=-1)
    return out.astype(np.float32)


def _pick_file_for_dir(result_dir: Path, sample_id: Optional[str]) -> Optional[Path]:
    if sample_id:
        p = result_dir / f"{sample_id}.npy"
        return p if p.is_file() else None
    files = sorted(result_dir.glob("*.npy"))
    return files[0] if files else None


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot one sample from each Benchmark_hub result_* directory."
    )
    parser.add_argument("--benchmark-root", type=str, default=None, help="Path to Benchmark_hub.")
    parser.add_argument("--result-prefix", type=str, default="result_", help="Result folder prefix.")
    parser.add_argument(
        "--sample-id",
        type=str,
        default=None,
        help="Sample id to plot for all folders. If omitted, auto-select first common id.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <benchmark_root>/plot_benchmark_samples",
    )
    parser.add_argument("--fps", type=int, default=20, help="Animation FPS.")
    parser.add_argument("--stride", type=int, default=2, help="Animation stride.")
    parser.add_argument("--no-animation", action="store_true", help="Disable mp4 animation output.")
    parser.add_argument("--no-static", action="store_true", help="Disable static png output.")
    args = parser.parse_args()

    benchmark_root = _resolve_benchmark_root(args.benchmark_root)
    result_dirs = _discover_result_dirs(benchmark_root, prefix=str(args.result_prefix))
    chosen_id = _choose_sample_id(result_dirs, explicit_id=args.sample_id)

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (benchmark_root / "plot_benchmark_samples").resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[plot_benchmark_samples] benchmark_root={benchmark_root}")
    print(f"[plot_benchmark_samples] output_dir={out_dir}")
    print(f"[plot_benchmark_samples] discovered_dirs={len(result_dirs)}")
    if chosen_id:
        print(f"[plot_benchmark_samples] sample_id={chosen_id}")
    else:
        print("[plot_benchmark_samples] no common sample id found; using first file per directory")

    plotted = 0
    skipped = 0
    summary: Dict[str, str] = {}

    for d in result_dirs:
        src = _pick_file_for_dir(d, sample_id=chosen_id)
        if src is None:
            skipped += 1
            summary[d.name] = "skipped:no_file_for_selected_id"
            continue

        try:
            arr9 = _load_9d(src)
            arr12 = _clatr9d_to_tkcam12d(arr9)
            base = _safe_name(d.name)
            sid = src.stem
            title = f"{d.name} | {sid}"

            if not args.no_static:
                png_path = out_dir / f"{base}__{sid}.png"
                plot_camera_trajectory(
                    arr12,
                    str(png_path),
                    title=title,
                    format_type=CameraDataFormat.FULL_12_ROTMAT,
                )

            if not args.no_animation:
                mp4_path = out_dir / f"{base}__{sid}.mp4"
                plot_camera_trajectory_animation(
                    arr12,
                    str(mp4_path),
                    title=title,
                    fps=int(args.fps),
                    format_type=CameraDataFormat.FULL_12_ROTMAT,
                    stride=int(args.stride),
                    rotate_view=True,
                )

            plotted += 1
            summary[d.name] = f"ok:{src.name}"
            print(f"[plot_benchmark_samples] plotted {d.name} <- {src.name}")
        except Exception as exc:
            skipped += 1
            summary[d.name] = f"error:{exc!r}"
            print(f"[plot_benchmark_samples] skip {d.name}: {exc!r}")

    summary_path = out_dir / "plot_summary.json"
    import json

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "benchmark_root": str(benchmark_root),
                "output_dir": str(out_dir),
                "sample_id": chosen_id,
                "plotted": plotted,
                "skipped": skipped,
                "details": summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"[plot_benchmark_samples] done plotted={plotted} skipped={skipped}")
    print(f"[plot_benchmark_samples] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
