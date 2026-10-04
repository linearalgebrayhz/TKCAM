#!/usr/bin/env python3
"""Visualize CLaTr 9D camera trajectories in Blender.

Run from Blender, for example:

    blender --python visualize_clatr9d_blender.py -- \
        --traj /home/haozhe/CamTraj/Benchmark_hub/reference/000120.npy \
        --layout clatr --sample-every 8

Or load a Benchmark_hub sample id:

    blender --python visualize_clatr9d_blender.py -- \
        --benchmark-dir /home/haozhe/CamTraj/Benchmark_hub \
        --sample-id 000120 --variant reference

Use comma-separated baselines or ``all`` for every registered hub folder (reference + result_* baselines), e.g.
``--variant ccd,director3d,gendop_with_rgbd`` or ``--variant all``.

For paper figures with consistent framing across baselines, use e.g.
``--world-origin reference --fit-bounds all`` so the camera and frustum scale follow the union of every
baseline's path for that sample ID (requires reference .npy). Default ``--camera-fit bounds`` fits the
trajectory plus ground in frame; use ``--camera-fit legacy`` for the previous heuristic only.
Frustum size uses path radius, step length, and camera distance (unless ``--no-frustum-camera-aware``).
For the same frustum size across every image in a batch, add ``--uniform-frustum-across-batch``.
Rendering several ``--variant`` for one ``--sample-id`` automatically uses the max frustum scale across those variants.

For **aligned scale across baselines for the same id**, use ``--world-origin reference --normalize-display per_sample_union``.
Every variant is scaled by the same factor so the union of all baselines fits a box of side ``--normalize-target-extent``;
frustums use a fixed multiplier (``--normalize-frustum-mul``) unless you pass ``--normalize-keep-frustum-auto``.
With reference + benchmark id, ``--ground-coherence sample`` (default) places one shared floor from the union of all baselines
(after normalization) so variants are directly comparable; use ``--ground-coherence scene`` for a per-render floor only.
are the first two columns of an OpenGL camera-to-world rotation matrix.
The camera looks down local -Z, so the world-space forward vector is -R[:, 2].
The script maps OpenGL world axes to Blender's Z-up world as:

    [x, y, z]_opengl -> [x, -z, y]_blender

The motion midpoint (middle frame) is translated to the world origin by default; optional
``--world-origin reference`` aligns all baselines to the reference trajectory's mid-frame for
comparable figures. ``--fit-bounds all`` unions every baseline's path so crop and frustum scale
match across variants when used with ``--world-origin reference``.
Ground size defaults to slightly larger than the trajectory XY extent. Frustum wire
colors run in a start-to-end gradient unless --color-by-stage is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import bpy
import mathutils
import numpy as np


def srgb_unit_to_linear(u: float) -> float:
    """One channel of sRGB in [0, 1] → linear light for Blender shader sockets."""

    if u <= 0.0:
        return 0.0
    if u <= 0.04045:
        return u / 12.92
    return ((u + 0.055) / 1.055) ** 2.4


def srgb8_to_linear_rgba(r: int, g: int, b: int, a: int = 255) -> Tuple[float, float, float, float]:
    """8-bit sRGB (+ alpha) → RGBA with linear RGB for Principled / World Background nodes."""

    return (
        srgb_unit_to_linear(r / 255.0),
        srgb_unit_to_linear(g / 255.0),
        srgb_unit_to_linear(b / 255.0),
        max(0.0, min(1.0, a / 255.0)),
    )


DEFAULT_STAGE_COLORS = {
    "default": (0.95, 0.55, 0.45, 1.0),
    "dolly": (0.1, 0.9, 0.25, 1.0),
    "pan": (0.15, 0.35, 1.0, 1.0),
    "tilt": (1.0, 0.48, 0.1, 1.0),
    "truck": (0.75, 0.25, 1.0, 1.0),
    "pedestal": (0.15, 0.9, 0.9, 1.0),
    "zoom": (1.0, 0.15, 0.55, 1.0),
}

# World + floor targets are specified in 8-bit sRGB (what you want in the saved image).
# Blender's Principled Base Color and World Background Color expect linear RGB, not sRGB/255.
VIS_BACKGROUND_RGBA = srgb8_to_linear_rgba(209, 209, 209)
# Floor: intended rgba(113, 113, 113) after render; base color must be linearized.
VIS_GROUND_PLANE_RGBA = srgb8_to_linear_rgba(113, 113, 113)
# Full strength on light-gray world still competes with the key light; keep modest fill.
VIS_WORLD_BACKGROUND_STRENGTH = 0.5

# Canonical variant keys (used in output filenames) → Benchmark_hub subdirectory.
_BENCHMARK_VARIANT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("reference", "reference"),
    ("ccd", "result_CCD"),
    ("director", "result_director"),
    ("director3d", "result_Director3D"),
    ("gendop_without_rgbd", "result_GenDoP_noRGB"),
    ("gendop_with_rgbd", "result_GenDoP_RGB"),
    ("tkcam", "result_TKCAM_firstFrame_bestfid_all_supported"),
    ("tkcam_noframe", "result_TKCAM_noFrame_bestfid_mixed"),
    ("tkcam_sparse_keyframe", "result_TKCAM_sparseKeyframe_full"),
)
VARIANT_SUBDIR: Dict[str, str] = dict(_BENCHMARK_VARIANT_ORDER)
VARIANT_ALIAS_TO_SLUG: Dict[str, str] = {}


def _norm_variant_token(s: str) -> str:
    return "".join(c.lower() for c in s.strip() if c.isalnum())


def _register_variant_aliases(slug: str, *extra_aliases: str) -> None:
    for token in (slug, *extra_aliases):
        VARIANT_ALIAS_TO_SLUG[_norm_variant_token(token)] = slug


_register_variant_aliases("reference", "ref", "gt", "groundtruth")
_register_variant_aliases("ccd")
_register_variant_aliases("director")
_register_variant_aliases("director3d", "director_3d")
_register_variant_aliases(
    "gendop_without_rgbd",
    "gendopwithoutrgbd",
    "GenDoP_withoutRGBD",
    "gendop_norgb",
    "gendop_no_rgb",
)
_register_variant_aliases(
    "gendop_with_rgbd",
    "gendopwithrgbd",
    "GenDoP_withRGBD",
    "gendop",
)
_register_variant_aliases(
    "tkcam",
    "tkcam_firstframe",
    "TKCAM",
    "TKCAM_firstFrame",
)
_register_variant_aliases("tkcam_noframe", "tkcamnoframe", "TKCAM_noframe", "tkcam_no_frame")
_register_variant_aliases(
    "tkcam_sparse_keyframe",
    "tkcamsparsekeyframe",
    "TKCAM_sparseKeyframe",
    "tkcam_sparse",
    "TKCAM_sprarseKeyframe",
)


def benchmark_variant_keys_all() -> List[str]:
    return [slug for slug, _ in _BENCHMARK_VARIANT_ORDER]


def resolve_variant_keys(variant_arg: str) -> List[str]:
    """Parse --variant: comma-separated slugs/aliases, or ``all``."""

    parts = [p.strip() for p in variant_arg.split(",") if p.strip()]
    if not parts:
        return ["reference"]
    keys: List[str] = []
    for p in parts:
        if _norm_variant_token(p) == "all":
            return benchmark_variant_keys_all()
        n = _norm_variant_token(p)
        slug = VARIANT_ALIAS_TO_SLUG.get(n)
        if slug is None:
            known = ", ".join(VARIANT_SUBDIR.keys())
            raise ValueError(
                f"Unknown variant {p!r}. Use comma-separated names or 'all'. "
                f"Keys: {known}. Aliases include gendop, ref, GenDoP_withRGBD, TKCAM_noframe, etc."
            )
        keys.append(slug)
    out: List[str] = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    script_args = argv[argv.index("--") + 1 :] if "--" in argv else []

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", type=Path, default=None, help="Path to a .npy trajectory.")
    parser.add_argument(
        "--benchmark-dir",
        type=Path,
        default=Path("/home/haozhe/CamTraj/Benchmark_hub"),
        help="Benchmark_hub root used with --sample-id.",
    )
    parser.add_argument("--sample-id", type=str, default=None, help="Benchmark_hub sample id, e.g. 000120.")
    parser.add_argument(
        "--variant",
        type=str,
        default="reference",
        help=(
            "Benchmark baseline(s) for --sample-id / batch: comma-separated keys or aliases, or 'all' "
            "(reference + CCD, director, Director3D, GenDoP ± RGBD, TKCAM first-frame / no-frame / sparse keyframe). "
            "Examples: gendop_with_rgbd,TKCAM_noframe or GenDoP_withRGBD or all."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=("clatr", "xyz_rot6d", "auto"),
        default="clatr",
        help="9D layout: clatr=[rot6d,xyz], xyz_rot6d=[xyz,rot6d].",
    )
    parser.add_argument(
        "--sample-every",
        type=int,
        default=8,
        help="Draw a frustum every N frames; use a larger N for sparser paths and less visual overlap.",
    )
    parser.add_argument("--scale", type=float, default=1.0, help="Scale camera translations.")
    parser.add_argument(
        "--world-origin",
        choices=("midframe", "reference", "bbox"),
        default="midframe",
        help=(
            "After OpenGL→Blender mapping: midframe=center on this trajectory's middle frame (default); "
            "reference=center on reference baseline's mid-frame (for fair multi-baseline comps); "
            "bbox=center on this trajectory's bounding box."
        ),
    )
    parser.add_argument(
        "--fit-bounds",
        choices=("scene", "reference", "all"),
        default="scene",
        help=(
            "What geometry sets camera distance and frustum auto-scale: only the drawn trajectory (scene), "
            "union with reference, or union with every variant .npy found for this sample id (use with "
            "--world-origin reference for comparable paper renders)."
        ),
    )
    parser.add_argument(
        "--normalize-display",
        choices=("none", "per_scene", "per_sample_union"),
        default="none",
        help=(
            "Uniformly scale positions before drawing: target max AABB edge = --normalize-target-extent. "
            "per_scene: scale from the current trajectory (+ fit-bounds extras). "
            "per_sample_union: one scale per sample id from the union of ALL benchmark variants (same id), "
            "requires --world-origin reference. Makes frustum/camera comparable across baselines."
        ),
    )
    parser.add_argument(
        "--normalize-target-extent",
        type=float,
        default=2.0,
        help="Max AABB edge length in Blender units after normalization.",
    )
    parser.add_argument(
        "--normalize-min-extent",
        type=float,
        default=1e-4,
        help="Minimum extent when computing scale (avoids blow-ups on degenerate paths).",
    )
    parser.add_argument(
        "--normalize-frustum-mul",
        type=float,
        default=1.12,
        help="When --normalize-display is on (without --normalize-keep-frustum-auto), frustum dims *= this.",
    )
    parser.add_argument(
        "--normalize-keep-frustum-auto",
        action="store_true",
        help="Keep camera-aware automatic frustum sizing when normalizing (default: fixed frustum after normalize).",
    )
    parser.add_argument(
        "--camera-fit",
        choices=("bounds", "legacy"),
        default="bounds",
        help=(
            "bounds: place camera along a fixed view axis so the trajectory (+ ground) fits the render frame; "
            "legacy: previous distance ~ max(extent, span) heuristic (can clip long paths)."
        ),
    )
    parser.add_argument(
        "--frame-margin",
        type=float,
        default=1.22,
        help="bounds fit: enlarge the virtual frame by this factor (>=1) so paths stay inside.",
    )
    parser.add_argument(
        "--camera-lens",
        type=float,
        default=32.0,
        help="Perspective camera focal length in mm (Blender default sensor).",
    )
    parser.add_argument(
        "--view-axis-elevation",
        type=float,
        default=0.95,
        help="bounds fit: Y-side view direction (0,-1,elev); normalized; larger = more top-down on the ground.",
    )
    parser.add_argument(
        "--no-frustum-auto-scale",
        action="store_true",
        help="Keep fixed --frustum-width/height/depth regardless of path size.",
    )
    parser.add_argument(
        "--frustum-scale-mode",
        choices=("combined", "aabb", "radius", "step"),
        default="combined",
        help=(
            "Characteristic path length for frustum sizing: combined=max(radius, step, AABB); "
            "aabb=max edge; radius=2*bounding-sphere radius; step≈18*median inter-frame step."
        ),
    )
    parser.add_argument(
        "--frustum-scale-ref-extent",
        type=float,
        default=2.0,
        help="Divide characteristic length by this (smaller value ⇒ larger frustums). Try 0.35–0.55.",
    )
    parser.add_argument(
        "--frustum-scale-min",
        type=float,
        default=0.5,
        help="Clamp multiplier from below (1.0 = never smaller than base frustum dimensions).",
    )
    parser.add_argument(
        "--frustum-scale-max",
        type=float,
        default=12.0,
        help="Clamp frustum size multiplier from above.",
    )
    parser.add_argument(
        "--no-frustum-camera-aware",
        action="store_true",
        help="Do not grow frustums when the bounds-fit camera moves farther away.",
    )
    parser.add_argument(
        "--frustum-ref-camera-distance",
        type=float,
        default=1.75,
        help="Camera-aware frustum: multiply by (fit camera distance / this), clamped.",
    )
    parser.add_argument(
        "--frustum-camera-aware-min",
        type=float,
        default=0.95,
        help="Lower clamp for the camera-distance ratio.",
    )
    parser.add_argument(
        "--frustum-camera-aware-max",
        type=float,
        default=8.0,
        help="Upper clamp for the camera-distance ratio.",
    )
    parser.add_argument(
        "--uniform-frustum-across-batch",
        action="store_true",
        help="Batch only: use the max frustum multiplier over all jobs so size is coherent across samples.",
    )
    parser.add_argument(
        "--extras-with-frustums",
        action="store_true",
        help="Scale wire thickness, path bevel, arrows, anchors with frustum auto-scale (off by default).",
    )
    parser.add_argument("--frustum-depth", type=float, default=0.055, help="Frustum pyramid depth.")
    parser.add_argument("--frustum-width", type=float, default=0.065, help="Frustum base width.")
    parser.add_argument("--frustum-height", type=float, default=0.04, help="Frustum base height.")
    parser.add_argument("--wire-thickness", type=float, default=0.1, help="Frustum wireframe thickness.")
    parser.add_argument("--path-thickness", type=float, default=0.0025, help="Trajectory curve bevel depth.")
    parser.add_argument("--arrow-length", type=float, default=0.045, help="Forward arrow length.")
    parser.add_argument(
        "--ground-size",
        type=float,
        default=None,
        help="Ground plane side length; omit for auto from trajectory XY extent.",
    )
    parser.add_argument(
        "--ground-margin",
        type=float,
        default=0.18,
        help="When ground size is auto, expand max XY span by this fraction.",
    )
    parser.add_argument("--grid-step", type=float, default=0.5, help="Ground grid spacing.")
    parser.add_argument("--show-ground-grid", action="store_true", help="Draw a wire grid on the ground plane.")
    parser.add_argument(
        "--ground-z",
        type=float,
        default=None,
        help="Explicit Blender-world Z height for the ground plane. Defaults below the trajectory.",
    )
    parser.add_argument(
        "--ground-offset",
        type=float,
        default=0.15,
        help="When --ground-z is unset, place ground this far below min trajectory Z (union min when --ground-coherence sample).",
    )
    parser.add_argument(
        "--ground-coherence",
        choices=("scene", "sample"),
        default="sample",
        help=(
            "scene: ground height and auto size follow only the drawn trajectory. "
            "sample: with --world-origin reference + benchmark id, use one floor for that id from the union "
            "of all baselines (after the same display normalization), fixing mismatched slabs between variants. "
            "Not used with --normalize-display per_scene (each scene has its own scale)."
        ),
    )
    parser.add_argument("--anchor-radius", type=float, default=0.01, help="Start/end anchor sphere radius.")
    parser.add_argument("--render-width", type=int, default=2560, help="Render output width in pixels.")
    parser.add_argument("--render-height", type=int, default=1440, help="Render output height in pixels.")
    parser.add_argument("--render-samples", type=int, default=128, help="Eevee/Cycles render anti-aliasing samples.")
    parser.add_argument("--show-velocity", action="store_true", help="Draw velocity vectors.")
    parser.add_argument(
        "--velocity-scale",
        type=float,
        default=1.0,
        help="Multiplier for finite-difference velocity indicators.",
    )
    parser.add_argument(
        "--stage-ids",
        type=str,
        default=None,
        help="Comma-separated stage ids, one per frame or one per sampled frustum.",
    )
    parser.add_argument(
        "--stage-id-file",
        type=Path,
        default=None,
        help="JSON list or text file containing one stage id per line/frame.",
    )
    parser.add_argument(
        "--stage-colors",
        type=str,
        default=None,
        help='JSON mapping, e.g. \'{"dolly":[0,1,0,1],"pan":[0,0.2,1,1]}\'.',
    )
    parser.add_argument(
        "--color-by-stage",
        action="store_true",
        help="Color frustum wires by motion stage (requires --stage-ids or --stage-id-file).",
    )
    parser.add_argument(
        "--trajectory-color",
        type=str,
        default="[1.0, 1.0, 1.0, 1.0]",
        help="Trajectory NURBS color as JSON RGB or RGBA (default white).",
    )
    parser.add_argument(
        "--gradient-start",
        type=str,
        default="[0.15, 0.95, 0.85, 1.0]",
        help="Start color (sampled frustums) as JSON RGBA.",
    )
    parser.add_argument(
        "--gradient-end",
        type=str,
        default="[0.98, 0.25, 0.55, 1.0]",
        help="End color (sampled frustums) as JSON RGBA.",
    )
    parser.add_argument(
        "--axis-triad-offset",
        type=float,
        default=0.0,
        help=(
            "Extra clearance in -X beyond automatic placement (past the ground's west edge + X-axis length). "
            "Larger values move the triad further left."
        ),
    )
    parser.add_argument(
        "--axis-triad-length",
        type=float,
        default=0.2,
        help="Length of each fluorescent axis arrow.",
    )
    parser.add_argument(
        "--axis-emission",
        type=float,
        default=4.0,
        help="Emission strength for axis materials (fluorescent look).",
    )
    parser.add_argument("--output-blend", type=Path, default=None, help="Optional path to save the .blend file.")
    parser.add_argument("--render-image", type=Path, default=None, help="Optional image path for rendering one scene.")
    parser.add_argument("--render-dir", type=Path, default=None, help="Output folder for batch rendered images.")
    parser.add_argument(
        "--batch-all",
        action="store_true",
        help="Batch: union of sample ids that have a .npy in any selected --variant folder.",
    )
    parser.add_argument("--ids", type=str, default=None, help="Comma-separated sample ids for batch rendering.")
    parser.add_argument("--ids-file", type=Path, default=None, help="Text file containing one sample id per line.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of batch scenes.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip batch images that already exist.")
    parser.add_argument(
        "--image-format",
        choices=("PNG", "JPEG"),
        default="PNG",
        help="Still render image format.",
    )
    ns = parser.parse_args(script_args)
    try:
        ns.variant_keys = resolve_variant_keys(ns.variant)
    except ValueError as exc:
        parser.error(str(exc))
    return ns


def trajectory_dir_for_variant(benchmark_dir: Path, variant_key: str) -> Path:
    sub = VARIANT_SUBDIR.get(variant_key)
    if sub is None:
        raise ValueError(
            f"Unknown variant key: {variant_key!r}. Known: {', '.join(sorted(VARIANT_SUBDIR.keys()))}"
        )
    return benchmark_dir / sub


def trajectory_path_for_sample(benchmark_dir: Path, variant_key: str, sample_id: str) -> Path:
    traj_dir = trajectory_dir_for_variant(benchmark_dir, variant_key)
    sample_id = str(sample_id)
    direct = traj_dir / f"{sample_id}.npy"
    if direct.is_file():
        return direct
    if sample_id.isdigit():
        for width in (6, 5, 4):
            padded = traj_dir / f"{int(sample_id):0{width}d}.npy"
            if padded.is_file():
                return padded
    return direct


def output_stem_for_sample(sample_id: str, traj_path: Path) -> str:
    return traj_path.stem if traj_path.is_file() else str(sample_id)


def paired_mid_index(n_frames: int, n_other: int) -> int:
    return min(n_frames, n_other) // 2


def load_blender_positions(path: Path, layout: str, scale: float) -> np.ndarray:
    positions, _ = load_trajectory(path, layout)
    return (opengl_to_blender_positions(positions) * float(scale)).astype(np.float32)


def collect_fit_bounds_positions(
    args: argparse.Namespace,
    benchmark_dir: Path,
    sample_id: str,
    current_traj_path: Path,
    reference_anchor: np.ndarray,
) -> np.ndarray:
    if args.fit_bounds == "scene":
        return np.empty((0, 3), dtype=np.float32)
    variant_keys = ["reference"] if args.fit_bounds == "reference" else benchmark_variant_keys_all()
    chunks: List[np.ndarray] = []
    bench = benchmark_dir.expanduser().resolve()
    anchor = np.asarray(reference_anchor, dtype=np.float64)
    for vk in variant_keys:
        pth = trajectory_path_for_sample(bench, vk, sample_id)
        if not pth.is_file() or pth.resolve() == current_traj_path.resolve():
            continue
        try:
            raw = load_blender_positions(pth, args.layout, args.scale).astype(np.float64)
            raw -= anchor
        except Exception as exc:
            print(f"Warning: skipped --fit-bounds from {pth}: {exc}")
            continue
        chunks.append(raw.astype(np.float32))
    if not chunks:
        return np.empty((0, 3), dtype=np.float32)
    return np.vstack(chunks)


def max_aabb_extent(points: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    return float(np.max(points.max(axis=0) - points.min(axis=0)))


def union_all_variant_positions_for_sample(
    args: argparse.Namespace,
    benchmark_dir: Path,
    sample_id: str,
    reference_anchor: np.ndarray,
) -> np.ndarray:
    """All registered benchmark variants for this id, rigidly aligned (reference mid-frame subtracted)."""

    bench = benchmark_dir.expanduser().resolve()
    anchor = np.asarray(reference_anchor, dtype=np.float64)
    chunks: List[np.ndarray] = []
    for vk in benchmark_variant_keys_all():
        pth = trajectory_path_for_sample(bench, vk, sample_id)
        if not pth.is_file():
            continue
        try:
            raw = load_blender_positions(pth, args.layout, args.scale).astype(np.float64)
            raw -= anchor
        except Exception as exc:
            print(f"Warning: skipped variant {vk} in normalization union ({pth}): {exc}")
            continue
        chunks.append(raw.astype(np.float32))
    if not chunks:
        raise ValueError(
            f"No trajectory .npy found for sample_id={sample_id!r} under any registered benchmark variant."
        )
    return np.vstack(chunks)


def resolve_display_normalization_scale(
    args: argparse.Namespace,
    *,
    norm_mode: str,
    bounds_stack: np.ndarray,
    benchmark_dir: Optional[Path],
    sample_id: Optional[str],
    reference_anchor: Optional[np.ndarray],
) -> float:
    target = float(args.normalize_target_extent)
    floor = float(args.normalize_min_extent)
    if norm_mode == "none":
        return 1.0
    if norm_mode == "per_scene":
        ext = max_aabb_extent(bounds_stack)
        return target / max(ext, floor)
    if norm_mode != "per_sample_union":
        raise ValueError(norm_mode)
    if reference_anchor is None or benchmark_dir is None or sample_id is None:
        raise ValueError(
            "--normalize-display per_sample_union requires --world-origin reference "
            "and benchmark context (--sample-id or batch mode)."
        )
    if args.world_origin != "reference":
        raise ValueError("--normalize-display per_sample_union requires --world-origin reference.")
    if not hasattr(args, "_normalize_scale_cache"):
        setattr(args, "_normalize_scale_cache", {})
    cache: Dict[str, float] = getattr(args, "_normalize_scale_cache")
    key = str(sample_id)
    if key in cache:
        return cache[key]
    union_pts = union_all_variant_positions_for_sample(
        args, benchmark_dir.expanduser().resolve(), key, reference_anchor
    )
    ext = max_aabb_extent(union_pts)
    s = target / max(ext, floor)
    cache[key] = s
    return s


def ground_coherence_uses_sample_union(
    args: argparse.Namespace,
    benchmark_dir: Optional[Path],
    sample_id: Optional[str],
    reference_anchor: Optional[np.ndarray],
) -> bool:
    if str(args.ground_coherence) != "sample":
        return False
    if str(args.normalize_display) == "per_scene":
        return False
    if benchmark_dir is None or sample_id is None or reference_anchor is None:
        return False
    return True


def union_all_variants_scaled(
    args: argparse.Namespace,
    benchmark_dir: Path,
    sample_id: str,
    reference_anchor: np.ndarray,
    scale_u: float,
) -> np.ndarray:
    u = union_all_variant_positions_for_sample(args, benchmark_dir, sample_id, reference_anchor)
    if abs(float(scale_u) - 1.0) > 1e-8:
        u = (u.astype(np.float64) * float(scale_u)).astype(np.float32)
    return u


def resolve_ground_plane(
    args: argparse.Namespace,
    positions: np.ndarray,
    benchmark_dir: Optional[Path],
    sample_id: Optional[str],
    reference_anchor: Optional[np.ndarray],
    scale_u: float,
) -> Tuple[float, float]:
    use_union = ground_coherence_uses_sample_union(args, benchmark_dir, sample_id, reference_anchor)
    cache_key: Optional[str] = None
    cache: Optional[Dict[str, Tuple[float, float]]] = None
    union_scaled: Optional[np.ndarray] = None

    if use_union:
        if not hasattr(args, "_ground_plane_cache"):
            setattr(args, "_ground_plane_cache", {})
        cache = getattr(args, "_ground_plane_cache")
        cache_key = str(sample_id)
        if cache_key in cache:
            ground_size, ground_z = cache[cache_key]
            if args.ground_size is not None:
                ground_size = float(args.ground_size)
            if args.ground_z is not None:
                ground_z = float(args.ground_z)
            return ground_size, ground_z
        union_scaled = union_all_variants_scaled(
            args, benchmark_dir.expanduser().resolve(), cache_key, reference_anchor, scale_u
        )

    if args.ground_size is not None:
        ground_size = float(args.ground_size)
    else:
        ref_pts = union_scaled if union_scaled is not None else positions
        ground_size = adaptive_ground_side_length(ref_pts, args.ground_margin)

    if args.ground_z is not None:
        ground_z = float(args.ground_z)
    else:
        ref_pts = union_scaled if union_scaled is not None else positions
        ground_z = float(np.min(ref_pts[:, 2]) - float(args.ground_offset))

    if use_union and cache is not None and cache_key is not None:
        cache[cache_key] = (ground_size, ground_z)

    return ground_size, ground_z


def scene_points_for_camera(
    positions: np.ndarray,
    ground_z: float,
    ground_half_extent: float,
) -> np.ndarray:
    gh = float(ground_half_extent)
    corners = np.array(
        [[x, y, float(ground_z)] for x in (-gh, gh) for y in (-gh, gh)],
        dtype=np.float64,
    )
    pts = np.vstack([positions.astype(np.float64), corners])
    return pts


def bounds_center_and_radius(points: np.ndarray) -> Tuple[np.ndarray, float]:
    c = 0.5 * (points.min(axis=0) + points.max(axis=0))
    r = float(np.max(np.linalg.norm(points - c, axis=1)))
    return c, max(r, 1e-4)


def perspective_tan_half_fovs(lens_mm: float, res_x: int, res_y: int) -> Tuple[float, float]:
    sensor_w = 36.0
    focal = max(float(lens_mm), 1e-3)
    hfov = 2.0 * math.atan(0.5 * sensor_w / focal)
    aspect = float(res_x) / float(max(res_y, 1))
    tan_half_h = math.tan(0.5 * hfov)
    tan_half_v = tan_half_h / aspect
    return tan_half_h, tan_half_v


def points_fit_camera(
    points: np.ndarray,
    look_center: mathutils.Vector,
    eye: mathutils.Vector,
    tan_half_h: float,
    tan_half_v: float,
    margin: float,
) -> bool:
    f_vec = look_center - eye
    z_view = float(f_vec.length)
    if z_view < 1e-8:
        return False
    f_vec_norm = f_vec / z_view
    up = mathutils.Vector((0.0, 0.0, 1.0))
    r_vec = f_vec_norm.cross(up)
    if r_vec.length < 1e-6:
        r_vec = f_vec_norm.cross(mathutils.Vector((0.0, 1.0, 0.0)))
    r_vec.normalize()
    t_vec = r_vec.cross(f_vec_norm).normalized()
    mt_h = tan_half_h * margin
    mt_v = tan_half_v * margin
    for p in points:
        w = mathutils.Vector((float(p[0]), float(p[1]), float(p[2]))) - eye
        z = w.dot(f_vec_norm)
        if z <= 1e-4:
            return False
        if abs(w.dot(r_vec)) > z * mt_h or abs(w.dot(t_vec)) > z * mt_v:
            return False
    return True


def min_camera_distance_for_bounds(
    scene_points: np.ndarray,
    look_center: np.ndarray,
    view_dir: mathutils.Vector,
    lens_mm: float,
    res_x: int,
    res_y: int,
    margin: float,
) -> float:
    tan_h, tan_v = perspective_tan_half_fovs(lens_mm, res_x, res_y)
    _, r = bounds_center_and_radius(scene_points)
    d_lo = r * 0.08
    d_hi = max(r * 80.0, 10.0)
    if d_lo >= d_hi:
        d_hi = d_lo * 2.0 + 1.0
    lc = mathutils.Vector((float(look_center[0]), float(look_center[1]), float(look_center[2])))
    for _ in range(56):
        d_mid = 0.5 * (d_lo + d_hi)
        eye = lc + view_dir * d_mid
        if points_fit_camera(scene_points, lc, eye, tan_h, tan_v, margin):
            d_hi = d_mid
        else:
            d_lo = d_mid
    return max(d_hi * 1.02, 0.25)


def frustum_characteristic_length(bounds_positions: np.ndarray, mode: str) -> float:
    if bounds_positions.size == 0:
        return 1e-4
    mins = bounds_positions.min(axis=0)
    maxs = bounds_positions.max(axis=0)
    max_edge = float(np.max(maxs - mins))
    _, r = bounds_center_and_radius(bounds_positions)
    n = len(bounds_positions)
    if n >= 2:
        steps = np.linalg.norm(np.diff(bounds_positions.astype(np.float64), axis=0), axis=1)
        med_step = float(np.median(steps))
    else:
        med_step = 0.0

    if mode == "aabb":
        return max(max_edge, 1e-6)
    if mode == "radius":
        return max(2.0 * r, 1e-6)
    if mode == "step":
        return max(18.0 * med_step, 1e-6)
    return max(
        2.45 * r,
        0.32 * max_edge,
        20.0 * med_step,
        1e-6,
    )


def compute_frustum_multiplier_from_geometry(
    bounds_stack: np.ndarray,
    cam_pts: np.ndarray,
    args: argparse.Namespace,
) -> float:
    if args.no_frustum_auto_scale:
        return 1.0
    char = frustum_characteristic_length(bounds_stack, str(args.frustum_scale_mode))
    ref = max(float(args.frustum_scale_ref_extent), 1e-4)
    m = char / ref
    if not args.no_frustum_camera_aware:
        elev = float(args.view_axis_elevation)
        view_dir = mathutils.Vector((0.0, -1.0, elev)).normalized()
        lc_arr, _ = bounds_center_and_radius(cam_pts)
        dist = min_camera_distance_for_bounds(
            cam_pts,
            lc_arr,
            view_dir,
            float(args.camera_lens),
            int(args.render_width),
            int(args.render_height),
            float(args.frame_margin),
        )
        ref_d = max(float(args.frustum_ref_camera_distance), 0.05)
        ratio = dist / ref_d
        ratio = max(float(args.frustum_camera_aware_min), min(float(args.frustum_camera_aware_max), ratio))
        m *= ratio
    return max(float(args.frustum_scale_min), min(float(args.frustum_scale_max), m))


def collect_batch_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.ids:
        ids.extend(part.strip() for part in args.ids.split(",") if part.strip())
    if args.ids_file:
        ids.extend(
            line.strip()
            for line in args.ids_file.expanduser().read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if args.batch_all:
        benchmark_dir = args.benchmark_dir.expanduser().resolve()
        stems: set[str] = set()
        for variant_key in args.variant_keys:
            traj_dir = trajectory_dir_for_variant(benchmark_dir, variant_key)
            if not traj_dir.is_dir():
                print(f"Warning: variant directory missing, skipped for --batch-all: {traj_dir}")
                continue
            stems.update(path.stem for path in traj_dir.glob("*.npy"))
        ids.extend(sorted(stems))

    seen = set()
    unique_ids = []
    for sample_id in ids:
        if sample_id in seen:
            continue
        seen.add(sample_id)
        unique_ids.append(sample_id)
    if args.limit is not None:
        unique_ids = unique_ids[: max(0, int(args.limit))]
    return unique_ids


def load_trajectory(path: Path, layout: str) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(np.load(path), dtype=np.float32)
    if arr.ndim == 1:
        if arr.size % 9 != 0:
            raise ValueError(f"Flat trajectory length must be divisible by 9, got {arr.shape}")
        arr = arr.reshape(-1, 9)
    if arr.ndim != 2 or arr.shape[1] != 9:
        raise ValueError(f"Expected trajectory shape (T,9), got {arr.shape} from {path}")

    resolved_layout = layout
    if layout == "auto":
        resolved_layout = infer_layout(arr)

    if resolved_layout == "clatr":
        rot6d = arr[:, :6]
        positions = arr[:, 6:9]
    elif resolved_layout == "xyz_rot6d":
        positions = arr[:, :3]
        rot6d = arr[:, 3:9]
    else:
        raise ValueError(f"Unsupported layout: {layout}")

    rotations = sixd_to_matrix(rot6d)
    return positions.astype(np.float32), rotations.astype(np.float32)


def infer_layout(arr: np.ndarray) -> str:
    clatr_err = rotation_score(arr[:, :6])
    xyz_err = rotation_score(arr[:, 3:9])
    return "clatr" if clatr_err <= xyz_err else "xyz_rot6d"


def rotation_score(rot6d: np.ndarray) -> float:
    r1 = rot6d[:, :3]
    r2 = rot6d[:, 3:6]
    n1 = np.linalg.norm(r1, axis=1)
    n2 = np.linalg.norm(r2, axis=1)
    dot = np.sum(r1 * r2, axis=1)
    return float(np.mean(np.abs(n1 - 1.0) + np.abs(n2 - 1.0) + np.abs(dot)))


def sixd_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Convert first-two-column 6D rotations into 3x3 matrices.

    The third column is reconstructed by the requested cross product:

        r3 = r1 x r2

    The output matrix columns are [r1, r2, r3], matching OpenGL camera-to-world
    convention. Blender can consume this as a mathutils.Matrix and convert it to
    a Quaternion with Matrix.to_quaternion().
    """
    rot6d = np.asarray(rot6d, dtype=np.float32)
    r1_raw = rot6d[..., :3]
    r2_raw = rot6d[..., 3:6]

    r1 = normalize_np(r1_raw)
    r2 = r2_raw - np.sum(r1 * r2_raw, axis=-1, keepdims=True) * r1
    r2 = normalize_np(r2)
    r3 = np.cross(r1, r2, axis=-1)
    return np.stack([r1, r2, r3], axis=-1).astype(np.float32)


def normalize_np(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def matrix_to_blender_quaternion(rot: np.ndarray) -> mathutils.Quaternion:
    mat = mathutils.Matrix(
        (
            (float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2])),
            (float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2])),
            (float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2])),
        )
    )
    return mat.to_quaternion()


def opengl_to_blender_positions(positions: np.ndarray) -> np.ndarray:
    """Map OpenGL world coordinates (+Y up) to Blender world coordinates (+Z up)."""
    x = positions[..., 0]
    y = positions[..., 1]
    z = positions[..., 2]
    return np.stack([x, -z, y], axis=-1).astype(np.float32)


def opengl_to_blender_rotations(rotations: np.ndarray) -> np.ndarray:
    """Map camera-to-world rotation columns from OpenGL world into Blender world.

    The camera local frame remains OpenGL/Blender camera style: +X right,
    +Y up, and -Z forward. Only the world basis changes, so each rotation
    column is transformed with the same vector mapping used for positions.
    """
    basis = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    return np.einsum("ij,tjk->tik", basis, rotations).astype(np.float32)


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def make_material(
    name: str,
    rgba: Tuple[float, float, float, float],
    *,
    alpha_blend: bool = False,
    roughness: float = 0.6,
) -> bpy.types.Material:
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Alpha"].default_value = rgba[3]
        bsdf.inputs["Roughness"].default_value = roughness
    mat.diffuse_color = rgba
    if alpha_blend or rgba[3] < 1.0:
        mat.blend_method = "BLEND"
        mat.use_screen_refraction = True
        mat.show_transparent_back = True
    return mat


def make_emissive_material(
    name: str,
    rgb: Tuple[float, float, float],
    *,
    strength: float = 4.0,
) -> bpy.types.Material:
    """Bright emissive material for fluorescent-style axes."""
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for node in list(nt.nodes):
        nt.nodes.remove(node)
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)
    emit.inputs["Strength"].default_value = float(strength)
    out.location = (280.0, 0.0)
    emit.location = (0.0, 0.0)
    nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    mat.diffuse_color = (*rgb, 1.0)
    return mat


def make_stage_materials(stage_colors: Dict[str, Tuple[float, float, float, float]]) -> Dict[str, bpy.types.Material]:
    return {
        str(stage).lower(): make_material(f"Stage_{stage}", tuple(color))
        for stage, color in stage_colors.items()
    }


def create_ground(size: float, step: float, z_height: float, show_grid: bool) -> None:
    ground_mat = make_material("GroundPlane", VIS_GROUND_PLANE_RGBA)
    if hasattr(ground_mat, "use_backface_culling"):
        ground_mat.use_backface_culling = False

    bpy.ops.mesh.primitive_plane_add(size=size, location=(0.0, 0.0, z_height))
    ground = bpy.context.object
    ground.name = "Ground_Lowered"
    ground.data.materials.append(ground_mat)
    if not show_grid:
        return

    grid_mat = make_material("GroundGrid", srgb8_to_linear_rgba(140, 140, 140))

    curve = bpy.data.curves.new("GroundGridCurve", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 1
    curve.bevel_depth = max(size * 0.00045, 0.002)
    curve.bevel_resolution = 0

    half = size * 0.5
    count = int(math.floor(size / step))
    values = np.linspace(-half, half, count + 1)
    z = z_height + 0.003
    for v in values:
        add_polyline(curve, [(-half, float(v), z), (half, float(v), z)])
        add_polyline(curve, [(float(v), -half, z), (float(v), half, z)])

    obj = bpy.data.objects.new("Ground_Wire_Grid", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(grid_mat)


def create_frustum_template(face_mat: bpy.types.Material) -> bpy.types.Object:
    # Apex at camera center. The base lies along local -Z, the OpenGL forward direction.
    verts = [
        (0.0, 0.0, 0.0),
        (-0.5, -0.5, -1.0),
        (0.5, -0.5, -1.0),
        (0.5, 0.5, -1.0),
        (-0.5, 0.5, -1.0),
    ]
    faces = [(0, 1, 2), (0, 2, 3), (0, 3, 4), (0, 4, 1), (1, 4, 3, 2)]
    mesh = bpy.data.meshes.new("CameraFrustumTemplateMesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    mesh.materials.append(face_mat)

    obj = bpy.data.objects.new("Camera_Frustum_Template", mesh)
    bpy.context.collection.objects.link(obj)
    obj.hide_viewport = True
    obj.hide_render = True
    return obj


def create_frustum_instance(
    template: bpy.types.Object,
    name: str,
    position: np.ndarray,
    rotation: np.ndarray,
    color_mat: bpy.types.Material,
    args: argparse.Namespace,
    *,
    frustum_xyz: Optional[Tuple[float, float, float]] = None,
    wire_thickness: Optional[float] = None,
) -> bpy.types.Object:
    obj = template.copy()
    obj.data = template.data.copy()
    obj.name = name
    obj.hide_viewport = False
    obj.hide_render = False
    bpy.context.collection.objects.link(obj)

    obj.data.materials.clear()
    obj.data.materials.append(template.data.materials[0])
    obj.data.materials.append(color_mat)

    obj.location = mathutils.Vector(tuple(float(x) for x in position))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = matrix_to_blender_quaternion(rotation)
    if frustum_xyz is None:
        frustum_xyz = (float(args.frustum_width), float(args.frustum_height), float(args.frustum_depth))
    obj.scale = frustum_xyz

    wire = obj.modifiers.new("Solid_Visible_Wire_Borders", type="WIREFRAME")
    wire.thickness = float(args.wire_thickness if wire_thickness is None else wire_thickness)
    wire.use_even_offset = True
    wire.use_replace = False
    wire.material_offset = 1
    return obj


def create_arrow(
    name: str,
    start: np.ndarray,
    direction: np.ndarray,
    length: float,
    radius: float,
    material: bpy.types.Material,
) -> None:
    direction_vec = mathutils.Vector(tuple(float(x) for x in direction))
    if direction_vec.length < 1e-8:
        return
    direction_vec.normalize()
    start_vec = mathutils.Vector(tuple(float(x) for x in start))
    shaft_len = length * 0.72
    cone_len = length * 0.28
    shaft_mid = start_vec + direction_vec * (shaft_len * 0.5)
    cone_mid = start_vec + direction_vec * (shaft_len + cone_len * 0.5)
    quat = direction_vec.to_track_quat("Z", "Y")

    bpy.ops.mesh.primitive_cylinder_add(vertices=16, radius=radius, depth=shaft_len, location=shaft_mid)
    shaft = bpy.context.object
    shaft.name = f"{name}_shaft"
    shaft.rotation_euler = quat.to_euler()
    shaft.data.materials.append(material)

    bpy.ops.mesh.primitive_cone_add(vertices=20, radius1=radius * 2.5, radius2=0.0, depth=cone_len, location=cone_mid)
    cone = bpy.context.object
    cone.name = f"{name}_head"
    cone.rotation_euler = quat.to_euler()
    cone.data.materials.append(material)


def create_velocity_line(
    name: str,
    start: np.ndarray,
    velocity: np.ndarray,
    scale: float,
    material: bpy.types.Material,
) -> None:
    vel = np.asarray(velocity, dtype=np.float32) * float(scale)
    if float(np.linalg.norm(vel)) < 1e-8:
        return
    curve = bpy.data.curves.new(name, type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.0025
    curve.bevel_resolution = 1
    add_polyline(curve, [tuple(start), tuple(start + vel)])
    obj = bpy.data.objects.new(name, curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)


def create_trajectory_path(positions: np.ndarray, bevel_depth: float, material: bpy.types.Material) -> None:
    curve = bpy.data.curves.new("Trajectory_NURBS_Path", type="CURVE")
    curve.dimensions = "3D"
    curve.resolution_u = 16
    curve.bevel_depth = bevel_depth
    curve.bevel_resolution = 3

    spline = curve.splines.new(type="NURBS")
    spline.points.add(len(positions) - 1)
    for point, pos in zip(spline.points, positions):
        point.co = (float(pos[0]), float(pos[1]), float(pos[2]), 1.0)
    spline.order_u = min(4, len(positions))
    spline.use_endpoint_u = True

    obj = bpy.data.objects.new("Trajectory_Path", curve)
    bpy.context.collection.objects.link(obj)
    obj.data.materials.append(material)


def create_anchor(
    name: str,
    position: np.ndarray,
    radius: float,
    material: bpy.types.Material,
) -> None:
    loc = tuple(float(x) for x in position)
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=loc)
    sphere = bpy.context.object
    sphere.name = name
    sphere.data.materials.append(material)


def create_axis_triad(
    origin: Tuple[float, float, float],
    length: float,
    shaft_radius: float,
    emission_strength: float,
) -> None:
    """Small RGB axes with +X/+Y/+Z arms from origin; place origin west of the ground to avoid the path."""

    def add_axis(name_prefix: str, mat: bpy.types.Material, direction: mathutils.Vector) -> None:
        start = mathutils.Vector(origin)
        d = direction.normalized()
        shaft_mid = start + d * (length * 0.5)
        quat = d.to_track_quat("Z", "Y")

        bpy.ops.mesh.primitive_cylinder_add(
            vertices=8, radius=shaft_radius, depth=length, location=shaft_mid.to_tuple()
        )
        cyl = bpy.context.object
        cyl.name = f"{name_prefix}_shaft"
        cyl.rotation_euler = quat.to_euler()
        cyl.data.materials.append(mat)

    mat_x = make_emissive_material("Axis_Fluor_X", (1.0, 0.15, 0.72), strength=emission_strength)
    mat_y = make_emissive_material("Axis_Fluor_Y", (0.2, 1.0, 0.35), strength=emission_strength)
    mat_z = make_emissive_material("Axis_Fluor_Z", (0.25, 0.92, 1.0), strength=emission_strength)

    add_axis("WorldAxis_X", mat_x, mathutils.Vector((1.0, 0.0, 0.0)))
    add_axis("WorldAxis_Y", mat_y, mathutils.Vector((0.0, 1.0, 0.0)))
    add_axis("WorldAxis_Z", mat_z, mathutils.Vector((0.0, 0.0, 1.0)))


def add_polyline(curve: bpy.types.Curve, points: Sequence[Tuple[float, float, float]]) -> None:
    spline = curve.splines.new(type="POLY")
    spline.points.add(len(points) - 1)
    for point, co in zip(spline.points, points):
        point.co = (float(co[0]), float(co[1]), float(co[2]), 1.0)


def load_stage_ids(args: argparse.Namespace, n_frames: int, n_samples: int) -> Optional[List[str]]:
    values: Optional[List[str]] = None
    if args.stage_ids:
        values = [part.strip() for part in args.stage_ids.split(",") if part.strip()]
    elif args.stage_id_file:
        text = args.stage_id_file.expanduser().read_text(encoding="utf-8").strip()
        if text.startswith("["):
            values = [str(v) for v in json.loads(text)]
        else:
            values = [line.strip() for line in text.splitlines() if line.strip()]

    if not values:
        return None
    if len(values) == n_frames:
        return values
    if len(values) == n_samples:
        return values

    # Resample stage ids to frame count so short annotations can still be useful.
    idx = np.linspace(0, len(values) - 1, n_frames).round().astype(int)
    return [values[int(i)] for i in idx]


def parse_rgba_json(raw: str, name: str) -> Tuple[float, float, float, float]:
    value = json.loads(raw)
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array, got {type(value)}")
    rgba = list(value)
    if len(rgba) == 3:
        rgba.append(1.0)
    if len(rgba) != 4:
        raise ValueError(f"{name} must have 3 or 4 channels")
    return tuple(float(x) for x in rgba)


def lerp_rgba(
    a: Tuple[float, float, float, float],
    b: Tuple[float, float, float, float],
    t: float,
) -> Tuple[float, float, float, float]:
    t = min(1.0, max(0.0, t))
    return (
        float(a[0] + (b[0] - a[0]) * t),
        float(a[1] + (b[1] - a[1]) * t),
        float(a[2] + (b[2] - a[2]) * t),
        float(a[3] + (b[3] - a[3]) * t),
    )


def adaptive_ground_side_length(positions: np.ndarray, margin: float) -> float:
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    span_xy = float(max(maxs[0] - mins[0], maxs[1] - mins[1]))
    return max(1.0, span_xy * (1.0 + float(margin)))


def parse_stage_colors(raw: Optional[str]) -> Dict[str, Tuple[float, float, float, float]]:
    colors = dict(DEFAULT_STAGE_COLORS)
    if not raw:
        return colors
    overrides = json.loads(raw)
    for key, value in overrides.items():
        rgba = list(value)
        if len(rgba) == 3:
            rgba.append(1.0)
        if len(rgba) != 4:
            raise ValueError(f"Stage color for {key!r} must have 3 or 4 channels")
        colors[str(key).lower()] = tuple(float(x) for x in rgba)
    return colors


def stage_for_sample(stage_ids: Optional[List[str]], frame_idx: int, sample_ordinal: int, n_frames: int) -> str:
    if not stage_ids:
        return "default"
    if len(stage_ids) == n_frames:
        return str(stage_ids[frame_idx]).lower()
    return str(stage_ids[min(sample_ordinal, len(stage_ids) - 1)]).lower()


def configure_world_background() -> None:
    """World background sRGB (209, 209, 209) as linear color; strength capped for balance with the sun."""
    scene = bpy.context.scene
    world = scene.world
    if world is None:
        world = bpy.data.worlds.new("VisWorld")
        scene.world = world
    world.use_nodes = True
    nt = world.node_tree
    if nt is None:
        return
    bg = nt.nodes.get("Background")
    if bg is None:
        bg = nt.nodes.new("ShaderNodeBackground")
        out = nt.nodes.get("World Output") or nt.nodes.new("ShaderNodeOutputWorld")
        nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    bg.inputs["Color"].default_value = VIS_BACKGROUND_RGBA
    bg.inputs["Strength"].default_value = VIS_WORLD_BACKGROUND_STRENGTH


def setup_scene_camera(
    args: argparse.Namespace,
    *,
    positions: np.ndarray,
    ground_z: float,
    camera_fit_points: np.ndarray,
) -> None:
    configure_world_background()
    if hasattr(bpy.context.scene.view_settings, "exposure"):
        bpy.context.scene.view_settings.exposure = 0.0

    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    extent = float(np.max(maxs - mins))
    span = float(np.max(np.linalg.norm(positions, axis=1)))
    sun_distance = max(1.35, extent * 1.35, span * 1.75, 3.0)

    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, sun_distance * 1.4))
    sun = bpy.context.object
    sun.name = "Sun_Key"
    sun.data.energy = 3.0
    sun.rotation_euler = (0.0, 0.0, 0.0)

    if args.camera_fit == "legacy":
        look_z = 0.5 * float(ground_z)
        look_at = mathutils.Vector((0.0, 0.0, look_z))
        distance = max(1.35, extent * 1.35, span * 1.75)
        camera_height = max(0.25, distance * 0.50)
        bpy.ops.object.camera_add(location=(0.0, -distance, camera_height))
        camera = bpy.context.object
        direction = look_at - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    else:
        elev = float(args.view_axis_elevation)
        view_dir = mathutils.Vector((0.0, -1.0, elev)).normalized()
        lc, _ = bounds_center_and_radius(camera_fit_points)
        dist = min_camera_distance_for_bounds(
            camera_fit_points,
            lc,
            view_dir,
            float(args.camera_lens),
            int(args.render_width),
            int(args.render_height),
            float(args.frame_margin),
        )
        eye = mathutils.Vector((float(lc[0]), float(lc[1]), float(lc[2]))) + view_dir * dist
        bpy.ops.object.camera_add(location=(float(eye.x), float(eye.y), float(eye.z)))
        camera = bpy.context.object
        look_at = mathutils.Vector((float(lc[0]), float(lc[1]), float(lc[2])))
        direction = look_at - camera.location
        camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()

    camera.data.lens = float(args.camera_lens)
    bpy.context.scene.camera = camera

    bpy.context.scene.render.engine = "BLENDER_EEVEE_NEXT" if bpy.app.version >= (4, 2, 0) else "BLENDER_EEVEE"
    bpy.context.scene.render.resolution_x = int(args.render_width)
    bpy.context.scene.render.resolution_y = int(args.render_height)
    bpy.context.scene.render.resolution_percentage = 100
    if hasattr(bpy.context.scene, "eevee"):
        samples = int(args.render_samples)
        if hasattr(bpy.context.scene.eevee, "taa_render_samples"):
            bpy.context.scene.eevee.taa_render_samples = samples
        if hasattr(bpy.context.scene.eevee, "taa_samples"):
            bpy.context.scene.eevee.taa_samples = samples
    if hasattr(bpy.context.scene, "cycles"):
        bpy.context.scene.cycles.samples = int(args.render_samples)
    bpy.context.scene.view_settings.view_transform = "Filmic"
    bpy.context.scene.view_settings.look = "Medium High Contrast"


def prepare_scene_kinematics(
    args: argparse.Namespace,
    traj_path: Path,
    *,
    benchmark_dir: Optional[Path],
    sample_id: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, float, float, np.ndarray]:
    """Load trajectory, apply world origin, union fit bounds, derive ground and camera point cloud."""

    traj_path = traj_path.expanduser().resolve()
    positions, rotations = load_trajectory(traj_path, args.layout)
    positions = opengl_to_blender_positions(positions) * float(args.scale)
    rotations = opengl_to_blender_rotations(rotations)
    n_frames = len(positions)
    mid_idx = n_frames // 2

    reference_anchor: Optional[np.ndarray] = None
    if args.world_origin == "reference":
        if benchmark_dir is None or sample_id is None:
            raise ValueError(
                "--world-origin reference needs --sample-id with --benchmark-dir (or use batch rendering)."
            )
        ref_path = trajectory_path_for_sample(benchmark_dir.expanduser().resolve(), "reference", str(sample_id))
        if not ref_path.is_file():
            raise FileNotFoundError(f"Reference trajectory not found: {ref_path}")
        ref_pos = load_blender_positions(ref_path, args.layout, args.scale)
        pair_mid = paired_mid_index(n_frames, len(ref_pos))
        reference_anchor = np.asarray(ref_pos[pair_mid], dtype=np.float64)
        positions = (positions.astype(np.float64) - reference_anchor).astype(np.float32)
    elif args.world_origin == "bbox":
        c = 0.5 * (positions.min(axis=0) + positions.max(axis=0))
        positions = positions - c
    else:
        positions = positions - positions[mid_idx]

    if args.fit_bounds != "scene" and args.world_origin != "reference":
        raise ValueError("--fit-bounds reference or all requires --world-origin reference.")

    fit_extra = np.empty((0, 3), dtype=np.float32)
    if (
        args.fit_bounds != "scene"
        and benchmark_dir is not None
        and sample_id is not None
        and reference_anchor is not None
    ):
        fit_extra = collect_fit_bounds_positions(
            args,
            benchmark_dir.expanduser().resolve(),
            str(sample_id),
            traj_path,
            reference_anchor,
        )

    bounds_stack = np.vstack([positions, fit_extra]) if fit_extra.size else positions

    scale_u = 1.0
    norm_mode = str(args.normalize_display)
    if norm_mode != "none":
        scale_u = resolve_display_normalization_scale(
            args,
            norm_mode=norm_mode,
            bounds_stack=bounds_stack,
            benchmark_dir=benchmark_dir,
            sample_id=sample_id,
            reference_anchor=reference_anchor,
        )
        positions = (positions.astype(np.float64) * scale_u).astype(np.float32)
        if fit_extra.size:
            fit_extra = (fit_extra.astype(np.float64) * scale_u).astype(np.float32)
        bounds_stack = np.vstack([positions, fit_extra]) if fit_extra.size else positions
        print(f"Display normalize ({norm_mode}): scale={scale_u:.6f}, target extent={args.normalize_target_extent}")

    ground_size, ground_z = resolve_ground_plane(
        args,
        positions,
        benchmark_dir,
        sample_id,
        reference_anchor,
        scale_u,
    )
    ground_half = 0.5 * float(ground_size)
    fit_cloud = bounds_stack if args.fit_bounds != "scene" else positions
    cam_pts = scene_points_for_camera(fit_cloud, ground_z, ground_half)

    return positions, rotations, bounds_stack, fit_cloud, n_frames, mid_idx, ground_size, ground_z, cam_pts


def build_scene(
    args: argparse.Namespace,
    traj_path: Path,
    *,
    benchmark_dir: Optional[Path] = None,
    sample_id: Optional[str] = None,
) -> Tuple[int, int]:
    positions, rotations, bounds_stack, _, n_frames, _mid_idx, ground_size, ground_z, cam_pts = (
        prepare_scene_kinematics(args, traj_path, benchmark_dir=benchmark_dir, sample_id=sample_id)
    )

    normalized = str(args.normalize_display) != "none"
    if args.no_frustum_auto_scale:
        frustum_m = 1.0
        ufix = getattr(args, "_uniform_frustum_m_global", None)
        if ufix is not None:
            frustum_m = float(ufix)
    elif normalized and not args.normalize_keep_frustum_auto:
        frustum_m = float(args.normalize_frustum_mul)
    else:
        frustum_m = compute_frustum_multiplier_from_geometry(bounds_stack, cam_pts, args)
        ufix = getattr(args, "_uniform_frustum_m_global", None)
        if ufix is not None:
            frustum_m = float(ufix)

    extras_m = 1.0
    if not args.no_frustum_auto_scale and args.extras_with_frustums:
        extras_m = max(0.55, min(3.5, math.sqrt(frustum_m)))

    fw = float(args.frustum_width) * frustum_m
    fh = float(args.frustum_height) * frustum_m
    fd = float(args.frustum_depth) * frustum_m
    wire_t = float(args.wire_thickness)
    if not args.no_frustum_auto_scale:
        wire_t *= max(0.7, min(2.5, math.sqrt(frustum_m)))
    wire_t *= extras_m if args.extras_with_frustums else 1.0
    path_t = float(args.path_thickness) * extras_m
    arr_len = float(args.arrow_length) * extras_m
    anch_r = float(args.anchor_radius) * extras_m

    sampled_indices = list(range(0, n_frames, args.sample_every))
    if sampled_indices[-1] != n_frames - 1:
        sampled_indices.append(n_frames - 1)

    clean_scene()

    traj_rgba = parse_rgba_json(args.trajectory_color, "--trajectory-color")
    grad_a = parse_rgba_json(args.gradient_start, "--gradient-start")
    grad_b = parse_rgba_json(args.gradient_end, "--gradient-end")

    face_mat = make_material("Fully_Transparent_Frustum_Faces", (0.0, 0.0, 0.0, 0.0), alpha_blend=True)
    path_mat = make_material("Trajectory_Path", traj_rgba)
    forward_mat = make_material("Forward_Arrow_Red", (1.0, 0.05, 0.03, 1.0))
    velocity_mat = make_material("Velocity_Vector_Cyan", (0.0, 0.75, 1.0, 1.0))
    start_mat = make_material("Start_Anchor_Green", (0.0, 1.0, 0.15, 1.0))
    end_mat = make_material("End_Anchor_Magenta", (1.0, 0.0, 0.8, 1.0))
    stage_materials = make_stage_materials(parse_stage_colors(args.stage_colors))

    create_ground(ground_size, args.grid_step, ground_z, args.show_ground_grid)

    triad_z = float(ground_z) + 0.02
    ground_half = 0.5 * float(ground_size)
    triad_len = float(args.axis_triad_length)
    # +X arm reaches to origin_x + triad_len; keep that west of -ground_half so the slab stays clear.
    triad_margin = max(0.06, 0.08 * ground_half)
    triad_x = -ground_half - triad_len - triad_margin - float(args.axis_triad_offset)
    create_axis_triad(
        (triad_x, 0.0, triad_z),
        triad_len,
        max(0.00045, triad_len * 0.014),
        float(args.axis_emission),
    )

    create_trajectory_path(positions, path_t, path_mat)
    create_anchor("Start_Anchor", positions[0], anch_r, start_mat)
    create_anchor("End_Anchor", positions[-1], anch_r, end_mat)
    template = create_frustum_template(face_mat)

    velocities = np.zeros_like(positions)
    if n_frames > 1:
        velocities[1:] = positions[1:] - positions[:-1]
        velocities[0] = velocities[1]

    stage_ids = load_stage_ids(args, n_frames, len(sampled_indices)) if args.color_by_stage else None
    if args.color_by_stage and stage_ids is None:
        print("Warning: --color-by-stage set but no --stage-ids or --stage-id-file; using gradient frustum colors.")

    n_sample = len(sampled_indices)
    for ordinal, frame_idx in enumerate(sampled_indices):
        if args.color_by_stage and stage_ids is not None:
            stage = stage_for_sample(stage_ids, frame_idx, ordinal, n_frames)
            edge_color_mat = stage_materials.get(stage, stage_materials["default"])
            obj_name = f"Camera_Frustum_{frame_idx:04d}_{stage}"
        else:
            t = 0.0 if n_sample <= 1 else float(ordinal) / float(n_sample - 1)
            gcol = lerp_rgba(grad_a, grad_b, t)
            edge_color_mat = make_material(f"FrustGrad_{uuid.uuid4().hex[:12]}", gcol)
            obj_name = f"Camera_Frustum_{frame_idx:04d}"
        pos = positions[frame_idx]
        rot = rotations[frame_idx]
        forward = -rot[:, 2]

        create_frustum_instance(
            template=template,
            name=obj_name,
            position=pos,
            rotation=rot,
            color_mat=edge_color_mat,
            args=args,
            frustum_xyz=(fw, fh, fd),
            wire_thickness=wire_t,
        )
        create_arrow(
            name=f"Forward_Arrow_{frame_idx:04d}",
            start=pos,
            direction=forward,
            length=arr_len,
            radius=arr_len * 0.016,
            material=forward_mat,
        )
        if args.show_velocity:
            create_velocity_line(
                name=f"Velocity_{frame_idx:04d}",
                start=pos,
                velocity=velocities[frame_idx],
                scale=args.velocity_scale,
                material=velocity_mat,
            )

    setup_scene_camera(
        args,
        positions=positions,
        ground_z=ground_z,
        camera_fit_points=cam_pts,
    )
    return n_frames, len(sampled_indices)


def render_still(path: Path, image_format: str) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.scene.render.image_settings.file_format = image_format
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print(f"Rendered {path}")


def _with_variant_suffix(path: Path, variant_key: str, multi_variant: bool) -> Path:
    if not multi_variant:
        return path
    return path.parent / f"{path.stem}_{variant_key}{path.suffix}"


def render_single(args: argparse.Namespace) -> None:
    if args.traj is not None:
        traj_path = args.traj.expanduser().resolve()
        n_frames, n_frustums = build_scene(args, traj_path, benchmark_dir=None, sample_id=None)
        print(f"Visualized {traj_path}")
        print(f"Frames: {n_frames}; frustums: {n_frustums}; layout: {args.layout}")
        if args.render_image:
            render_still(args.render_image, args.image_format)
        if args.output_blend:
            args.output_blend.parent.mkdir(parents=True, exist_ok=True)
            bpy.ops.wm.save_as_mainfile(filepath=str(args.output_blend))
            print(f"Saved {args.output_blend}")
        return

    if not args.sample_id:
        raise ValueError("Pass either --traj or --sample-id.")

    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    multi = len(args.variant_keys) > 1
    run_uniform = (
        multi
        and not args.no_frustum_auto_scale
        and (str(args.normalize_display) == "none" or args.normalize_keep_frustum_auto)
    )
    if run_uniform:
        ms: List[float] = []
        for vk in args.variant_keys:
            tp = trajectory_path_for_sample(benchmark_dir, vk, args.sample_id)
            if not tp.is_file():
                continue
            try:
                _, _, bstack, _, _, _, _, _, cpts = prepare_scene_kinematics(
                    args, tp, benchmark_dir=benchmark_dir, sample_id=str(args.sample_id)
                )
                ms.append(compute_frustum_multiplier_from_geometry(bstack, cpts, args))
            except Exception as exc:
                print(f"Warning: uniform frustum (multi-variant) skipped {vk}: {exc}")
        if ms:
            args._uniform_frustum_m_global = float(max(ms))
            print(f"Uniform frustum multiplier for this sample: {args._uniform_frustum_m_global:.4f}")
        else:
            args._uniform_frustum_m_global = None
    else:
        args._uniform_frustum_m_global = None

    any_ok = False
    try:
        for variant_key in args.variant_keys:
            traj_path = trajectory_path_for_sample(benchmark_dir, variant_key, args.sample_id)
            if not traj_path.is_file():
                print(f"Warning: no trajectory for variant {variant_key}: {traj_path}")
                continue
            any_ok = True
            n_frames, n_frustums = build_scene(
                args, traj_path, benchmark_dir=benchmark_dir, sample_id=str(args.sample_id)
            )
            print(f"Visualized {traj_path} (variant={variant_key})")
            print(f"Frames: {n_frames}; frustums: {n_frustums}; layout: {args.layout}")
            if args.render_image:
                render_still(_with_variant_suffix(args.render_image, variant_key, multi), args.image_format)
            if args.output_blend:
                out_blend = _with_variant_suffix(args.output_blend, variant_key, multi)
                out_blend.parent.mkdir(parents=True, exist_ok=True)
                bpy.ops.wm.save_as_mainfile(filepath=str(out_blend))
                print(f"Saved {out_blend}")

    finally:
        args._uniform_frustum_m_global = None
        if hasattr(args, "_normalize_scale_cache"):
            args._normalize_scale_cache.clear()
        if hasattr(args, "_ground_plane_cache"):
            args._ground_plane_cache.clear()

    if not any_ok:
        raise FileNotFoundError(
            f"No trajectory found for sample_id={args.sample_id!r} in variants {args.variant_keys}"
        )


def _batch_uniform_frustum_pre_pass_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "uniform_frustum_across_batch", False)
        and not args.no_frustum_auto_scale
        and (str(args.normalize_display) == "none" or args.normalize_keep_frustum_auto)
    )


def render_batch(args: argparse.Namespace, sample_ids: Sequence[str]) -> None:
    if args.render_dir is None:
        raise ValueError("Batch rendering requires --render-dir.")
    render_dir = args.render_dir.expanduser().resolve()
    extension = ".jpg" if args.image_format == "JPEG" else ".png"
    benchmark_dir = args.benchmark_dir.expanduser().resolve()
    variant_keys = args.variant_keys

    total = len(sample_ids) * len(variant_keys)
    if len(sample_ids) == 0:
        raise ValueError("No batch ids found. Use --batch-all, --ids, or --ids-file.")

    if _batch_uniform_frustum_pre_pass_enabled(args):
        ms: List[float] = []
        for sid in sample_ids:
            for vk in variant_keys:
                tp = trajectory_path_for_sample(benchmark_dir, vk, sid)
                if not tp.is_file():
                    continue
                try:
                    _, _, bstack, _, _, _, _, _, cpts = prepare_scene_kinematics(
                        args, tp, benchmark_dir=benchmark_dir, sample_id=str(sid)
                    )
                    ms.append(compute_frustum_multiplier_from_geometry(bstack, cpts, args))
                except Exception as exc:
                    print(f"Warning: uniform frustum pre-pass skipped {sid} ({vk}): {exc}")
        if ms:
            args._uniform_frustum_m_global = float(max(ms))
            print(
                f"Uniform frustum multiplier for batch: {args._uniform_frustum_m_global:.4f} "
                f"(max over {len(ms)} trajectories)"
            )
        else:
            args._uniform_frustum_m_global = None
    else:
        args._uniform_frustum_m_global = None

    rendered = 0
    skipped = 0
    failed: List[Tuple[str, str, str]] = []
    job = 0
    try:
        for sample_id in sample_ids:
            for variant_key in variant_keys:
                job += 1
                traj_path = trajectory_path_for_sample(benchmark_dir, variant_key, sample_id)
                out_stem = output_stem_for_sample(sample_id, traj_path)
                out_path = render_dir / f"{out_stem}_{variant_key}{extension}"
                if args.skip_existing and out_path.exists():
                    skipped += 1
                    print(f"[{job}/{total}] Skipping existing {out_path}")
                    continue

                if not traj_path.is_file():
                    failed.append((sample_id, variant_key, f"missing trajectory: {traj_path}"))
                    print(f"[{job}/{total}] {sample_id} ({variant_key}): missing {traj_path}")
                    continue

                try:
                    n_frames, n_frustums = build_scene(
                        args, traj_path, benchmark_dir=benchmark_dir, sample_id=str(sample_id)
                    )
                    print(f"[{job}/{total}] {sample_id} ({variant_key}): frames={n_frames}, frustums={n_frustums}")
                    render_still(out_path, args.image_format)
                    rendered += 1
                except Exception as exc:
                    failed.append((sample_id, variant_key, str(exc)))
                    print(f"[{job}/{total}] Failed {sample_id} ({variant_key}): {exc}")

    finally:
        args._uniform_frustum_m_global = None
        if hasattr(args, "_normalize_scale_cache"):
            args._normalize_scale_cache.clear()
        if hasattr(args, "_ground_plane_cache"):
            args._ground_plane_cache.clear()

    print(f"Batch complete: rendered={rendered}, skipped={skipped}, failed={len(failed)}, output={render_dir}")
    if failed:
        print("Failed samples:")
        for sample_id, variant_key, reason in failed[:20]:
            print(f"  {sample_id} [{variant_key}]: {reason}")


def main() -> None:
    args = parse_args()
    if args.sample_every < 1:
        raise ValueError("--sample-every must be >= 1")

    batch_ids = collect_batch_ids(args)
    if batch_ids:
        render_batch(args, batch_ids)
    else:
        render_single(args)


if __name__ == "__main__":
    main()
