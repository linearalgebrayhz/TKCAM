#!/usr/bin/env python3
"""
Rebuild RealEstate10K trajectories with the corrected pose convention.

RealEstate10K camera files store each pose as a row-major 3x4 WORLD-TO-CAMERA
matrix in OpenCV camera axes (+X right, +Y down, +Z forward).
`RealEstate10KProcessor.parse_camera_file` reads it with
`ensure_twc(raw, "twc")`, i.e. as camera-to-world with OpenGL camera axes.
This script replays the processor's own code on the RE10K camera files and
changes exactly one thing, the function that turns the raw 3x4 block into
(R_wc, t_wc):

    original : R_wc, t_wc = raw                       (ensure_twc(raw, "twc"))
    fixed    : R_wc, t_wc = inverse(raw)              (ensure_twc(raw, "tcw"))
               R_wc       = R_wc @ diag(1, -1, -1)    (OpenCV -> OpenGL camera axes)

A second, independent fix (on by default, --keep-duplicate-first-frame to
disable): parse_camera_file adds frame 1 twice (a hand-made identity entry,
then frame 1 relative to itself), so trajectories have N+1 rows for N frames.
The hand-made entry is dropped, giving N rows: row k <-> frame k+1.

Everything else (relative-to-first-frame transform, 12-D features, Gaussian
smoothing, motion guidance) runs through the unmodified processor methods. As a self-check, every clip is also run with the
original function and compared with the existing trajectory file, so the
convention is the only difference between the old and new outputs.

The source dataset is only read. The output directory must not exist yet.

Output layout:
    <output>/trajectories/<id>.npy     fixed (T, 12) trajectories
    <output>/texts_old/, untagged_text_old/   copies of the OLD captions
                                       (they describe the old trajectories)
    <output>/guidance.json             {id: {"old": ..., "fixed": ...}} motion guidance
    <output>/metadata/                 copied; num_frames updated to the new trajectory length
    <output>/splits, scene_id_mapping.json   copied unchanged
    <output>/Mean.npy, Std.npy         recomputed over all clips (as curate_dataset.py does)
    <output>/posefix_report.json

Usage (TKCAM env):
    python fix_re10k_pose_convention.py \
        --source ./dataset/RealEstate10K_rotmat_25k \
        --output /data5/haozhe/CamTraj/data/posefix/RealEstate10K_rotmat_25k_posefix
"""

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import tkcam.utils.realestate10k_processor as rp
from tkcam.utils.camera_geometry import ensure_twc
from tkcam.utils.dataset_layout import TRAJ_DIR, traj_dir
from tkcam.utils.unified_data_format import CameraDataFormat, UnifiedCameraData

RE10K_ROOT = "/data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K"

# Camera-axis change OpenCV (+X right, +Y down, +Z fwd) -> OpenGL (+X right, +Y up, -Z fwd).
CV_TO_GL = np.diag([1.0, -1.0, -1.0])


def ensure_twc_original(block, semantics):
    """What the processor does today: the raw block is taken as T_wc."""
    return ensure_twc(block, semantics)


def ensure_twc_fixed(block, semantics):
    """RE10K blocks are T_cw (OpenCV axes): invert, then switch camera axes to OpenGL."""
    R_wc, t_wc = ensure_twc(block, "tcw")
    return R_wc @ CV_TO_GL, t_wc


def run_processor(proc, camera_file, ensure_fn, drop_duplicate_first=False):
    """Replay RealEstate10KProcessor.process_dataset's per-scene steps (Phase 1)."""
    rp.ensure_twc = ensure_fn
    try:
        _, camera_data = proc.parse_camera_file(str(camera_file))
        if drop_duplicate_first:
            # entry 0 is the hand-made identity for frame 1; entry 1 is frame 1 relative to itself
            assert np.allclose(camera_data[1]["pose_matrix"], np.eye(3, 4), atol=1e-6)
            camera_data = camera_data[1:]
            proc.relative_motion_data = proc.relative_motion_data[1:]
        if len(camera_data) > proc.max_sequence_length:
            camera_data = camera_data[: proc.max_sequence_length]
        traj = proc.extract_camera_trajectory(camera_data)
        traj = proc.smooth_trajectory(traj)
        unified = UnifiedCameraData(traj, CameraDataFormat.FULL_12_ROTMAT)
        final = unified.to_format(proc.target_format).get_momask_compatible_data().numpy()
        guidance = proc._generate_guidance()
    finally:
        rp.ensure_twc = ensure_twc_original
    return final, guidance


def find_camera_file(re10k_root, clip_id):
    for split in ("train", "test"):
        path = Path(re10k_root) / split / f"{clip_id}.txt"
        if path.exists():
            return path
    raise FileNotFoundError(f"RE10K camera file not found for {clip_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="./dataset/RealEstate10K_rotmat_25k")
    parser.add_argument("--output", required=True)
    parser.add_argument("--re10k-root", default=RE10K_ROOT)
    parser.add_argument("--max-sequence-length", type=int, default=450,
                        help="process_realestate10k.py --max-length default")
    parser.add_argument("--keep-duplicate-first-frame", action="store_true",
                        help="keep the N+1-row layout of the original pipeline")
    parser.add_argument("--reproduce-tol", type=float, default=1e-5,
                        help="max abs error allowed when re-running the original convention")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if output.exists():
        sys.exit(f"{output} already exists; refusing to overwrite")

    proc = rp.RealEstate10KProcessor(
        output_format="rotmat",
        transform="relative",
        max_sequence_length=args.max_sequence_length,
    )

    mapping = json.loads((source / "scene_id_mapping.json").read_text())
    src_traj = Path(traj_dir(source))
    ids = sorted(p.stem for p in src_traj.glob("*.npy"))

    output.mkdir(parents=True)
    (output / TRAJ_DIR).mkdir()

    guidance = {}
    num_frames = {}
    worst_repro = 0.0
    all_fixed = []
    for sid in tqdm(ids, desc="Rebuilding"):
        camera_file = find_camera_file(args.re10k_root, mapping[sid])
        stored = np.load(src_traj / f"{sid}.npy")

        old, g_old = run_processor(proc, camera_file, ensure_twc_original)
        err = float(np.abs(old - stored).max()) if old.shape == stored.shape else float("inf")
        if err > args.reproduce_tol:
            sys.exit(f"{sid}: original convention does not reproduce the stored trajectory "
                     f"(shape {old.shape} vs {stored.shape}, max err {err:.3g}); aborting")
        worst_repro = max(worst_repro, err)

        fixed, g_fixed = run_processor(proc, camera_file, ensure_twc_fixed,
                                       drop_duplicate_first=not args.keep_duplicate_first_frame)
        np.save(output / TRAJ_DIR / f"{sid}.npy", fixed)
        num_frames[sid] = len(fixed)
        all_fixed.append(fixed)
        guidance[sid] = {"old": g_old, "fixed": g_fixed}

    combined = np.concatenate(all_fixed, axis=0)
    np.save(output / "Mean.npy", np.mean(combined, axis=0))
    np.save(output / "Std.npy", np.std(combined, axis=0))

    for name in ("train.txt", "val.txt", "test.txt", "scene_id_mapping.json", "curation_report.json"):
        shutil.copy2(source / name, output / name)
    (output / "metadata").mkdir()
    for sid in ids:
        meta = json.loads((source / "metadata" / f"{sid}.json").read_text())
        meta["num_frames"] = num_frames[sid]
        (output / "metadata" / f"{sid}.json").write_text(json.dumps(meta, indent=2))
    shutil.copytree(source / "texts", output / "texts_old")
    shutil.copytree(source / "untagged_text", output / "untagged_text_old")
    (output / "guidance.json").write_text(json.dumps(guidance, indent=1))

    changed = sum(g["old"] != g["fixed"] for g in guidance.values())
    words = ("tilts up", "tilts down", "tracks left", "tracks right", "pans left", "pans right",
             "arcs left", "arcs right", "moves up", "moves down", "dollies forward", "dollies backward")
    count = lambda key: dict(Counter(w for g in guidance.values() for w in words if w in g[key]))
    report = {
        "source": str(source.resolve()),
        "clips": len(ids),
        "original_convention_max_abs_reproduction_error": worst_repro,
        "duplicate_first_frame_dropped": not args.keep_duplicate_first_frame,
        "guidance_changed": changed,
        "guidance_word_counts_old": count("old"),
        "guidance_word_counts_fixed": count("fixed"),
    }
    (output / "posefix_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
