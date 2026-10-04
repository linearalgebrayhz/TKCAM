#!/usr/bin/env python3
"""
Compatibility wrapper for the old TKCAM first-frame benchmark CLI.

The corrected benchmark inference implementation lives in
``run_tkcam_benchmark_inference.py``. This file only translates legacy
arguments and forwards execution there, so there is a single generation and
12D-to-9D conversion path to maintain.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Deprecated wrapper. Use run_tkcam_benchmark_inference.py --track firstframe "
            "for corrected TKCAM benchmark inference."
        )
    )
    parser.add_argument("--test-json", type=str, default="/home/haozhe/CamTraj/Benchmark_hub/test.json")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/home/haozhe/CamTraj/Benchmark_hub/result_TKCAM_firstFrame",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default="/home/haozhe/CamTraj/momask-codes/checkpoints",
    )
    parser.add_argument("--dataset-name", type=str, default="realestate10k_rotmat")
    parser.add_argument("--mtrans-name", type=str, default="mtrans_25k_1frame_r4")
    parser.add_argument("--rtrans-name", type=str, default="rtrans_25k_1frame_r4")
    parser.add_argument("--mtrans-ckpt", type=str, default="net_best_fid")
    parser.add_argument("--rtrans-ckpt", type=str, default=None)
    parser.add_argument("--apply-residual", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--numeric-id-only", action="store_true")
    parser.add_argument("--length-key", type=str, default="traj_len")
    parser.add_argument("--default-traj-len", type=int, default=200)
    parser.add_argument("--timesteps", type=int, default=18)
    parser.add_argument("--cond-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--topkr", type=float, default=0.9)
    parser.add_argument("--gsample", action="store_true")
    parser.add_argument("--res-cond-scale", type=float, default=5.0)
    args = parser.parse_args()

    if args.numeric_id_only:
        raise SystemExit(
            "--numeric-id-only is not supported by the canonical benchmark runner. "
            "Create a filtered prompts JSON and pass it via --test-json instead."
        )

    unsupported = []
    if args.default_traj_len != 200:
        unsupported.append("--default-traj-len")
    if args.timesteps != 18:
        unsupported.append("--timesteps")
    if args.cond_scale != 3.0:
        unsupported.append("--cond-scale")
    if args.temperature != 1.0:
        unsupported.append("--temperature")
    if args.topkr != 0.9:
        unsupported.append("--topkr")
    if args.gsample:
        unsupported.append("--gsample")
    if args.res_cond_scale != 5.0:
        unsupported.append("--res-cond-scale")
    if unsupported:
        raise SystemExit(
            "These legacy overrides are now read from the checkpoint opt.txt by "
            f"run_tkcam_benchmark_inference.py: {', '.join(unsupported)}"
        )

    # Legacy wrapper: run_tkcam_benchmark_inference lives in ../scripts in the release layout.
    import pathlib as _pathlib
    sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[1] / "scripts"))
    from run_tkcam_benchmark_inference import main as canonical_main

    forwarded = [
        sys.argv[0],
        "--track", "firstframe",
        "--prompts-json", args.test_json,
        "--out-dir", args.out_dir,
        "--checkpoints-dir", args.checkpoints_dir,
        "--dataset-name", args.dataset_name,
        "--mtrans-name", args.mtrans_name,
        "--rtrans-name", args.rtrans_name,
        "--which-epoch", args.mtrans_ckpt,
        "--device", args.device,
        "--seed", str(args.seed),
        "--batch-size", str(args.batch_size),
        "--length-key", args.length_key,
    ]
    if args.rtrans_ckpt:
        forwarded.extend(["--res-which-epoch", args.rtrans_ckpt])
    if args.apply_residual:
        forwarded.append("--apply-residual")
    if args.max_samples is not None:
        forwarded.extend(["--max-samples", str(args.max_samples)])

    sys.argv = forwarded
    return canonical_main()


if __name__ == "__main__":
    raise SystemExit(main())
