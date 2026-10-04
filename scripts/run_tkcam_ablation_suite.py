#!/usr/bin/env python3
"""Launch TKCAM benchmark ablation inference/evaluation suites.

By default this script prints commands. Add --execute to run them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parent


def _cmd_to_str(cmd: List[str]) -> str:
    return subprocess.list2cmdline(cmd)


def _run_or_print(cmds: Iterable[List[str]], execute: bool) -> int:
    for cmd in cmds:
        print(_cmd_to_str(cmd))
        if execute:
            completed = subprocess.run(cmd, cwd=str(ROOT))
            if completed.returncode != 0:
                return completed.returncode
    return 0


def _inference_cmd(
    *,
    track: str,
    output_name: str,
    benchmark_root: Path,
    prompts_json: Optional[Path],
    dataset_name: str,
    checkpoints_dir: Optional[Path],
    device: str,
    batch_size: int,
    seed: int,
    mtrans_name: Optional[str],
    rtrans_name: Optional[str],
    apply_residual: bool,
    num_res_layers: int,
    max_samples: Optional[int],
) -> List[str]:
    cmd = [
        sys.executable,
        str(ROOT / "run_tkcam_benchmark_inference.py"),
        "--track",
        track,
        "--benchmark-root",
        str(benchmark_root),
        "--dataset-name",
        dataset_name,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
        "--seed",
        str(seed),
        "--out-dir",
        str((benchmark_root / output_name).resolve()),
    ]
    if prompts_json is not None:
        cmd += ["--prompts-json", str(prompts_json)]
    if checkpoints_dir is not None:
        cmd += ["--checkpoints-dir", str(checkpoints_dir)]
    if mtrans_name:
        cmd += ["--mtrans-name", mtrans_name]
    if rtrans_name:
        cmd += ["--rtrans-name", rtrans_name]
    if apply_residual:
        cmd += ["--apply-residual", "--num-res-layers", str(num_res_layers)]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    return cmd


def _eval_cmd(
    *,
    benchmark_root: Path,
    prompts_json: Optional[Path],
    result_names: List[str],
    evaluator_ckpt: Optional[str],
    reference_result_dir: Optional[Path],
    device: str,
    max_samples: Optional[int],
) -> List[str]:
    baseline_names = ",".join(name[len("result_"):] if name.startswith("result_") else name for name in result_names)
    cmd = [
        sys.executable,
        str(ROOT / "eval_benchmark_clatr.py"),
        "--benchmark-root",
        str(benchmark_root),
        "--result-prefix",
        "result_",
        "--baseline-names",
        baseline_names,
        "--device",
        device,
    ]
    if prompts_json is not None:
        cmd += ["--prompts-json", str(prompts_json)]
    if evaluator_ckpt:
        cmd += ["--evaluator-ckpt", evaluator_ckpt]
    if reference_result_dir is not None:
        cmd += ["--reference-result-dir", str(reference_result_dir)]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    return cmd


def build_residual_suite(args: argparse.Namespace) -> List[List[str]]:
    variants = [
        ("result_TKCAM_textOnly_base", False, -1),
        ("result_TKCAM_textOnly_levels2", True, 1),
        ("result_TKCAM_textOnly_levels3", True, 2),
        ("result_TKCAM_textOnly_full", True, -1),
    ]
    cmds = [
        _inference_cmd(
            track=args.track,
            output_name=name,
            benchmark_root=args.benchmark_root,
            prompts_json=args.prompts_json,
            dataset_name=args.dataset_name,
            checkpoints_dir=args.checkpoints_dir,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            mtrans_name=args.mtrans_name,
            rtrans_name=args.rtrans_name,
            apply_residual=apply,
            num_res_layers=num_layers,
            max_samples=args.max_samples,
        )
        for name, apply, num_layers in variants
    ]
    cmds.append(_eval_cmd(
        benchmark_root=args.benchmark_root,
        prompts_json=args.prompts_json,
        result_names=[name for name, _, _ in variants],
        evaluator_ckpt=args.evaluator_ckpt,
        reference_result_dir=args.reference_result_dir,
        device=args.device,
        max_samples=args.max_samples,
    ))
    return cmds


def build_keyframe_suite(args: argparse.Namespace) -> List[List[str]]:
    specs = [
        ("textonly", "result_TKCAM_textOnly_full", args.text_mtrans_name, args.text_rtrans_name),
        ("firstframe", "result_TKCAM_firstFrame_full", args.firstframe_mtrans_name, args.firstframe_rtrans_name),
        ("sparsekeyframe", "result_TKCAM_sparseKeyframe_full", args.sparse_mtrans_name, args.sparse_rtrans_name),
    ]
    cmds = [
        _inference_cmd(
            track=track,
            output_name=name,
            benchmark_root=args.benchmark_root,
            prompts_json=args.prompts_json,
            dataset_name=args.dataset_name,
            checkpoints_dir=args.checkpoints_dir,
            device=args.device,
            batch_size=args.batch_size,
            seed=args.seed,
            mtrans_name=m_name,
            rtrans_name=r_name,
            apply_residual=True,
            num_res_layers=-1,
            max_samples=args.max_samples,
        )
        for track, name, m_name, r_name in specs
    ]
    cmds.append(_eval_cmd(
        benchmark_root=args.benchmark_root,
        prompts_json=args.prompts_json,
        result_names=[name for _, name, _, _ in specs],
        evaluator_ckpt=args.evaluator_ckpt,
        reference_result_dir=args.reference_result_dir,
        device=args.device,
        max_samples=args.max_samples,
    ))
    return cmds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["residual", "keyframe"], required=True)
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument("--benchmark-root", type=Path, default=(ROOT.parent / "Benchmark_hub"))
    parser.add_argument("--prompts-json", type=Path, default=None)
    parser.add_argument("--checkpoints-dir", type=Path, default=None)
    parser.add_argument("--dataset-name", default="realestate10k_rotmat")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--evaluator-ckpt", type=str, default=None)
    parser.add_argument(
        "--reference-result-dir",
        type=Path,
        default=None,
        help=(
            "Reference trajectory folder for FID. Defaults to <benchmark-root>/reference; "
            "pass --no-reference-fid to omit it."
        ),
    )
    parser.add_argument("--no-reference-fid", action="store_true", help="Do not pass a reference folder to eval_benchmark_clatr.py.")

    parser.add_argument("--track", default="textonly", choices=["textonly", "noframe", "firstframe", "sparsekeyframe"],
                        help="Track used by the residual suite.")
    parser.add_argument("--mtrans-name", default=None, help="Mask transformer for residual suite.")
    parser.add_argument("--rtrans-name", default=None, help="Residual transformer for residual suite.")

    parser.add_argument("--text-mtrans-name", default=None)
    parser.add_argument("--text-rtrans-name", default=None)
    parser.add_argument("--firstframe-mtrans-name", default=None)
    parser.add_argument("--firstframe-rtrans-name", default=None)
    parser.add_argument("--sparse-mtrans-name", default=None)
    parser.add_argument("--sparse-rtrans-name", default=None)
    args = parser.parse_args()

    args.benchmark_root = args.benchmark_root.expanduser().resolve()
    if args.prompts_json is not None:
        args.prompts_json = args.prompts_json.expanduser().resolve()
    if args.checkpoints_dir is not None:
        args.checkpoints_dir = args.checkpoints_dir.expanduser().resolve()
    if args.no_reference_fid:
        args.reference_result_dir = None
    elif args.reference_result_dir is None:
        args.reference_result_dir = (args.benchmark_root / "reference").resolve()
    else:
        args.reference_result_dir = args.reference_result_dir.expanduser().resolve()

    cmds = build_residual_suite(args) if args.suite == "residual" else build_keyframe_suite(args)
    return _run_or_print(cmds, execute=args.execute)


if __name__ == "__main__":
    raise SystemExit(main())

"""
CUDA_VISIBLE_DEVICES=4 python3 run_tkcam_ablation_suite.py \
  --suite residual \
  --execute \
  --device cuda:0 \
  --benchmark-root /home/haozhe/CamTraj/Benchmark_hub \
  --prompts-json /home/haozhe/CamTraj/Benchmark_hub/test.json \
  --dataset-name realestate10k_rotmat \
  --track textonly \
  --mtrans-name mtrans_25k_noframe_r4 \
  --rtrans-name rtrans_25k_noframe_r4 &&

CUDA_VISIBLE_DEVICES=4 python3 run_tkcam_ablation_suite.py \
  --suite keyframe \
  --execute \
  --device cuda:0 \
  --benchmark-root /home/haozhe/CamTraj/Benchmark_hub \
  --prompts-json /home/haozhe/CamTraj/Benchmark_hub/test.json \
  --dataset-name realestate10k_rotmat \
  --text-mtrans-name mtrans_25k_noframe_r4 \
  --text-rtrans-name rtrans_25k_noframe_r4 \
  --firstframe-mtrans-name mtrans_25k_1frame_r4 \
  --firstframe-rtrans-name rtrans_25k_1frame_r4 \
  --sparse-mtrans-name mtrans_25k_xframe_r4 \
  --sparse-rtrans-name rtrans_25k_xframe_r4
"""