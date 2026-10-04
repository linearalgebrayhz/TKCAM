#!/usr/bin/env python3
"""
Real Estate 10K Dataset Processing Script

Processes RE10K camera trajectories with AI-assisted captioning (Qwen VL).
Scenes without video frames are automatically skipped.

Usage:
    python process_realestate10k.py \\
        /path/to/RealEstate10K/train \\
        ./dataset/output \\
        --video-frames-dir /path/to/frames \\
        --max-scenes 100
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from tkcam.utils.realestate10k_processor import RealEstate10KProcessor


def main():
    parser = argparse.ArgumentParser(
        description="Process Real Estate 10K camera trajectories with AI captioning",
    )

    # Positional
    parser.add_argument(
        "input_dir",
        help="Directory containing RE10K camera parameter .txt files",
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for processed dataset",
    )
    parser.add_argument(
        "--video-frames-dir",
        required=True,
        help="Directory containing extracted video frames (required, one subfolder per scene)",
    )

    # Output format
    parser.add_argument(
        "--format",
        choices=["rotmat", "quat"],
        default="rotmat",
        help="Output format: 'rotmat' = 12D [pos,vel,rot6d], 'quat' = 10D [pos,vel,quat] (default: rotmat)",
    )

    # Scene selection
    parser.add_argument(
        "--scene-list",
        help="Text file with scene IDs to process (one per line)",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        help="Maximum number of scenes to process",
    )

    # Sequence filtering
    parser.add_argument(
        "--min-length",
        type=int,
        default=120,
        help="Minimum sequence length (default: 120)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=450,
        help="Maximum sequence length (default: 450)",
    )
    parser.add_argument(
        "--filter-min-frames",
        type=int,
        default=0,
        help="Skip scenes with fewer raw frames (0 = disabled)",
    )

    # Processing options
    parser.add_argument(
        "--no-smoothing",
        action="store_true",
        help="Disable Gaussian smoothing of trajectories",
    )
    parser.add_argument(
        "--transform",
        choices=["relative", "absolute"],
        default="relative",
        help="Pose transform type (default: relative)",
    )

    # AI captioning
    parser.add_argument(
        "--ai-model",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Qwen VL model for captioning (default: Qwen/Qwen3-VL-8B-Instruct)",
    )
    parser.add_argument(
        "--ai-max-frames",
        type=int,
        default=32,
        help="Max frames per scene for AI captioning (default: 32)",
    )
    parser.add_argument(
        "--ai-batch-size",
        type=int,
        default=2,
        help="Batch size for AI captioning (default: 2)",
    )

    parser.add_argument(
        "--ai-with-scene",
        action="store_true",
        help="Use scene-aware prompt (describes what the camera reveals); "
             "default is motion-only prompt for pipeline verification",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run, skip already-captioned scenes",
    )

    args = parser.parse_args()

    # Validate
    input_path = Path(args.input_dir)
    if not input_path.exists():
        print(f"Error: Input directory does not exist: {input_path}")
        sys.exit(1)
    if not list(input_path.glob("*.txt")):
        print(f"Error: No .txt files in {input_path}")
        sys.exit(1)

    # Create processor
    processor = RealEstate10KProcessor(
        output_format=args.format,
        min_sequence_length=args.min_length,
        max_sequence_length=args.max_length,
        transform=args.transform,
        ai_model_name=args.ai_model,
        ai_max_frames=args.ai_max_frames,
        video_source_dir=args.video_frames_dir,
        ai_batch_size=args.ai_batch_size,
        resume=args.resume,
        filter_min_frames=args.filter_min_frames,
        ai_with_scene=args.ai_with_scene,
    )

    # Print config
    print(f"Processing RE10K dataset")
    print(f"  Input:     {input_path}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Format:    {args.format}")
    print(f"  Transform: {args.transform}")
    print(f"  Model:     {args.ai_model}")
    print(f"  Frames:    {args.video_frames_dir}")
    print(f"  Sequence:  {args.min_length}-{args.max_length} frames")
    print(f"  Smoothing: {'disabled' if args.no_smoothing else 'enabled'}")
    print(f"  Batch:     {args.ai_batch_size}")
    if args.max_scenes:
        print(f"  Max scenes: {args.max_scenes}")
    if args.filter_min_frames:
        print(f"  Min raw frames: {args.filter_min_frames}")
    if args.resume:
        print(f"  Resume: enabled")
    print("-" * 50)

    # Run
    stats = processor.process_dataset(
        str(input_path),
        args.output_dir,
        args.scene_list,
        args.max_scenes,
        apply_smoothing=not args.no_smoothing,
    )

    print(
        f"\nCompleted: {stats['processed']} processed, "
        f"{stats['failed']} failed, {stats.get('skipped', 0)} skipped"
    )


if __name__ == "__main__":
    main()


"""
CUDA_VISIBLE_DEVICES=7 python process_realestate10k.py \
    /data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train \
    ./dataset/RealEstate10K_rotmat1 \
    --video-frames-dir /data4/haozhe/CamTraj/data/processed_estate/train_frames \
    --format rotmat \
    --ai-model Qwen/Qwen3-VL-8B-Instruct \
    --ai-batch-size 4 \
    --max-scenes 10000

"""