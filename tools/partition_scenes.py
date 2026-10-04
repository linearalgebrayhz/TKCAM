#!/usr/bin/env python3
"""
Partition RealEstate10K scenes into multiple txt files for parallel processing.
Each file contains scene hash IDs (one per line) for use with --scene-list flag.
"""

import os
import argparse
from pathlib import Path


def partition_scenes(input_dir, output_dir, num_partitions=5, prefix="scenes"):
    """
    Partition scene files into multiple text files.
    
    Args:
        input_dir: Directory containing .txt camera parameter files
        output_dir: Directory to write partition files
        num_partitions: Number of partitions to create
        prefix: Prefix for output files (e.g., "train_scenes" or "test_scenes")
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get all scene files (*.txt)
    scene_files = sorted(input_path.glob("*.txt"))
    scene_ids = [f.stem for f in scene_files]
    
    total_scenes = len(scene_ids)
    scenes_per_partition = total_scenes // num_partitions
    remainder = total_scenes % num_partitions
    
    print(f"Total scenes: {total_scenes}")
    print(f"Partitions: {num_partitions}")
    print(f"Base scenes per partition: {scenes_per_partition}")
    print(f"Remainder: {remainder} (distributed to first partitions)")
    print()
    
    # Distribute scenes across partitions
    partition_files = []
    start_idx = 0
    
    for i in range(num_partitions):
        # Add extra scene to first 'remainder' partitions
        partition_size = scenes_per_partition + (1 if i < remainder else 0)
        end_idx = start_idx + partition_size
        
        partition_scenes = scene_ids[start_idx:end_idx]
        
        # Write partition file
        partition_file = output_path / f"{prefix}_part{i+1}.txt"
        with open(partition_file, 'w') as f:
            for scene_id in partition_scenes:
                f.write(f"{scene_id}\n")
        
        partition_files.append(partition_file)
        print(f"Part {i+1}: {len(partition_scenes)} scenes → {partition_file}")
        print(f"  Range: {partition_scenes[0]} to {partition_scenes[-1]}")
        print(f"  Indices: {start_idx} to {end_idx-1}")
        print()
        
        start_idx = end_idx
    
    return partition_files


def main():
    parser = argparse.ArgumentParser(
        description="Partition RealEstate10K scenes for parallel processing"
    )
    parser.add_argument(
        "--train-dir",
        default="/data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/train",
        help="Train directory containing camera parameter files"
    )
    parser.add_argument(
        "--test-dir",
        default="/data5/haozhe/CamTraj/data/real-state-10k/RealEstate10K/test",
        help="Test directory containing camera parameter files"
    )
    parser.add_argument(
        "--output-dir",
        default="./scene_partitions",
        help="Output directory for partition files"
    )
    parser.add_argument(
        "--num-partitions",
        type=int,
        default=5,
        help="Number of partitions to create"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only partition train set"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only partition test set"
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("RealEstate10K Scene Partitioner")
    print("=" * 80)
    print()
    
    # Partition train set
    if not args.test_only:
        print("Partitioning TRAIN set:")
        print("-" * 80)
        train_files = partition_scenes(
            args.train_dir,
            args.output_dir,
            args.num_partitions,
            prefix="train_scenes"
        )
    
    # Partition test set
    if not args.train_only:
        print("Partitioning TEST set:")
        print("-" * 80)
        test_files = partition_scenes(
            args.test_dir,
            args.output_dir,
            args.num_partitions,
            prefix="test_scenes"
        )
    
    print("=" * 80)
    print("Partitioning complete!")
    print()
    print("Usage examples:")
    print("-" * 80)
    
    if not args.test_only:
        print("\nTrain set (4 GPUs for parts 1-4):")
        for i in range(1, min(5, args.num_partitions + 1)):
            print(f"  GPU {i-1}: python process_realestate10k.py \\")
            print(f"    --input-dir {args.train_dir} \\")
            print(f"    --output-dir ./dataset/train_motion_part{i} \\")
            print(f"    --scene-list {args.output_dir}/train_scenes_part{i}.txt \\")
            print(f"    --use-ai-captioning \\")
            print(f"    --ai-batch-size 32")
            print()
    
    if not args.train_only:
        print("Test set (1 GPU):")
        print(f"  GPU 4: python process_realestate10k.py \\")
        print(f"    --input-dir {args.test_dir} \\")
        print(f"    --output-dir ./dataset/test_motion \\")
        print(f"    --scene-list {args.output_dir}/test_scenes_part1.txt \\")
        print(f"    --use-ai-captioning \\")
        print(f"    --ai-batch-size 32")


if __name__ == "__main__":
    main()
