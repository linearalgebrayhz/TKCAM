#!/usr/bin/env python3
"""
Merge multiple dataset splits into a unified dataset with sequential numbering.

This script:
1. Collects all scenes from multiple splits
2. Renumbers them sequentially (000000, 000001, ...)
3. Merges scene_id_mapping.json files
4. Recalculates dataset statistics (Mean.npy, Std.npy)
5. Regenerates train/val/test splits
"""

import json
import shutil
import numpy as np
from pathlib import Path
import argparse

# Optional tqdm import with fallback
try:
    from tqdm import tqdm
except ImportError:
    # Fallback: simple progress indicator
    def tqdm(iterable, desc="", total=None):
        """Simple fallback for tqdm"""
        print(f"{desc}...")
        return iterable


def collect_split_info(splits_root: Path):
    """Collect information about all splits"""
    splits = []
    
    # Find all split directories
    split_dirs = sorted([d for d in splits_root.iterdir() if d.is_dir()])
    
    print(f"Found {len(split_dirs)} splits:")
    for split_dir in split_dirs:
        # Count scenes
        npy_files = list((split_dir / "new_joint_vecs").glob("*.npy"))
        txt_files = list((split_dir / "untagged_text").glob("*.txt"))
        
        # Load scene_id_mapping
        mapping_file = split_dir / "scene_id_mapping.json"
        if mapping_file.exists():
            with open(mapping_file, 'r') as f:
                scene_mapping = json.load(f)
        else:
            scene_mapping = {}
            print(f"  Warning: {split_dir.name} has no scene_id_mapping.json")
        
        splits.append({
            'name': split_dir.name,
            'path': split_dir,
            'num_scenes': len(npy_files),
            'num_texts': len(txt_files),
            'scene_mapping': scene_mapping
        })
        
        print(f"  {split_dir.name}: {len(npy_files)} scenes, {len(txt_files)} captions")
    
    total_scenes = sum(s['num_scenes'] for s in splits)
    print(f"\nTotal scenes across all splits: {total_scenes}")
    
    return splits, total_scenes


def merge_datasets(splits_root: Path, output_root: Path, splits_info: list):
    """Merge all splits into a unified dataset"""
    
    # Create output directories
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "new_joint_vecs").mkdir(exist_ok=True)
    (output_root / "untagged_text").mkdir(exist_ok=True)
    (output_root / "metadata").mkdir(exist_ok=True)
    
    # Initialize unified scene_id_mapping
    unified_mapping = {}
    current_id = 0
    
    # Track all scene IDs for later split generation
    all_scene_ids = []
    
    # Process each split
    print("\nMerging splits...")
    for split_info in tqdm(splits_info, desc="Processing splits"):
        split_path = split_info['path']
        split_mapping = split_info['scene_mapping']
        
        # Get all scenes in this split (sorted by numeric ID)
        npy_dir = split_path / "new_joint_vecs"
        npy_files = sorted(npy_dir.glob("*.npy"), key=lambda x: int(x.stem))
        
        # Process each scene
        for npy_file in npy_files:
            old_numeric_id = npy_file.stem
            new_numeric_id = f"{current_id:06d}"
            
            # Get original hex ID from mapping
            original_hex_id = split_mapping.get(old_numeric_id, old_numeric_id)
            
            # Update unified mapping
            unified_mapping[new_numeric_id] = original_hex_id
            all_scene_ids.append(new_numeric_id)
            
            # Copy trajectory data
            src_npy = npy_file
            dst_npy = output_root / "new_joint_vecs" / f"{new_numeric_id}.npy"
            shutil.copy2(src_npy, dst_npy)
            
            # Copy caption text
            src_txt = split_path / "untagged_text" / f"{old_numeric_id}.txt"
            dst_txt = output_root / "untagged_text" / f"{new_numeric_id}.txt"
            if src_txt.exists():
                shutil.copy2(src_txt, dst_txt)
            
            # Copy metadata
            src_meta = split_path / "metadata" / f"{old_numeric_id}.json"
            dst_meta = output_root / "metadata" / f"{new_numeric_id}.json"
            if src_meta.exists():
                # Load, update IDs, and save
                with open(src_meta, 'r') as f:
                    metadata = json.load(f)
                metadata['scene_id'] = new_numeric_id
                metadata['original_scene_id'] = original_hex_id
                with open(dst_meta, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            current_id += 1
    
    print(f"\n✓ Merged {current_id} scenes")
    
    return unified_mapping, all_scene_ids


def save_unified_mapping(output_root: Path, unified_mapping: dict):
    """Save the unified scene_id_mapping.json"""
    mapping_file = output_root / "scene_id_mapping.json"
    with open(mapping_file, 'w') as f:
        json.dump(unified_mapping, f, indent=2)
    print(f"✓ Saved unified scene_id_mapping.json with {len(unified_mapping)} mappings")


def generate_splits(output_root: Path, scene_ids: list, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Generate train/val/test splits"""
    np.random.seed(42)  # For reproducibility
    
    # Shuffle scene IDs
    shuffled_ids = scene_ids.copy()
    np.random.shuffle(shuffled_ids)
    
    # Calculate split sizes
    n_total = len(shuffled_ids)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    
    # Split
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    # Save split files
    with open(output_root / "train.txt", 'w') as f:
        for scene_id in train_ids:
            f.write(f"{scene_id}\n")
    
    with open(output_root / "val.txt", 'w') as f:
        for scene_id in val_ids:
            f.write(f"{scene_id}\n")
    
    with open(output_root / "test.txt", 'w') as f:
        for scene_id in test_ids:
            f.write(f"{scene_id}\n")
    
    print(f"✓ Generated splits: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")


def calculate_statistics(output_root: Path, scene_ids: list):
    """Calculate dataset statistics (Mean.npy, Std.npy)"""
    print("\nCalculating dataset statistics...")
    
    motion_dir = output_root / "new_joint_vecs"
    
    # Collect all data
    all_data = []
    for scene_id in tqdm(scene_ids, desc="Loading data"):
        motion_file = motion_dir / f"{scene_id}.npy"
        if motion_file.exists():
            data = np.load(motion_file)
            all_data.append(data)
    
    if all_data:
        # Concatenate all data
        combined_data = np.concatenate(all_data, axis=0)
        
        # Calculate mean and std
        mean = np.mean(combined_data, axis=0)
        std = np.std(combined_data, axis=0)
        
        # Save statistics
        np.save(output_root / "Mean.npy", mean)
        np.save(output_root / "Std.npy", std)
        
        print(f"✓ Calculated statistics from {len(all_data)} scenes")
        print(f"  Mean shape: {mean.shape}")
        print(f"  Std shape: {std.shape}")
        print(f"  Mean: {mean}")
        print(f"  Std: {std}")
    else:
        print("✗ No data found for statistics calculation")


def main():
    parser = argparse.ArgumentParser(description="Merge dataset splits into unified dataset")
    parser.add_argument(
        "splits_root",
        help="Root directory containing all splits (e.g., RealEstate10K_6_75K)"
    )
    parser.add_argument(
        "output_root",
        help="Output directory for merged dataset (e.g., RealEstate10K_6_merged)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.15)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio (default: 0.15)"
    )
    
    args = parser.parse_args()
    
    splits_root = Path(args.splits_root)
    output_root = Path(args.output_root)
    
    if not splits_root.exists():
        print(f"Error: Splits root directory does not exist: {splits_root}")
        return 1
    
    # Validate split ratios
    if not abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) < 0.001:
        print(f"Error: Split ratios must sum to 1.0")
        return 1
    
    print("=" * 80)
    print("Dataset Merge Tool")
    print("=" * 80)
    print(f"Splits root: {splits_root}")
    print(f"Output root: {output_root}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    print("=" * 80)
    
    # Step 1: Collect split information
    print("\n[1/5] Collecting split information...")
    splits_info, total_scenes = collect_split_info(splits_root)
    
    # Step 2: Merge all splits
    print("\n[2/5] Merging splits...")
    unified_mapping, all_scene_ids = merge_datasets(splits_root, output_root, splits_info)
    
    # Step 3: Save unified scene_id_mapping.json
    print("\n[3/5] Saving unified scene ID mapping...")
    save_unified_mapping(output_root, unified_mapping)
    
    # Step 4: Generate train/val/test splits
    print("\n[4/5] Generating dataset splits...")
    generate_splits(output_root, all_scene_ids, args.train_ratio, args.val_ratio, args.test_ratio)
    
    # Step 5: Calculate dataset statistics
    print("\n[5/5] Calculating dataset statistics...")
    calculate_statistics(output_root, all_scene_ids)
    
    print("\n" + "=" * 80)
    print("✓ Dataset merge completed successfully!")
    print("=" * 80)
    print(f"Output directory: {output_root}")
    print(f"Total scenes: {len(all_scene_ids)}")
    print(f"Scene ID range: 000000 - {len(all_scene_ids)-1:06d}")
    print("\nGenerated files:")
    print(f"  - new_joint_vecs/      ({len(all_scene_ids)} .npy files)")
    print(f"  - untagged_text/       ({len(list((output_root / 'untagged_text').glob('*.txt')))} .txt files)")
    print(f"  - metadata/            ({len(list((output_root / 'metadata').glob('*.json')))} .json files)")
    print(f"  - scene_id_mapping.json ({len(unified_mapping)} mappings)")
    print(f"  - train.txt, val.txt, test.txt")
    print(f"  - Mean.npy, Std.npy")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

