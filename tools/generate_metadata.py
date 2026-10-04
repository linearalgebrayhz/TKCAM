#!/usr/bin/env python3
"""
Script to generate metadata (mean.npy, std.npy, opt.txt) for custom datasets
This solves the shape mismatch issue by creating proper normalization statistics
"""

import os
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from os.path import join as pjoin

def compute_dataset_stats(data_root, motion_dir_name='new_joint_vecs', train_split='train.txt'):
    """
    Compute mean and std statistics for a dataset
    """
    motion_dir = pjoin(data_root, motion_dir_name)
    split_file = pjoin(data_root, train_split)
    
    print(f"Computing statistics from: {motion_dir}")
    print(f"Using split file: {split_file}")
    
    # Read training file list
    with open(split_file, 'r') as f:
        file_list = [line.strip() for line in f.readlines()]
    
    print(f"Found {len(file_list)} training files")
    
    # Collect all motion data
    all_motions = []
    valid_files = 0
    
    for filename in tqdm(file_list, desc="Loading motion files"):
        motion_file = pjoin(motion_dir, filename + '.npy')
        try:
            motion = np.load(motion_file)
            if len(motion.shape) == 2 and motion.shape[0] > 0:
                all_motions.append(motion)
                valid_files += 1
            else:
                print(f"Skipping invalid motion file: {filename} (shape: {motion.shape})")
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    if not all_motions:
        raise ValueError("No valid motion files found!")
    
    print(f"Successfully loaded {valid_files} motion files")
    
    # Concatenate all motions
    print("Concatenating motions...")
    all_data = np.concatenate(all_motions, axis=0)
    
    print(f"Total motion frames: {all_data.shape[0]}")
    print(f"Motion feature dimensions: {all_data.shape[1]}")
    
    # Compute statistics
    mean = np.mean(all_data, axis=0)
    std = np.std(all_data, axis=0)
    
    print(f"Computed mean: {mean}")
    print(f"Computed std: {std}")
    
    return mean, std, all_data.shape[1]

def create_opt_file(output_dir, dataset_name, dim_pose, experiment_name="Comp_v6_KLD005"):
    """
    Create opt.txt file with proper configuration
    """
    opt_content = f"""------------ Options -------------
batch_size: 256
checkpoints_dir: ./checkpoints
dataset_name: {dataset_name}
decomp_name: Decomp_SP001_SM001_H512
dim_att_vec: 512
dim_dec_hidden: 1024
dim_movement2_dec_hidden: 512
dim_movement_dec_hidden: 512
dim_movement_enc_hidden: 512
dim_movement_latent: 512
dim_msd_hidden: 512
dim_pos_hidden: 1024
dim_pri_hidden: 1024
dim_seq_de_hidden: 512
dim_seq_en_hidden: 512
dim_text_hidden: 512
dim_z: 128
early_stop_count: 3
estimator_mod: bigru
eval_every_e: 5
feat_bias: 5
fixed_steps: 5
gpu_id: 0
input_z: False
is_continue: False
is_train: True
lambda_fake: 10
lambda_gan_l: 0.1
lambda_gan_mt: 0.1
lambda_gan_mv: 0.1
lambda_kld: 0.005
lambda_rec: 1
lambda_rec_init: 1
lambda_rec_mot: 1
lambda_rec_mov: 1
log_every: 50
lr: 0.0002
max_sub_epoch: 50
max_text_len: 20
n_layers_dec: 1
n_layers_msd: 2
n_layers_pos: 1
n_layers_pri: 1
n_layers_seq_de: 2
n_layers_seq_en: 1
name: {experiment_name}
num_experts: 4
save_every_e: 10
save_latest: 500
text_enc_mod: bigru
tf_ratio: 0.4
unit_length: 4
-------------- End ----------------
"""
    
    opt_file = pjoin(output_dir, 'opt.txt')
    with open(opt_file, 'w') as f:
        f.write(opt_content)
    
    print(f"Created opt.txt: {opt_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate metadata for custom datasets')
    parser.add_argument('--dataset_name', type=str, required=True, 
                       help='Dataset name (e.g., realestate10k_6)')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to dataset root directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: ./checkpoints/{dataset_name}/Comp_v6_KLD005)')
    parser.add_argument('--experiment_name', type=str, default='Comp_v6_KLD005',
                       help='Experiment name for the metadata')
    parser.add_argument('--motion_dir_name', type=str, default='new_joint_vecs',
                       help='Name of motion directory within data_root')
    parser.add_argument('--train_split', type=str, default='train.txt',
                       help='Training split file name')
    
    args = parser.parse_args()
    
    # Set default output directory
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/{args.dataset_name}/{args.experiment_name}"
    
    print(f"Generating metadata for dataset: {args.dataset_name}")
    print(f"Data root: {args.data_root}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    meta_dir = output_dir / 'meta'
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(exist_ok=True)
    
    # Compute dataset statistics
    print("\n=== Computing Dataset Statistics ===")
    mean, std, dim_pose = compute_dataset_stats(
        args.data_root, 
        args.motion_dir_name, 
        args.train_split
    )
    
    # Save mean and std
    mean_file = meta_dir / 'mean.npy'
    std_file = meta_dir / 'std.npy'
    
    np.save(mean_file, mean)
    np.save(std_file, std)
    
    print(f"\n=== Saved Statistics ===")
    print(f"Mean saved to: {mean_file}")
    print(f"Std saved to: {std_file}")
    print(f"Feature dimensions: {dim_pose}")
    
    # Create opt.txt file
    print(f"\n=== Creating Configuration ===")
    create_opt_file(output_dir, args.dataset_name, dim_pose, args.experiment_name)
    
    print(f"\n=== Metadata Generation Complete ===")
    print(f"Generated files:")
    print(f"  - {mean_file}")
    print(f"  - {std_file}")
    print(f"  - {output_dir / 'opt.txt'}")
    
    print(f"\nYou can now use this metadata for training:")
    print(f"  - Dataset: {args.dataset_name}")
    print(f"  - Features: {dim_pose}")
    print(f"  - Checkpoint: {args.output_dir}")

if __name__ == "__main__":
    main()

