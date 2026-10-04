import os
from os.path import join as pjoin

import torch
from torch.utils.data.dataloader import DataLoader

from tkcam.models.vq.model import RVQVAE
from tkcam.models.vq.vq_trainer import RVQTokenizerTrainer
from tkcam.options.vq_option import arg_parse
from tkcam.data.t2m_dataset import MotionDataset, collate_fn_camera
from tkcam.utils import paramUtil
import numpy as np

from tkcam.models.t2m_eval_wrapper import EvaluatorModelWrapper
from tkcam.models.evaluator.clatr_models import CLaTrEvalWrapper
from tkcam.utils.get_opt import get_opt
from tkcam.motion_loaders.dataset_motion_loader import get_dataset_motion_loader

from tkcam.utils.motion_process import recover_from_ric
from tkcam.utils.plot_script import plot_3d_motion
from tkcam.utils.fixseed import fixseed
from tkcam.utils.dataset_config import get_unified_dataset_config

os.environ["OMP_NUM_THREADS"] = "1"

def plot_t2m(data, save_dir):
    data = train_dataset.inv_transform(data)
    
    # Check if this is a camera dataset
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    
    if is_camera_dataset:
        # For camera data, use GT vs Pred comparison visualizations
        from tkcam.gen_camera import (
            plot_camera_trajectory_animation,
            plot_camera_trajectory_animation_vel_integrated,
        )
        from tkcam.utils.camera_plot import plot_camera_trajectory_3d
        from tkcam.utils.camera_geometry import integrate_velocity_to_positions
        from tkcam.utils.unified_data_format import detect_format_from_dataset_name
        import os
        
        # Detect format from dataset name
        viz_format_type = detect_format_from_dataset_name(opt.dataset_name)
        os.makedirs(save_dir, exist_ok=True)
        
        # Data is structured as: [GT_0, GT_1, GT_2, GT_3, Pred_0, Pred_1, Pred_2, Pred_3]
        # Split into GT and Pred
        num_samples = len(data) // 2
        gt_data = data[:num_samples]
        pred_data = data[num_samples:]
        
        for i in range(num_samples):
            gt_trajectory = gt_data[i]
            pred_trajectory = pred_data[i]

            def _prepare_velint_trajectory(trajectory):
                """Replace xyz with velocity-integrated positions when available."""
                if trajectory.shape[-1] < 6:
                    return trajectory
                try:
                    trajectory_velint = trajectory.copy()
                    trajectory_velint[:, :3] = integrate_velocity_to_positions(
                        trajectory,
                        smooth=True,
                        smooth_sigma=1.5,
                    )
                    return trajectory_velint
                except Exception as e:
                    print(f"[Warning] Vel-integration fallback to direct xyz: {e}")
                    return trajectory

            gt_trajectory_velint = _prepare_velint_trajectory(gt_trajectory)
            pred_trajectory_velint = _prepare_velint_trajectory(pred_trajectory)
            
            # 1. Generate comprehensive GT vs Pred comparison plot
            # This includes: 3D trajectory, top-down view, position error, orientation comparison
            comparison_path = pjoin(save_dir, f'camera_viz_sample_{i:02d}_comparison.png')
            try:
                plot_camera_trajectory_3d(
                    gt_data=gt_trajectory_velint[None, ...],  # Add batch dimension
                    pred_data=pred_trajectory_velint[None, ...],  # Add batch dimension
                    save_path=comparison_path,
                    title=f"RVQ Training Sample {i:02d} - GT vs Pred Comparison (Vel. Integrated)",
                    seq_idx=0,
                    format_type=viz_format_type
                )
                print(f"Camera comparison plot saved: {comparison_path}")
            except Exception as e:
                print(f"Error creating comparison plot {i}: {e}")
            
            # 2. Generate MP4 animations (GT direct xyz, Pred velocity-integrated)
            gt_video_path = pjoin(save_dir, f'camera_viz_sample_{i:02d}_gt.mp4')
            pred_video_path = pjoin(save_dir, f'camera_viz_sample_{i:02d}_pred.mp4')
            try:
                plot_camera_trajectory_animation(
                    data=gt_trajectory,
                    save_path=gt_video_path,
                    title=f"Sample {i:02d} - Ground Truth",
                    fps=30,
                    show_trail=True,
                    trail_length=30,
                    figsize=(10, 8),
                    format_type=viz_format_type
                )
                plot_camera_trajectory_animation_vel_integrated(
                    data=pred_trajectory,
                    save_path=pred_video_path,
                    title=f"Sample {i:02d} - Predicted (Vel. Integrated)",
                    fps=30,
                    show_trail=True,
                    trail_length=30,
                    figsize=(10, 8),
                    format_type=viz_format_type,
                    smooth=True,
                )
                print(f"Camera videos saved: {gt_video_path}, {pred_video_path}")
            except Exception as e:
                print(f"Error creating camera videos {i}: {e}")
    else:
        # For human motion data, use original plotting
        for i in range(len(data)):
            joint_data = data[i]
            joint = recover_from_ric(torch.from_numpy(joint_data).float(), opt.joints_num).numpy()
            save_path = pjoin(save_dir, '%02d.mp4' % (i))
            plot_3d_motion(save_path, kinematic_chain, joint, title="None", fps=fps, radius=radius)


if __name__ == "__main__":
    # torch.autograd.set_detect_anomaly(True)  # Disabled for performance with replicated samples
    opt = arg_parse(True)
    fixseed(opt.seed)
    # print(f"opt: {opt}")
    # exit()
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log/vq/', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    # Get unified dataset configuration with automatic format detection
    dataset_config = get_unified_dataset_config(opt)
    
    # Extract configuration values
    dim_pose = dataset_config['dim_pose']
    fps = dataset_config['fps']
    radius = dataset_config['radius']
    kinematic_chain = dataset_config['kinematic_chain']
    dataset_opt_path = dataset_config['dataset_opt_path']
    
    print(f"Dataset: {opt.dataset_name}")
    print(f"Data root: {opt.data_root}")
    print(f"Detected format: {dataset_config.get('detected_format', 'N/A')}")
    print(f"Feature dimensions: {dim_pose}")
    
    # Validate that dataset exists
    if not os.path.exists(opt.data_root):
        raise FileNotFoundError(f"Dataset directory does not exist: {opt.data_root}")
    if not os.path.exists(opt.motion_dir):
        raise FileNotFoundError(f"Motion directory does not exist: {opt.motion_dir}")

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    wrapper_opt.eval_on = opt.eval_on

    # Use CLaTr evaluator if checkpoint provided, otherwise fall back to legacy
    evaluator_ckpt = getattr(opt, 'evaluator_ckpt', None)
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if evaluator_ckpt and is_camera_dataset:
        eval_wrapper = CLaTrEvalWrapper(
            ckpt_path=evaluator_ckpt,
            device=opt.device,
            input_dim=dim_pose,
        )
        print(f"[CLaTr] Using CLaTr evaluator from {evaluator_ckpt}")
    else:
        eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        if is_camera_dataset and not evaluator_ckpt:
            print("[Warning] No --evaluator_ckpt provided for camera dataset. "
                  "Using legacy evaluator (metrics may be unreliable). "
                  "Run train_evaluator.py first to get a CLaTr checkpoint.")

    mean = np.load(pjoin(opt.data_root, 'Mean.npy'), allow_pickle=False)
    std = np.load(pjoin(opt.data_root, 'Std.npy'), allow_pickle=False)

    # Register pipeline normalization stats with CLaTr evaluator
    if evaluator_ckpt and is_camera_dataset and hasattr(eval_wrapper, 'set_pipeline_stats'):
        eval_wrapper.set_pipeline_stats(
            torch.from_numpy(mean), torch.from_numpy(std), dataset_name=opt.dataset_name)

    train_split_file = pjoin(opt.data_root, 'train.txt')
    val_split_file = pjoin(opt.data_root, 'val.txt')


    net = RVQVAE(opt,
                dim_pose,
                opt.nb_code,
                opt.code_dim,
                opt.code_dim,
                opt.down_t,
                opt.stride_t,
                opt.width,
                opt.depth,
                opt.dilation_growth_rate,
                opt.vq_act,
                opt.vq_norm)

    pc_vq = sum(param.numel() for param in net.parameters())
    print(net)
    # print("Total parameters of discriminator net: {}".format(pc_vq))
    # all_params += pc_vq_dis

    print('Total parameters of all models: {}M'.format(pc_vq/1000_000))

    trainer = RVQTokenizerTrainer(opt, vq_model=net)

    train_dataset = MotionDataset(opt, mean, std, train_split_file)
    val_dataset = MotionDataset(opt, mean, std, val_split_file)

    # Use camera-specific collate function for camera datasets
    is_camera_dataset = any(name in opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    
    if is_camera_dataset:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                  shuffle=True, pin_memory=True, collate_fn=collate_fn_camera)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                shuffle=True, pin_memory=True, collate_fn=collate_fn_camera)
    else:
        train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                  shuffle=True, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                                shuffle=True, pin_memory=True)
    
    # Eval loader uses meta from dataset_opt_path (e.g. Comp_v6_KLD005/meta). Ensure that meta
    # has format-specific mean.npy, std.npy matching the dataset (e.g. copy from dataset or from
    # a VQ run's meta once). The evaluator is trained on the dataset, so mean/std must match.
    if opt.eval_on:
        eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device,
                                                       data_root_override=opt.data_root)
    else:
        eval_val_loader = None

    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'val', device=opt.device,
                                                   data_root_override=opt.data_root)
    trainer.train(train_loader, val_loader, eval_val_loader, eval_wrapper, plot_t2m)

## train_vq.py --dataset_name kit --batch_size 512 --name VQVAE_dp2 --gpu_id 3
## train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp2_b256 --gpu_id 2
## train_vq.py --dataset_name kit --batch_size 1024 --name VQVAE_dp2_b1024 --gpu_id 1
## python train_vq.py --dataset_name kit --batch_size 256 --name VQVAE_dp1_b256 --gpu_id 2