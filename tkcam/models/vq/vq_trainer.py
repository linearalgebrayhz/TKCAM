import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
from os.path import join as pjoin
import torch.nn.functional as F
import torch.optim as optim

import time
import numpy as np
from collections import OrderedDict, defaultdict
from tkcam.utils.eval_t2m import evaluation_vqvae
from tkcam.utils.camera_eval import evaluation_camera_vqvae
from tkcam.utils.clatr_camera_eval import evaluation_camera_vqvae_clatr
from tkcam.utils.utils import print_current_loss
from tkcam.utils.logging_utils import VQLogger, create_loss_dict, aggregate_losses

import os
import sys
from tqdm import tqdm

def def_value():
    return 0.0


def compute_smoothness_loss(pred_motion, gt_motion):
    """
    Compute temporal smoothness loss for camera trajectories.
    Penalizes large changes in velocity (acceleration) and jerk.
    
    Args:
        pred_motion: Predicted motion [B, T, D]
        gt_motion: Ground truth motion [B, T, D]
    
    Returns:
        smoothness_loss: Combined acceleration and jerk penalty
    """
    feature_dim = pred_motion.shape[-1]
    
    if feature_dim >= 10:  # Has explicit velocity components (10D quat, 12D formats)
        # Extract velocity (dims 3-5 for all formats with velocity)
        pred_vel = pred_motion[..., 3:6]  # [B, T, 3]
        gt_vel = gt_motion[..., 3:6]
        
        # Compute acceleration (velocity changes between frames)
        pred_accel = pred_vel[:, 1:] - pred_vel[:, :-1]  # [B, T-1, 3]
        gt_accel = gt_vel[:, 1:] - gt_vel[:, :-1]
        
        # L2 penalty on acceleration difference
        loss_accel = F.mse_loss(pred_accel, gt_accel)
        
        # Jerk penalty (acceleration changes) - smoother higher-order motion
        if pred_accel.shape[1] > 1:  # Need at least 2 frames for jerk
            pred_jerk = pred_accel[:, 1:] - pred_accel[:, :-1]  # [B, T-2, 3]
            gt_jerk = gt_accel[:, 1:] - gt_accel[:, :-1]
            loss_jerk = F.mse_loss(pred_jerk, gt_jerk)
            return loss_accel + 0.5 * loss_jerk
        else:
            return loss_accel
    else:
        # For legacy formats and 9D rotmat without explicit velocity, use position-based smoothness.
        pred_pos = pred_motion[..., :3]
        gt_pos = gt_motion[..., :3]
        
        # Compute velocity via finite differences
        pred_vel_fd = pred_pos[:, 1:] - pred_pos[:, :-1]  # [B, T-1, 3]
        gt_vel_fd = gt_pos[:, 1:] - gt_pos[:, :-1]
        
        # Compute acceleration from finite differences
        if pred_vel_fd.shape[1] > 1:
            pred_accel = pred_vel_fd[:, 1:] - pred_vel_fd[:, :-1]  # [B, T-2, 3]
            gt_accel = gt_vel_fd[:, 1:] - gt_vel_fd[:, :-1]
            return F.mse_loss(pred_accel, gt_accel)
        else:
            return torch.tensor(0.0, device=pred_motion.device)


class RVQTokenizerTrainer:
    def __init__(self, args, vq_model):
        self.opt = args
        self.vq_model = vq_model
        self.device = args.device

        if args.is_train:
            self.logger = VQLogger(args)
            if args.recons_loss == 'l1':
                self.l1_criterion = nn.L1Loss()
            elif args.recons_loss == 'l1_smooth':
                self.l1_criterion = nn.SmoothL1Loss()

        # self.critic = CriticWrapper(self.opt.dataset_name, self.opt.device)

    def forward(self, batch_data, step=None):
        motions = batch_data.detach().to(self.device).float()
        pred_motion, loss_commit, perplexity = self.vq_model(motions)
        
        self.motions = motions
        self.pred_motion = pred_motion

        loss_rec = self.l1_criterion(pred_motion, motions)
        
        # Check if this is a camera dataset
        is_camera_dataset = any(name in self.opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
        
        if is_camera_dataset:
            # For camera data, handle different feature dimensions
            feature_dim = motions.shape[-1]
            
            if feature_dim == 5:
                # Legacy 5-feature: [x, y, z, pitch, yaw]
                pred_pos = pred_motion[..., :3]  # x, y, z
                gt_pos = motions[..., :3]
                pred_ori = pred_motion[..., 3:5]  # pitch, yaw
                gt_ori = motions[..., 3:5]
                
                loss_pos = self.l1_criterion(pred_pos, gt_pos)
                loss_ori = self.l1_criterion(pred_ori, gt_ori)
                loss_explicit = loss_pos + loss_ori
                
            elif feature_dim == 6:
                # 6-feature: [x, y, z, pitch, yaw, roll]
                pred_pos = pred_motion[..., :3]  # x, y, z
                gt_pos = motions[..., :3]
                pred_ori = pred_motion[..., 3:6]  # pitch, yaw, roll
                gt_ori = motions[..., 3:6]
                
                loss_pos = self.l1_criterion(pred_pos, gt_pos)
                loss_ori = self.l1_criterion(pred_ori, gt_ori)
                loss_explicit = loss_pos + loss_ori

            elif feature_dim == 9:
                # 9D rotmat: [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]
                pred_pos = pred_motion[..., :3]
                gt_pos = motions[..., :3]
                pred_rot = pred_motion[..., 3:9]
                gt_rot = motions[..., 3:9]

                loss_pos = self.l1_criterion(pred_pos, gt_pos)
                loss_rot = self.l1_criterion(pred_rot, gt_rot)

                if 'rotmat' in self.opt.dataset_name.lower():
                    from tkcam.utils.unified_data_format import compute_orthogonality_loss
                    loss_orth = compute_orthogonality_loss(pred_rot)
                    orth_weight = getattr(self.opt, 'loss_orthogonality', 0.1)
                    loss_explicit = loss_pos + loss_rot + orth_weight * loss_orth
                    self.loss_orth = loss_orth
                else:
                    loss_explicit = loss_pos + loss_rot
                    self.loss_orth = torch.tensor(0.0, device=self.device)
                
            elif feature_dim == 10:
                # 10D quaternion: [x, y, z, dx, dy, dz, qw, qx, qy, qz]
                pred_pos_vel = pred_motion[..., :6]   # position + velocity
                pred_quat = pred_motion[..., 6:10]    # quaternion
                
                gt_pos_vel = motions[..., :6]
                gt_quat = motions[..., 6:10]
                
                loss_pos_vel = self.l1_criterion(pred_pos_vel, gt_pos_vel)
                loss_quat = self.l1_criterion(pred_quat, gt_quat)
                
                # Simple combination - normalized data should have similar scales
                loss_explicit = loss_pos_vel + loss_quat
                
            elif feature_dim == 12:
                # 12D format - could be Euler or rotation matrix
                # Both have structure: [x, y, z, dx, dy, dz, rot1, rot2, rot3, rot4, rot5, rot6]
                # Euler: rot = [pitch, yaw, roll, dpitch, dyaw, droll]
                # RotMat: rot = [r1x, r1y, r1z, r2x, r2y, r2z]
                
                pred_pos_vel = pred_motion[..., :6]    # position + velocity
                pred_rot = pred_motion[..., 6:12]      # rotation representation (6D)
                
                gt_pos_vel = motions[..., :6]
                gt_rot = motions[..., 6:12]
                
                loss_pos_vel = self.l1_criterion(pred_pos_vel, gt_pos_vel)
                loss_rot = self.l1_criterion(pred_rot, gt_rot)
                
                # Check if this is rotation matrix format (need orthogonality)
                is_rotmat = 'rotmat' in self.opt.dataset_name.lower()
                
                if is_rotmat:
                    # Add orthogonality loss for rotation matrix format
                    from tkcam.utils.unified_data_format import compute_orthogonality_loss
                    loss_orth = compute_orthogonality_loss(pred_rot)
                    
                    # Weight for orthogonality loss (default 0.1)
                    orth_weight = getattr(self.opt, 'loss_orthogonality', 0.1)
                    
                    loss_explicit = loss_pos_vel + loss_rot + orth_weight * loss_orth
                    
                    # Store for logging
                    self.loss_orth = loss_orth
                else:
                    loss_explicit = loss_pos_vel + loss_rot
                    self.loss_orth = torch.tensor(0.0, device=self.device)
                
            else:
                # Fallback for unknown camera data format
                print(f"Warning: Unknown camera data format with {feature_dim} features, using simple reconstruction loss")
                loss_explicit = loss_rec
        else:
            # For human motion data, use original local position loss
            pred_local_pos = pred_motion[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            local_pos = motions[..., 4 : (self.opt.joints_num - 1) * 3 + 4]
            loss_explicit = self.l1_criterion(pred_local_pos, local_pos)

        # Compute temporal smoothness loss (acceleration/jerk penalty)
        # Only compute if weight > 0 to allow disabling for comparison experiments
        loss_smoothness = torch.tensor(0.0, device=self.device)
        smoothness_weight = getattr(self.opt, 'loss_smoothness', 0.0)  # Default 0.0 (disabled)
        
        if is_camera_dataset and smoothness_weight > 0 and motions.shape[1] > 2:
            loss_smoothness = compute_smoothness_loss(pred_motion, motions)
        
        # Initialize loss_orth if not set (for non-rotmat datasets)
        if not hasattr(self, 'loss_orth'):
            self.loss_orth = torch.tensor(0.0, device=self.device)
        
        # Combine all losses
        loss = loss_rec + self.opt.loss_vel * loss_explicit + \
               self.opt.commit * loss_commit + \
               smoothness_weight * loss_smoothness
        
        # Debug NaN detection
        if torch.isnan(loss).any() or torch.isnan(loss_explicit).any():
            print(f"NaN detected! Feature dim: {motions.shape[-1]}, Dataset: {self.opt.dataset_name}")
            print(f"loss_rec: {loss_rec.item():.6f}, loss_explicit: {loss_explicit.item():.6f}, loss_commit: {loss_commit.item():.6f}")
            print(f"Motions range: [{motions.min().item():.6f}, {motions.max().item():.6f}]")
            print(f"Pred_motion range: [{pred_motion.min().item():.6f}, {pred_motion.max().item():.6f}]")
            if is_camera_dataset and feature_dim == 12:
                print(f"Angular velocity range: [{motions[..., 9:12].min().item():.6f}, {motions[..., 9:12].max().item():.6f}]")
            # Set loss to a small positive value to continue training
            loss = torch.tensor(1e-6, device=self.device, requires_grad=True)

        return loss, loss_rec, loss_explicit, loss_commit, perplexity, loss_smoothness, self.loss_orth


    # @staticmethod
    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_vq_model.param_groups:
            param_group["lr"] = current_lr

        return current_lr

    def save(self, file_name, ep, total_it, metrics=None):
        state = {
            "vq_model": self.vq_model.state_dict(),
            "opt_vq_model": self.opt_vq_model.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        torch.save(state, file_name)
        
        # Log checkpoint info if metrics provided
        if metrics and hasattr(self.logger, 'save_checkpoint_info'):
            self.logger.save_checkpoint_info(ep, total_it, file_name, metrics)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        self.vq_model.load_state_dict(checkpoint['vq_model'])
        self.opt_vq_model.load_state_dict(checkpoint['opt_vq_model'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval=None):
        self.vq_model.to(self.device)
        
        # Watch model for gradient/weight logging if enabled
        if hasattr(self.logger, 'watch_model'):
            self.logger.watch_model(self.vq_model)
        
        # Log dataset information
        if hasattr(self.logger, 'log_dataset_info'):
            motion_length_stats = {
                'mean': self.opt.window_size,  # This is the training window size
                'dataset': self.opt.dataset_name
            }
            self.logger.log_dataset_info(len(train_loader.dataset), len(val_loader.dataset), motion_length_stats)

        self.opt_vq_model = optim.AdamW(self.vq_model.parameters(), lr=self.opt.lr, betas=(0.9, 0.99), weight_decay=self.opt.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_vq_model, milestones=self.opt.milestones, gamma=self.opt.gamma)

        epoch = 0
        it = 0
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)
            print("Load model epoch:%d iterations:%d"%(epoch, it))

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader))) #????
        # val_loss = 0
        # min_val_loss = np.inf
        # min_val_epoch = epoch
        current_lr = self.opt.lr
        logs = defaultdict(def_value, OrderedDict())

        # sys.exit()
        # Initialize best metrics
        best_fid = 1000
        best_div = 100
        best_top1 = 0
        best_top2 = 0
        best_top3 = 0
        best_matching = 100
        best_recon = float('inf')
        best_smoothness = float('inf')
        best_position_error = float('inf')
        best_orientation_error = float('inf')
        
        # if self.opt.eval_on:
        is_camera_dataset = any(name in self.opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
        use_clatr = is_camera_dataset and hasattr(eval_wrapper, 'traj_encoder')  # CLaTrEvalWrapper
        if is_camera_dataset:
            eval_fn = evaluation_camera_vqvae_clatr if use_clatr else evaluation_camera_vqvae
            # Use camera-specific evaluation (now includes FID and other motion metrics)
            _clatr_extra = dict(vis_vel_integration=True) if use_clatr else {}
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_recon, best_smoothness, best_position_error, best_orientation_error, writer = eval_fn(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, 
                best_recon=best_recon, best_smoothness=best_smoothness,
                best_position_error=best_position_error, best_orientation_error=best_orientation_error,
                best_fid=best_fid, best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, save=False, **_clatr_extra)
        else:
            # Use original evaluation for human motion
            best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                best_div=best_div, best_top1=best_top1,
                best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                eval_wrapper=eval_wrapper, save=False)

        while epoch < self.opt.max_epoch:
            self.vq_model.train()
            for i, batch_data in tqdm(enumerate(train_loader), desc = f"[Train RVQ epoch {epoch}/{self.opt.max_epoch}]", total=len(train_loader)):
                it += 1
                if it < self.opt.warm_up_iter:
                    current_lr = self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                loss, loss_rec, loss_vel, loss_commit, perplexity, loss_smoothness, loss_orth = self.forward(batch_data, step=it)
                self.opt_vq_model.zero_grad()
                loss.backward()
                
                # Add gradient clipping to prevent NaN from gradient explosions
                clip_grad_norm_(self.vq_model.parameters(), max_norm=1.0)
                
                self.opt_vq_model.step()

                if it >= self.opt.warm_up_iter:
                    self.scheduler.step()
                
                logs['loss'] += loss.item()
                logs['loss_rec'] += loss_rec.item()
                # Note it not necessarily velocity, too lazy to change the name now
                logs['loss_vel'] += loss_vel.item()
                logs['loss_commit'] += loss_commit.item()
                logs['perplexity'] += perplexity.item()
                logs['lr'] += self.opt_vq_model.param_groups[0]['lr']
                logs['loss_smoothness'] += loss_smoothness.item() if isinstance(loss_smoothness, torch.Tensor) else loss_smoothness
                logs['loss_orth'] += loss_orth.item() if isinstance(loss_orth, torch.Tensor) else loss_orth

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    # Normalize accumulated logs
                    for tag, value in logs.items():
                        mean_loss[tag] = value / self.opt.log_every
                    
                    # Log training metrics
                    lr = self.opt_vq_model.param_groups[0]['lr']
                    self.logger.log_training_metrics(mean_loss, lr, it)
                    
                    # Log additional metrics if enabled
                    if self.logger.should_log_gradients:
                        self.logger.log_gradients(self.vq_model, it)
                    
                    if self.logger.should_log_codebook_usage:
                        self.logger.log_codebook_usage(self.vq_model.quantizer, it)
                    
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            # Calculate epoch metrics for checkpoint saving
            epoch_metrics = {
                'epoch': epoch,
                'iteration': it,
                'train_loss': logs.get('loss', 0) / max(1, len(train_loader)),
                'lr': self.opt_vq_model.param_groups[0]['lr']
            }
            
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it, epoch_metrics)

            epoch += 1
            # if epoch % self.opt.save_every_e == 0:
            #     self.save(pjoin(self.opt.model_dir, 'E%04d.tar' % (epoch)), epoch, total_it=it)

            print('Validation time:')
            self.vq_model.eval()
            val_loss_rec = []
            val_loss_vel = []
            val_loss_commit = []
            val_loss = []
            val_perpexity = []
            val_loss_smoothness = []
            val_loss_orth = []
            with torch.no_grad():
                for i, batch_data in tqdm(enumerate(val_loader), desc = f"[Validation RVQ epoch {epoch}/{self.opt.max_epoch}]", total=len(val_loader)):
                    loss, loss_rec, loss_vel, loss_commit, perplexity, loss_smoothness, loss_orth = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_loss_rec.append(loss_rec.item())
                    val_loss_vel.append(loss_vel.item())
                    val_loss_commit.append(loss_commit.item())
                    val_perpexity.append(perplexity.item())
                    val_loss_smoothness.append(loss_smoothness.item() if isinstance(loss_smoothness, torch.Tensor) else loss_smoothness)
                    val_loss_orth.append(loss_orth.item() if isinstance(loss_orth, torch.Tensor) else loss_orth)

            # Calculate validation metrics
            val_metrics = {
                'loss': sum(val_loss) / len(val_loss),
                'loss_rec': sum(val_loss_rec) / len(val_loss_rec),
                'loss_explicit': sum(val_loss_vel) / len(val_loss_vel),
                'loss_commit': sum(val_loss_commit) / len(val_loss_commit),
                'perplexity': sum(val_perpexity) / len(val_perpexity),
                'loss_smoothness': sum(val_loss_smoothness) / len(val_loss_smoothness) if val_loss_smoothness else 0.0,
                'loss_orth': sum(val_loss_orth) / len(val_loss_orth) if val_loss_orth else 0.0
            }
            
            # Log validation metrics
            self.logger.log_validation_metrics(val_metrics, epoch)

            # Print validation metrics
            smoothness_str = f", Smoothness: {sum(val_loss_smoothness)/len(val_loss_smoothness):.5f}" if val_loss_smoothness and any(val_loss_smoothness) else ""
            print(f'Validation Loss: {sum(val_loss)/len(val_loss):.5f} Reconstruction: {sum(val_loss_rec)/len(val_loss):.5f}, '
                  f'Velocity: {sum(val_loss_vel)/len(val_loss):.5f}, Commit: {sum(val_loss_commit)/len(val_loss):.5f}{smoothness_str}')

            # if sum(val_loss) / len(val_loss) < min_val_loss:
            #     min_val_loss = sum(val_loss) / len(val_loss)
            # # if sum(val_loss_vel) / len(val_loss_vel) < min_val_loss:
            # #     min_val_loss = sum(val_loss_vel) / len(val_loss_vel)
            #     min_val_epoch = epoch
            #     self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
            #     print('Best Validation Model So Far!~')
            # if self.opt.eval_on:
            is_camera_dataset = any(name in self.opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
            use_clatr = is_camera_dataset and hasattr(eval_wrapper, 'traj_encoder')  # CLaTrEvalWrapper
            if is_camera_dataset:
                eval_fn = evaluation_camera_vqvae_clatr if use_clatr else evaluation_camera_vqvae
                # Use camera-specific evaluation (now includes FID and other motion metrics)
                _clatr_extra = dict(vis_vel_integration=True) if use_clatr else {}
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_recon, best_smoothness, best_position_error, best_orientation_error, writer = eval_fn(
                    self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, 
                    best_recon=best_recon, best_smoothness=best_smoothness,
                    best_position_error=best_position_error, best_orientation_error=best_orientation_error,
                    best_fid=best_fid, best_div=best_div, best_top1=best_top1,
                    best_top2=best_top2, best_top3=best_top3, best_matching=best_matching,
                    eval_wrapper=eval_wrapper, **_clatr_extra)
            else:
                # Use original evaluation for human motion
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_vqvae(
                    self.opt.model_dir, eval_val_loader, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1,
                    best_top2=best_top2, best_top3=best_top3, best_matching=best_matching, eval_wrapper=eval_wrapper)


            # if self.opt.eval_on and epoch % self.opt.eval_every_e == 0:
            n_vis = getattr(self.opt, 'num_vis_samples', 4)
            data = torch.cat([self.motions[:n_vis], self.pred_motion[:n_vis]], dim=0).detach().cpu().numpy()
            # np.save(pjoin(self.opt.eval_dir, 'E%04d.npy' % (epoch)), data)
            save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
            os.makedirs(save_dir, exist_ok=True)
            if plot_eval is not None:
                plot_eval(data, save_dir)
                # if plot_eval is not None:
                #     save_dir = pjoin(self.opt.eval_dir, 'E%04d' % (epoch))
                #     os.makedirs(save_dir, exist_ok=True)
                #     plot_eval(data, save_dir)

            # if epoch - min_val_epoch >= self.opt.early_stop_e:
            #     print('Early Stopping!~')


class LengthEstTrainer(object):

    def __init__(self, args, estimator, text_encoder, encode_fnc):
        self.opt = args
        self.estimator = estimator
        self.text_encoder = text_encoder
        self.encode_fnc = encode_fnc
        self.device = args.device

        if args.is_train:
            # self.motion_dis
            try:
                from torch.utils.tensorboard import SummaryWriter  # type: ignore
            except ImportError:
                from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore
            self.logger = SummaryWriter(args.log_dir)
            self.mul_cls_criterion = nn.CrossEntropyLoss()

    def resume(self, model_dir):
        checkpoints = torch.load(model_dir, map_location=self.device)
        self.estimator.load_state_dict(checkpoints['estimator'])
        # self.opt_estimator.load_state_dict(checkpoints['opt_estimator'])
        return checkpoints['epoch'], checkpoints['iter']

    def save(self, model_dir, epoch, niter):
        state = {
            'estimator': self.estimator.state_dict(),
            # 'opt_estimator': self.opt_estimator.state_dict(),
            'epoch': epoch,
            'niter': niter,
        }
        torch.save(state, model_dir)

    @staticmethod
    def zero_grad(opt_list):
        for opt in opt_list:
            opt.zero_grad()

    @staticmethod
    def clip_norm(network_list):
        for network in network_list:
            clip_grad_norm_(network.parameters(), 0.5)

    @staticmethod
    def step(opt_list):
        for opt in opt_list:
            opt.step()

    def train(self, train_dataloader, val_dataloader):
        self.estimator.to(self.device)
        self.text_encoder.to(self.device)

        self.opt_estimator = optim.Adam(self.estimator.parameters(), lr=self.opt.lr)

        epoch = 0
        it = 0

        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            epoch, it = self.resume(model_dir)

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_dataloader)
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_dataloader), len(val_dataloader)))
        val_loss = 0
        min_val_loss = np.inf
        logs = defaultdict(float)
        while epoch < self.opt.max_epoch:
            # time0 = time.time()
            for i, batch_data in enumerate(train_dataloader):
                self.estimator.train()

                conds, _, m_lens = batch_data
                # word_emb = word_emb.detach().to(self.device).float()
                # pos_ohot = pos_ohot.detach().to(self.device).float()
                # m_lens = m_lens.to(self.device).long()
                text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device).detach()
                # print(text_embs.shape, text_embs.device)

                pred_dis = self.estimator(text_embs)

                self.zero_grad([self.opt_estimator])

                gt_labels = m_lens // self.opt.unit_length
                gt_labels = gt_labels.long().to(self.device)
                # print(gt_labels.shape, pred_dis.shape)
                # print(gt_labels.max(), gt_labels.min())
                # print(pred_dis)
                acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)
                loss = self.mul_cls_criterion(pred_dis, gt_labels)

                loss.backward()

                self.clip_norm([self.estimator])
                self.step([self.opt_estimator])

                logs['loss'] += loss.item()
                logs['acc'] += acc.item()

                it += 1
                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict({'val_loss': val_loss})
                    # self.logger.add_scalar('Val/loss', val_loss, it)

                    for tag, value in logs.items():
                        self.logger.add_scalar("Train/%s"%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(float)
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                    if it % self.opt.save_latest == 0:
                        self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1

            print('Validation time:')

            val_loss = 0
            val_acc = 0
            # self.estimator.eval()
            with torch.no_grad():
                for i, batch_data in enumerate(val_dataloader):
                    self.estimator.eval()

                    conds, _, m_lens = batch_data
                    # word_emb = word_emb.detach().to(self.device).float()
                    # pos_ohot = pos_ohot.detach().to(self.device).float()
                    # m_lens = m_lens.to(self.device).long()
                    text_embs = self.encode_fnc(self.text_encoder, conds, self.opt.device)
                    pred_dis = self.estimator(text_embs)

                    gt_labels = m_lens // self.opt.unit_length
                    gt_labels = gt_labels.long().to(self.device)
                    loss = self.mul_cls_criterion(pred_dis, gt_labels)
                    acc = (gt_labels == pred_dis.argmax(dim=-1)).sum() / len(gt_labels)

                    val_loss += loss.item()
                    val_acc += acc.item()


            val_loss = val_loss / len(val_dataloader)
            val_acc = val_acc / len(val_dataloader)
            print('Validation Loss: %.5f Validation Acc: %.5f' % (val_loss, val_acc))

            if val_loss < min_val_loss:
                self.save(pjoin(self.opt.model_dir, 'finest.tar'), epoch, it)
                min_val_loss = val_loss
