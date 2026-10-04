# This evaluation is deprecated, use the new clatr evaluation

import numpy as np
import torch
import torch.nn.functional as F
from tkcam.utils.camera_process import calculate_camera_metrics
from tkcam.utils.unified_data_format import detect_format_from_dataset_name
from tkcam.utils.metrics import calculate_R_precision, euclidean_distance_matrix, calculate_activation_statistics, calculate_frechet_distance, calculate_diversity, calculate_multimodality
import os
from os.path import join as pjoin
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import io
from PIL import Image

def log_trajectory_to_tensorboard(writer, gt_trajectory, pred_trajectory, caption, epoch, batch_idx=0):
    """
    Log trajectory comparison plots to tensorboard
    
    Args:
        writer: TensorBoard writer
        gt_trajectory: Ground truth trajectory (T, 5) - [x, y, z, pitch, yaw]
        pred_trajectory: Predicted trajectory (T, 5) 
        caption: Text description
        epoch: Current epoch
        batch_idx: Batch index for multiple trajectories
    """
    try:
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Extract positions and orientations
        gt_pos = gt_trajectory[:, :3]  # x, y, z
        pred_pos = pred_trajectory[:, :3]
        gt_orient = gt_trajectory[:, 3:]  # pitch, yaw
        pred_orient = pred_trajectory[:, 3:]
        
        # 1. 3D trajectory plot
        ax1.remove()
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', linewidth=2, label='Ground Truth')
        ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r--', linewidth=2, label='Predicted')
        ax1.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c='green', s=50, label='Start')
        ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], c='red', s=50, label='End')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title('3D Trajectory')
        ax1.legend()
        
        # 2. Top-down view (X-Y plane)
        ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', linewidth=2, label='Ground Truth')
        ax2.plot(pred_pos[:, 0], pred_pos[:, 1], 'r--', linewidth=2, label='Predicted')
        ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=50, label='Start')
        ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='red', s=50, label='End')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Top-Down View (X-Y)')
        ax2.legend()
        ax2.grid(True)
        
        # 3. Position error over time
        pos_error = np.linalg.norm(gt_pos - pred_pos, axis=1)
        ax3.plot(pos_error, 'g-', linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Position Error')
        ax3.set_title('Position Error Over Time')
        ax3.grid(True)
        
        # 4. Orientation comparison
        time_steps = np.arange(len(gt_orient))
        ax4.plot(time_steps, gt_orient[:, 0], 'b-', label='GT Pitch')
        ax4.plot(time_steps, pred_orient[:, 0], 'r--', label='Pred Pitch')
        ax4.plot(time_steps, gt_orient[:, 1], 'b:', label='GT Yaw')
        ax4.plot(time_steps, pred_orient[:, 1], 'r:', label='Pred Yaw')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Angle (radians)')
        ax4.set_title('Orientation Comparison')
        ax4.legend()
        ax4.grid(True)
        
        # Add caption as suptitle
        fig.suptitle(f'Trajectory Comparison - Epoch {epoch}\nCaption: {caption}', fontsize=12, wrap=True)
        
        # Convert plot to image and log to tensorboard
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to PIL Image and then to tensor
        image = Image.open(buf)
        image_array = np.array(image)
        # Convert to CHW format for tensorboard
        if len(image_array.shape) == 3:
            image_array = image_array.transpose(2, 0, 1)
        
        # Log to tensorboard
        writer.add_image(f'Trajectories/Batch_{batch_idx}', image_array, epoch)
        
        plt.close(fig)
        buf.close()
        
    except Exception as e:
        print(f"Error logging trajectory to tensorboard: {e}")

@torch.no_grad()
def evaluation_camera_vqvae(out_dir, val_loader, net, writer, ep, best_recon, best_smoothness, 
                           best_position_error, best_orientation_error, eval_wrapper, 
                           best_fid=float('inf'), best_div=float('inf'), best_top1=0, 
                           best_top2=0, best_top3=0, best_matching=float('inf'), 
                           save=True, draw=True):
    """
    Evaluate camera VQ-VAE model with camera-specific metrics and standard motion metrics
    """
    net.eval()
    
    print(f"Starting evaluation with {len(val_loader)} batches in val_loader")
    
    # Format type for metrics (rotmat vs euler)
    fmt_type = None
    if hasattr(val_loader.dataset, 'opt') and hasattr(val_loader.dataset.opt, 'dataset_name'):
        fmt_type = detect_format_from_dataset_name(val_loader.dataset.opt.dataset_name)
    
    # Camera-specific metrics
    total_recon_loss = 0
    total_position_error = 0
    total_orientation_error = 0
    total_smoothness = 0
    total_velocity_error = 0
    
    # Standard motion metrics
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    
    nb_sample = 0
    
    # For trajectory logging
    trajectory_log_count = 0
    max_trajectory_logs = 5  # Log first 5 trajectories per epoch
    
    batch_count = 0
    for batch in val_loader:
        batch_count += 1
        # print(f"Processing batch {batch_count}, batch type: {type(batch)}")
        
        # Handle different batch formats - VQ training vs text-to-motion evaluation
        if isinstance(batch, torch.Tensor):
            # VQ training format: just motion tensor
            motion = batch.cuda()
            bs, seq = motion.shape[0], motion.shape[1]
            m_length = [seq] * bs  # All sequences have full length
            
            # Skip text-based evaluation metrics for VQ training
            skip_text_metrics = True
            word_embeddings = pos_one_hots = caption = sent_len = token = None
        else:
            # Text-to-motion format: full batch with text data
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
            motion = motion.cuda()
            bs, seq = motion.shape[0], motion.shape[1]
            skip_text_metrics = False
        
        # Forward pass
        pred_motion, loss_commit, perplexity = net(motion)
        
        if not skip_text_metrics:
            # Get co-embeddings for standard motion metrics (only for text-to-motion)
            et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
            
            # Add to motion lists for FID/diversity calculation
            motion_pred_list.append(em_pred)
            motion_annotation_list.append(em)
        
        if not skip_text_metrics:
            # Calculate R-precision and matching score (only for text-to-motion)
            temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
            R_precision_real += temp_R
            matching_score_real += temp_match
            temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
            temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
            R_precision += temp_R
            matching_score_pred += temp_match
        
        # Convert to numpy for metric calculation
        pred_np = pred_motion.detach().cpu().numpy()
        gt_np = motion.detach().cpu().numpy()
        
        # Calculate camera-specific metrics and log trajectories
        for i in range(bs):
            pred_data = pred_np[i, :m_length[i]]
            gt_data = gt_np[i, :m_length[i]]
            
            metrics = calculate_camera_metrics(pred_data[None, ...], gt_data[None, ...], format_type=fmt_type)
            
            total_position_error += metrics['mean_position_error']
            total_orientation_error += metrics['mean_orientation_error']
            total_smoothness += metrics['pred_smoothness']
            total_velocity_error += metrics['velocity_error']
            
            # Log trajectory to tensorboard (limit to first few per epoch), disabled, something wrong with add_image
            # if trajectory_log_count < max_trajectory_logs and draw:
            #     caption_text = "VQ Training Sample" if skip_text_metrics else (caption[i] if i < len(caption) else "No caption")
            #     log_trajectory_to_tensorboard(
            #         writer, 
            #         gt_data, 
            #         pred_data, 
            #         caption_text,
            #         ep, 
            #         trajectory_log_count
            #     )
            #     trajectory_log_count += 1
        
        # Reconstruction loss
        recon_loss = F.l1_loss(pred_motion, motion)
        total_recon_loss += recon_loss.item()
        
        nb_sample += bs
    
    # Calculate standard motion metrics (only if we have text-based data)
    if motion_annotation_list and motion_pred_list:
        motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
        motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
        gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
        mu, cov = calculate_activation_statistics(motion_pred_np)
        
        diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
        diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
        
        R_precision_real = R_precision_real / nb_sample
        R_precision = R_precision / nb_sample
        matching_score_real = matching_score_real / nb_sample
        matching_score_pred = matching_score_pred / nb_sample
    else:
        # For VQ training without text data, set default values
        gt_mu = mu = np.zeros(512)  # Default embedding size
        gt_cov = cov = np.eye(512)
        diversity_real = diversity = 0.0
        R_precision_real = R_precision = 0.0
        matching_score_real = matching_score_pred = 0.0
    
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    
    # Average camera-specific metrics (with safety check for division by zero)
    if nb_sample > 0:
        avg_recon_loss = total_recon_loss / nb_sample
        avg_position_error = total_position_error / nb_sample
        avg_orientation_error = total_orientation_error / nb_sample
        avg_smoothness = total_smoothness / nb_sample
        avg_velocity_error = total_velocity_error / nb_sample
    else:
        print("Warning: No samples processed in validation, setting metrics to 0")
        avg_recon_loss = 0.0
        avg_position_error = 0.0
        avg_orientation_error = 0.0
        avg_smoothness = 0.0
        avg_velocity_error = 0.0
    
    # Ensure R_precision is an array
    if not isinstance(R_precision, np.ndarray):
        R_precision = np.array([R_precision, R_precision, R_precision])
    if not isinstance(R_precision_real, np.ndarray):
        R_precision_real = np.array([R_precision_real, R_precision_real, R_precision_real])
    
    # Add warning message about R-precision accuracy for camera datasets
    r_precision_warning = " (WARNING: R-precision may be inaccurate for camera datasets due to randomly initialized eval_wrapper)"
    
    msg = f"--> Eva. Ep {ep}: FID: {fid:.4f}, Diversity: {diversity:.4f}, " \
          f"R_precision: ({R_precision[0]:.4f}, {R_precision[1]:.4f}, {R_precision[2]:.4f}), " \
          f"Matching: {matching_score_pred:.4f}, " \
          f"Recon: {avg_recon_loss:.4f}, Pos_Error: {avg_position_error:.4f}, " \
          f"Ori_Error: {avg_orientation_error:.4f}, Smoothness: {avg_smoothness:.4f}, " \
          f"Vel_Error: {avg_velocity_error:.4f}"
    print(msg)
    
    if draw:
        # Standard motion metrics
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        
        # Camera-specific metrics
        writer.add_scalar('./Test/Reconstruction_Loss', avg_recon_loss, ep)
        writer.add_scalar('./Test/Position_Error', avg_position_error, ep)
        writer.add_scalar('./Test/Orientation_Error', avg_orientation_error, ep)
        writer.add_scalar('./Test/Smoothness', avg_smoothness, ep)
        writer.add_scalar('./Test/Velocity_Error', avg_velocity_error, ep)
        
        # Log trajectory count
        writer.add_scalar('./Test/Logged_Trajectories', trajectory_log_count, ep)
    
    # Save best models - Standard motion metrics
    if fid < best_fid:
        msg = f"--> --> FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        if draw: print(msg)
        best_fid = fid
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_fid.tar'))
    
    if abs(diversity_real - diversity) < abs(diversity_real - best_div):
        msg = f"--> --> Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        if draw: print(msg)
        best_div = diversity
    
    if R_precision[0] > best_top1:
        msg = f"--> --> Top1 Improved from {best_top1:.5f} to {R_precision[0]:.5f} !!!"
        if draw: print(msg)
        best_top1 = R_precision[0]
    
    if R_precision[1] > best_top2:
        msg = f"--> --> Top2 Improved from {best_top2:.5f} to {R_precision[1]:.5f} !!!"
        if draw: print(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3:
        msg = f"--> --> Top3 Improved from {best_top3:.5f} to {R_precision[2]:.5f} !!!"
        if draw: print(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching:
        msg = f"--> --> Matching Score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        if draw: print(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_matching.tar'))
    
    # Save best models - Camera-specific metrics
    if avg_recon_loss < best_recon:
        msg = f"--> --> Reconstruction Loss Improved from {best_recon:.5f} to {avg_recon_loss:.5f} !!!"
        if draw: print(msg)
        best_recon = avg_recon_loss
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_recon.tar'))
    
    if avg_position_error < best_position_error:
        msg = f"--> --> Position Error Improved from {best_position_error:.5f} to {avg_position_error:.5f} !!!"
        if draw: print(msg)
        best_position_error = avg_position_error
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_position.tar'))
    
    if avg_orientation_error < best_orientation_error:
        msg = f"--> --> Orientation Error Improved from {best_orientation_error:.5f} to {avg_orientation_error:.5f} !!!"
        if draw: print(msg)
        best_orientation_error = avg_orientation_error
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_orientation.tar'))
    
    if avg_smoothness < best_smoothness:
        msg = f"--> --> Smoothness Improved from {best_smoothness:.5f} to {avg_smoothness:.5f} !!!"
        if draw: print(msg)
        best_smoothness = avg_smoothness
        if save:
            torch.save({'vq_model': net.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_smoothness.tar'))
    
    net.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, best_recon, best_smoothness, best_position_error, best_orientation_error, writer

@torch.no_grad()
def evaluation_camera_transformer(out_dir, val_loader, trans, vq_model, writer, ep, 
                                 best_fid, best_div, best_top1, best_top2, best_top3, 
                                 best_matching, best_position_error, eval_wrapper, plot_func, save_ckpt=False, 
                                 save_anim=False, cond_scale=3, temperature=1, topkr=0.9):
    """
    Evaluate camera transformer model with camera-specific metrics and standard motion metrics
    """
    trans.eval()
    vq_model.eval()
    
    # Format type for metrics (rotmat vs euler)
    fmt_type = None
    if hasattr(val_loader.dataset, 'opt') and hasattr(val_loader.dataset.opt, 'dataset_name'):
        fmt_type = detect_format_from_dataset_name(val_loader.dataset.opt.dataset_name)
    
    # Standard motion metrics
    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0
    
    # Camera-specific metrics
    total_position_error = 0
    total_orientation_error = 0
    total_smoothness = 0
    
    nb_sample = 0
    
    # For trajectory logging
    trajectory_log_count = 0
    max_trajectory_logs = 5  # Log first 5 trajectories per epoch
    
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = batch
        motion = motion.cuda()
        
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]
        
        # Generate motion tokens
        token_lens = m_length // 4
        mids = trans.generate(caption, token_lens, timesteps=10, cond_scale=cond_scale, 
                             temperature=temperature, topk_filter_thres=topkr, gsample=True)
        
        # Decode to motion
        pred_motion = vq_model.forward_decoder(mids)
        
        # Get embeddings for evaluation
        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_motion, m_length)
        
        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
        
        # Calculate R-precision and matching score
        temp_R = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et.cpu().numpy(), em.cpu().numpy()).trace()
        R_precision_real += temp_R
        matching_score_real += temp_match
        
        temp_R = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        temp_match = euclidean_distance_matrix(et_pred.cpu().numpy(), em_pred.cpu().numpy()).trace()
        R_precision += temp_R
        matching_score_pred += temp_match
        
        # Calculate camera-specific metrics and log trajectories
        pred_np = pred_motion.detach().cpu().numpy()
        gt_np = motion.detach().cpu().numpy()
        
        for i in range(bs):
            pred_data = pred_np[i, :m_length[i]]
            gt_data = gt_np[i, :m_length[i]]
            
            metrics = calculate_camera_metrics(pred_data[None, ...], gt_data[None, ...], format_type=fmt_type)
            total_position_error += metrics['mean_position_error']
            total_orientation_error += metrics['mean_orientation_error']
            total_smoothness += metrics['pred_smoothness']
            
            # Log trajectory to tensorboard (limit to first few per epoch)
            if trajectory_log_count < max_trajectory_logs and writer is not None:
                log_trajectory_to_tensorboard(
                    writer, 
                    gt_data, 
                    pred_data, 
                    caption[i] if i < len(caption) else "No caption",
                    ep, 
                    trajectory_log_count
                )
                trajectory_log_count += 1
        
        nb_sample += bs
    
    # Calculate standard motion metrics
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    
    # Calculate FID and diversity
    gt_mu, gt_cov = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)
    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)
    
    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample
    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample
    
    # Ensure R_precision is an array
    if not isinstance(R_precision, np.ndarray):
        R_precision = np.array([R_precision, R_precision, R_precision])
    if not isinstance(R_precision_real, np.ndarray):
        R_precision_real = np.array([R_precision_real, R_precision_real, R_precision_real])
    
    avg_position_error = total_position_error / nb_sample
    avg_orientation_error = total_orientation_error / nb_sample
    avg_smoothness = total_smoothness / nb_sample
    
    # Add warning message about R-precision accuracy for camera datasets
    r_precision_warning = " (WARNING: R-precision may be inaccurate for camera datasets due to randomly initialized eval_wrapper)"
    
    msg = f"--> Eva. Ep {ep}: FID: {fid:.4f}, Diversity: {diversity:.4f}, " \
          f"R_precision: ({R_precision[0]:.4f}, {R_precision[1]:.4f}, {R_precision[2]:.4f}){r_precision_warning}, " \
          f"Matching: {matching_score_pred:.4f}, " \
          f"Pos_Error: {avg_position_error:.4f}, Ori_Error: {avg_orientation_error:.4f}, " \
          f"Smoothness: {avg_smoothness:.4f}"
    print(msg)
    
    if writer is not None:
        # Standard motion metrics
        writer.add_scalar('./Test/FID', fid, ep)
        writer.add_scalar('./Test/Diversity', diversity, ep)
        writer.add_scalar('./Test/Position_Error', avg_position_error, ep)
        writer.add_scalar('./Test/Orientation_Error', avg_orientation_error, ep)
        writer.add_scalar('./Test/Smoothness', avg_smoothness, ep)
        writer.add_scalar('./Test/top1', R_precision[0], ep)
        writer.add_scalar('./Test/top2', R_precision[1], ep)
        writer.add_scalar('./Test/top3', R_precision[2], ep)
        writer.add_scalar('./Test/matching_score', matching_score_pred, ep)
        
        # Log trajectory count
        writer.add_scalar('./Test/Logged_Trajectories', trajectory_log_count, ep)
    
    # Save best models
    if fid < best_fid:
        msg = f"--> --> FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        print(msg)
        best_fid = fid
        if save_ckpt:
            torch.save({'trans': trans.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_fid.tar'))
    
    if avg_position_error < best_position_error:
        msg = f"--> --> Position Error Improved from {best_position_error:.5f} to {avg_position_error:.5f} !!!"
        print(msg)
        best_position_error = avg_position_error
        if save_ckpt:
            torch.save({'trans': trans.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_position.tar'))
    
    if R_precision[0] > best_top1:
        msg = f"--> --> Top1 Improved from {best_top1:.5f} to {R_precision[0]:.5f} !!!"
        print(msg)
        best_top1 = R_precision[0]
    
    if R_precision[1] > best_top2:
        msg = f"--> --> Top2 Improved from {best_top2:.5f} to {R_precision[1]:.5f} !!!"
        print(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3:
        msg = f"--> --> Top3 Improved from {best_top3:.5f} to {R_precision[2]:.5f} !!!"
        print(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching:
        msg = f"--> --> Matching Score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        print(msg)
        best_matching = matching_score_pred
        if save_ckpt:
            torch.save({'trans': trans.state_dict(), 'ep': ep}, 
                      os.path.join(out_dir, 'net_best_mm.tar'))
    
    trans.train()
    vq_model.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer 