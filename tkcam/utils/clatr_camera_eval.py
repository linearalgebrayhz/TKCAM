"""
CLaTr-based Camera Evaluation Functions.

Drop-in replacements for the functions in utils/camera_eval.py,
using the pre-trained CLaTr evaluator instead of the legacy
GloVe+BiGRU evaluator.

These functions compute:
  - FID (from trajectory encoder latents)
  - R-Precision (Top-1/2/3)
  - Matching Score (avg cosine similarity)
  - Diversity
  - Camera-specific metrics (position error, orientation error, smoothness)
"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from os.path import join as pjoin
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – needed for 3-D projection

from tkcam.utils.camera_process import calculate_camera_metrics
from tkcam.utils.unified_data_format import UnifiedCameraData, CameraDataFormat, detect_format_from_dataset_name
from tkcam.utils.camera_geometry import forward_from_sixd, sixd_to_matrix, to_mpl
from tkcam.models.evaluator.clatr_models import (
    CLaTrEvalWrapper,
    compute_fid,
    compute_r_precision,
    compute_matching_score,
    compute_diversity,
)


# ---------------------------------------------------------------------------
#  Trajectory comparison plotting
# ---------------------------------------------------------------------------

def _positions_from_trajectory(traj: np.ndarray) -> np.ndarray:
    """Extract (T, 3) positions from a trajectory of arbitrary dim."""
    return traj[:, :3]

# already implemented in camera_geometry.py
def _to_mpl(v: np.ndarray) -> np.ndarray:
    """OpenGL [x, y, z] -> Matplotlib Z-up [x, -z, y]."""
    return np.stack([v[..., 0], -v[..., 2], v[..., 1]], axis=-1)


def _positions_from_vel_integration(
    traj: np.ndarray,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    smooth: bool = True,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """Denormalise *traj* (if mean/std given), then integrate velocity channels.

    Feature layout assumed: [x, y, z, dx, dy, dz, rot6d(6)].

    Returns:
        (T, 3) positions in OpenGL frame (not yet mapped to MPL).
        Returns None if the trajectory has fewer than 6 channels.
    """
    from tkcam.utils.camera_geometry import integrate_velocity_to_positions
    raw = _inv_transform(traj, mean, std) if (mean is not None and std is not None) else traj
    if raw.shape[1] < 6:
        return None
    return integrate_velocity_to_positions(raw, smooth=smooth, smooth_sigma=smooth_sigma)

# ? Check mean std file load logic.
def _inv_transform(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Convert from normalised feature space back to raw feature space."""
    return data * std + mean

# reviewed 
def _compute_orientation_vectors(traj: np.ndarray, format_type=None) -> np.ndarray:
    """Compute camera forward direction vectors for each frame in MPL convention.

    Returns:
        (T, 3) array of unit forward vectors in Matplotlib [x, depth, z] convention.
    """
    unified = UnifiedCameraData(traj, format_type=format_type)
    positions = unified.positions.numpy()
    orientations = unified.orientations.numpy()
    T = len(positions)
    fwd_mpl = np.zeros((T, 3))

    for t in range(T):
        ori = orientations[t]
        if unified.format_type == CameraDataFormat.QUATERNION_10:
            from tkcam.common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified.format_type == CameraDataFormat.FULL_12_ROTMAT:
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            pitch, yaw = ori[0], ori[1]
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),
                -np.sin(pitch),
                -np.cos(pitch) * np.cos(yaw)
            ])
        fwd_mpl[t] = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        norm = np.linalg.norm(fwd_mpl[t])
        if norm > 1e-6:
            fwd_mpl[t] /= norm
    return fwd_mpl


def plot_trajectory_comparison(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    caption: str,
    save_path: str,
    epoch: int = 0,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    format_type=None,
):
    """Save a multi-panel PNG comparing GT and predicted camera trajectories.

    Panels:
      1. 3-D overlay of GT (blue) vs Pred (red) with start/end markers and camera direction arrows
      2. Top-down view (X vs Depth) with direction arrows
      3. Per-frame position error
      4. Per-axis position over time

    Parameters
    ----------
    gt_traj   : (T, D) ground-truth trajectory — normalised feature space
    pred_traj : (T, D) predicted trajectory    — normalised feature space
    caption   : text prompt
    save_path : output .png path
    epoch     : current training epoch (shown in title)
    mean      : (D,) normalisation mean; if provided trajectories are converted
                to raw feature space before plotting
    std       : (D,) normalisation std; must be provided together with mean
    format_type : CameraDataFormat for orientation extraction (e.g. FULL_12_ROTMAT)
    """
    if mean is not None and std is not None:
        gt_traj  = _inv_transform(gt_traj,  mean, std)
        pred_traj = _inv_transform(pred_traj, mean, std)
        space_label = 'raw units'
    else:
        space_label = 'normalised'

    gt_pos = _to_mpl(_positions_from_trajectory(gt_traj))
    pr_pos = _to_mpl(_positions_from_trajectory(pred_traj))

    # Compute camera forward direction vectors for orientation arrows
    gt_fwd = _compute_orientation_vectors(gt_traj, format_type)
    pr_fwd = _compute_orientation_vectors(pred_traj, format_type)

    # Arrow scale based on trajectory extent
    all_pos = np.concatenate([gt_pos, pr_pos], axis=0)
    arrow_scale = max(np.ptp(all_pos, axis=0).max() * 0.08, 1e-3)
    step = max(1, min(len(gt_pos), len(pr_pos)) // 8)  # ~8 arrows per trajectory

    fig = plt.figure(figsize=(16, 12))

    # ---- 1. 3-D overlay with orientation arrows ----
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(*gt_pos.T, 'b-', lw=2, label='GT')
    ax1.plot(*pr_pos.T, 'r--', lw=2, label='Pred')
    ax1.scatter(*gt_pos[0], c='green', s=80, marker='^', label='Start')
    ax1.scatter(*gt_pos[-1], c='orange', s=80, marker='v', label='End')
    for i in range(0, len(gt_pos), step):
        dx, dy, dz = gt_fwd[i]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax1.quiver(gt_pos[i, 0], gt_pos[i, 1], gt_pos[i, 2],
                       dx, dy, dz, length=arrow_scale, color='blue', alpha=0.7)
    for i in range(0, len(pr_pos), step):
        dx, dy, dz = pr_fwd[i]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax1.quiver(pr_pos[i, 0], pr_pos[i, 1], pr_pos[i, 2],
                       dx, dy, dz, length=arrow_scale, color='red', alpha=0.7)
    ax1.set_xlabel('X');  ax1.set_ylabel('Depth');  ax1.set_zlabel('Y')
    ax1.set_title('3-D Trajectory (arrows = camera direction)')
    ax1.legend(fontsize=8)

    # ---- 2. Top-down with orientation arrows ----
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', lw=2, label='GT')
    ax2.plot(pr_pos[:, 0], pr_pos[:, 1], 'r--', lw=2, label='Pred')
    ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=60, marker='^')
    ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='orange', s=60, marker='v')
    arrow_scale_2d = max(np.ptp(all_pos[:, :2], axis=0).max() * 0.08, 1e-3)
    for i in range(0, len(gt_pos), step):
        dx, dy = gt_fwd[i, 0], gt_fwd[i, 1]
        if np.sqrt(dx**2 + dy**2) > 1e-6:
            ax2.arrow(gt_pos[i, 0], gt_pos[i, 1], dx * arrow_scale_2d, dy * arrow_scale_2d,
                     head_width=arrow_scale_2d * 0.3, head_length=arrow_scale_2d * 0.2,
                     fc='blue', ec='blue', alpha=0.7)
    for i in range(0, len(pr_pos), step):
        dx, dy = pr_fwd[i, 0], pr_fwd[i, 1]
        if np.sqrt(dx**2 + dy**2) > 1e-6:
            ax2.arrow(pr_pos[i, 0], pr_pos[i, 1], dx * arrow_scale_2d, dy * arrow_scale_2d,
                     head_width=arrow_scale_2d * 0.3, head_length=arrow_scale_2d * 0.2,
                     fc='red', ec='red', alpha=0.7)
    ax2.set_xlabel('X');  ax2.set_ylabel('Depth')
    ax2.set_title('Top-Down (X vs Depth, arrows = camera direction)')
    ax2.legend(fontsize=8);  ax2.grid(True);  ax2.set_aspect('equal', 'datalim')

    # ---- 3. Position error ----
    min_len = min(len(gt_pos), len(pr_pos))
    pos_err = np.linalg.norm(gt_pos[:min_len] - pr_pos[:min_len], axis=1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pos_err, 'g-', lw=2)
    ax3.set_xlabel('Frame');  ax3.set_ylabel(f'L2 Position Error [{space_label}]')
    ax3.set_title(f'Position Error  (mean={pos_err.mean():.4f}  [{space_label}])')
    ax3.grid(True)

    # ---- 4. Per-axis position ----
    ax4 = fig.add_subplot(2, 2, 4)
    frames = np.arange(len(gt_pos))
    for dim, (name, ls) in enumerate(zip(['X', 'Depth', 'Y'], ['-', '--', ':'])):
        ax4.plot(frames, gt_pos[:, dim], ls, color='blue', lw=1.5,
                 label=f'GT {name}')
    frames_p = np.arange(len(pr_pos))
    for dim, (name, ls) in enumerate(zip(['X', 'Depth', 'Y'], ['-', '--', ':'])):
        ax4.plot(frames_p, pr_pos[:, dim], ls, color='red', lw=1.5,
                 label=f'Pred {name}')
    ax4.set_xlabel('Frame');  ax4.set_ylabel('Position')
    ax4.set_title('Per-Axis Position')
    ax4.legend(fontsize=7, ncol=2);  ax4.grid(True)

    # ---- Suptitle ----
    wrapped = '\n'.join(textwrap.wrap(caption, width=90))
    fig.suptitle(f'Epoch {epoch}  |  {wrapped}', fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory_comparison_animation(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    caption: str,
    save_path: str,
    epoch: int = 0,
    fps: int = 30,
    trail_length: int = 30,
    stride: int = 1,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    format_type=None,
):
    """Save a side-by-side MP4 animation comparing GT and predicted trajectories.

    Left panel  : Ground Truth (blue path, green trail, purple direction arrow)
    Right panel : Predicted    (red path, orange trail, purple direction arrow)
    Both panels show the camera pointing direction at the current frame.
    Both panels share the same axis limits for fair visual comparison.
    The caption is displayed as the figure suptitle.

    Parameters
    ----------
    gt_traj   : (T, D) ground-truth trajectory — normalised feature space
    pred_traj : (T, D) predicted trajectory    — normalised feature space
    caption   : text prompt
    save_path : output .mp4 path
    epoch     : current training epoch
    fps       : frames per second
    trail_length : number of recent frames highlighted
    stride    : render every N-th frame (speeds up rendering)
    mean      : (D,) normalisation mean; if provided, convert to raw space before plotting
    std       : (D,) normalisation std
    format_type : CameraDataFormat for orientation extraction (e.g. FULL_12_ROTMAT)
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    if mean is not None and std is not None:
        gt_traj   = _inv_transform(gt_traj,   mean, std)
        pred_traj = _inv_transform(pred_traj, mean, std)

    gt_pos = _to_mpl(_positions_from_trajectory(gt_traj))
    pr_pos = _to_mpl(_positions_from_trajectory(pred_traj))

    # Pre-compute camera forward direction vectors for orientation arrows
    gt_fwd = _compute_orientation_vectors(gt_traj, format_type)
    pr_fwd = _compute_orientation_vectors(pred_traj, format_type)

    # Shared axis limits and arrow scale
    all_pos = np.concatenate([gt_pos, pr_pos], axis=0)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    padding = max(np.ptp(all_pos, axis=0).max() * 0.12, 1e-3)
    arrow_scale = max(np.ptp(all_pos, axis=0).max() * 0.1, 1e-3)

    fig = plt.figure(figsize=(18, 8))
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pr = fig.add_subplot(1, 2, 2, projection='3d')

    for ax, label in [(ax_gt, 'Ground Truth'), (ax_pr, 'Predicted')]:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_zlim(mins[2] - padding, maxs[2] + padding)
        ax.set_xlabel('X');  ax.set_ylabel('Depth');  ax.set_zlabel('Y')
        ax.set_title(label, fontsize=13)

    wrapped = '\n'.join(textwrap.wrap(caption, width=100))
    fig.suptitle(f'Epoch {epoch}  |  {wrapped}', fontsize=10, y=0.98)

    # Static elements
    for ax, pos, clr in [(ax_gt, gt_pos, 'blue'), (ax_pr, pr_pos, 'red')]:
        ax.scatter(*pos[0], c='green', s=120, marker='^', zorder=5, label='Start')
        ax.scatter(*pos[-1], c='orange', s=120, marker='v', zorder=5, label='End')
        ax.legend(fontsize=7, loc='upper left')

    # Artists to update each frame
    gt_full, = ax_gt.plot([], [], [], 'b-', lw=1.2, alpha=0.35)
    gt_trail, = ax_gt.plot([], [], [], 'b-', lw=3, alpha=0.85)
    gt_dot = ax_gt.scatter([], [], [], c='blue', s=160, zorder=5)

    pr_full, = ax_pr.plot([], [], [], 'r-', lw=1.2, alpha=0.35)
    pr_trail, = ax_pr.plot([], [], [], 'r-', lw=3, alpha=0.85)
    pr_dot = ax_pr.scatter([], [], [], c='red', s=160, zorder=5)

    max_frames = max(len(gt_pos), len(pr_pos))
    gt_arrow_ref = [None]  # Store quiver artist for removal
    pr_arrow_ref = [None]

    def _update(frame):
        # Remove previous orientation arrows (quiver adds to collections)
        if gt_arrow_ref[0] is not None:
            try:
                gt_arrow_ref[0].remove()
            except (ValueError, AttributeError):
                pass
            gt_arrow_ref[0] = None
        if pr_arrow_ref[0] is not None:
            try:
                pr_arrow_ref[0].remove()
            except (ValueError, AttributeError):
                pass
            pr_arrow_ref[0] = None

        # GT
        f_gt = min(frame, len(gt_pos) - 1)
        gt_full.set_data_3d(gt_pos[:f_gt + 1, 0], gt_pos[:f_gt + 1, 1], gt_pos[:f_gt + 1, 2])
        ts = max(0, f_gt - trail_length)
        seg = gt_pos[ts:f_gt + 1]
        gt_trail.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        gt_dot._offsets3d = ([gt_pos[f_gt, 0]], [gt_pos[f_gt, 1]], [gt_pos[f_gt, 2]])

        # Draw GT orientation arrow
        dx, dy, dz = gt_fwd[f_gt]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            q = ax_gt.quiver(gt_pos[f_gt, 0], gt_pos[f_gt, 1], gt_pos[f_gt, 2],
                             dx, dy, dz, length=arrow_scale, color='purple', alpha=0.9)
            gt_arrow_ref[0] = q

        # Pred
        f_pr = min(frame, len(pr_pos) - 1)
        pr_full.set_data_3d(pr_pos[:f_pr + 1, 0], pr_pos[:f_pr + 1, 1], pr_pos[:f_pr + 1, 2])
        ts = max(0, f_pr - trail_length)
        seg = pr_pos[ts:f_pr + 1]
        pr_trail.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        pr_dot._offsets3d = ([pr_pos[f_pr, 0]], [pr_pos[f_pr, 1]], [pr_pos[f_pr, 2]])

        # Draw Pred orientation arrow
        dx, dy, dz = pr_fwd[f_pr]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            q = ax_pr.quiver(pr_pos[f_pr, 0], pr_pos[f_pr, 1], pr_pos[f_pr, 2],
                             dx, dy, dz, length=arrow_scale, color='purple', alpha=0.9)
            pr_arrow_ref[0] = q

        return gt_full, gt_trail, gt_dot, pr_full, pr_trail, pr_dot

    frame_indices = list(range(0, max_frames, stride))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    anim = FuncAnimation(fig, _update, frames=frame_indices,
                         interval=1000 / fps, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=100)
    except Exception as e:
        # Fallback: save as GIF if ffmpeg is unavailable
        gif_path = save_path.rsplit('.', 1)[0] + '.gif'
        from matplotlib.animation import PillowWriter
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=80)
        print(f"[Warning] FFMpeg unavailable, saved GIF instead: {gif_path}")
    del anim
    plt.close(fig)


# ---------------------------------------------------------------------------
#  Velocity-integrated comparison plots
# ---------------------------------------------------------------------------

def plot_trajectory_comparison_vel_integrated(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    caption: str,
    save_path: str,
    epoch: int = 0,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    format_type=None,
    smooth_sigma: float = 1.5,
):
    """4-panel PNG comparing GT and predicted trajectories via velocity integration.

    Positions are reconstructed by cumulative summation of the velocity channels
    (indices 3-5) anchored at the first frame's direct position.  Gaussian
    smoothing (sigma=smooth_sigma) is applied to the integrated positions.

    The current position-based visualizations do **not** apply smoothing; this
    function adds it only for the velocity-integrated trajectory.

    Panels:
      1. 3-D overlay (GT=blue, Pred=red) with camera-direction arrows
      2. Top-down view (X vs Depth)
      3. Per-frame position error between integrated GT and integrated Pred
      4. Per-axis integrated position over time

    Skips silently when trajectories have < 6 channels (no velocity information).
    """
    gt_pos_raw = _positions_from_vel_integration(
        gt_traj, mean, std, smooth=True, smooth_sigma=smooth_sigma)
    pr_pos_raw = _positions_from_vel_integration(
        pred_traj, mean, std, smooth=True, smooth_sigma=smooth_sigma)

    if gt_pos_raw is None or pr_pos_raw is None:
        print(f"[vel_integrated] Skipping comparison plot — insufficient channels.")
        return

    gt_pos = _to_mpl(gt_pos_raw)
    pr_pos = _to_mpl(pr_pos_raw)

    # Orientation arrows from original (denormalised) features
    gt_raw = _inv_transform(gt_traj, mean, std) if (mean is not None and std is not None) else gt_traj
    pr_raw = _inv_transform(pred_traj, mean, std) if (mean is not None and std is not None) else pred_traj
    gt_fwd = _compute_orientation_vectors(gt_raw, format_type)
    pr_fwd = _compute_orientation_vectors(pr_raw, format_type)

    all_pos = np.concatenate([gt_pos, pr_pos], axis=0)
    arrow_scale = max(np.ptp(all_pos, axis=0).max() * 0.08, 1e-3)
    step = max(1, min(len(gt_pos), len(pr_pos)) // 8)

    fig = plt.figure(figsize=(16, 12))

    # ---- 1. 3-D overlay ----
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.plot(*gt_pos.T, 'b-', lw=2, label='GT (integrated)')
    ax1.plot(*pr_pos.T, 'r--', lw=2, label='Pred (integrated)')
    ax1.scatter(*gt_pos[0], c='green', s=80, marker='^', label='Start')
    ax1.scatter(*gt_pos[-1], c='orange', s=80, marker='v', label='End')
    for i in range(0, len(gt_pos), step):
        dx, dy, dz = gt_fwd[i]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax1.quiver(gt_pos[i, 0], gt_pos[i, 1], gt_pos[i, 2],
                       dx, dy, dz, length=arrow_scale, color='blue', alpha=0.7)
    for i in range(0, len(pr_pos), step):
        dx, dy, dz = pr_fwd[i]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax1.quiver(pr_pos[i, 0], pr_pos[i, 1], pr_pos[i, 2],
                       dx, dy, dz, length=arrow_scale, color='red', alpha=0.7)
    ax1.set_xlabel('X');  ax1.set_ylabel('Depth');  ax1.set_zlabel('Y')
    ax1.set_title(f'3-D Trajectory — vel-integrated, σ={smooth_sigma}')
    ax1.legend(fontsize=8)

    # ---- 2. Top-down ----
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 'b-', lw=2, label='GT')
    ax2.plot(pr_pos[:, 0], pr_pos[:, 1], 'r--', lw=2, label='Pred')
    ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=60, marker='^')
    ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='orange', s=60, marker='v')
    arrow_scale_2d = max(np.ptp(all_pos[:, :2], axis=0).max() * 0.08, 1e-3)
    for i in range(0, len(gt_pos), step):
        dx, dy = gt_fwd[i, 0], gt_fwd[i, 1]
        if np.sqrt(dx**2 + dy**2) > 1e-6:
            ax2.arrow(gt_pos[i, 0], gt_pos[i, 1], dx * arrow_scale_2d, dy * arrow_scale_2d,
                     head_width=arrow_scale_2d * 0.3, head_length=arrow_scale_2d * 0.2,
                     fc='blue', ec='blue', alpha=0.7)
    for i in range(0, len(pr_pos), step):
        dx, dy = pr_fwd[i, 0], pr_fwd[i, 1]
        if np.sqrt(dx**2 + dy**2) > 1e-6:
            ax2.arrow(pr_pos[i, 0], pr_pos[i, 1], dx * arrow_scale_2d, dy * arrow_scale_2d,
                     head_width=arrow_scale_2d * 0.3, head_length=arrow_scale_2d * 0.2,
                     fc='red', ec='red', alpha=0.7)
    ax2.set_xlabel('X');  ax2.set_ylabel('Depth')
    ax2.set_title('Top-Down (vel-integrated)')
    ax2.legend(fontsize=8);  ax2.grid(True);  ax2.set_aspect('equal', 'datalim')

    # ---- 3. Position error between integrated trajectories ----
    min_len = min(len(gt_pos), len(pr_pos))
    pos_err = np.linalg.norm(gt_pos[:min_len] - pr_pos[:min_len], axis=1)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(pos_err, 'g-', lw=2)
    ax3.set_xlabel('Frame');  ax3.set_ylabel('L2 Position Error')
    ax3.set_title(f'Position Error (integrated)  mean={pos_err.mean():.4f}')
    ax3.grid(True)

    # ---- 4. Per-axis integrated position ----
    ax4 = fig.add_subplot(2, 2, 4)
    frames = np.arange(len(gt_pos))
    for dim, (name, ls) in enumerate(zip(['X', 'Depth', 'Y'], ['-', '--', ':'])):
        ax4.plot(frames, gt_pos[:, dim], ls, color='blue', lw=1.5, label=f'GT {name}')
    frames_p = np.arange(len(pr_pos))
    for dim, (name, ls) in enumerate(zip(['X', 'Depth', 'Y'], ['-', '--', ':'])):
        ax4.plot(frames_p, pr_pos[:, dim], ls, color='red', lw=1.5, label=f'Pred {name}')
    ax4.set_xlabel('Frame');  ax4.set_ylabel('Integrated Position')
    ax4.set_title('Per-Axis Integrated Position')
    ax4.legend(fontsize=7, ncol=2);  ax4.grid(True)

    wrapped = '\n'.join(textwrap.wrap(caption, width=90))
    fig.suptitle(f'Epoch {epoch} [vel-integrated]  |  {wrapped}', fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_trajectory_comparison_animation_vel_integrated(
    gt_traj: np.ndarray,
    pred_traj: np.ndarray,
    caption: str,
    save_path: str,
    epoch: int = 0,
    fps: int = 30,
    trail_length: int = 30,
    stride: int = 1,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    format_type=None,
    smooth_sigma: float = 1.5,
):
    """Side-by-side animated MP4 comparing GT and predicted trajectories using
    **velocity-channel time integration** for position reconstruction.

    Positions are reconstructed by cumulative summation of the velocity channels
    (indices 3-5) anchored at the first frame's direct position.  Gaussian
    smoothing (sigma=smooth_sigma frames) is applied after integration.

    Layout mirrors ``plot_trajectory_comparison_animation``:
      Left panel  — Ground Truth  (blue path, green trail, purple direction arrow)
      Right panel — Predicted     (red path, orange trail, purple direction arrow)
    Both panels share the same axis limits.

    Silently skips saving if the trajectories have < 6 channels.

    Parameters
    ----------
    gt_traj, pred_traj : (T, D) arrays — normalised feature space
    caption   : text prompt (shown as suptitle)
    save_path : output .mp4 path
    epoch     : current training epoch
    fps, trail_length, stride : animation settings
    mean, std : if provided, trajectories are denormalised before integration
    format_type : CameraDataFormat for orientation extraction
    smooth_sigma : Gaussian sigma (frames) applied after integration
    """
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    gt_pos_raw = _positions_from_vel_integration(
        gt_traj, mean, std, smooth=True, smooth_sigma=smooth_sigma)
    pr_pos_raw = _positions_from_vel_integration(
        pred_traj, mean, std, smooth=True, smooth_sigma=smooth_sigma)

    if gt_pos_raw is None or pr_pos_raw is None:
        print(f"[vel_integrated_anim] Skipping — insufficient channels.")
        return

    gt_pos = _to_mpl(gt_pos_raw)
    pr_pos = _to_mpl(pr_pos_raw)

    # Orientation arrows from original (denormalised) features
    gt_raw = _inv_transform(gt_traj, mean, std) if (mean is not None and std is not None) else gt_traj
    pr_raw = _inv_transform(pred_traj, mean, std) if (mean is not None and std is not None) else pred_traj
    gt_fwd = _compute_orientation_vectors(gt_raw, format_type)
    pr_fwd = _compute_orientation_vectors(pr_raw, format_type)

    all_pos = np.concatenate([gt_pos, pr_pos], axis=0)
    mins = all_pos.min(axis=0)
    maxs = all_pos.max(axis=0)
    padding = max(np.ptp(all_pos, axis=0).max() * 0.12, 1e-3)
    arrow_scale = max(np.ptp(all_pos, axis=0).max() * 0.1, 1e-3)

    fig = plt.figure(figsize=(18, 8))
    ax_gt = fig.add_subplot(1, 2, 1, projection='3d')
    ax_pr = fig.add_subplot(1, 2, 2, projection='3d')

    for ax, label, clr in [
        (ax_gt, f'GT (vel-integrated, σ={smooth_sigma})', 'blue'),
        (ax_pr, f'Pred (vel-integrated, σ={smooth_sigma})', 'red'),
    ]:
        ax.set_xlim(mins[0] - padding, maxs[0] + padding)
        ax.set_ylim(mins[1] - padding, maxs[1] + padding)
        ax.set_zlim(mins[2] - padding, maxs[2] + padding)
        ax.set_xlabel('X');  ax.set_ylabel('Depth');  ax.set_zlabel('Y')
        ax.set_title(label, fontsize=12)

    wrapped = '\n'.join(textwrap.wrap(caption, width=100))
    fig.suptitle(f'Epoch {epoch} [vel-integrated]  |  {wrapped}', fontsize=10, y=0.98)

    # Static start / end markers
    for ax, pos in [(ax_gt, gt_pos), (ax_pr, pr_pos)]:
        ax.scatter(*pos[0], c='green', s=120, marker='^', zorder=5, label='Start')
        ax.scatter(*pos[-1], c='orange', s=120, marker='v', zorder=5, label='End')
        ax.legend(fontsize=7, loc='upper left')

    # Dynamic artists
    gt_full,  = ax_gt.plot([], [], [], 'b-', lw=1.2, alpha=0.35)
    gt_trail, = ax_gt.plot([], [], [], 'b-', lw=3,   alpha=0.85)
    gt_dot    = ax_gt.scatter([], [], [], c='blue', s=160, zorder=5)

    pr_full,  = ax_pr.plot([], [], [], 'r-', lw=1.2, alpha=0.35)
    pr_trail, = ax_pr.plot([], [], [], 'r-', lw=3,   alpha=0.85)
    pr_dot    = ax_pr.scatter([], [], [], c='red', s=160, zorder=5)

    max_frames = max(len(gt_pos), len(pr_pos))
    gt_arrow_ref = [None]
    pr_arrow_ref = [None]

    def _update(frame):
        for ref, ax in [(gt_arrow_ref, ax_gt), (pr_arrow_ref, ax_pr)]:
            if ref[0] is not None:
                try:
                    ref[0].remove()
                except (ValueError, AttributeError):
                    pass
                ref[0] = None

        f_gt = min(frame, len(gt_pos) - 1)
        gt_full.set_data_3d(gt_pos[:f_gt + 1, 0], gt_pos[:f_gt + 1, 1], gt_pos[:f_gt + 1, 2])
        ts = max(0, f_gt - trail_length)
        seg = gt_pos[ts:f_gt + 1]
        gt_trail.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        gt_dot._offsets3d = ([gt_pos[f_gt, 0]], [gt_pos[f_gt, 1]], [gt_pos[f_gt, 2]])
        dx, dy, dz = gt_fwd[f_gt]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            gt_arrow_ref[0] = ax_gt.quiver(
                gt_pos[f_gt, 0], gt_pos[f_gt, 1], gt_pos[f_gt, 2],
                dx, dy, dz, length=arrow_scale, color='purple', alpha=0.9)

        f_pr = min(frame, len(pr_pos) - 1)
        pr_full.set_data_3d(pr_pos[:f_pr + 1, 0], pr_pos[:f_pr + 1, 1], pr_pos[:f_pr + 1, 2])
        ts = max(0, f_pr - trail_length)
        seg = pr_pos[ts:f_pr + 1]
        pr_trail.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        pr_dot._offsets3d = ([pr_pos[f_pr, 0]], [pr_pos[f_pr, 1]], [pr_pos[f_pr, 2]])
        dx, dy, dz = pr_fwd[f_pr]
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            pr_arrow_ref[0] = ax_pr.quiver(
                pr_pos[f_pr, 0], pr_pos[f_pr, 1], pr_pos[f_pr, 2],
                dx, dy, dz, length=arrow_scale, color='purple', alpha=0.9)

        return gt_full, gt_trail, gt_dot, pr_full, pr_trail, pr_dot

    frame_indices = list(range(0, max_frames, stride))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    anim = FuncAnimation(fig, _update, frames=frame_indices,
                         interval=1000 / fps, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=100)
    except Exception as e:
        gif_path = save_path.rsplit('.', 1)[0] + '.gif'
        from matplotlib.animation import PillowWriter
        anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=80)
        print(f"[Warning] FFMpeg unavailable, saved GIF instead: {gif_path}")
    del anim
    plt.close(fig)


#  VQ-VAE Evaluation (camera datasets)
# ---------------------------------------------------------------------------

def _rotation_geodesic_deg(rot6d_a: np.ndarray, rot6d_b: np.ndarray) -> np.ndarray:
    """Per-frame angle (degrees) between the rotations encoded by two (T, 6) rot6d arrays."""
    Ra = sixd_to_matrix(rot6d_a)
    Rb = sixd_to_matrix(rot6d_b)
    cos = (np.einsum('tij,tij->t', Ra, Rb) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


@torch.no_grad()
def evaluation_camera_vqvae_clatr(
    out_dir, val_loader, net, logger, ep, best,
    eval_wrapper=None,       # CLaTrEvalWrapper instance, or None for reconstruction-only eval
    save=True, draw=True,
    vis_vel_integration: bool = False,
    seed: int = 0,
):
    """
    Evaluate a camera VQ-VAE on the validation split.

    Checkpoint selection uses the masked L1 reconstruction error in the VQ's normalized space
    (the training objective), over the valid frames of every clip. Each clip is encoded at its
    own length, so the zero padding of the batch never enters the VQ. Real-unit errors (after
    de-normalizing with the loader's mean/std) and, when ``eval_wrapper`` is a CLaTrEvalWrapper,
    CLaTr metrics are logged but not used for selection. The current CLaTr checkpoint was trained
    on data with the old RealEstate10K pose convention, so its numbers are a relative signal only.

    ``val_loader`` must yield (caption, motion, m_length) batches (see unpack_clatr_val_batch)
    from a single process: the RNG is re-seeded on entry and restored on exit, so the random
    caption choice and temporal crop of Text2MotionDataset are identical every epoch.

    ``best`` is a dict {'recon': float, 'fid': float}. With ``save=True`` it is updated and
    net_best_recon.tar / net_best_fid.tar are written on improvement; with ``save=False``
    (e.g. the untrained model before training) the metrics are only logged.

    Returns (best, metrics).
    """
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    net.eval()
    device = next(net.parameters()).device
    mean = np.asarray(val_loader.dataset.mean)
    std = np.asarray(val_loader.dataset.std)
    fmt_type = None
    if hasattr(val_loader.dataset, 'opt') and hasattr(val_loader.dataset.opt, 'dataset_name'):
        fmt_type = detect_format_from_dataset_name(val_loader.dataset.opt.dataset_name)
    use_clatr = isinstance(eval_wrapper, CLaTrEvalWrapper)

    all_gt_emb, all_pred_emb, all_text_emb = [], [], []
    l1_sum, n_values = 0.0, 0
    pos_err_sum, extent_sum, rot_err_sum, vel_err_sum, n_frames = 0.0, 0.0, 0.0, 0.0, 0
    nb_sample = 0
    plot_count = 0
    max_plots = 5  # comparison plots per epoch

    for batch in val_loader:
        motion, m_length, captions = unpack_clatr_val_batch(batch)[:3]
        motion = motion.to(device).float()
        m_len_list = m_length.tolist() if isinstance(m_length, torch.Tensor) else list(m_length)
        bs = motion.shape[0]

        # VQ forward, one clip at a time at its valid length
        pred_motion = torch.zeros_like(motion)
        for i in range(bs):
            ml = int(m_len_list[i])
            pred_motion[i, :ml] = net(motion[i:i + 1, :ml])[0][0]

        if use_clatr and captions is not None:
            m_lens_tensor = torch.as_tensor(m_len_list, device=device)
            all_gt_emb.append(eval_wrapper.get_trajectory_embeddings(motion, m_lens_tensor))
            all_pred_emb.append(eval_wrapper.get_trajectory_embeddings(pred_motion, m_lens_tensor))
            all_text_emb.append(eval_wrapper.get_text_embeddings(captions))

        gt_np = motion.cpu().numpy()
        pr_np = pred_motion.cpu().numpy()
        for i in range(bs):
            ml = int(m_len_list[i])
            gt_n, pr_n = gt_np[i, :ml], pr_np[i, :ml]
            l1_sum += float(np.abs(pr_n - gt_n).sum())
            n_values += gt_n.size
            gt, pr = gt_n * std + mean, pr_n * std + mean
            pos_err_sum += float(np.linalg.norm(pr[:, :3] - gt[:, :3], axis=1).sum())
            extent_sum += float(np.linalg.norm(gt[:, :3] - gt[:1, :3], axis=1).max()) * ml
            rot_err_sum += float(_rotation_geodesic_deg(gt[:, 6:12], pr[:, 6:12]).sum())
            vel_err_sum += float(np.linalg.norm(pr[:, 3:6] - gt[:, 3:6], axis=1).sum())
            n_frames += ml
        nb_sample += bs

        # ── Comparison plots ──
        if draw and captions is not None and plot_count < max_plots:
            plot_dir = pjoin(out_dir, "eval_plots", f"E{ep:04d}_vq")
            for i in range(min(bs, max_plots - plot_count)):
                ml = int(m_len_list[i])
                cap_str = captions[i] if i < len(captions) else "N/A"
                plot_trajectory_comparison(
                    gt_np[i, :ml], pr_np[i, :ml], cap_str,
                    pjoin(plot_dir, f"cmp_{plot_count:02d}.png"), epoch=ep,
                    mean=mean, std=std, format_type=fmt_type,
                )
                try:
                    plot_trajectory_comparison_animation(
                        gt_np[i, :ml], pr_np[i, :ml], cap_str,
                        pjoin(plot_dir, f"cmp_{plot_count:02d}.mp4"),
                        epoch=ep, fps=20, stride=2,
                        mean=mean, std=std, format_type=fmt_type,
                    )
                except Exception as e:
                    print(f"[Warning] Animation failed for VQ sample {plot_count}: {e}")
                if vis_vel_integration:
                    try:
                        plot_trajectory_comparison_vel_integrated(
                            gt_np[i, :ml], pr_np[i, :ml], cap_str,
                            pjoin(plot_dir, f"cmp_{plot_count:02d}_velint.png"), epoch=ep,
                            mean=mean, std=std, format_type=fmt_type,
                        )
                    except Exception as e:
                        print(f"[Warning] Vel-integrated plot failed for VQ sample {plot_count}: {e}")
                plot_count += 1

    metrics = {
        "recon_l1": l1_sum / max(n_values, 1),
        "pos_err": pos_err_sum / max(n_frames, 1),
        "pos_err_rel": pos_err_sum / max(extent_sum, 1e-12),
        "rot_err_deg": rot_err_sum / max(n_frames, 1),
        "vel_err": vel_err_sum / max(n_frames, 1),
        "num_clips": nb_sample,
    }
    if all_gt_emb:
        gt_emb_all = np.concatenate(all_gt_emb, axis=0)
        pred_emb_all = np.concatenate(all_pred_emb, axis=0)
        text_emb_all = np.concatenate(all_text_emb, axis=0)
        r_prec = np.mean([compute_r_precision(text_emb_all, pred_emb_all, top_k=3, seed=s) for s in range(3)], axis=0)
        metrics.update({
            "clatr_fid": float(compute_fid(gt_emb_all, pred_emb_all)),
            "clatr_top1": float(r_prec[0]), "clatr_top2": float(r_prec[1]), "clatr_top3": float(r_prec[2]),
            "clatr_matching": float(compute_matching_score(text_emb_all, pred_emb_all)),
            "clatr_diversity": float(np.mean([compute_diversity(pred_emb_all, seed=s) for s in range(3)])),
        })

    print(f"--> VQ Eva. Ep {ep} ({nb_sample} clips): " + ", ".join(
        f"{k}={v:.5f}" for k, v in metrics.items() if k != "num_clips"))
    if draw:
        for k, v in metrics.items():
            if k != "num_clips":
                logger.add_scalar(f"./Test/{k}", v, ep)

    if save:
        if metrics["recon_l1"] < best["recon"]:
            print(f"--> --> Recon L1 Improved from {best['recon']:.5f} to {metrics['recon_l1']:.5f} !!!")
            best["recon"] = metrics["recon_l1"]
            torch.save({"vq_model": net.state_dict(), "ep": ep, "metrics": metrics},
                       os.path.join(out_dir, "net_best_recon.tar"))
        if "clatr_fid" in metrics and metrics["clatr_fid"] < best["fid"]:
            print(f"--> --> CLaTr FID Improved from {best['fid']:.5f} to {metrics['clatr_fid']:.5f} !!!")
            best["fid"] = metrics["clatr_fid"]
            torch.save({"vq_model": net.state_dict(), "ep": ep, "metrics": metrics},
                       os.path.join(out_dir, "net_best_fid.tar"))

    net.train()
    random.setstate(rng_state[0])
    np.random.set_state(rng_state[1])
    torch.set_rng_state(rng_state[2])
    if rng_state[3] is not None:
        torch.cuda.set_rng_state_all(rng_state[3])
    return best, metrics


# ---------------------------------------------------------------------------
#  MaskTransformer Evaluation (replaces evaluation_mask_transformer for camera)
# ---------------------------------------------------------------------------

def unpack_clatr_val_batch(batch):
    """Unpack batches from train collates, id_embedding, or Text2MotionDatasetEval.

    Returns
    -------
    motion, m_length, captions, id_batch,
    first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask

    * captions is None for id_embedding 3-tuples.
    * visual_indices are **frame**-level when from sparse train collate; caller maps with //4 for VQ.
    """
    n = len(batch)
    id_batch = False
    captions = None
    first_frame_pixels = None
    sparse_frames = None
    visual_indices = None
    visual_valid_mask = None

    if n == 3:
        a, motion, m_length = batch
        if isinstance(a, torch.Tensor):
            id_batch = True
        else:
            captions = list(a) if not isinstance(a, list) else a
        return (motion, m_length, captions, id_batch, first_frame_pixels,
                sparse_frames, visual_indices, visual_valid_mask)

    if n == 4:
        cap, motion, m_length, fourth = batch
        captions = list(cap) if not isinstance(cap, list) else cap
        if isinstance(fourth, torch.Tensor) and fourth.dim() == 4:
            first_frame_pixels = fourth
        else:
            raise TypeError(
                "4-field batch must be (caption, motion, m_len, first_frame_tensor[B,3,224,224])")
        return (motion, m_length, captions, id_batch, first_frame_pixels,
                sparse_frames, visual_indices, visual_valid_mask)

    if n == 6:
        cap, motion, m_length, sparse_frames, visual_valid_mask, visual_indices = batch
        captions = list(cap) if not isinstance(cap, list) else cap
        return (motion, m_length, captions, id_batch, first_frame_pixels,
                sparse_frames, visual_indices, visual_valid_mask)

    if n >= 7:
        _, _, caption, _, motion, m_length, _ = batch[:7]
        captions = list(caption) if not isinstance(caption, list) else caption
        return (motion, m_length, captions, id_batch, first_frame_pixels,
                sparse_frames, visual_indices, visual_valid_mask)

    raise ValueError(f"unpack_clatr_val_batch: unsupported batch with {n} elements")


def _move_visual_to_device(first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask,
                           device):
    """Downsample sparse indices to VQ grid and move tensors to ``device``."""
    if visual_indices is not None:
        visual_indices = torch.div(
            visual_indices.long(), 4, rounding_mode='floor').to(device)
    if first_frame_pixels is not None:
        first_frame_pixels = first_frame_pixels.float().to(device)
    if sparse_frames is not None:
        sparse_frames = sparse_frames.float().to(device)
    if visual_valid_mask is not None:
        visual_valid_mask = visual_valid_mask.to(device)
    return first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask


@torch.no_grad()
def evaluation_mask_transformer_clatr(
    out_dir, val_loader, trans, vq_model, logger, ep,
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
    eval_wrapper,        # CLaTrEvalWrapper
    plot_func=None, save_ckpt=False, save_anim=False,
    cond_scale=3, temperature=1, topkr=0.9,
    num_repeat: int = 3,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    format_type=None,
    timesteps: int = 18,
    gsample: bool = False,
    vis_vel_integration: bool = False,
):
    """
    Evaluate mask transformer on camera trajectories using CLaTr metrics.

    When the val loader matches training (captions + optional CLIP visual tensors),
    the same conditioning is passed to ``generate()`` as in training / ``gen_camera.py``.

    Returns
    -------
    (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger)
    """
    trans.eval()
    vq_model.eval()

    all_gt_emb = []
    all_gen_emb = []
    all_text_emb = []
    nb_sample = 0
    plot_count = 0
    max_plots = 5

    gen_motion = None
    for batch in val_loader:
        (motion, m_length, captions, _id_batch, first_frame_pixels, sparse_frames,
         visual_indices, visual_valid_mask) = unpack_clatr_val_batch(batch)

        motion = motion.cuda()
        bs = motion.shape[0]
        m_lens_tensor = m_length if isinstance(m_length, torch.Tensor) else torch.tensor(m_length)
        m_lens_tensor = m_lens_tensor.cuda()

        # Generate trajectories
        if captions is not None:
            # Text-conditioned generation
            clip_text = captions
            token_lens = torch.ceil(m_lens_tensor.float() / 4).long()
            dev = motion.device

            ff, sp, vi, vm = _move_visual_to_device(
                first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask, dev)

            mids = trans.generate(
                clip_text, token_lens, timesteps=timesteps,
                cond_scale=cond_scale, temperature=temperature,
                topk_filter_thres=topkr, gsample=gsample,
                first_frame_pixels=ff,
                sparse_frames=sp,
                visual_indices=vi,
                visual_valid_mask=vm,
            )
            # MaskTransformer returns 2D (batch, seq_len); forward_decoder needs 3D (batch, seq_len, nq)
            if mids.dim() == 2:
                mids = mids.unsqueeze(-1)
            # Decode back to trajectory space
            gen_motion = vq_model.forward_decoder(mids)

            # Trim to original length
            max_gen_len = motion.shape[1]
            if gen_motion.shape[1] < max_gen_len:
                pad = torch.zeros(
                    bs, max_gen_len - gen_motion.shape[1], gen_motion.shape[2],
                    device=gen_motion.device,
                )
                gen_motion = torch.cat([gen_motion, pad], dim=1)
            elif gen_motion.shape[1] > max_gen_len:
                gen_motion = gen_motion[:, :max_gen_len]

            # CLaTr embeddings
            if isinstance(eval_wrapper, CLaTrEvalWrapper):
                gt_emb = eval_wrapper.get_trajectory_embeddings(motion, m_lens_tensor)
                gen_emb = eval_wrapper.get_trajectory_embeddings(gen_motion, m_lens_tensor)
                text_emb = eval_wrapper.get_text_embeddings(captions)
                all_gt_emb.append(gt_emb)
                all_gen_emb.append(gen_emb)
                all_text_emb.append(text_emb)

            # ── Comparison plots ──
            if plot_count < max_plots:
                plot_dir = pjoin(out_dir, "eval_plots", f"E{ep:04d}_mtrans")
                gt_np = motion.detach().cpu().numpy()
                gen_np = gen_motion.detach().cpu().numpy()
                m_len_list = m_lens_tensor.cpu().tolist()
                for i in range(min(bs, max_plots - plot_count)):
                    ml = int(m_len_list[i])
                    cap_str = captions[i] if i < len(captions) else "N/A"
                    plot_trajectory_comparison(
                        gt_np[i, :ml], gen_np[i, :ml], cap_str,
                        pjoin(plot_dir, f"cmp_{plot_count:02d}.png"), epoch=ep,
                        mean=mean, std=std, format_type=format_type,
                    )
                    try:
                        plot_trajectory_comparison_animation(
                            gt_np[i, :ml], gen_np[i, :ml], cap_str,
                            pjoin(plot_dir, f"cmp_{plot_count:02d}.mp4"),
                            epoch=ep, fps=20, stride=2,
                            mean=mean, std=std, format_type=format_type,
                        )
                    except Exception as e:
                        print(f"[Warning] Animation failed for MTrans sample {plot_count}: {e}")
                    if vis_vel_integration:
                        try:
                            plot_trajectory_comparison_vel_integrated(
                                gt_np[i, :ml], gen_np[i, :ml], cap_str,
                                pjoin(plot_dir, f"cmp_{plot_count:02d}_velint.png"), epoch=ep,
                                mean=mean, std=std, format_type=format_type,
                            )
                            plot_trajectory_comparison_animation_vel_integrated(
                                gt_np[i, :ml], gen_np[i, :ml], cap_str,
                                pjoin(plot_dir, f"cmp_{plot_count:02d}_velint.mp4"), epoch=ep,
                                fps=20, stride=2, mean=mean, std=std, format_type=format_type,
                            )
                        except Exception as e:
                            print(f"[Warning] Vel-integrated comparison failed for MTrans sample {plot_count}: {e}")
                    plot_count += 1

        nb_sample += bs

        # Save animations
        if save_anim and captions is not None and nb_sample <= 20:
            save_dir = pjoin(out_dir, "animations", f"E{ep:04d}")
            os.makedirs(save_dir, exist_ok=True)
            if plot_func is not None:
                data = torch.cat([motion[:4], gen_motion[:4]], dim=0).detach().cpu().numpy()
                plot_func(data, save_dir)

    # ── Compute metrics (averaged over num_repeat runs for stability) ──
    if all_gt_emb:
        gt_emb_all = np.concatenate(all_gt_emb, axis=0)
        gen_emb_all = np.concatenate(all_gen_emb, axis=0)
        text_emb_all = np.concatenate(all_text_emb, axis=0)

        # FID is deterministic (no sampling), compute once
        fid = compute_fid(gt_emb_all, gen_emb_all)

        # R-Precision and Diversity use random sampling — average over
        # multiple seeds for statistical stability.
        r_prec_runs = []
        div_runs = []
        gt_div_runs = []
        for run_i in range(num_repeat):
            r_prec_runs.append(compute_r_precision(
                text_emb_all, gen_emb_all, top_k=3, seed=run_i))
            div_runs.append(compute_diversity(gen_emb_all, seed=run_i))
            gt_div_runs.append(compute_diversity(gt_emb_all, seed=run_i))

        r_prec = np.mean(r_prec_runs, axis=0)
        match_score = compute_matching_score(text_emb_all, gen_emb_all)
        diversity = float(np.mean(div_runs))
        gt_diversity = float(np.mean(gt_div_runs))

    else:
        fid = float("inf")
        r_prec = np.zeros(3)
        match_score = 0.0
        diversity = 0.0
        gt_diversity = 0.0

    msg = (
        f"--> CLaTr MaskTrans Eva. Ep {ep}: FID={fid:.4f}, Div={diversity:.4f}, "
        f"R-Prec=({r_prec[0]:.4f}, {r_prec[1]:.4f}, {r_prec[2]:.4f}), "
        f"Match={match_score:.4f}"
    )
    print(msg)

    logger.add_scalar("./Test/CLaTr_FID", fid, ep)
    logger.add_scalar("./Test/CLaTr_Diversity", diversity, ep)
    logger.add_scalar("./Test/CLaTr_top1", r_prec[0], ep)
    logger.add_scalar("./Test/CLaTr_top2", r_prec[1], ep)
    logger.add_scalar("./Test/CLaTr_top3", r_prec[2], ep)
    logger.add_scalar("./Test/CLaTr_MatchingScore", match_score, ep)

    # ── Track best ──
    if fid < best_fid:
        print(f"--> --> CLaTr FID Improved: {best_fid:.5f} → {fid:.5f}")
        best_fid = fid
        if save_ckpt:
            torch.save(
                {"t2m_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(gt_diversity - diversity) < abs(gt_diversity - best_div):
        best_div = diversity

    if r_prec[0] > best_top1:
        print(f"--> --> Top1 Improved: {best_top1:.5f} → {r_prec[0]:.5f}")
        best_top1 = r_prec[0]
    if r_prec[1] > best_top2:
        best_top2 = r_prec[1]
    if r_prec[2] > best_top3:
        best_top3 = r_prec[2]

    if match_score > best_matching:
        print(f"--> --> Match Improved: {best_matching:.5f} → {match_score:.5f}")
        best_matching = match_score
        if save_ckpt:
            torch.save(
                {"t2m_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_matching.tar"),
            )

    trans.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger


# ---------------------------------------------------------------------------
#  ResidualTransformer Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluation_res_transformer_clatr(
    out_dir, val_loader, trans, vq_model, logger, ep,
    best_fid, best_div, best_top1, best_top2, best_top3, best_matching,
    eval_wrapper,
    plot_func=None, save_ckpt=False, save_anim=False,
    cond_scale=5, temperature=1,
    num_repeat: int = 3,
    mean: np.ndarray = None,
    std: np.ndarray = None,
    format_type=None,
    vis_vel_integration: bool = False,
):
    """
    Evaluate residual transformer using CLaTr metrics.

    ``cond_scale`` default matches ``gen_camera.py`` residual stage (5).

    Returns
    -------
    (best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger)
    """
    trans.eval()
    vq_model.eval()

    all_gt_emb = []
    all_gen_emb = []
    all_text_emb = []
    nb_sample = 0
    plot_count = 0
    max_plots = 5

    gen_motion = None
    for batch in val_loader:
        (motion, m_length, captions, _id_batch, first_frame_pixels, sparse_frames,
         visual_indices, visual_valid_mask) = unpack_clatr_val_batch(batch)

        motion = motion.cuda()
        bs = motion.shape[0]
        m_lens_tensor = m_length if isinstance(m_length, torch.Tensor) else torch.tensor(m_length)
        m_lens_tensor = m_lens_tensor.cuda()

        if captions is not None:
            clip_text = captions
            token_lens = torch.ceil(m_lens_tensor.float() / 4).long()
            dev = motion.device

            ff, sp, vi, vm = _move_visual_to_device(
                first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask, dev)

            # Encode GT → get base codes
            code_idx, all_codes = vq_model.encode(motion)

            if ep == 0:
                # At epoch 0, use base layer directly (no residual generation)
                pred_ids = code_idx[..., 0:1]
            else:
                pred_ids = trans.generate(
                    code_idx[..., 0], clip_text, token_lens,
                    cond_scale=cond_scale, temperature=temperature,
                    first_frame_pixels=ff,
                    sparse_frames=sp,
                    visual_indices=vi,
                    visual_valid_mask=vm,
                )

            gen_motion = vq_model.forward_decoder(pred_ids)

            max_gen_len = motion.shape[1]
            if gen_motion.shape[1] < max_gen_len:
                pad = torch.zeros(
                    bs, max_gen_len - gen_motion.shape[1], gen_motion.shape[2],
                    device=gen_motion.device,
                )
                gen_motion = torch.cat([gen_motion, pad], dim=1)
            elif gen_motion.shape[1] > max_gen_len:
                gen_motion = gen_motion[:, :max_gen_len]

            if isinstance(eval_wrapper, CLaTrEvalWrapper):
                gt_emb = eval_wrapper.get_trajectory_embeddings(motion, m_lens_tensor)
                gen_emb = eval_wrapper.get_trajectory_embeddings(gen_motion, m_lens_tensor)
                text_emb = eval_wrapper.get_text_embeddings(captions)
                all_gt_emb.append(gt_emb)
                all_gen_emb.append(gen_emb)
                all_text_emb.append(text_emb)

            # ── Comparison plots ──
            if plot_count < max_plots:
                plot_dir = pjoin(out_dir, "eval_plots", f"E{ep:04d}_rtrans")
                gt_np = motion.detach().cpu().numpy()
                gen_np = gen_motion.detach().cpu().numpy()
                m_len_list = m_lens_tensor.cpu().tolist()
                for i in range(min(bs, max_plots - plot_count)):
                    ml = int(m_len_list[i])
                    cap_str = captions[i] if i < len(captions) else "N/A"
                    plot_trajectory_comparison(
                        gt_np[i, :ml], gen_np[i, :ml], cap_str,
                        pjoin(plot_dir, f"cmp_{plot_count:02d}.png"), epoch=ep,
                        mean=mean, std=std, format_type=format_type,
                    )
                    try:
                        plot_trajectory_comparison_animation(
                            gt_np[i, :ml], gen_np[i, :ml], cap_str,
                            pjoin(plot_dir, f"cmp_{plot_count:02d}.mp4"),
                            epoch=ep, fps=20, stride=2,
                            mean=mean, std=std, format_type=format_type,
                        )
                    except Exception as e:
                        print(f"[Warning] Animation failed for RTrans sample {plot_count}: {e}")
                    if vis_vel_integration:
                        try:
                            plot_trajectory_comparison_vel_integrated(
                                gt_np[i, :ml], gen_np[i, :ml], cap_str,
                                pjoin(plot_dir, f"cmp_{plot_count:02d}_velint.png"), epoch=ep,
                                mean=mean, std=std, format_type=format_type,
                            )
                            plot_trajectory_comparison_animation_vel_integrated(
                                gt_np[i, :ml], gen_np[i, :ml], cap_str,
                                pjoin(plot_dir, f"cmp_{plot_count:02d}_velint.mp4"), epoch=ep,
                                fps=20, stride=2, mean=mean, std=std, format_type=format_type,
                            )
                        except Exception as e:
                            print(f"[Warning] Vel-integrated comparison failed for RTrans sample {plot_count}: {e}")
                    plot_count += 1

        nb_sample += bs

    # ── Compute metrics (averaged over num_repeat seeds for stability) ──
    if all_gt_emb:
        gt_emb_all = np.concatenate(all_gt_emb, axis=0)
        gen_emb_all = np.concatenate(all_gen_emb, axis=0)
        text_emb_all = np.concatenate(all_text_emb, axis=0)

        fid = compute_fid(gt_emb_all, gen_emb_all)

        r_prec_runs = []
        div_runs = []
        gt_div_runs = []
        for run_i in range(num_repeat):
            r_prec_runs.append(compute_r_precision(
                text_emb_all, gen_emb_all, top_k=3, seed=run_i))
            div_runs.append(compute_diversity(gen_emb_all, seed=run_i))
            gt_div_runs.append(compute_diversity(gt_emb_all, seed=run_i))

        r_prec = np.mean(r_prec_runs, axis=0)
        match_score = compute_matching_score(text_emb_all, gen_emb_all)
        diversity = float(np.mean(div_runs))
        gt_diversity = float(np.mean(gt_div_runs))
    else:
        fid = float("inf")
        r_prec = np.zeros(3)
        match_score = 0.0
        diversity = 0.0
        gt_diversity = 0.0

    msg = (
        f"--> CLaTr ResTrans Eva. Ep {ep}: FID={fid:.4f}, Div={diversity:.4f}, "
        f"R-Prec=({r_prec[0]:.4f}, {r_prec[1]:.4f}, {r_prec[2]:.4f}), "
        f"Match={match_score:.4f}"
    )
    print(msg)

    logger.add_scalar("./Test/CLaTr_FID", fid, ep)
    logger.add_scalar("./Test/CLaTr_Diversity", diversity, ep)
    logger.add_scalar("./Test/CLaTr_top1", r_prec[0], ep)
    logger.add_scalar("./Test/CLaTr_top2", r_prec[1], ep)
    logger.add_scalar("./Test/CLaTr_top3", r_prec[2], ep)
    logger.add_scalar("./Test/CLaTr_MatchingScore", match_score, ep)

    if fid < best_fid:
        print(f"--> --> CLaTr FID Improved: {best_fid:.5f} → {fid:.5f}")
        best_fid = fid
        if save_ckpt:
            torch.save(
                {"res_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_fid.tar"),
            )

    if abs(gt_diversity - diversity) < abs(gt_diversity - best_div):
        best_div = diversity

    if r_prec[0] > best_top1:
        best_top1 = r_prec[0]
    if r_prec[1] > best_top2:
        best_top2 = r_prec[1]
    if r_prec[2] > best_top3:
        best_top3 = r_prec[2]

    if match_score > best_matching:
        print(f"--> --> Match Improved: {best_matching:.5f} → {match_score:.5f}")
        best_matching = match_score
        if save_ckpt:
            torch.save(
                {"res_transformer": trans.state_dict(), "ep": ep},
                os.path.join(out_dir, "net_best_matching.tar"),
            )

    trans.train()
    return best_fid, best_div, best_top1, best_top2, best_top3, best_matching, logger
