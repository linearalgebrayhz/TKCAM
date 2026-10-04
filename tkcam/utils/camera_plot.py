"""
Static GT-vs-Predicted camera trajectory comparison plot.

Used during VQ-VAE training validation to visualize reconstruction quality.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .unified_data_format import UnifiedCameraData, CameraDataFormat


def plot_camera_trajectory_3d(
    gt_data,
    pred_data,
    save_path,
    title="Camera Trajectory Comparison",
    seq_idx=0,
    format_type=None,
):
    """Plot 3D comparison of ground truth vs predicted camera trajectories.

    Creates a 4-panel figure: 3D trajectory, top-down view, position error,
    and orientation comparison.

    Args:
        gt_data:   Ground truth camera data (batch, seq_len, features) or (seq_len, features).
        pred_data: Predicted camera data, same shape as *gt_data*.
        save_path: Path to save the PNG.
        title:     Figure title.
        seq_idx:   Which sequence to plot when a batch dimension is present.
        format_type: Explicit ``CameraDataFormat``; auto-detected if *None*.
    """
    # Select a single sequence
    gt_seq = gt_data[seq_idx] if gt_data.ndim == 3 else gt_data
    pred_seq = pred_data[seq_idx] if pred_data.ndim == 3 else pred_data

    # Use unified data format for component extraction
    gt_u = UnifiedCameraData(gt_seq, format_type=format_type)
    pred_u = UnifiedCameraData(pred_seq, format_type=format_type)

    gt_pos = gt_u.positions.numpy()
    pred_pos = pred_u.positions.numpy()
    gt_ori = gt_u.orientations.numpy()
    pred_ori = pred_u.orientations.numpy()

    fmt_label = f"{gt_u.num_features}-feature"

    # ---- figure ----
    fig = plt.figure(figsize=(16, 12))
    time_steps = np.arange(len(gt_pos))

    # 1. 3D trajectory
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], "b-", lw=2, label="GT", alpha=0.8)
    ax1.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], "r--", lw=2, label="Pred", alpha=0.8)
    ax1.scatter(*gt_pos[0], color="green", s=100, marker="o", label="Start")
    ax1.scatter(*gt_pos[-1], color="orange", s=100, marker="s", label="End")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title("3D Camera Trajectory")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Top-down (X-Y)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], "b-", lw=2, label="GT", alpha=0.8)
    ax2.plot(pred_pos[:, 0], pred_pos[:, 1], "r--", lw=2, label="Pred", alpha=0.8)
    ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], color="green", s=100, marker="o", label="Start")
    ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], color="orange", s=100, marker="s", label="End")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_title("Top-down View (X-Y)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    # 3. Position error
    ax3 = fig.add_subplot(2, 2, 3)
    pos_err = np.linalg.norm(gt_pos - pred_pos, axis=1)
    ax3.plot(time_steps, pos_err, "r-", lw=2)
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Position Error (L2)")
    ax3.set_title("Position Error Over Time")
    ax3.grid(True, alpha=0.3)

    # 4. Orientation comparison
    ax4 = fig.add_subplot(2, 2, 4)
    n_ori = gt_ori.shape[-1]

    # Choose meaningful labels based on the orientation format
    if n_ori == 6:
        # Rotation matrix columns: first column (r1) and second column (r2)
        ori_labels = ["r1x", "r1y", "r1z", "r2x", "r2y", "r2z"]
    elif n_ori == 4:
        ori_labels = ["qw", "qx", "qy", "qz"]
    elif n_ori == 3:
        ori_labels = ["pitch", "yaw", "roll"]
    elif n_ori == 2:
        ori_labels = ["pitch", "yaw"]
    else:
        ori_labels = [f"Comp{k}" for k in range(n_ori)]

    # Use a colour cycle that supports up to 6 pairs
    gt_colors = ["b", "g", "c", "tab:brown", "tab:purple", "tab:olive"]
    pred_colors = ["r", "m", "y", "tab:orange", "tab:pink", "tab:gray"]
    for k in range(n_ori):
        ax4.plot(time_steps, gt_ori[:, k], gt_colors[k % len(gt_colors)], lw=2,
                 label=f"GT {ori_labels[k]}", alpha=0.8)
        ax4.plot(time_steps, pred_ori[:, k], pred_colors[k % len(pred_colors)], ls="--", lw=2,
                 label=f"Pred {ori_labels[k]}", alpha=0.8)
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Value")
    ax4.set_title(f"Orientation Comparison ({fmt_label})")
    ax4.legend(fontsize=6, ncol=2)
    ax4.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"3D trajectory plot saved to: {save_path} ({fmt_label})")
