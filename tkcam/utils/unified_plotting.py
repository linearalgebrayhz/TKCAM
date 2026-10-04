"""
Unified plotting factory for transformer training scripts.

Returns a single plot function that handles both camera and human-motion
datasets, auto-detecting the appropriate visualisation.
"""

from os.path import join as pjoin

import numpy as np
import torch

from .unified_data_format import CameraDataFormat, detect_format_from_dataset_name


def create_plotting_function_for_transformer(dataset_name: str,
                                             vis_vel_integration: bool = False):
    """Return a plot function suitable for transformer training callbacks.

    For camera datasets the function generates animated MP4s via
    ``gen_camera.plot_camera_trajectory_animation``.

    When *vis_vel_integration* is True **and** data is in the paired
    GT+Pred format (first N samples GT, last N samples Pred — as emitted
    by the evaluation loop), each pair is additionally rendered as a
    side-by-side velocity-integrated comparison animation.

    For human-motion datasets it falls back to the standard skeleton renderer.

    Args:
        dataset_name:       Dataset identifier (e.g. ``realestate10k_rotmat``).
        vis_vel_integration: When True, also generate vel-integrated GT vs Pred
                            side-by-side MP4s for camera datasets.
    """

    is_camera = any(
        tag in dataset_name.lower() for tag in ("cam", "estate", "realestate")
    )
    fmt = detect_format_from_dataset_name(dataset_name) if is_camera else None

    def plot_function(data, save_dir, captions=None, m_lengths=None, **kwargs):
        fps    = kwargs.get("fps", 30)
        radius = kwargs.get("radius", 4)
        mean   = kwargs.get("mean", None)
        std    = kwargs.get("std",  None)

        if is_camera:
            from tkcam.gen_camera import (
                plot_camera_trajectory_animation,
                plot_camera_trajectory_animation_vel_integrated,
            )

            n = len(data)

            # ── Individual position-based trajectory animations ─────────────
            for i in range(n):
                seq = data[i]
                if m_lengths is not None:
                    seq = seq[: m_lengths[i]]
                cap = (captions[i] if captions is not None and i < len(captions)
                       else f"sample {i:02d}")
                save_path = pjoin(save_dir, "%02d.mp4" % i)
                try:
                    plot_camera_trajectory_animation(
                        data=seq,
                        save_path=save_path,
                        title=cap,
                        fps=fps,
                        show_trail=True,
                        trail_length=30,
                        format_type=fmt,
                    )
                except Exception as e:
                    print(f"Warning: camera plot failed for sample {i}: {e}")

            # ── Velocity-integrated GT vs Pred side-by-side ─────────────────
            # Detected when n is even: first half = GT, second half = Pred.
            if vis_vel_integration and n >= 2 and n % 2 == 0:
                from tkcam.utils.clatr_camera_eval import (
                    plot_trajectory_comparison_animation_vel_integrated,
                )
                n_pairs = n // 2
                for i in range(n_pairs):
                    gt_seq  = data[i]
                    pr_seq  = data[i + n_pairs]
                    if m_lengths is not None:
                        gt_seq = gt_seq[: m_lengths[i]]
                        pr_seq = pr_seq[: m_lengths[i + n_pairs]]
                    cap = (captions[i] if captions is not None and i < len(captions)
                           else f"sample {i:02d}")
                    vel_path = pjoin(save_dir, "velint_%02d.mp4" % i)
                    try:
                        plot_trajectory_comparison_animation_vel_integrated(
                            gt_traj=gt_seq,
                            pred_traj=pr_seq,
                            caption=cap,
                            save_path=vel_path,
                            fps=fps,
                            trail_length=30,
                            stride=2,
                            mean=mean,
                            std=std,
                            format_type=fmt,
                        )
                    except Exception as e:
                        print(f"Warning: vel-integrated comparison failed for pair {i}: {e}")

        else:
            from tkcam.utils.motion_process import recover_from_ric
            from tkcam.utils.plot_script import plot_3d_motion

            joints_num     = kwargs.get("joints_num", 22)
            kinematic_chain = kwargs.get("kinematic_chain", None)
            n = len(data)
            for i in range(n):
                seq = data[i]
                if m_lengths is not None:
                    seq = seq[: m_lengths[i]]
                cap = (captions[i] if captions is not None and i < len(captions)
                       else f"sample {i:02d}")
                joint = recover_from_ric(
                    torch.from_numpy(seq).float(), joints_num
                ).numpy()
                save_path = pjoin(save_dir, "%02d.mp4" % i)
                plot_3d_motion(
                    save_path, kinematic_chain, joint,
                    title=cap, fps=fps, radius=radius,
                )

    return plot_function
