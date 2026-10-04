"""
Camera process utilities.

Provides camera-specific metric computation used by camera_eval.py.
Format handling is delegated to unified_data_format.py.
"""

import numpy as np

from .unified_data_format import UnifiedCameraData, CameraDataFormat
from .camera_geometry import forward_from_sixd


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def orientation_to_vector(orientations: np.ndarray) -> np.ndarray:
    """Convert pitch/yaw angles to unit direction vectors.

    Args:
        orientations: (..., 2+) array where dim-0 = pitch, dim-1 = yaw.

    Returns:
        Unit direction vectors (..., 3).
    """
    pitch = orientations[..., 0]
    yaw = orientations[..., 1]

    x = np.cos(pitch) * np.sin(yaw)
    y = -np.sin(pitch)
    z = np.cos(pitch) * np.cos(yaw)

    return np.stack([x, y, z], axis=-1)


def _orientation_to_forward_vector(orientations: np.ndarray, format_type) -> np.ndarray:
    """Convert orientation representation to camera forward (unit) vectors.

    Args:
        orientations: Format-dependent orientation data.
        format_type: CameraDataFormat enum.

    Returns:
        (..., 3) unit forward vectors in world frame.
    """
    if format_type == CameraDataFormat.FULL_12_ROTMAT:
        # orientations is (..., 6) = [r1x, r1y, r1z, r2x, r2y, r2z]
        return forward_from_sixd(orientations)
    elif format_type == CameraDataFormat.QUATERNION_10:
        from tkcam.common.quaternion import qrot
        import torch
        quat = np.asarray(orientations, dtype=np.float32)
        if quat.ndim == 1:
            quat = quat.reshape(1, -1)
        forward_local = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        if forward_local.shape[0] != quat.shape[0]:
            forward_local = np.tile(forward_local, (quat.shape[0], 1))
        quat_t = torch.from_numpy(quat)
        fwd_t = torch.from_numpy(forward_local)
        fwd_gl = qrot(quat_t, fwd_t).numpy()
        norm = np.linalg.norm(fwd_gl, axis=-1, keepdims=True)
        norm = np.maximum(norm, 1e-8)
        return fwd_gl / norm
    else:
        # Euler angles: pitch, yaw (and optionally roll)
        return orientation_to_vector(orientations)


def calculate_trajectory_smoothness(
    positions: np.ndarray, orientations: np.ndarray
) -> float:
    """Smoothness score based on acceleration magnitude (lower = smoother).

    Args:
        positions: (T, 3) camera positions.
        orientations: (T, 2+) camera orientations.

    Returns:
        Combined positional + orientational acceleration magnitude.
    """
    if positions.shape[0] < 3:
        return float("inf")

    pos_accel = np.diff(positions, n=2, axis=0)
    pos_smoothness = np.mean(np.linalg.norm(pos_accel, axis=-1))

    ori_accel = np.diff(orientations, n=2, axis=0)
    ori_smoothness = np.mean(np.linalg.norm(ori_accel, axis=-1))

    return pos_smoothness + ori_smoothness


def calculate_camera_metrics(
    pred_data: np.ndarray, gt_data: np.ndarray, format_type=None
) -> dict:
    """Calculate camera-specific evaluation metrics.

    Supports all formats handled by UnifiedCameraData (10D quat, 12D rotmat, etc.).

    Args:
        pred_data: (batch, seq_len, features) predicted camera data.
        gt_data:   (batch, seq_len, features) ground-truth camera data.
        format_type: Optional CameraDataFormat. If None, auto-detected (12D defaults to
            FULL_12_EULER; for rotmat datasets pass FULL_12_ROTMAT explicitly).

    Returns:
        Dictionary of metric values.
    """
    # Collapse batch dimension if present (UnifiedCameraData expects 2D)
    pred_seq = pred_data[0] if pred_data.ndim == 3 else pred_data
    gt_seq = gt_data[0] if gt_data.ndim == 3 else gt_data

    pred_unified = UnifiedCameraData(pred_seq, format_type=format_type)
    gt_unified = UnifiedCameraData(gt_seq, format_type=format_type)

    pred_pos = pred_unified.positions.numpy()
    gt_pos = gt_unified.positions.numpy()
    pred_ori = pred_unified.orientations.numpy()
    gt_ori = gt_unified.orientations.numpy()

    # Position error
    pos_error = np.linalg.norm(pred_pos - gt_pos, axis=-1)

    # Orientation error via direction-vector angular distance
    # Use format-aware conversion (rotmat needs forward_from_sixd, not pitch/yaw)
    fmt = pred_unified.format_type
    pred_vec = _orientation_to_forward_vector(pred_ori, fmt)
    gt_vec = _orientation_to_forward_vector(gt_ori, fmt)
    dot = np.clip(np.sum(pred_vec * gt_vec, axis=-1), -1.0, 1.0)
    angle_error = np.arccos(dot)

    # Smoothness
    pred_smoothness = calculate_trajectory_smoothness(pred_pos, pred_ori)
    gt_smoothness = calculate_trajectory_smoothness(gt_pos, gt_ori)

    # Velocity error (finite-difference)
    vel_error = np.mean(
        np.linalg.norm(np.diff(pred_pos, axis=0) - np.diff(gt_pos, axis=0), axis=-1)
    )

    metrics = {
        "mean_position_error": float(np.mean(pos_error)),
        "mean_orientation_error": float(np.mean(angle_error)),
        "pred_smoothness": pred_smoothness,
        "gt_smoothness": gt_smoothness,
        "velocity_error": vel_error,
        "format": gt_unified.num_features,
    }

    # Direct velocity metrics when available (10D/12D)
    pred_vel = pred_unified.velocities
    gt_vel = gt_unified.velocities
    if pred_vel is not None and gt_vel is not None:
        pv = pred_vel.numpy()
        gv = gt_vel.numpy()
        vel_err = np.linalg.norm(pv - gv, axis=-1)
        metrics["direct_velocity_error"] = float(np.mean(vel_err))
        metrics["direct_velocity_error_std"] = float(np.std(vel_err))

    return metrics
