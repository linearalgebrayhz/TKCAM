"""
Camera Geometry Module

Coordinate Convention: Strict OpenGL
    +X : Right
    +Y : Up
    -Z : Forward  (camera looks down -Z)

Pose Representation: Camera-to-World  T_wc  (columns of R are camera axes in world)
    Column 0 of R → local X (right)   in world
    Column 1 of R → local Y (up)      in world
    Column 2 of R → local Z           in world
    Forward       = -Column2           (camera looks down -Z)

12-D Feature Vector:
    [pos(3), vel(3), rot6d(6)]
    where rot6d = [col0, col1] of the 3×3 rotation matrix, flattened.

Matplotlib 3-D Mapping (Z-up display):
    OpenGL [x, y, z]  →  MPL [x, -z, y]
    Applied identically to positions AND direction vectors.

This module is the ONLY place where these transforms are defined.
All other code must call into this module.
"""

import numpy as np
from typing import Union, Tuple

# ============================================================================
# 6-D  ⟷  3×3  Rotation Matrix  (Gram–Schmidt)
# ============================================================================

def matrix_to_sixd(R: np.ndarray) -> np.ndarray:
    """Extract the first two columns of a 3×3 rotation matrix as a 6-D vector.

    Args:
        R: (..., 3, 3) rotation matrix (columns = camera axes in world).

    Returns:
        (..., 6) vector [col0_x, col0_y, col0_z, col1_x, col1_y, col1_z].
    """
    col0 = R[..., :, 0]  # (..., 3)
    col1 = R[..., :, 1]  # (..., 3)
    return np.concatenate([col0, col1], axis=-1)


def sixd_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """Reconstruct a proper rotation matrix from a 6-D representation via
    Gram–Schmidt orthonormalization.

    The third column is computed as  col2 = col0 × col1,  which gives
    a right-handed frame.  Because the camera forward is  -col2  in OpenGL,
    there is **no** extra sign flip here — the cross product naturally produces
    the +Z axis, and forward = -Z is applied later when needed.

    Args:
        rot6d: (..., 6)  [col0(3), col1(3)]

    Returns:
        (..., 3, 3) proper rotation matrix  (det = +1).
    """
    a = rot6d[..., :3]
    b = rot6d[..., 3:6]

    # Normalize col0
    col0 = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

    # Orthogonalize col1 against col0
    dot = np.sum(col0 * b, axis=-1, keepdims=True)
    col1 = b - dot * col0
    col1 = col1 / (np.linalg.norm(col1, axis=-1, keepdims=True) + 1e-8)

    # col2 = col0 × col1  (right-hand rule)
    col2 = np.cross(col0, col1, axis=-1)

    # Stack as columns: R = [col0 | col1 | col2]
    return np.stack([col0, col1, col2], axis=-1)  # (..., 3, 3)


# ============================================================================
# Forward / Up / Right  Vectors  (OpenGL Convention)
# ============================================================================

def forward_from_sixd(rot6d: np.ndarray) -> np.ndarray:
    """Camera forward direction in world coordinates.

    OpenGL: camera looks down **-Z**, so forward = -col2 = -(col0 × col1).

    Args:
        rot6d: (..., 6)

    Returns:
        (..., 3) unit forward vector in world frame.
    """
    R = sixd_to_matrix(rot6d)
    col2 = R[..., :, 2]   # local +Z in world
    forward = -col2        # camera looks down -Z
    return forward


def up_from_sixd(rot6d: np.ndarray) -> np.ndarray:
    """Camera up direction in world coordinates (= col1 of R).

    Args:
        rot6d: (..., 6)

    Returns:
        (..., 3) unit up vector in world frame.
    """
    R = sixd_to_matrix(rot6d)
    return R[..., :, 1]


def right_from_sixd(rot6d: np.ndarray) -> np.ndarray:
    """Camera right direction in world coordinates (= col0 of R).

    Args:
        rot6d: (..., 6)

    Returns:
        (..., 3) unit right vector in world frame.
    """
    R = sixd_to_matrix(rot6d)
    return R[..., :, 0]


# ============================================================================
# Matplotlib Coordinate Mapping
# ============================================================================

def to_mpl(v: np.ndarray) -> np.ndarray:
    """Transform an OpenGL vector to Matplotlib 3-D convention (Z-up).

    Mapping:  [x, y, z]_gl  →  [x, -z, y]_mpl

    This is applied **identically** to both positions and direction vectors
    to maintain geometric consistency.

    Args:
        v: (..., 3) vector(s) in OpenGL convention.

    Returns:
        (..., 3) vector(s) in Matplotlib convention.
    """
    x = v[..., 0]
    y = v[..., 1]
    z = v[..., 2]
    return np.stack([x, -z, y], axis=-1)


# ============================================================================
# Relative Pose Computation
# ============================================================================

def compute_relative_pose(R0: np.ndarray, t0: np.ndarray,
                          R_abs: np.ndarray, t_abs: np.ndarray
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the relative pose of frame *i* w.r.t. frame 0.

    R_rel = R0^T @ R_abs
    t_rel = R0^T @ (t_abs - t0)

    Args:
        R0:    (3, 3)  rotation of frame 0  (T_wc).
        t0:    (3,)    translation of frame 0.
        R_abs: (3, 3)  rotation of frame i  (T_wc).
        t_abs: (3,)    translation of frame i.

    Returns:
        R_rel: (3, 3)
        t_rel: (3,)
    """
    R_rel = R0.T @ R_abs
    t_rel = R0.T @ (t_abs - t0)
    return R_rel, t_rel


def relativize_trajectory(positions: np.ndarray,
                          rotations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Make an entire trajectory relative to its first frame.

    After this transform:
        positions[0]  = [0, 0, 0]
        rotations[0]  = I

    Args:
        positions:  (N, 3)
        rotations:  (N, 3, 3)

    Returns:
        rel_positions:  (N, 3)
        rel_rotations:  (N, 3, 3)
    """
    R0 = rotations[0]
    t0 = positions[0]

    N = len(positions)
    rel_positions = np.zeros_like(positions)
    rel_rotations = np.zeros_like(rotations)

    for i in range(N):
        rel_rotations[i], rel_positions[i] = compute_relative_pose(
            R0, t0, rotations[i], positions[i]
        )

    return rel_positions, rel_rotations


# ============================================================================
# 12-D Feature Construction
# ============================================================================

def build_12d_features(positions: np.ndarray,
                       rotations: np.ndarray) -> np.ndarray:
    """Construct the 12-D feature vector from positions & rotation matrices.

    Format:  [pos(3), vel(3), rot6d(6)]
    
    Velocity is first-order finite difference of position:
        vel[0]   = pos[1] - pos[0]        (forward diff)
        vel[i]   = pos[i] - pos[i-1]      (backward diff, i ≥ 1)

    rot6d = first two columns of R, flattened.

    Args:
        positions:  (N, 3)
        rotations:  (N, 3, 3)

    Returns:
        features: (N, 12)
    """
    N = len(positions)
    assert rotations.shape == (N, 3, 3), f"Expected (N,3,3), got {rotations.shape}"

    # Velocity (first-order difference)
    vel = np.zeros_like(positions)
    if N > 1:
        vel[1:] = positions[1:] - positions[:-1]
        vel[0] = vel[1]  # Copy first frame velocity from second

    # 6-D rotation
    rot6d = matrix_to_sixd(rotations)  # (N, 6)

    return np.concatenate([positions, vel, rot6d], axis=-1)  # (N, 12)


# ============================================================================
# Twc / Tcw Handling
# ============================================================================

def tcw_to_twc(R_cw: np.ndarray, t_cw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Invert a world-to-camera transform to get camera-to-world.

    T_wc = T_cw^{-1}
    R_wc = R_cw^T
    t_wc = -R_cw^T @ t_cw

    Args:
        R_cw: (3, 3)
        t_cw: (3,)

    Returns:
        R_wc: (3, 3)
        t_wc: (3,)
    """
    R_wc = R_cw.T
    t_wc = -R_cw.T @ t_cw
    return R_wc, t_wc


def parse_3x4_to_Rt(block: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a 3x4 matrix into R (3x3) and t (3,).

    Args:
        block: (3, 4)

    Returns:
        R: (3, 3)
        t: (3,)
    """
    return block[:, :3].copy(), block[:, 3].copy()


def ensure_twc(block: np.ndarray, semantics: str) -> Tuple[np.ndarray, np.ndarray]:
    """Parse a 3×4 block and return (R, t) guaranteed to be T_wc.

    Args:
        block:     (3, 4) raw matrix from file.
        semantics: 'twc' or 'tcw'.

    Returns:
        R_wc: (3, 3)
        t_wc: (3,)
    """
    R, t = parse_3x4_to_Rt(block)
    if semantics == 'twc':
        return R, t
    elif semantics == 'tcw':
        return tcw_to_twc(R, t)
    else:
        raise ValueError(f"Unknown semantics: {semantics!r}. Use 'twc' or 'tcw'.")


# ============================================================================
# Validation Helpers
# ============================================================================

def orthogonalize_rotation(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via SVD.

    Args:
        R: (3, 3) approximately orthogonal matrix.

    Returns:
        R_clean: (3, 3) proper rotation matrix (det = +1).
    """
    U, _, Vt = np.linalg.svd(R)
    R_clean = U @ Vt
    if np.linalg.det(R_clean) < 0:
        U[:, -1] *= -1
        R_clean = U @ Vt
    return R_clean


def integrate_velocity_to_positions(
    features: np.ndarray,
    smooth: bool = True,
    smooth_sigma: float = 1.5,
) -> np.ndarray:
    """Reconstruct camera positions by time-integrating velocity channels.

    Assumes the 12-D feature layout  [x, y, z, dx, dy, dz, rot6d(6)].
    The first frame's direct position is used as the integration anchor so
    the reconstructed trajectory shares the same global origin as the
    position-channel trajectory.

    Integration rule:
        pos[0]  = features[0, 0:3]          (anchor from position channels)
        pos[t]  = pos[t-1] + features[t, 3:6]   for t >= 1

    Optionally applies per-axis Gaussian smoothing to remove high-frequency
    jitter introduced by the discrete summation.

    Args:
        features:     (N, D) array with D >= 6.  Channels 3-5 must hold the
                      per-frame velocity (first-order finite difference of
                      position, as produced by build_12d_features).
        smooth:       If True, apply 1-D Gaussian smoothing after integration.
        smooth_sigma: Standard deviation (in frames) for the Gaussian kernel.

    Returns:
        positions: (N, 3) integrated (and optionally smoothed) positions.

    Raises:
        ValueError: if features has fewer than 6 channels.
    """
    if features.ndim != 2 or features.shape[1] < 6:
        raise ValueError(
            f"integrate_velocity_to_positions requires (N, D>=6) features; "
            f"got shape {features.shape}"
        )
    from scipy.ndimage import gaussian_filter1d

    N = len(features)
    velocities = features[:, 3:6]  # (N, 3)

    positions = np.zeros((N, 3), dtype=np.float64)
    positions[0] = features[0, :3]  # anchor at measured first-frame position
    for t in range(1, N):
        positions[t] = positions[t - 1] + velocities[t]

    if smooth and N > 1:
        for dim in range(3):
            positions[:, dim] = gaussian_filter1d(
                positions[:, dim], sigma=smooth_sigma, mode='nearest'
            )

    return positions


def validate_rotation(R: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """Validate and optionally repair a rotation matrix. SO(3)

    Args:
        R:   (3, 3)
        tol: tolerance for RR^T - I Frobenius norm.

    Returns:
        R_valid: (3, 3) valid rotation matrix.
    """
    err = np.linalg.norm(R @ R.T - np.eye(3))
    if err > tol:
        R = orthogonalize_rotation(R)
    if np.linalg.det(R) < 0:
        R = -R  # flip reflection to rotation
        if np.linalg.det(R) < 0:
            R = orthogonalize_rotation(R)
    return R
