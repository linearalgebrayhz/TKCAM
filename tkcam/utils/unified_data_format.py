"""
Unified Data Format Module for Camera Trajectory Processing

This module provides a unified interface for handling different camera trajectory data formats:
- 5-feature: [x, y, z, pitch, yaw] (legacy cam dataset)
- 6-feature: [x, y, z, pitch, yaw, roll] (position + full orientation)
- 10-feature: [x, y, z, dx, dy, dz, qw, qx, qy, qz] (position + velocity + quaternion)
- 9-feature RotMat: [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z] (position + 2 rotation matrix columns)
- 12-feature Euler: [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll] (position + velocity + Euler + angular velocity)
- 12-feature RotMat: [x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z] (position + velocity + 2 rotation matrix columns)

The module automatically detects format and provides conversion utilities.
"""

import numpy as np
import torch
from typing import Union, Tuple, Dict, Optional
from enum import Enum

class CameraDataFormat(Enum):
    """Enumeration of supported camera data formats"""
    LEGACY_5 = 5  # [x, y, z, pitch, yaw]
    POSITION_ORIENTATION_6 = 6  # [x, y, z, pitch, yaw, roll]
    POSITION_ROTMAT_9 = 9  # [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]
    QUATERNION_10 = 10  # [x, y, z, dx, dy, dz, qw, qx, qy, qz]
    FULL_12_EULER = 12  # [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]
    FULL_12_ROTMAT = 13  # [x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z] - uses 13 as unique ID

class UnifiedCameraData:
    """
    Unified camera data representation that can handle multiple formats
    """
    
    def __init__(self, data: Union[np.ndarray, torch.Tensor], format_type: Optional[CameraDataFormat] = None):
        """
        Initialize unified camera data
        
        Args:
            data: Camera trajectory data of shape (seq_len, features)
            format_type: Explicit format type, if None will auto-detect
        """
        self.data = torch.tensor(data) if isinstance(data, np.ndarray) else data
        self.seq_len, self.num_features = self.data.shape
        
        # Auto-detect format if not provided
        if format_type is None:
            self.format_type = self._detect_format()
        else:
            self.format_type = format_type
            
        self._validate_format()
        
    def _detect_format(self) -> CameraDataFormat:
        """Auto-detect data format based on number of features"""
        if self.num_features == 5:
            return CameraDataFormat.LEGACY_5
        elif self.num_features == 6:
            return CameraDataFormat.POSITION_ORIENTATION_6
        elif self.num_features == 9:
            return CameraDataFormat.POSITION_ROTMAT_9
        elif self.num_features == 10:
            return CameraDataFormat.QUATERNION_10
        elif self.num_features == 12:
            # Ambiguous in principle, but the active TKCAM pipeline uses 12D rotmat.
            # Prefer ROTMAT by default; pass format_type explicitly for Euler datasets.
            return CameraDataFormat.FULL_12_ROTMAT
        else:
            raise ValueError(f"Unsupported data format with {self.num_features} features. "
                           f"Supported formats: 5, 6, 9, 10, or 12 features.")
    
    def _validate_format(self):
        """Validate that data matches declared format"""
        # FULL_12_ROTMAT uses 13 as enum value but has 12 features
        expected_features = 12 if self.format_type == CameraDataFormat.FULL_12_ROTMAT else self.format_type.value
        if self.num_features != expected_features:
            raise ValueError(f"Data has {self.num_features} features but format expects {expected_features}")
    
    @property
    def positions(self) -> torch.Tensor:
        """Extract position data [x, y, z]"""
        return self.data[:, :3]
    
    @property
    def orientations(self) -> torch.Tensor:
        """Extract orientation data based on format (Euler angles or quaternion or rotation matrix)"""
        if self.format_type == CameraDataFormat.LEGACY_5:
            return self.data[:, 3:5]  # [pitch, yaw]
        elif self.format_type == CameraDataFormat.POSITION_ORIENTATION_6:
            return self.data[:, 3:6]  # [pitch, yaw, roll]
        elif self.format_type == CameraDataFormat.POSITION_ROTMAT_9:
            return self.data[:, 3:9]  # [r1x, r1y, r1z, r2x, r2y, r2z]
        elif self.format_type == CameraDataFormat.QUATERNION_10:
            return self.data[:, 6:10]  # [qw, qx, qy, qz]
        elif self.format_type == CameraDataFormat.FULL_12_EULER:
            return self.data[:, 6:9]  # [pitch, yaw, roll]
        else:  # FULL_12_ROTMAT
            return self.data[:, 6:12]  # [r1x, r1y, r1z, r2x, r2y, r2z]
    
    @property
    def velocities(self) -> Optional[torch.Tensor]:
        """Extract velocity data if available"""
        if self.format_type in [CameraDataFormat.QUATERNION_10, CameraDataFormat.FULL_12_EULER, CameraDataFormat.FULL_12_ROTMAT]:
            return self.data[:, 3:6]  # [dx, dy, dz]
        return None
    
    @property
    def angular_velocities(self) -> Optional[torch.Tensor]:
        """Extract angular velocity data if available (only for FULL_12_EULER)"""
        if self.format_type == CameraDataFormat.FULL_12_EULER:
            return self.data[:, 9:12]  # [dpitch, dyaw, droll]
        return None
    
    @property
    def quaternions(self) -> Optional[torch.Tensor]:
        """Extract quaternion data if available"""
        if self.format_type == CameraDataFormat.QUATERNION_10:
            return self.data[:, 6:10]  # [qw, qx, qy, qz]
        return None
    
    @property
    def rotation_matrix_columns(self) -> Optional[torch.Tensor]:
        """Extract rotation matrix columns if available (returns first 2 columns as 6D vector)"""
        if self.format_type == CameraDataFormat.FULL_12_ROTMAT:
            return self.data[:, 6:12]  # [r1x, r1y, r1z, r2x, r2y, r2z]
        if self.format_type == CameraDataFormat.POSITION_ROTMAT_9:
            return self.data[:, 3:9]  # [r1x, r1y, r1z, r2x, r2y, r2z]
        return None
    
    def to_format(self, target_format: CameraDataFormat) -> 'UnifiedCameraData':
        """
        Convert to target format
        
        Args:
            target_format: Target data format
            
        Returns:
            New UnifiedCameraData in target format
        """
        if self.format_type == target_format:
            return UnifiedCameraData(self.data.clone(), target_format)
        
        return self._convert_format(target_format)
    
    def _convert_format(self, target_format: CameraDataFormat) -> 'UnifiedCameraData':
        """Internal format conversion logic"""
        
        # Get base components
        positions = self.positions  # [seq_len, 3]
        orientations = self.orientations  # [seq_len, 2 or 3 or 4 or 6]
        
        if target_format == CameraDataFormat.LEGACY_5:
            # Convert to [x, y, z, pitch, yaw]
            if orientations.shape[1] == 2:  # Already [pitch, yaw]
                converted_data = torch.cat([positions, orientations], dim=1)
            else:  # [pitch, yaw, roll] -> [pitch, yaw]
                converted_data = torch.cat([positions, orientations[:, :2]], dim=1)
                
        elif target_format == CameraDataFormat.POSITION_ORIENTATION_6:
            # Convert to [x, y, z, pitch, yaw, roll]
            if orientations.shape[1] == 2:  # [pitch, yaw] -> [pitch, yaw, roll]
                # Add zero roll if not present
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_orientations = torch.cat([orientations, roll], dim=1)
            else:  # Already [pitch, yaw, roll]
                full_orientations = orientations
            converted_data = torch.cat([positions, full_orientations], dim=1)

        elif target_format == CameraDataFormat.POSITION_ROTMAT_9:
            # Convert to [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]
            if self.format_type in [CameraDataFormat.POSITION_ROTMAT_9, CameraDataFormat.FULL_12_ROTMAT]:
                rot_mat_cols = orientations
            elif orientations.shape[1] == 2:
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_euler = torch.cat([orientations, roll], dim=1)
                rot_mat_cols = euler_to_rotation_matrix(full_euler)
            elif orientations.shape[1] == 3:
                rot_mat_cols = euler_to_rotation_matrix(orientations)
            elif orientations.shape[1] == 4:
                euler = quaternion_to_euler(orientations)
                rot_mat_cols = euler_to_rotation_matrix(euler)
            else:
                raise ValueError(f"Cannot convert orientations of shape {orientations.shape} to 9D rotmat")
            converted_data = torch.cat([positions, rot_mat_cols], dim=1)
        
        elif target_format == CameraDataFormat.QUATERNION_10:
            # Convert to [x, y, z, dx, dy, dz, qw, qx, qy, qz]
            
            # Calculate velocities if not available
            if self.velocities is not None:
                velocities = self.velocities
            else:
                velocities = self._calculate_velocities(positions)
            
            # Ensure full Euler orientations [pitch, yaw, roll]
            if self.format_type == CameraDataFormat.QUATERNION_10:
                # Already quaternion, just use it
                quaternions = orientations
            elif orientations.shape[1] == 2:  # [pitch, yaw] -> [pitch, yaw, roll]
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_euler = torch.cat([orientations, roll], dim=1)
                quaternions = euler_to_quaternion(full_euler)
            elif orientations.shape[1] == 3:  # [pitch, yaw, roll] Euler angles
                quaternions = euler_to_quaternion(orientations)
            elif orientations.shape[1] == 6:  # Rotation matrix columns
                # Convert rotation matrix to quaternion
                euler = rotation_matrix_to_euler(orientations)
                quaternions = euler_to_quaternion(euler)
            else:
                raise ValueError(f"Cannot convert orientations of shape {orientations.shape} to quaternion")
            
            converted_data = torch.cat([
                positions,     # [x, y, z]
                velocities,    # [dx, dy, dz]
                quaternions    # [qw, qx, qy, qz]
            ], dim=1)
        
        elif target_format == CameraDataFormat.FULL_12_ROTMAT:
            # Convert to [x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]
            
            # Calculate velocities if not available
            if self.velocities is not None:
                velocities = self.velocities
            else:
                velocities = self._calculate_velocities(positions)
            
            # Convert orientation to rotation matrix columns
            if self.format_type in [CameraDataFormat.FULL_12_ROTMAT, CameraDataFormat.POSITION_ROTMAT_9]:
                # Already rotation matrix columns, just use it
                rot_mat_cols = orientations
            elif orientations.shape[1] == 2:  # [pitch, yaw] -> [pitch, yaw, roll]
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_euler = torch.cat([orientations, roll], dim=1)
                rot_mat_cols = euler_to_rotation_matrix(full_euler)
            elif orientations.shape[1] == 3:  # [pitch, yaw, roll] Euler angles
                rot_mat_cols = euler_to_rotation_matrix(orientations)
            elif orientations.shape[1] == 4:  # Quaternion
                euler = quaternion_to_euler(orientations)
                rot_mat_cols = euler_to_rotation_matrix(euler)
            else:
                raise ValueError(f"Cannot convert orientations of shape {orientations.shape} to rotation matrix")
            
            converted_data = torch.cat([
                positions,      # [x, y, z]
                velocities,     # [dx, dy, dz]
                rot_mat_cols    # [r1x, r1y, r1z, r2x, r2y, r2z]
            ], dim=1)
            
        else:  # target_format == CameraDataFormat.FULL_12_EULER
            # Convert to [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]
            
            # Calculate velocities if not available
            if self.velocities is not None:
                velocities = self.velocities
            else:
                velocities = self._calculate_velocities(positions)
            
            # Ensure full orientations [pitch, yaw, roll]
            if orientations.shape[1] == 2:  # [pitch, yaw] -> [pitch, yaw, roll]
                roll = torch.zeros(self.seq_len, 1, device=orientations.device)
                full_orientations = torch.cat([orientations, roll], dim=1)
            elif orientations.shape[1] == 3:  # Already [pitch, yaw, roll]
                full_orientations = orientations
            elif orientations.shape[1] == 4:  # Quaternion
                full_orientations = quaternion_to_euler(orientations)
            elif orientations.shape[1] == 6:  # Rotation matrix columns
                full_orientations = rotation_matrix_to_euler(orientations)
            else:
                raise ValueError(f"Cannot convert orientations of shape {orientations.shape}")
            
            # Calculate angular velocities if not available
            if self.angular_velocities is not None:
                angular_velocities = self.angular_velocities
            else:
                angular_velocities = self._calculate_angular_velocities(full_orientations)
            
            converted_data = torch.cat([
                positions,          # [x, y, z]
                velocities,         # [dx, dy, dz]
                full_orientations,  # [pitch, yaw, roll]
                angular_velocities  # [dpitch, dyaw, droll]
            ], dim=1)
        
        return UnifiedCameraData(converted_data, target_format)
    
    def _calculate_velocities(self, positions: torch.Tensor) -> torch.Tensor:
        """Calculate linear velocities from positions"""
        if len(positions) < 2:
            return torch.zeros_like(positions)
        
        velocities = torch.zeros_like(positions)
        velocities[1:] = positions[1:] - positions[:-1]
        velocities[0] = velocities[1]  # Copy first velocity
        
        return velocities
    
    def _calculate_angular_velocities(self, orientations: torch.Tensor) -> torch.Tensor:
        """Calculate angular velocities from orientations"""
        if len(orientations) < 2:
            return torch.zeros_like(orientations)
        
        angular_velocities = torch.zeros_like(orientations)
        angular_velocities[1:] = orientations[1:] - orientations[:-1]
        
        # Handle angle wrapping for yaw
        yaw_diff = angular_velocities[:, 1]
        yaw_diff = torch.where(yaw_diff > np.pi, yaw_diff - 2*np.pi, yaw_diff)
        yaw_diff = torch.where(yaw_diff < -np.pi, yaw_diff + 2*np.pi, yaw_diff)
        angular_velocities[:, 1] = yaw_diff
        
        angular_velocities[0] = angular_velocities[1]  # Copy first angular velocity
        
        return angular_velocities
    
    def get_momask_compatible_data(self) -> torch.Tensor:
        """
        Get data in format compatible with MoMask training
        Returns the raw tensor data
        """
        return self.data
    
    def get_feature_info(self) -> Dict[str, any]:
        """Get information about the current data format"""
        return {
            'format_type': self.format_type,
            'num_features': self.num_features,
            'seq_len': self.seq_len,
            'has_velocities': self.velocities is not None,
            'has_angular_velocities': self.angular_velocities is not None,
            'orientation_dims': self.orientations.shape[1]
        }

def detect_dataset_format(data_root: str, sample_file: str = None) -> CameraDataFormat:
    """
    Detect the format of a dataset by examining sample files
    
    Args:
        data_root: Path to dataset root
        sample_file: Optional specific file to check, otherwise checks first available
        
    Returns:
        Detected format type
    """
    from pathlib import Path
    
    motion_dir = Path(data_root) / "new_joint_vecs"
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
    
    # Find a sample file
    if sample_file is None:
        npy_files = list(motion_dir.glob("*.npy"))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in {motion_dir}")
        sample_file = npy_files[0]
    else:
        sample_file = motion_dir / f"{sample_file}.npy"
    
    # Load and check dimensions
    try:
        data = np.load(sample_file)
        if len(data.shape) != 2:
            raise ValueError(f"Expected 2D data, got shape {data.shape}")
        
        num_features = data.shape[1]
        if num_features == 5:
            return CameraDataFormat.LEGACY_5
        elif num_features == 6:
            return CameraDataFormat.POSITION_ORIENTATION_6
        elif num_features == 9:
            return CameraDataFormat.POSITION_ROTMAT_9
        elif num_features == 10:
            return CameraDataFormat.QUATERNION_10
        elif num_features == 12:
            # Disambiguate 12D: check dataset name/path. Default to rotmat.
            path_str = str(data_root).lower()
            if 'euler' in path_str:
                return CameraDataFormat.FULL_12_EULER
            return CameraDataFormat.FULL_12_ROTMAT
        else:
            raise ValueError(f"Unsupported format with {num_features} features")
            
    except Exception as e:
        raise RuntimeError(f"Failed to detect format from {sample_file}: {e}")

def create_unified_data_from_file(file_path: str, format_type: Optional[CameraDataFormat] = None) -> UnifiedCameraData:
    """
    Create UnifiedCameraData from a file
    
    Args:
        file_path: Path to .npy file
        format_type: Optional explicit format type
        
    Returns:
        UnifiedCameraData instance
    """
    data = np.load(file_path)
    return UnifiedCameraData(data, format_type)

def batch_convert_dataset(input_dir: str, output_dir: str, 
                         source_format: CameraDataFormat, 
                         target_format: CameraDataFormat):
    """
    Convert an entire dataset from one format to another
    
    Args:
        input_dir: Input dataset directory
        output_dir: Output dataset directory
        source_format: Source data format
        target_format: Target data format
    """
    from pathlib import Path
    import shutil
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory structure
    output_motion_dir = output_path / "new_joint_vecs"
    output_motion_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy text directory if it exists
    input_text_dir = input_path / "texts"
    if input_text_dir.exists():
        output_text_dir = output_path / "texts"
        if output_text_dir.exists():
            shutil.rmtree(output_text_dir)
        shutil.copytree(input_text_dir, output_text_dir)
    
    # Convert motion files
    input_motion_dir = input_path / "new_joint_vecs"
    npy_files = list(input_motion_dir.glob("*.npy"))
    
    print(f"Converting {len(npy_files)} files from {source_format.name} to {target_format.name}")
    
    for npy_file in npy_files:
        # Load and convert
        unified_data = create_unified_data_from_file(str(npy_file), source_format)
        converted_data = unified_data.to_format(target_format)
        
        # Save converted data
        output_file = output_motion_dir / npy_file.name
        np.save(output_file, converted_data.get_momask_compatible_data().numpy())
    
    # Copy other files (train.txt, val.txt, etc.)
    for txt_file in input_path.glob("*.txt"):
        shutil.copy2(txt_file, output_path / txt_file.name)
    
    # Create format info file
    format_info = {
        'format_type': target_format.name,
        'num_features': 12 if target_format == CameraDataFormat.FULL_12_ROTMAT else target_format.value,
        'converted_from': source_format.name,
        'conversion_date': str(np.datetime64('now'))
    }
    
    with open(output_path / "format_info.json", 'w') as f:
        import json
        json.dump(format_info, f, indent=2)
    
    print(f"Dataset conversion completed. Output saved to: {output_path}")


# ============================================================================
# Conversion Utilities for Quaternion and Rotation Matrix
# ============================================================================

def euler_to_quaternion(euler: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Euler angles (pitch, yaw, roll) to quaternion (qw, qx, qy, qz)
    
    Args:
        euler: [..., 3] Euler angles in radians [pitch, yaw, roll]
    
    Returns:
        [..., 4] Quaternion [qw, qx, qy, qz] (scalar-first convention)
    """
    from scipy.spatial.transform import Rotation as R
    
    is_torch = isinstance(euler, torch.Tensor)
    if is_torch:
        euler_np = euler.detach().cpu().numpy()
    else:
        euler_np = euler
    
    # scipy uses 'xyz' intrinsic rotations by default
    # Our convention: pitch (X), yaw (Y), roll (Z)
    rot = R.from_euler('xyz', euler_np, degrees=False)
    quat = rot.as_quat()  # scipy returns [qx, qy, qz, qw]
    
    # Convert to scalar-first: [qw, qx, qy, qz]
    quat_scalar_first = np.concatenate([quat[..., 3:4], quat[..., :3]], axis=-1)
    
    if is_torch:
        return torch.tensor(quat_scalar_first, dtype=euler.dtype, device=euler.device)
    return quat_scalar_first


def quaternion_to_euler(quat: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert quaternion (qw, qx, qy, qz) to Euler angles (pitch, yaw, roll)
    
    Args:
        quat: [..., 4] Quaternion [qw, qx, qy, qz] (scalar-first convention)
    
    Returns:
        [..., 3] Euler angles [pitch, yaw, roll] in radians
    """
    from scipy.spatial.transform import Rotation as R
    
    is_torch = isinstance(quat, torch.Tensor)
    if is_torch:
        quat_np = quat.detach().cpu().numpy()
    else:
        quat_np = quat
    
    # Convert to scipy format: [qx, qy, qz, qw]
    quat_scipy = np.concatenate([quat_np[..., 1:4], quat_np[..., 0:1]], axis=-1)
    
    rot = R.from_quat(quat_scipy)
    euler = rot.as_euler('xyz', degrees=False)  # [pitch, yaw, roll]
    
    if is_torch:
        return torch.tensor(euler, dtype=quat.dtype, device=quat.device)
    return euler


def euler_to_rotation_matrix(euler: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Euler angles to rotation matrix (extract first 2 columns as 6D representation)
    
    Args:
        euler: [..., 3] Euler angles [pitch, yaw, roll] in radians
    
    Returns:
        [..., 6] First 2 columns of rotation matrix [r1x, r1y, r1z, r2x, r2y, r2z]
    """
    from scipy.spatial.transform import Rotation as R
    
    is_torch = isinstance(euler, torch.Tensor)
    if is_torch:
        euler_np = euler.detach().cpu().numpy()
    else:
        euler_np = euler
    
    rot = R.from_euler('xyz', euler_np, degrees=False)
    rot_mat = rot.as_matrix()  # [..., 3, 3]
    
    # Extract first 2 columns
    rot_6d = np.concatenate([rot_mat[..., :, 0], rot_mat[..., :, 1]], axis=-1)
    
    if is_torch:
        return torch.tensor(rot_6d, dtype=euler.dtype, device=euler.device)
    return rot_6d


def rotation_matrix_to_euler(rot_6d: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert 6D rotation representation to Euler angles
    
    Args:
        rot_6d: [..., 6] First 2 columns of rotation matrix [r1x, r1y, r1z, r2x, r2y, r2z]
    
    Returns:
        [..., 3] Euler angles [pitch, yaw, roll] in radians
    """
    from scipy.spatial.transform import Rotation as R
    
    is_torch = isinstance(rot_6d, torch.Tensor)
    if is_torch:
        rot_6d_np = rot_6d.detach().cpu().numpy()
    else:
        rot_6d_np = rot_6d
    
    # Reconstruct full rotation matrix from 6D representation
    r1 = rot_6d_np[..., :3]
    r2 = rot_6d_np[..., 3:6]
    
    # Gram-Schmidt orthogonalization
    r1_norm = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)
    r2_orth = r2 - (r1_norm * (r1_norm * r2).sum(axis=-1, keepdims=True))
    r2_norm = r2_orth / (np.linalg.norm(r2_orth, axis=-1, keepdims=True) + 1e-8)
    r3 = np.cross(r1_norm, r2_norm, axis=-1)
    
    # Stack into rotation matrix
    rot_mat = np.stack([r1_norm, r2_norm, r3], axis=-1)  # [..., 3, 3]
    
    rot = R.from_matrix(rot_mat)
    euler = rot.as_euler('xyz', degrees=False)
    
    if is_torch:
        return torch.tensor(euler, dtype=rot_6d.dtype, device=rot_6d.device)
    return euler


def detect_format_from_dataset_name(dataset_name: str) -> Optional[CameraDataFormat]:
    """
    Detect format from dataset name convention
    
    Args:
        dataset_name: Dataset name (e.g., 'realestate10k_quat', 'realestate10k_rotmat')
    
    Returns:
        Detected format or None if cannot determine
    """
    dataset_name_lower = dataset_name.lower()
    
    # Check for explicit Euler marker first.
    if 'euler' in dataset_name_lower:
        return CameraDataFormat.FULL_12_EULER
    if 'rotmat9' in dataset_name_lower or 'rotmat_9' in dataset_name_lower or '9d' in dataset_name_lower:
        return CameraDataFormat.POSITION_ROTMAT_9
    # Check for rotation matrix format first (more specific)
    if 'rotmat' in dataset_name_lower or 'rot_mat' in dataset_name_lower:
        return CameraDataFormat.FULL_12_ROTMAT
    # Then check for quaternion
    elif 'quat' in dataset_name_lower:
        return CameraDataFormat.QUATERNION_10
    # Check for specific dimension markers
    elif '12' in dataset_name_lower:
        # Ambiguous in naming, but pipeline default is now 12D rotmat.
        return CameraDataFormat.FULL_12_ROTMAT
    elif '10' in dataset_name_lower:
        return CameraDataFormat.QUATERNION_10
    elif '9' in dataset_name_lower:
        return CameraDataFormat.POSITION_ROTMAT_9
    elif '6' in dataset_name_lower:
        return CameraDataFormat.POSITION_ORIENTATION_6
    elif '5' in dataset_name_lower or 'legacy' in dataset_name_lower:
        return CameraDataFormat.LEGACY_5
    
    return None


def gram_schmidt_orthogonalize(rot_6d: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    """
    Apply Gram-Schmidt orthogonalization to 6D rotation representation.
    Ensures r1 and r2 are orthonormal and reconstructs valid rotation matrix.
    
    Args:
        rot_6d: [..., 6] First 2 columns of rotation matrix [r1x, r1y, r1z, r2x, r2y, r2z]
    
    Returns:
        [..., 6] Orthogonalized rotation matrix columns with unit norm and orthogonality
    """
    is_torch = isinstance(rot_6d, torch.Tensor)
    
    if is_torch:
        # Torch implementation
        r1 = rot_6d[..., :3]
        r2 = rot_6d[..., 3:6]
        
        # Normalize r1
        r1_norm = r1 / (torch.norm(r1, dim=-1, keepdim=True) + 1e-8)
        
        # Orthogonalize r2 w.r.t r1 using Gram-Schmidt
        r2_orth = r2 - (r1_norm * (r1_norm * r2).sum(dim=-1, keepdim=True))
        
        # Normalize r2
        r2_norm = r2_orth / (torch.norm(r2_orth, dim=-1, keepdim=True) + 1e-8)
        
        # Concatenate back
        return torch.cat([r1_norm, r2_norm], dim=-1)
    else:
        # NumPy implementation
        r1 = rot_6d[..., :3]
        r2 = rot_6d[..., 3:6]
        
        # Normalize r1
        r1_norm = r1 / (np.linalg.norm(r1, axis=-1, keepdims=True) + 1e-8)
        
        # Orthogonalize r2 w.r.t r1 using Gram-Schmidt
        r2_orth = r2 - (r1_norm * (r1_norm * r2).sum(axis=-1, keepdims=True))
        
        # Normalize r2
        r2_norm = r2_orth / (np.linalg.norm(r2_orth, axis=-1, keepdims=True) + 1e-8)
        
        # Concatenate back
        return np.concatenate([r1_norm, r2_norm], axis=-1)


def compute_orthogonality_loss(rot_6d: torch.Tensor) -> torch.Tensor:
    """
    Compute orthogonality loss for 6D rotation representation.
    Penalizes deviation from unit norm and orthogonality.
    
    Args:
        rot_6d: [..., 6] Rotation matrix columns [r1x, r1y, r1z, r2x, r2y, r2z]
    
    Returns:
        Scalar loss value
    """
    r1 = rot_6d[..., :3]
    r2 = rot_6d[..., 3:6]
    
    # Unit norm loss: ||r1|| and ||r2|| should be 1
    r1_norm = torch.norm(r1, dim=-1)
    r2_norm = torch.norm(r2, dim=-1)
    loss_norm = torch.mean((r1_norm - 1.0)**2 + (r2_norm - 1.0)**2)
    
    # Orthogonality loss: r1 · r2 should be 0
    dot_product = (r1 * r2).sum(dim=-1)
    loss_orth = torch.mean(dot_product**2)
    
    return loss_norm + loss_orth

    
    print(f"Dataset conversion completed. Output saved to: {output_path}")
