"""
Unified Dataset Configuration Module

This module provides unified dataset configuration that automatically detects
and handles different camera trajectory formats (5, 6, or 12 features).
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
from .unified_data_format import detect_dataset_format, CameraDataFormat
from tkcam.utils import paramUtil

class DatasetConfig:
    """Unified dataset configuration"""
    
    def __init__(self, dataset_name: str, data_root: str, auto_detect_format: bool = True):
        self.dataset_name = dataset_name
        self.data_root = Path(data_root)
        self.auto_detect_format = auto_detect_format
        
        # Initialize with defaults
        self._init_defaults()
        
        # Auto-detect format if enabled and this is a camera dataset
        if auto_detect_format and self._is_camera_dataset():
            self._detect_and_set_format()
    
    def _init_defaults(self):
        """Initialize default configuration values"""
        self.motion_dir = self.data_root / 'new_joint_vecs'
        self.text_dir = self.data_root / 'texts'
        self.joints_num = 1 if self._is_camera_dataset() else 22
        self.radius = 240 * 8 if self._is_camera_dataset() else 4
        self.fps = 30 if self._is_camera_dataset() else 20
        self.max_motion_length = 500 if self._is_camera_dataset() else 196
        self.kinematic_chain = paramUtil.kit_kinematic_chain if self._is_camera_dataset() else paramUtil.t2m_kinematic_chain
        
        # Set dim_pose based on dataset type (known defaults per dataset name)
        _dim_defaults = {
            't2m': 263,
            'kit': 251,
            'cam': 5,
            'realestate10k_6': 6,
            'realestate10k_12': 12,
            'realestate10k_quat': 10,
            'realestate10k_rotmat': 12,
            'realestate10k_rotmat9': 9,
        }
        self.dim_pose = _dim_defaults.get(self.dataset_name, 6)
    
    def _is_camera_dataset(self) -> bool:
        """Check if this is a camera trajectory dataset"""
        camera_datasets = ["cam", "estate", "estate_v", "realestate", "realestate10k"]
        return any(name in self.dataset_name.lower() for name in camera_datasets)
    
    def _detect_and_set_format(self):
        """Auto-detect data format and set dim_pose accordingly"""
        try:
            if self.motion_dir.exists():
                detected_format = detect_dataset_format(str(self.data_root))
                # FULL_12_ROTMAT uses enum value 13 as unique ID but has 12 actual features
                self.dim_pose = 12 if detected_format == CameraDataFormat.FULL_12_ROTMAT else detected_format.value
                self.detected_format = detected_format
                print(f"Auto-detected format: {detected_format.name} ({self.dim_pose}D features)")
            else:
                print(f"Motion directory not found: {self.motion_dir}. Using default dim_pose={self.dim_pose}")
                self.detected_format = CameraDataFormat.POSITION_ORIENTATION_6  # Default
        except Exception as e:
            print(f"Failed to auto-detect format for {self.dataset_name}: {e}")
            print(f"Using default dim_pose={self.dim_pose}")
            self.detected_format = CameraDataFormat.POSITION_ORIENTATION_6  # Default
    
    def get_dataset_opt_path(self, checkpoints_dir: str = './checkpoints') -> str:
        """Get the dataset option path for evaluation"""
        if self.dataset_name == "t2m":
            return f'{checkpoints_dir}/t2m/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "kit":
            return f'{checkpoints_dir}/kit/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "cam":
            # For camera datasets, use cam checkpoint as fallback
            return f'{checkpoints_dir}/cam/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "realestate10k_6":
            return f'{checkpoints_dir}/realestate10k_6/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "realestate10k_12":
            return f'{checkpoints_dir}/realestate10k_12/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "realestate10k_quat":
            return f'{checkpoints_dir}/realestate10k_quat/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "realestate10k_rotmat":
            return f'{checkpoints_dir}/realestate10k_rotmat/Comp_v6_KLD005/opt.txt'
        elif self.dataset_name == "realestate10k_rotmat9":
            return f'{checkpoints_dir}/realestate10k_rotmat9/Comp_v6_KLD005/opt.txt'
        else:
            raise ValueError(f"Unknown dataset '{self.dataset_name}'. Please provide custom_data_root or use: {list(default_roots.keys())}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'dataset_name': self.dataset_name,
            'data_root': str(self.data_root),
            'motion_dir': str(self.motion_dir),
            'text_dir': str(self.text_dir),
            'joints_num': self.joints_num,
            'dim_pose': self.dim_pose,
            'radius': self.radius,
            'fps': self.fps,
            'max_motion_length': self.max_motion_length,
            'kinematic_chain': self.kinematic_chain,
            'detected_format': getattr(self, 'detected_format', None)
        }
    
    def apply_to_opt(self, opt):
        """Apply configuration to options object"""
        opt.data_root = str(self.data_root)
        opt.motion_dir = str(self.motion_dir)
        opt.text_dir = str(self.text_dir)
        opt.joints_num = self.joints_num
        opt.max_motion_length = self.max_motion_length
        
        # Set format-specific attributes
        if hasattr(self, 'detected_format'):
            opt.detected_format = self.detected_format
            opt.dim_pose = self.dim_pose

def create_dataset_config(dataset_name: str, custom_data_root: Optional[str] = None) -> DatasetConfig:
    """
    Create dataset configuration with automatic format detection
    
    Args:
        dataset_name: Name of the dataset
        custom_data_root: Custom data root path, if None uses default mapping
        
    Returns:
        DatasetConfig instance
    """
    
    # Default data root mapping
    default_roots = {
        "t2m": "./dataset/HumanML3D/",
        "kit": "./dataset/KIT-ML/",
        "cam": "./dataset/CameraTraj/",
        "estate": "./dataset/Estate/",
        "estate_v": "./dataset/Estate/",
        "realestate": "./dataset/RealEstate10K/",
        "realestate10k": "./dataset/RealEstate10K/",
        "realestate10k_6": "./dataset/RealEstate10K_6feat/",
        "realestate10k_12": "./dataset/RealEstate10K_12feat/",
        "realestate10k_quat": "./dataset/RealEstate10K_quat/",
        "realestate10k_rotmat": "./dataset/RealEstate10K_rotmat/",
        "realestate10k_rotmat9": "./dataset/RealEstate10K_rotmat9/"
    }
    
    # Use custom root or default
    if custom_data_root:
        data_root = custom_data_root
    else:
        data_root = default_roots.get(dataset_name)
        if data_root is None:
            raise ValueError(f"Unknown dataset '{dataset_name}'. Please provide custom_data_root or use: {list(default_roots.keys())}")
    
    return DatasetConfig(dataset_name, data_root)

def get_unified_dataset_config(opt) -> Dict[str, Any]:
    """
    Get unified dataset configuration for training scripts
    
    Args:
        opt: Options object with dataset_name and optionally data_root
        
    Returns:
        Dictionary with dataset configuration
    """
    
    # Use custom data root if provided
    custom_root = getattr(opt, 'data_root', None)
    
    # Create configuration
    config = create_dataset_config(opt.dataset_name, custom_root)
    
    # Apply to opt object
    config.apply_to_opt(opt)
    
    # Return configuration info
    config_dict = config.to_dict()
    config_dict['dataset_opt_path'] = config.get_dataset_opt_path(getattr(opt, 'checkpoints_dir', './checkpoints'))
    
    return config_dict

# Legacy support functions for backward compatibility
def get_legacy_config(dataset_name: str) -> Dict[str, Any]:
    """Get legacy dataset configuration (for backward compatibility)"""
    
    if dataset_name == "t2m":
        return {
            'data_root': './dataset/HumanML3D/',
            'joints_num': 22,
            'dim_pose': 263,
            'fps': 20,
            'radius': 4,
            'kinematic_chain': paramUtil.t2m_kinematic_chain,
            'dataset_opt_path': './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        }
    elif dataset_name == "kit":
        return {
            'data_root': './dataset/KIT-ML/',
            'joints_num': 21,
            'dim_pose': 251,
            'fps': 12.5,
            'radius': 240 * 8,
            'max_motion_length': 196,
            'kinematic_chain': paramUtil.kit_kinematic_chain,
            'dataset_opt_path': './checkpoints/kit/Comp_v6_KLD005/opt.txt'
        }
    elif dataset_name == "cam":
        return {
            'data_root': './dataset/CameraTraj/',
            'joints_num': 1,
            'dim_pose': 5,
            'fps': 30,
            'radius': 240 * 8,
            'max_motion_length': 240,
            'kinematic_chain': paramUtil.kit_kinematic_chain,
            'dataset_opt_path': './checkpoints/cam/Comp_v6_KLD005/opt.txt'
        }
    elif dataset_name == "realestate10k_6":
        return {
            'data_root': './dataset/RealEstate10K_6feat/',
            'joints_num': 1,
            'dim_pose': 6,
            'fps': 30,
            'radius': 240 * 8,
            'max_motion_length': 240,
            'kinematic_chain': paramUtil.kit_kinematic_chain,
            'dataset_opt_path': './checkpoints/realestate10k_6/Comp_v6_KLD005/opt.txt' # to be updated, may need to train need evaluation model
        }
    elif dataset_name == "realestate10k_12":
        return {
            'data_root': './dataset/RealEstate10K_12feat/',
            'joints_num': 1,
            'dim_pose': 12,
            'fps': 30,
            'radius': 240 * 8,
            'max_motion_length': 240,
            'kinematic_chain': paramUtil.kit_kinematic_chain,
            'dataset_opt_path': './checkpoints/realestate10k_12/Comp_v6_KLD005/opt.txt'
        }
    elif dataset_name == "realestate10k_rotmat9":
        return {
            'data_root': './dataset/RealEstate10K_rotmat9/',
            'joints_num': 1,
            'dim_pose': 9,
            'fps': 30,
            'radius': 240 * 8,
            'max_motion_length': 500,
            'kinematic_chain': paramUtil.kit_kinematic_chain,
            'dataset_opt_path': './checkpoints/realestate10k_rotmat9/Comp_v6_KLD005/opt.txt'
        }
    else:
        raise KeyError(f'Dataset {dataset_name} does not exist')
