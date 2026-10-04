"""
Cached Sparse Keyframe Encoder (FAST VERSION)

Loads precomputed ResNet features from disk instead of computing on-the-fly.
Eliminates image loading bottleneck (~10-20x speedup).

Usage:
    1. Precompute features: python precompute_frame_features.py ...
    2. Use CachedSparseKeyframeEncoder with --cached-features-dir flag
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List
import random


class CachedSparseKeyframeEncoder(nn.Module):
    """
    Loads precomputed ResNet features and applies temporal downsampling.
    
    Args:
        cached_features_dir: Directory containing precomputed .npy feature files
        latent_dim: Output dimension (should match motion embedding dim)
        resnet_dim: ResNet output dimension (512 for ResNet18/34)
    """
    
    def __init__(self, cached_features_dir: str, latent_dim=384, resnet_dim=512):
        super().__init__()
        
        self.cached_features_dir = Path(cached_features_dir)
        self.latent_dim = latent_dim
        self.resnet_dim = resnet_dim
        
        # Verify cache directory exists
        if not self.cached_features_dir.exists():
            raise FileNotFoundError(f"Cached features directory not found: {cached_features_dir}")
        
        # Load metadata
        metadata_file = self.cached_features_dir / 'metadata.json'
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded cached features: {self.metadata['total_scenes']} scenes")
            print(f"  ResNet arch: {self.metadata['resnet_arch']}")
            print(f"  Feature dim: {self.metadata['feature_dim']}")
        else:
            print(f"Warning: No metadata.json found in {cached_features_dir}")
            self.metadata = {}
        
        # Temporal downsampling: (T, 512) -> (T//4, latent_dim)
        self.temporal_downsample = nn.Sequential(
            nn.Conv1d(resnet_dim, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        print(f"CachedSparseKeyframeEncoder: Loading from {cached_features_dir}")
        print(f"  Temporal Conv: (T, {resnet_dim}) → (T//4, {latent_dim})")
    
    def load_cached_features(self, scene_id: str) -> torch.Tensor:
        """
        Load precomputed features for a scene.
        
        Args:
            scene_id: Scene identifier (numeric ID like "000123")
        
        Returns:
            features: (T, resnet_dim) tensor
        """
        feature_file = self.cached_features_dir / f"{scene_id}.npy"
        
        if not feature_file.exists():
            # Scene not found in cache - return None
            return None
        
        try:
            features = np.load(feature_file)  # (T, resnet_dim)
            return torch.from_numpy(features).float()
        except Exception as e:
            print(f"Error loading cached features for {scene_id}: {e}")
            return None
    
    def forward(self, scene_ids: List[str], m_lens: torch.Tensor, deterministic=False):
        """
        Main forward pass: load cached features → sparse sample → temporal downsample.
        
        Args:
            scene_ids: List of scene IDs (numeric format like "000123"), length B
            m_lens: Tensor (B,) - original motion lengths (before VQ downsampling)
            deterministic: If True, always sample N=2 frames at fixed positions
        
        Returns:
            frame_embeddings: Tensor (T_motion, B, latent_dim)
                Where T_motion = T_original // 4 (matches motion token sequence length)
            has_any_frames: bool - whether any sample in batch has frames
        """
        device = next(self.parameters()).device
        batch_size = len(scene_ids)
        
        sparse_features_list = []
        has_any_frames = False
        
        for b_idx, scene_id in enumerate(scene_ids):
            T = int(m_lens[b_idx].item())  # Motion trajectory length
            
            # Load ALL precomputed features for this scene
            cached_features = self.load_cached_features(scene_id)
            
            if cached_features is None:
                # No cached features - create zeros
                sparse_features = torch.zeros(T, self.resnet_dim, device=device)
                sparse_features_list.append(sparse_features)
                continue
            
            # cached_features: (T_full, resnet_dim)
            num_available_frames = cached_features.shape[0]
            
            # Sample keyframes
            if deterministic:
                # Validation mode: fixed N=2 frames at 1/3 and 2/3 positions
                N = 2
                max_frame_idx = min(num_available_frames, T) - 1
                if max_frame_idx < 1:
                    sampled_indices = [0]
                    N = 1
                else:
                    idx1 = max_frame_idx // 3
                    idx2 = (max_frame_idx * 2) // 3
                    sampled_indices = [idx1, idx2]
            else:
                # Training mode: random N ∈ [0, 1, 2, 3, 4]
                N = random.randint(0, 4)
                
                if N == 0:
                    # No keyframes - create all-zeros
                    sparse_features = torch.zeros(T, self.resnet_dim, device=device)
                    sparse_features_list.append(sparse_features)
                    continue
                
                # Sample from available frames, but respect motion length T
                max_frame_idx = min(num_available_frames, T) - 1
                if max_frame_idx < 0:
                    max_frame_idx = 0
                
                # Sample N unique indices
                perm = torch.randperm(max_frame_idx + 1)[:N]
                sampled_indices = sorted(perm.tolist())
            
            # Create sparse tensor: (T, resnet_dim) with zeros except at sampled positions
            sparse_features = torch.zeros(T, self.resnet_dim, device=device)
            
            for idx in sampled_indices:
                if idx < num_available_frames:
                    sparse_features[idx] = cached_features[idx].to(device)
            
            sparse_features_list.append(sparse_features)
            has_any_frames = True
        
        # Pad to same length (max T in batch)
        max_T = max(sf.shape[0] for sf in sparse_features_list)
        
        padded_features = []
        for sf in sparse_features_list:
            T = sf.shape[0]
            if T < max_T:
                padding = torch.zeros(max_T - T, self.resnet_dim, device=device)
                sf = torch.cat([sf, padding], dim=0)
            padded_features.append(sf)
        
        # Stack: (B, T, resnet_dim)
        sparse_batch = torch.stack(padded_features)
        
        # Temporal downsampling: (B, T, resnet_dim) → (B, T//4, latent_dim)
        sparse_batch = sparse_batch.permute(0, 2, 1)  # (B, resnet_dim, T)
        downsampled = self.temporal_downsample(sparse_batch)  # (B, latent_dim, T//4)
        
        # Permute to (T//4, B, latent_dim) for transformer
        downsampled = downsampled.permute(2, 0, 1)
        
        return downsampled, has_any_frames


class HybridKeyframeEncoder(nn.Module):
    """
    Hybrid encoder: tries cached features first, falls back to on-the-fly loading.
    Useful during transition period when cache is being built.
    """
    
    def __init__(self, cached_features_dir: str = None, resnet_arch='resnet18', latent_dim=384):
        super().__init__()
        
        self.use_cache = cached_features_dir is not None
        self.latent_dim = latent_dim
        
        if self.use_cache:
            self.cached_encoder = CachedSparseKeyframeEncoder(
                cached_features_dir, latent_dim
            )
            print("HybridKeyframeEncoder: Using cached features (fast mode)")
        else:
            # Fall back to original on-the-fly encoder
            from tkcam.models.sparse_keyframe_encoder import SparseKeyframeEncoder
            self.live_encoder = SparseKeyframeEncoder(resnet_arch, latent_dim)
            print("HybridKeyframeEncoder: Using on-the-fly loading (slow mode)")
    
    def forward(self, input_data, m_lens, deterministic=False):
        """
        Forward pass: tries cached first, falls back to live loading.
        
        Args:
            input_data: Either List[str] (scene_ids) for cached mode,
                       or List[List[Path]] (frame_paths) for live mode
            m_lens: Motion lengths
            deterministic: Fixed sampling for validation
        """
        if self.use_cache:
            # Assume input_data is scene_ids
            return self.cached_encoder(input_data, m_lens, deterministic)
        else:
            # Assume input_data is frame_paths_batch
            return self.live_encoder(input_data, m_lens, deterministic)

