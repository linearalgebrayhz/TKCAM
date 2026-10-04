"""
Sparse Keyframe Encoder for MoMask Camera Trajectory Generation

Randomly samples N: [1,4] keyframes from T frames, encodes with ResNet (trainable),
creates sparse tensor, and downsamples to match motion token sequence length.

Architecture:
    1. Random sparse sampling: N keyframes from T frames
    2. ResNet18/34 encoding: (N frames) → (N, 512)
    3. Sparse tensor creation: (N, 512) → (T, 512) with zeros
    4. Temporal conv downsampling: (T, 512) → (T//4, latent_dim)
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from typing import List
from pathlib import Path


class SparseKeyframeEncoder(nn.Module):
    """
    Encodes sparse keyframes using ResNet + Temporal Convolution.
    
    Args:
        resnet_arch: 'resnet18' or 'resnet34'
        latent_dim: Output dimension (should match motion embedding dim, e.g., 384)
        pretrained: Use ImageNet pretrained weights
    """
    
    def __init__(self, resnet_arch='resnet18', latent_dim=384, pretrained=True):
        super().__init__()
        
        self.resnet_arch = resnet_arch
        self.latent_dim = latent_dim
        
        # Load ResNet backbone (trainable)
        if resnet_arch == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.resnet_dim = 512
        elif resnet_arch == 'resnet34':
            resnet = models.resnet34(pretrained=pretrained)
            self.resnet_dim = 512
        else:
            raise ValueError(f"Unsupported ResNet architecture: {resnet_arch}")
        
        # Remove final FC layer, keep feature extractor
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Temporal downsampling: (T, 512) -> (T//4, latent_dim)
        # Use 2 conv layers with stride=2 each: T -> T/2 -> T/4
        self.temporal_downsample = nn.Sequential(
            nn.Conv1d(self.resnet_dim, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, latent_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(inplace=True)
        )
        
        # Pre-create image transform (avoid recreating every forward pass)
        import torchvision.transforms as transforms
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"SparseKeyframeEncoder: {resnet_arch} (trainable) + Temporal Conv → {latent_dim}D")
    
    def encode_frames(self, images):
        """
        Encode a batch of images with ResNet.
        
        Args:
            images: (N_total, C, H, W) - all keyframes from batch stacked
        
        Returns:
            features: (N_total, resnet_dim) - ResNet features
        """
        # ResNet forward
        features = self.resnet_features(images)  # (N_total, resnet_dim, 1, 1)
        features = features.squeeze(-1).squeeze(-1)  # (N_total, resnet_dim)
        return features
    
    def forward(self, frames_batch, m_lens, deterministic=False):
        """
        OPTIMIZED forward pass with GPU-native image decoding and batch processing.
        
        Args:
            frames_batch: List of frame paths, length B
                Each element is List[Path] - paths to ALL available frames for that scene
            m_lens: Tensor (B,) - original motion lengths (before VQ downsampling)
            deterministic: If True, always sample N=2 frames at fixed positions (for validation)
        
        Returns:
            frame_embeddings: Tensor (T_motion, B, latent_dim)
                Where T_motion = T_original // 4 (matches motion token sequence length)
            has_frames: bool - Whether any sample in batch has actual frames
        """
        import random
        import numpy as np
        
        device = next(self.parameters()).device
        batch_size = len(frames_batch)
        
        # ===== PHASE 1: Sample keyframe INDICES for entire batch =====
        batch_metadata = []
        total_sampled_frames = 0
        
        for b_idx, frame_paths in enumerate(frames_batch):
            T = int(m_lens[b_idx].item())
            num_available_frames = len(frame_paths) if frame_paths else 0
            
            if num_available_frames == 0 or T == 0:
                batch_metadata.append({
                    'b_idx': b_idx,
                    'T': T,
                    'N': 0,
                    'sampled_indices': [],
                    'sampled_paths': []
                })
                continue
            
            # Sample keyframe indices
            if deterministic:
                N = 2
                max_frame_idx = min(num_available_frames, T) - 1
                if max_frame_idx < 1:
                    sampled_indices = [0]
                    N = 1
                else:
                    idx1 = max_frame_idx // 3
                    idx2 = (max_frame_idx * 2) // 3
                    sampled_indices = sorted([idx1, idx2])
            else:
                N = random.randint(0, 4)
                
                if N == 0:
                    batch_metadata.append({
                        'b_idx': b_idx,
                        'T': T,
                        'N': 0,
                        'sampled_indices': [],
                        'sampled_paths': []
                    })
                    continue
                
                max_frame_idx = min(num_available_frames, T) - 1
                if max_frame_idx < 0:
                    max_frame_idx = 0
                
                perm = torch.randperm(max_frame_idx + 1)[:N]
                sampled_indices = sorted(perm.tolist())
            
            # Collect sampled frame paths
            sampled_paths = [frame_paths[idx] for idx in sampled_indices]
            
            batch_metadata.append({
                'b_idx': b_idx,
                'T': T,
                'N': N,
                'sampled_indices': sampled_indices,
                'sampled_paths': sampled_paths
            })
            total_sampled_frames += N
        
        # ===== PHASE 2: Batch load and process ALL sampled frames at once =====
        if total_sampled_frames > 0:
            try:
                # Try GPU-native decoding with torchvision.io (fastest)
                import torchvision.io as tvio
                
                # Read all images as byte streams and decode on GPU
                all_images_list = []
                for meta in batch_metadata:
                    for path in meta['sampled_paths']:
                        try:
                            # Read image file as bytes
                            img_bytes = tvio.read_file(str(path))
                            # Decode directly to GPU tensor (RGB, uint8)
                            img_tensor = tvio.decode_jpeg(img_bytes, device=device)  # (H, W, 3)
                            all_images_list.append(img_tensor)
                        except Exception as e:
                            # Fallback: create dummy tensor on GPU
                            all_images_list.append(torch.zeros(224, 224, 3, dtype=torch.uint8, device=device))
                
                if len(all_images_list) > 0:
                    # Stack all images: (total_N, H, W, 3)
                    all_images_tensor = torch.stack(all_images_list)
                    
                    # Batch resize and normalize on GPU
                    # Convert to float and permute to (total_N, 3, H, W)
                    all_images_tensor = all_images_tensor.permute(0, 3, 1, 2).float() / 255.0
                    
                    # Batch resize
                    all_images_tensor = torch.nn.functional.interpolate(
                        all_images_tensor, 
                        size=(224, 224), 
                        mode='bilinear', 
                        align_corners=False
                    )
                    
                    # Normalize (ImageNet stats)
                    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
                    all_images_tensor = (all_images_tensor - mean) / std
                    
                    # Single ResNet forward pass for ALL images
                    with torch.set_grad_enabled(self.training):
                        all_features = self.encode_frames(all_images_tensor)  # (total_N, resnet_dim)
                else:
                    all_features = None
                    
            except (ImportError, AttributeError, RuntimeError) as e:
                # Fallback to CPU PIL processing if GPU decode fails
                print(f"Warning: GPU image decoding failed ({e}), falling back to CPU PIL")
                from PIL import Image
                
                all_images_list = []
                for meta in batch_metadata:
                    for path in meta['sampled_paths']:
                        try:
                            img = Image.open(path).convert('RGB')
                            img_tensor = self.image_transform(img)
                            all_images_list.append(img_tensor)
                        except Exception:
                            all_images_list.append(torch.zeros(3, 224, 224))
                
                if len(all_images_list) > 0:
                    all_images_tensor = torch.stack(all_images_list).to(device)
                    with torch.set_grad_enabled(self.training):
                        all_features = self.encode_frames(all_images_tensor)
                else:
                    all_features = None
        else:
            all_features = None
        
        # ===== PHASE 3: Reconstruct sparse tensors (optimized with scatter) =====
        sparse_features_list = []
        has_any_frames = total_sampled_frames > 0
        
        feature_offset = 0
        max_T = max(meta['T'] for meta in batch_metadata)
        
        for meta in batch_metadata:
            T = meta['T']
            N = meta['N']
            
            if N == 0 or all_features is None:
                sparse_features = torch.zeros(T, self.resnet_dim, device=device)
            else:
                # Extract this sample's features
                sample_features = all_features[feature_offset:feature_offset + N]
                feature_offset += N
                
                # Create sparse tensor using scatter (faster than loop)
                sparse_features = torch.zeros(T, self.resnet_dim, device=device)
                indices_tensor = torch.tensor(meta['sampled_indices'], device=device, dtype=torch.long)
                sparse_features.index_copy_(0, indices_tensor, sample_features)
            
            # Pad to max_T
            if T < max_T:
                padding = torch.zeros(max_T - T, self.resnet_dim, device=device)
                sparse_features = torch.cat([sparse_features, padding], dim=0)
            
            sparse_features_list.append(sparse_features)
        
        # Stack: (B, max_T, resnet_dim)
        sparse_batch = torch.stack(sparse_features_list)
        
        # Temporal downsampling: (B, max_T, resnet_dim) → (B, max_T//4, latent_dim)
        sparse_batch = sparse_batch.permute(0, 2, 1)  # (B, resnet_dim, max_T)
        downsampled = self.temporal_downsample(sparse_batch)  # (B, latent_dim, max_T//4)
        downsampled = downsampled.permute(2, 0, 1)  # (max_T//4, B, latent_dim)
        
        return downsampled, has_any_frames


class SparseKeyframeEncoderWithCaching(SparseKeyframeEncoder):
    """
    Version with optional frame caching for faster training.
    Can cache ResNet features to avoid repeated encoding.
    """
    
    def __init__(self, resnet_arch='resnet18', latent_dim=384, pretrained=True, use_cache=False):
        super().__init__(resnet_arch, latent_dim, pretrained)
        self.use_cache = use_cache
        self.feature_cache = {}  # hash_id -> (T, resnet_dim)
        print(f"Frame caching: {'enabled' if use_cache else 'disabled'}")
    
    def cache_scene_features(self, hash_id, features):
        """Cache ResNet features for a scene."""
        if self.use_cache:
            self.feature_cache[hash_id] = features.detach().cpu()
    
    def get_cached_features(self, hash_id):
        """Retrieve cached features if available."""
        if self.use_cache and hash_id in self.feature_cache:
            return self.feature_cache[hash_id].to(next(self.parameters()).device)
        return None
