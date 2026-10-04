"""Video encoder module for CLaTr frame-trajectory evaluator.

This module provides a temporal-aware video encoder that processes
sequences of frames and outputs embeddings compatible with the CLaTr
trajectory encoder.

Supports two backends:
1. SimpleTemporalEncoder: Lightweight 2D CNN + temporal conv
2. TC-CLIP: State-of-the-art video transformer with temporal contextualization
"""

from pathlib import Path
import torch
import torch.nn as nn
from einops import rearrange


class SimpleTemporalEncoder(nn.Module):
    """Simple temporal encoder using 3D convolutions and temporal pooling.
    
    This is a lightweight alternative to TC-CLIP for initial testing.
    Can be replaced with TC-CLIP's vision transformer later.
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_frames: int = 8,
        image_size: int = 224,
    ):
        super().__init__()
        
        self.num_frames = num_frames
        self.output_dim = output_dim
        
        # Spatial feature extractor (2D CNN backbone)
        self.spatial_encoder = nn.Sequential(
            # Input: (B, C, T, H, W) -> (B*T, C, H, W)
            nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Conv blocks
            self._make_conv_block(64, 128, stride=2),
            self._make_conv_block(128, 256, stride=2),
            self._make_conv_block(256, hidden_dim, stride=2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        
        # Temporal encoder (1D conv over time dimension)
        self.temporal_encoder = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Projection to output dimension
        self.projection = nn.Linear(hidden_dim, output_dim)
        
    def _make_conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames.
        
        Args:
            frames: (B, T, C, H, W) where T is num_frames
            
        Returns:
            features: (B, output_dim) - aggregated video embedding
        """
        B, T, C, H, W = frames.shape
        
        # Process each frame independently
        frames_flat = rearrange(frames, 'b t c h w -> (b t) c h w')
        spatial_feats = self.spatial_encoder(frames_flat)  # (B*T, hidden_dim, 1, 1)
        spatial_feats = spatial_feats.squeeze(-1).squeeze(-1)  # (B*T, hidden_dim)
        
        # Reshape to temporal sequence
        spatial_feats = rearrange(spatial_feats, '(b t) d -> b d t', b=B, t=T)
        
        # Temporal encoding
        temporal_feats = self.temporal_encoder(spatial_feats)  # (B, hidden_dim, T)
        
        # Temporal pooling (mean over time)
        pooled_feats = temporal_feats.mean(dim=-1)  # (B, hidden_dim)
        
        # Project to output dimension
        output = self.projection(pooled_feats)  # (B, output_dim)
        
        return output


class TCCLIPEncoder(nn.Module):
    """TC-CLIP vision encoder for video understanding.
    
    Uses the temporal contextualization mechanism from TC-CLIP paper (ECCV 2024).
    Loads pretrained weights from TC-CLIP checkpoints.
    """
    
    def __init__(
        self,
        output_dim: int = 512,
        num_frames: int = 8,
        image_size: int = 224,
        pretrained_path: str = None,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.output_dim = output_dim
        self.num_frames = num_frames
        
        try:
            from tkcam.models.tc_clip import vision_transformer_tc
        except ImportError:
            raise ImportError(
                "TC-CLIP modules not found. Please ensure models/tc_clip/ exists "
                "with vision_transformer_tc.py and dependencies"
            )
        
        # TC-CLIP design details (matching their default config)
        design_details = {
            'vision_model': 'TCVisionTransformer',
            'vision_block': 'TCAttentionBlock',
            'positional_embedding_type': 'joint',  # joint spatio-temporal
            'temporal_length': num_frames,
            'context_token_k': 8,  # number of context tokens
            'seed_token_a': 0.5,  # seed token ratio (0.5 * 196 = 98 seed tokens per frame)
            'local_global_bias': 0.5,  # local-global attention bias
            'tome_r': [0],  # ToMe token reduction schedule (disabled)
            'tome_d': 0,  # ToMe depth
        }
                # Create TC-CLIP vision transformer (ViT-B/16 architecture)
        self.visual = vision_transformer_tc.TCVisionTransformer(
            input_resolution=image_size,
            patch_size=16,
            num_frames=num_frames,
            width=768,  # ViT-B embedding dimension
            layers=12,  # ViT-B depth
            heads=12,  # ViT-B attention heads
            output_dim=512,  # CLIP projection dimension
            design_details=design_details,
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            self.load_pretrained(pretrained_path)
        
        # Optionally freeze the backbone
        if freeze_backbone:
            for param in self.visual.parameters():
                param.requires_grad = False
            print("TC-CLIP backbone frozen")
        
        # Projection head to match CLaTr output dimension
        self.projection = nn.Linear(512, output_dim) if output_dim != 512 else nn.Identity()
        
    def load_pretrained(self, checkpoint_path):
        """Load pretrained TC-CLIP weights."""
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            print(f"Warning: Pretrained checkpoint not found at {checkpoint_path}")
            print("Training from scratch...")
            return
        
        print(f"Loading TC-CLIP pretrained weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter visual/image encoder weights
        visual_state_dict = {}
        for k, v in state_dict.items():
            # Try different key patterns
            new_key = None
            if k.startswith('visual.'):
                new_key = k.replace('visual.', '')
            elif k.startswith('model.visual.'):
                new_key = k.replace('model.visual.', '')
            elif k.startswith('module.image_encoder.'):
                new_key = k.replace('module.image_encoder.', '')
            elif k.startswith('image_encoder.'):
                new_key = k.replace('image_encoder.', '')
            
            if new_key:
                visual_state_dict[new_key] = v
        
        if visual_state_dict:
            # Filter out incompatible keys (positional_embedding has different shape)
            incompatible_keys = []
            for k in list(visual_state_dict.keys()):
                if k == 'positional_embedding':
                    # Skip positional embedding - different shape for multi-frame
                    incompatible_keys.append(k)
                    del visual_state_dict[k]
            
            missing, unexpected = self.visual.load_state_dict(visual_state_dict, strict=False)
            print(f"✓ Loaded {len(visual_state_dict)} visual encoder params")
            if incompatible_keys:
                print(f"  Skipped incompatible keys: {incompatible_keys} (will be randomly initialized)")
            if missing:
                print(f"  Missing keys: {len(missing)} (expected for new layers)")
            if unexpected:
                print(f"  Unexpected keys: {unexpected[:3]}..." if len(unexpected) > 3 else f"  Unexpected: {unexpected}")
        else:
            print("Warning: No visual encoder weights found in checkpoint")
            print(f"Available keys: {list(state_dict.keys())[:5]}")
    
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        """Encode video frames using TC-CLIP.
        
        Args:
            frames: (B, T, C, H, W) where T is num_frames
            
        Returns:
            features: (B, output_dim) - aggregated video embedding
        """
        # TC-CLIP expects (B, T, C, H, W)
        cls_tokens, context_tokens, _, _ = self.visual(frames)
        
        # cls_tokens: [n_layers, B, T, D] - CLS tokens from each layer for each frame
        # context_tokens: [n_layers, B, K, D] - Context tokens from each layer
        
        # Use final layer CLS tokens
        final_cls = cls_tokens[-1]  # (B, T, D)
        
        # Aggregate over time (mean pooling)
        video_embedding = final_cls.mean(dim=1)  # (B, D)
        
        # Project to output dimension
        output = self.projection(video_embedding)  # (B, output_dim)
        
        return output


class VideoEncoder(nn.Module):
    """Video encoder wrapper for CLaTr.
    
    Supports two backends:
    - "simple": Lightweight 2D CNN + temporal convolution
    - "tc_clip": TC-CLIP vision transformer with temporal contextualization
    """
    
    def __init__(
        self,
        encoder_type: str = "simple",
        hidden_dim: int = 512,
        output_dim: int = 512,
        num_frames: int = 8,
        image_size: int = 224,
        pretrained_path: str = None,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.num_frames = num_frames
        self.output_dim = output_dim
        
        # Set num_feats for compatibility with TEMOS encoder interface
        # This should be different from trajectory num_feats to allow auto-detection
        # We use a large value since frames are (T, C, H, W)
        self.num_feats = num_frames * 3 * image_size * image_size
        
        if encoder_type == "simple":
            self.encoder = SimpleTemporalEncoder(
                input_channels=3,
                hidden_dim=hidden_dim,
                output_dim=output_dim,
                num_frames=num_frames,
                image_size=image_size,
            )
        elif encoder_type == "tc_clip":
            self.encoder = TCCLIPEncoder(
                output_dim=output_dim,
                num_frames=num_frames,
                image_size=image_size,
                pretrained_path=pretrained_path,
                freeze_backbone=freeze_backbone,
            )
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
    
    def forward(self, x, mask=None):
        """Forward pass compatible with TEMOS encoder interface.
        
        Args:
            x: Either frames tensor (B, T, C, H, W) or flattened (B, num_feats)
               For VideoEncoder, we expect the frames to be passed directly
            mask: Optional padding mask (ignored for now, can be used for frame masking)
            
        Returns:
            For VAE compatibility, returns (B, 2, output_dim) with [mu, logvar]
            where logvar is set to zeros (deterministic encoding)
        """
        # If x is already frames (B, T, C, H, W), use directly
        if x.dim() == 5:
            frames = x
        # If x is flattened, we can't reconstruct frames - this shouldn't happen
        # in our setup since we pass frames directly in compute_loss
        else:
            raise ValueError(f"Expected 5D frames tensor, got shape {x.shape}")
        
        # Encode frames
        embeddings = self.encoder(frames)  # (B, output_dim)
        
        # For VAE compatibility, return [mu, logvar] where logvar=0 (deterministic)
        # Shape: (B, 2, output_dim)
        mu = embeddings.unsqueeze(1)  # (B, 1, output_dim)
        logvar = torch.zeros_like(mu)  # (B, 1, output_dim)
        
        return torch.cat([mu, logvar], dim=1)  # (B, 2, output_dim)E
