import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
from tkcam.models.mask_transformer.tools import *
from tkcam.models.mask_transformer.conditioning import ConditioningProvider
from torch.distributions.categorical import Categorical


# ──────────────────── Helper Modules ────────────────────

class InputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim):
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.poseEmbedding = nn.Linear(self.input_feats, self.latent_dim)

    def forward(self, x):
        # [bs, ntokens, input_feats]
        x = x.permute((1, 0, 2))  # [seqlen, bs, input_feats]
        x = self.poseEmbedding(x)  # [seqlen, bs, d]
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [max_len, 1, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)


class OutputProcess_Bert(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, c, seqlen]
        return output


class OutputProcess(nn.Module):
    def __init__(self, out_feats, latent_dim):
        super().__init__()
        self.dense = nn.Linear(latent_dim, latent_dim)
        self.transform_act_fn = F.gelu
        self.LayerNorm = nn.LayerNorm(latent_dim, eps=1e-12)
        self.poseFinal = nn.Linear(latent_dim, out_feats)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        output = self.poseFinal(hidden_states)  # [seqlen, bs, out_feats]
        output = output.permute(1, 2, 0)  # [bs, e, seqlen]
        return output


# ──────────────────── Cross-Attention Block ────────────────────

class CrossAttentionBlock(nn.Module):
    """Pre-Norm Transformer block with Self-Attention → Cross-Attention → FFN.

    Dimension convention (mirrors nn.TransformerEncoder default): seq-first.
      x    : (S, B, D)  — motion tokens
      cond : (T, B, D)  — conditioning tokens already projected to D

    Masks (True = IGNORE / padding, matching nn.MultiheadAttention convention):
      motion_key_padding_mask : (B, S)  — motion padding positions
      cond_key_padding_mask   : (B, T)  — T5 padding / null-token positions
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()

        # ── Sub-layer 1: Self-Attention ──
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)

        # ── Sub-layer 2: Cross-Attention ──
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=False)
        self.norm2 = nn.LayerNorm(d_model)

        # ── Sub-layer 3: Feed-Forward Network ──
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        motion_key_padding_mask: Optional[torch.Tensor] = None,
        cond_key_padding_mask: Optional[torch.Tensor] = None,
        return_cross_attn_weights: bool = False,
    ) -> tuple:
        """Forward pass.

        Returns
        -------
        x : Tensor (S, B, D)
        attn_weights : Optional[Tensor] (B, query_len, key_len)
            Attention weights averaged over heads.  ``None`` when
            ``return_cross_attn_weights=False`` (default).
        """
        # ── 1. Self-Attention (temporal consistency among motion tokens) ──
        x = x + self.self_attn(
            self.norm1(x), self.norm1(x), self.norm1(x),
            key_padding_mask=motion_key_padding_mask,
            need_weights=False,
        )[0]

        # ── 2. Cross-Attention (inject T5 / ID semantics into motion) ──
        x_norm = self.norm2(x)
        cond_norm = self.norm2(cond)          # reuse same norm weights (tied)
        ca_out, ca_weights = self.cross_attn(
            x_norm, cond_norm, cond_norm,
            key_padding_mask=cond_key_padding_mask,
            need_weights=return_cross_attn_weights,
        )
        x = x + ca_out

        # ── 3. Feed-Forward ──
        x = x + self.ff(self.norm3(x))

        # ca_weights shape (when not None): (B, query_len, key_len)
        return x, ca_weights


# ──────────────────── Shared Base Class ────────────────────

class BaseCondTransformer(nn.Module):
    """Shared base for MaskTransformer and ResidualTransformer.

    Consolidates:
      - CLIP loading, freezing, encoding (with built-in finetune guard)
      - Condition encoding (text / action / uncond)
      - Condition masking (classifier-free guidance)
      - Weight initialization
      - First-frame / sparse CLIP image conditioning
      - Core transformer encoder
    """

    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None,
                 finetune_clip=False, finetune_clip_layers=2,
                 conditioning_mode='clip', num_id_samples=50,
                 t5_model_name='t5-base', use_first_frame=False,
                 use_sparse_frames=False, max_sparse_frames=4,
                 visual_drop_prob=0.0, condition_fusion='cross_attn', **kargs):
        super().__init__()
        kargs.pop('use_frames', None)
        kargs.pop('frame_dim', None)
        print(f'latent_dim: {latent_dim}, ff_size: {ff_size}, nlayers: {num_layers}, nheads: {num_heads}, dropout: {dropout}')

        self.code_dim = code_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.opt = opt
        self.finetune_clip = finetune_clip
        self.finetune_clip_layers = finetune_clip_layers
        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        # ── New: conditioning mode (clip / t5 / id_embedding) ──
        self.conditioning_mode = conditioning_mode
        self.condition_fusion = getattr(opt, 'condition_fusion', condition_fusion)
        if self.condition_fusion not in ('cross_attn', 'prefix_self_attn'):
            raise ValueError(f"Unsupported condition_fusion: {self.condition_fusion}")
        self.num_id_samples = num_id_samples
        self.t5_model_name = t5_model_name
        self._use_new_provider = conditioning_mode in ('t5', 'id_embedding')
        self.use_first_frame = use_first_frame
        self.use_sparse_frames = use_sparse_frames
        self.max_sparse_frames = max_sparse_frames
        self.visual_drop_prob = visual_drop_prob
        print(f'Conditioning mode: {conditioning_mode}')
        print(f'Condition fusion: {self.condition_fusion}')
        if visual_drop_prob > 0:
            print(f'Visual CFG dropout: {visual_drop_prob:.2f} '
                  f'(force_mask=True always drops visual for true unconditional)')

        if self._use_new_provider:
            self.cond_provider = ConditioningProvider(
                mode=conditioning_mode,
                latent_dim=latent_dim,
                clip_dim=clip_dim,
                num_samples=num_id_samples,
                t5_model_name=t5_model_name,
                cond_drop_prob=cond_drop_prob,
                device=getattr(opt, 'device', None),
            )

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
        self.num_actions = kargs.get('num_actions', 1)

        # ── Core network layers ──
        self.input_process = InputProcess(self.code_dim, self.latent_dim)
        self.position_enc = PositionalEncoding(self.latent_dim, self.dropout)

        # Cross-attention stack — replaces the old prefix-based encoder.
        self.cross_attn_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.encode_action = partial(F.one_hot, num_classes=self.num_actions)

        # ── Condition embedding (legacy CLIP / action / uncond path) ──
        if not self._use_new_provider:
            if self.cond_mode == 'text':
                self.cond_emb = nn.Linear(self.clip_dim, self.latent_dim)
            elif self.cond_mode == 'action':
                self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)
            elif self.cond_mode == 'uncond':
                self.cond_emb = nn.Identity()
            else:
                raise KeyError("Unsupported condition mode!")
        else:
            # For new provider modes, cond_emb is inside ConditioningProvider
            # but we still keep a dummy for action/uncond fallback
            if self.cond_mode == 'action':
                self.cond_emb = nn.Linear(self.num_actions, self.latent_dim)

        # ── First-Frame CLIP Image Conditioning ──
        if self.use_first_frame:
            print('First-Frame conditioning: ENABLED')
            # Frozen CLIP Vision Encoder (lazy import to avoid hard dep on new transformers)
            from transformers import CLIPVisionModel
            self.clip_image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
            for p in self.clip_image_encoder.parameters():
                p.requires_grad = False
            self.clip_image_encoder.eval()
            # CLIP ViT-B/32 pooler_output dim = 768
            clip_vision_dim = self.clip_image_encoder.config.hidden_size  # 768
            self.frame_proj = nn.Linear(clip_vision_dim, self.latent_dim)
            self.frame_ln = nn.LayerNorm(self.latent_dim)
            print(f'  CLIP Vision: openai/clip-vit-base-patch32 (frozen)')
            print(f'  Projection: {clip_vision_dim} -> {self.latent_dim} + LayerNorm')

        # ── Sparse Keyframe CLIP Image Conditioning (0-4 frames) ──
        if self.use_sparse_frames:
            print(f'Sparse Keyframe conditioning: ENABLED (0-{max_sparse_frames} frames)')
            # Frozen CLIP Vision Encoder (lazy import)
            from transformers import CLIPVisionModel
            if not hasattr(self, 'clip_image_encoder'):
                self.clip_image_encoder = CLIPVisionModel.from_pretrained('openai/clip-vit-base-patch32')
                for p in self.clip_image_encoder.parameters():
                    p.requires_grad = False
                self.clip_image_encoder.eval()
            # Trainable projection layers (may already exist from use_first_frame)
            clip_vision_dim = self.clip_image_encoder.config.hidden_size  # 768
            if not hasattr(self, 'frame_proj'):
                self.frame_proj = nn.Linear(clip_vision_dim, self.latent_dim)
                self.frame_ln = nn.LayerNorm(self.latent_dim)
            # Learnable temporal positional embedding for visual tokens
            max_motion_len = getattr(opt, 'max_motion_length', 196)
            self.visual_pos_embed = nn.Embedding(max_motion_len, self.latent_dim)
            print(f'  CLIP Vision: openai/clip-vit-base-patch32 (frozen)')
            print(f'  Projection: {clip_vision_dim} -> {self.latent_dim} + LayerNorm')
            print(f'  Visual Positional Embedding: {max_motion_len} -> {self.latent_dim}')

    # ── CLIP ────────────────────────────────────────────────

    def _init_clip(self, clip_version):
        """Initialize CLIP model. Call AFTER self.apply(_init_weights)."""
        if self.cond_mode == 'text' and self.conditioning_mode == 'clip':
            print('Loading CLIP...')
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)
            if self.finetune_clip:
                print(f'CLIP Fine-tuning: ENABLED (last {self.finetune_clip_layers} layers)')
            else:
                print('CLIP Fine-tuning: DISABLED (fully frozen)')
        elif self._use_new_provider and self.conditioning_mode == 'clip':
            # Initialise CLIP via the ConditioningProvider
            self.cond_provider.init_clip(clip_version, device=getattr(self.opt, 'device', None))

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        # Convert to FP16 only if NOT fine-tuning (fine-tuning requires FP32 for stability)
        if str(self.opt.device) != "cpu" and not self.finetune_clip:
            clip.model.convert_weights(clip_model)

        if self.finetune_clip:
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False
            total_layers = len(clip_model.transformer.resblocks)
            for i in range(total_layers - self.finetune_clip_layers, total_layers):
                for p in clip_model.transformer.resblocks[i].parameters():
                    p.requires_grad = True
            for p in clip_model.ln_final.parameters():
                p.requires_grad = True
            if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
                clip_model.text_projection.requires_grad = True
            print(f'Unfroze last {self.finetune_clip_layers} transformer layers + final LN + projection')
        else:
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False

        return clip_model

    def encode_text(self, raw_text):
        """Encode text via CLIP. Handles finetune_clip gradient guard internally."""
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        if self.finetune_clip:
            feat_clip_text = self.clip_model.encode_text(text).float()
        else:
            with torch.no_grad():
                feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def encode_first_frame(self, first_frame_pixels):
        """Encode the first frame via frozen CLIP Vision + trainable projection.

        Args:
            first_frame_pixels: (B, 3, 224, 224) preprocessed RGB tensor.

        Returns:
            visual_token: (1, B, D) seq-first, projected and layer-normed.
        """
        with torch.no_grad():
            vision_out = self.clip_image_encoder(pixel_values=first_frame_pixels)
            # pooler_output: (B, hidden_size=768) — CLS pooled representation
            pooled = vision_out.pooler_output.float()  # (B, 768)
        projected = self.frame_proj(pooled)      # (B, latent_dim)
        projected = self.frame_ln(projected)     # (B, latent_dim)
        return projected.unsqueeze(0)            # (1, B, latent_dim)

    def encode_sparse_frames(self, sparse_frames, visual_indices, visual_valid_mask):
        """Encode 0-4 sparse keyframes via frozen CLIP Vision + trainable projection + positional embedding.

        Args:
            sparse_frames: (B, 4, 3, 224, 224) preprocessed RGB tensor — K valid frames + zeros.
            visual_indices: (B, 4) long tensor — frame indices in [0, max_motion_length).
            visual_valid_mask: (B, 4) bool tensor — True for valid frame slots.

        Returns:
            visual_tokens: (4, B, D) seq-first, projected, layer-normed, and positionally embedded.
            visual_ignore_mask: (B, 4) bool tensor — True for positions to IGNORE in attention.
        """
        B, K, C, H, W = sparse_frames.shape  # K = max_sparse_frames = 4
        device = sparse_frames.device

        # Reshape for batch encoding: (B*4, 3, 224, 224)
        flat_frames = sparse_frames.view(B * K, C, H, W)

        with torch.no_grad():
            vision_out = self.clip_image_encoder(pixel_values=flat_frames)
            pooled = vision_out.pooler_output.float()  # (B*4, 768)

        # Project and normalize
        projected = self.frame_proj(pooled)      # (B*4, latent_dim)
        projected = self.frame_ln(projected)     # (B*4, latent_dim)

        # Reshape back: (B, 4, latent_dim)
        visual_tokens = projected.view(B, K, self.latent_dim)

        # Add temporal positional embeddings based on frame indices
        # visual_indices: (B, 4) contains the temporal position of each keyframe
        pos_emb = self.visual_pos_embed(visual_indices)  # (B, 4, latent_dim)
        visual_tokens = visual_tokens + pos_emb

        # Convert to seq-first: (4, B, latent_dim)
        visual_tokens = visual_tokens.permute(1, 0, 2)

        # Convert valid mask to ignore mask (True = IGNORE in PyTorch attention)
        visual_ignore_mask = ~visual_valid_mask  # (B, 4)

        return visual_tokens, visual_ignore_mask

    def encode_condition(self, y, bs, device):
        """Encode condition vector from text / action / uncond.

        For new conditioning modes (t5, id_embedding), delegates to
        ConditioningProvider and returns a tuple:
            (cond_vector, force_mask)              — for clip / action / uncond
            (cond_vector, force_mask, cond_mask)   — for t5 (sequence condition)

        Returns:
            cond_vector: (B, clip_dim) or (B, T, t5_dim) or (B, num_actions) or (B, latent_dim)
            force_mask: bool -- True for uncond mode
            cond_mask: Optional (B, T) bool -- only for t5 mode
        """
        # ── New provider modes ──
        if self._use_new_provider:
            cond, cond_mask, force_mask = self.cond_provider.encode(y, bs, device)
            return cond, force_mask, cond_mask

        # ── Legacy clip / action / uncond path ──
        force_mask = False
        if self.cond_mode == 'text':
            cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.encode_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(bs, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!")
        return cond_vector, force_mask, None

    # ── Shared utilities ───────────────────────────────────

    def mask_cond(self, cond, force_mask=False):
        """Apply CFG dropout. Works for (B, D) single-vector conditions."""
        bs = cond.shape[0]
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Pretrained, frozen encoders created in __init__ (T5 text, CLIP vision). Weight init and
    # train() must leave them alone. (CLIP text, `clip_model`, is loaded after init.)
    PRETRAINED_ENCODERS = ('cond_provider.t5_encoder', 'clip_image_encoder')

    def _pretrained_encoders(self):
        return [m for name, m in self.named_modules() if name in self.PRETRAINED_ENCODERS]

    def _init_trainable_weights(self):
        """Apply _init_weights to every module except the pretrained encoders."""
        frozen = {id(sub) for enc in self._pretrained_encoders() for sub in enc.modules()}
        for module in self.modules():
            if id(module) not in frozen:
                self._init_weights(module)

    def train(self, mode=True):
        super().train(mode)
        for enc in self._pretrained_encoders():
            enc.eval()  # frozen: keeps T5 dropout off during training
        return self

    def _prepare_crossattn_cond(
        self,
        cond: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        force_mask: bool,
        visual_token: Optional[torch.Tensor] = None,
        visual_tokens: Optional[torch.Tensor] = None,
        visual_ignore_mask: Optional[torch.Tensor] = None,
    ):
        """Project and format the conditioning signal for cross-attention.

        Parameters
        ----------
        visual_token : Optional[Tensor]  (1, B, D)
            First-frame visual token, already projected.  When provided, it is
            prepended to the conditioning sequence and its mask position is set
            to *attend* (never masked out).
        visual_tokens : Optional[Tensor]  (K, B, D) where K = max_sparse_frames
            Sparse keyframe visual tokens, already projected and positionally embedded.
            Prepended to the conditioning sequence.
        visual_ignore_mask : Optional[Tensor]  (B, K)
            Mask for sparse visual tokens — True = IGNORE (padding position).
            Required when visual_tokens is provided.

        Returns
        -------
        cond_seq : Tensor  (T_cond, B, D)  — seq-first, projected to latent_dim.
            T_cond = 1  for CLIP / ID / null branches (possibly +1 with visual).
            T_cond = T_text  for T5 conditional branches (possibly +K with visual).
        cond_kp : Optional[Tensor]  (B, T_cond)  — True = ignore (key_padding_mask).
            None when every position should be attended to (all modes except T5
            conditional branch with variable-length text).

        Mask routing summary
        --------------------
        motion tokens  → motion_key_padding_mask (B, S)  fed to self-attention.
        cond tokens    → cond_key_padding_mask   (B, T)  fed to cross-attention.
        These two masks are NEVER mixed together; each sub-layer sees only its
        own mask, eliminating the old seq-length dilution from prefix concat.
        """
        if self._use_new_provider:
            prefix, prefix_kp, _ = self.cond_provider.project_and_prepare(
                cond, cond_mask, force_mask=force_mask)

            # Prepend sparse visual tokens if provided (priority over single visual_token)
            if visual_tokens is not None and visual_ignore_mask is not None:
                # visual_tokens: (K, B, D)  prefix: (T, B, D)
                prefix = torch.cat([visual_tokens, prefix], dim=0)  # (K+T, B, D)
                # Concatenate ignore masks
                if prefix_kp is not None:
                    prefix_kp = torch.cat([visual_ignore_mask, prefix_kp], dim=1)  # (B, K+T)
                else:
                    # All text tokens were attended; use visual_ignore_mask for visual part
                    B = visual_tokens.shape[1]
                    T = prefix.shape[0] - visual_tokens.shape[0]  # text seq len
                    text_attend_mask = torch.zeros(B, T, dtype=torch.bool, device=visual_tokens.device)
                    prefix_kp = torch.cat([visual_ignore_mask, text_attend_mask], dim=1)  # (B, K+T)

            # Prepend single visual token if provided (for first_frame mode)
            elif visual_token is not None:
                # visual_token: (1, B, D)  prefix: (T, B, D)
                prefix = torch.cat([visual_token, prefix], dim=0)  # (1+T, B, D)
                B = visual_token.shape[1]
                # Visual token is ALWAYS valid (False = attend in key_padding_mask)
                vis_mask = torch.zeros(B, 1, dtype=torch.bool, device=visual_token.device)
                if prefix_kp is not None:
                    prefix_kp = torch.cat([vis_mask, prefix_kp], dim=1)  # (B, 1+T)
                else:
                    # All text tokens were attended; add False for visual too → still all-attend
                    # Return None to keep the "attend all" semantics
                    pass

            return prefix, prefix_kp  # (T, B, D), Optional (B, T)
        else:
            # Legacy path: CLIP / action / uncond
            cond = self.mask_cond(cond, force_mask=force_mask)   # (B, D)
            cond_seq = self.cond_emb(cond).unsqueeze(0)          # (1, B, D)

            cond_kp = None
            
            # Prepend sparse visual tokens if provided
            if visual_tokens is not None and visual_ignore_mask is not None:
                cond_seq = torch.cat([visual_tokens, cond_seq], dim=0)  # (K+1, B, D)
                # Need to create mask: visual_ignore_mask + attend for text token
                B = visual_tokens.shape[1]
                text_mask = torch.zeros(B, 1, dtype=torch.bool, device=visual_tokens.device)
                cond_kp = torch.cat([visual_ignore_mask, text_mask], dim=1)  # (B, K+1)
            # Prepend single visual token if provided
            elif visual_token is not None:
                cond_seq = torch.cat([visual_token, cond_seq], dim=0)  # (2, B, D)
                # Both tokens always valid → cond_kp stays None

            return cond_seq, cond_kp

    def parameters_wo_clip(self):
        """Return parameters excluding frozen encoders (CLIP text / T5 / CLIP vision)."""
        frozen_prefixes = ('clip_model.', 'clip_image_encoder.')
        if self._use_new_provider:
            return self.cond_provider.parameters_wo_clip() + [
                p for n, p in self.named_parameters()
                if not n.startswith('cond_provider.') and not any(n.startswith(fp) for fp in frozen_prefixes)]
        if self.finetune_clip:
            return [p for name, p in self.named_parameters()
                    if not name.startswith('clip_image_encoder.')]
        else:
            return [p for name, p in self.named_parameters()
                    if not any(name.startswith(fp) for fp in frozen_prefixes)]

    def _run_cross_attn_blocks(
        self,
        x: torch.Tensor,
        cond_seq: torch.Tensor,
        motion_key_padding_mask: Optional[torch.Tensor],
        cond_key_padding_mask: Optional[torch.Tensor],
        return_attn_weights: bool = False,
        num_capture_layers: int = 2,
    ) -> tuple:
        """Run the cross-attention block stack.

        Optionally captures cross-attention weight tensors from the last
        ``num_capture_layers`` blocks.  This is used for modality-collapse
        diagnostics — the caller should pass ``return_attn_weights=False``
        (the default) during normal training / inference so there is zero
        overhead.

        Parameters
        ----------
        x : Tensor (S, B, D)
        cond_seq : Tensor (T_cond, B, D)
        motion_key_padding_mask : Optional[Tensor] (B, S)
        cond_key_padding_mask : Optional[Tensor] (B, T_cond)
        return_attn_weights : bool
            When True, capture cross-attention weights from the last
            ``num_capture_layers`` blocks.
        num_capture_layers : int
            How many tail blocks to capture weights from (default 2).

        Returns
        -------
        x : Tensor (S, B, D)
        captured_weights : Optional[List[Tensor]]
            ``None`` when ``return_attn_weights=False``.
            Otherwise a list of length ``min(num_capture_layers, num_blocks)``
            where each element is a Tensor of shape (B, S, T_cond) —
            cross-attention weights averaged over heads.
        """
        n = len(self.cross_attn_blocks)
        captured_weights: Optional[List] = [] if return_attn_weights else None

        for i, block in enumerate(self.cross_attn_blocks):
            capture = return_attn_weights and (i >= n - num_capture_layers)
            x, w = block(
                x, cond_seq,
                motion_key_padding_mask=motion_key_padding_mask,
                cond_key_padding_mask=cond_key_padding_mask,
                return_cross_attn_weights=capture,
            )
            if capture and w is not None:
                captured_weights.append(w)

        return x, captured_weights

    def _run_prefix_self_attn_blocks(
        self,
        x: torch.Tensor,
        cond_seq: torch.Tensor,
        motion_key_padding_mask: Optional[torch.Tensor],
        cond_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run MoMask-style condition-prefix self-attention and return motion tokens."""
        cond_len, batch_size, _ = cond_seq.shape
        if cond_key_padding_mask is None:
            cond_key_padding_mask = torch.zeros(
                batch_size, cond_len, dtype=torch.bool, device=cond_seq.device)
        if motion_key_padding_mask is None:
            motion_key_padding_mask = torch.zeros(
                batch_size, x.shape[0], dtype=torch.bool, device=x.device)
        joint = torch.cat([cond_seq, x], dim=0)
        joint_key_padding_mask = torch.cat(
            [cond_key_padding_mask.to(joint.device), motion_key_padding_mask.to(joint.device)],
            dim=1,
        )

        for block in self.cross_attn_blocks:
            joint_norm = block.norm1(joint)
            joint = joint + block.self_attn(
                joint_norm, joint_norm, joint_norm,
                key_padding_mask=joint_key_padding_mask,
                need_weights=False,
            )[0]
            joint = joint + block.ff(block.norm3(joint))
        return joint[cond_len:]


# ──────────────────── Mask Transformer ────────────────────

class MaskTransformer(BaseCondTransformer):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0.1, clip_dim=512, cond_drop_prob=0.1,
                 clip_version=None, opt=None,
                 finetune_clip=False, finetune_clip_layers=2,
                 conditioning_mode='clip', num_id_samples=50,
                 t5_model_name='t5-base', use_first_frame=False,
                 use_sparse_frames=False, max_sparse_frames=4,
                 visual_drop_prob=0.0, **kargs):
        super().__init__(
            code_dim, cond_mode, latent_dim=latent_dim, ff_size=ff_size,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            clip_dim=clip_dim, cond_drop_prob=cond_drop_prob,
            clip_version=clip_version, opt=opt, finetune_clip=finetune_clip,
            finetune_clip_layers=finetune_clip_layers,
            conditioning_mode=conditioning_mode,
            num_id_samples=num_id_samples,
            t5_model_name=t5_model_name,
            use_first_frame=use_first_frame,
            use_sparse_frames=use_sparse_frames,
            max_sparse_frames=max_sparse_frames,
            visual_drop_prob=visual_drop_prob, **kargs)

        # ── Mask-specific layers ──
        _num_tokens = opt.num_tokens + 2  # mask + pad dummies
        self.mask_id = opt.num_tokens
        self.pad_id = opt.num_tokens + 1

        self.output_process = OutputProcess_Bert(out_feats=opt.num_tokens, latent_dim=latent_dim)
        self.token_emb = nn.Embedding(_num_tokens, self.code_dim)

        self._init_trainable_weights()
        self._init_clip(clip_version)

        self.noise_schedule = cosine_schedule
        # BERT-style token noise probabilities (controllable for overfitting experiments)
        self.mask_replace_prob = getattr(opt, 'mask_replace_prob', 0.1)
        print(f'Token noise: mask_replace_prob={self.mask_replace_prob} '
              f'({"pure masking" if self.mask_replace_prob == 0.0 else f"{self.mask_replace_prob*100:.0f}% replace + 88% of rest masked"})')

    def load_and_freeze_token_emb(self, codebook):
        '''
        :param codebook: (c, d)
        '''
        assert self.training, 'Only necessary in training mode'
        c, d = codebook.shape
        self.token_emb.weight = nn.Parameter(
            torch.cat([codebook, torch.zeros(size=(2, d), device=codebook.device)], dim=0))
        self.token_emb.requires_grad_(False)
        print("Token embedding initialized!")

    def trans_forward(self, motion_ids, cond, padding_mask, force_mask=False,
                      cond_mask=None,
                      first_frame_pixels=None,
                      sparse_frames=None, visual_indices=None, visual_valid_mask=None,
                      return_attn_weights=False):
        '''
        Cross-attention forward pass — no prefix tokens.

        :param motion_ids:          (B, S)  — VQ token indices
        :param cond:                (B, raw_cond_dim) or (B, T_text, t5_dim)
        :param padding_mask:        (B, S)  — True = padding (motion_key_padding_mask)
        :param force_mask:          bool   — True activates CFG null branch (drops ALL conditioning
                                    including visual, for a truly unconditional baseline)
        :param cond_mask:           (B, T_text) bool optional — True = VALID T5 token
        :param first_frame_pixels:  (B, 3, 224, 224) optional — preprocessed RGB for frame-0
        :param sparse_frames:       (B, K, 3, 224, 224) optional — sparse keyframe images
        :param visual_indices:      (B, K) long optional — VQ-level temporal indices of keyframes
        :param visual_valid_mask:   (B, K) bool optional — True for valid frame slots
        :param return_attn_weights: bool  — When True, also return cross-attention weights
                                    from the last 2 blocks as a list of (B, S, T_cond) tensors.
                                    Default False (zero overhead in normal operation).
        :return: logits (B, num_tokens, S)  [normal]
                 or (logits, attn_weights_list) when return_attn_weights=True
        '''
        # ── Motion token embedding ──────────────────────────────────────────
        x = self.token_emb(motion_ids)      # (B, S, code_dim)
        x = self.input_process(x)           # (S, B, D)  seq-first
        x = self.position_enc(x)

        # ── Visual tokens (first-frame and/or sparse CLIP) ──────────────────
        # When force_mask=True (CFG null branch) we skip ALL visual encoding so
        # the unconditional baseline is truly unconditioned (no text, no visual).
        visual_token = None
        visual_tokens_sparse = None
        visual_ignore_mask = None

        if not force_mask:
            # ── First-frame conditioning ──
            if self.use_first_frame and first_frame_pixels is not None:
                visual_token = self.encode_first_frame(first_frame_pixels)  # (1, B, D)
                # Independent per-sample stochastic dropout (training only)
                if self.training and self.visual_drop_prob > 0:
                    B = visual_token.shape[1]
                    keep = torch.bernoulli(
                        torch.full((B,), 1.0 - self.visual_drop_prob,
                                   device=visual_token.device)
                    ).view(1, B, 1)  # 1 = keep, 0 = drop
                    visual_token = visual_token * keep

            # ── Sparse keyframe conditioning ──
            if self.use_sparse_frames and sparse_frames is not None:
                visual_tokens_sparse, visual_ignore_mask = self.encode_sparse_frames(
                    sparse_frames, visual_indices, visual_valid_mask)  # (K, B, D), (B, K)
                # Per-sample stochastic dropout: randomly invalidate all K slots for a sample
                if self.training and self.visual_drop_prob > 0:
                    B = visual_tokens_sparse.shape[1]
                    drop = torch.bernoulli(
                        torch.full((B,), self.visual_drop_prob,
                                   device=visual_tokens_sparse.device)
                    ).bool()  # True = drop all frames for that sample
                    visual_ignore_mask = visual_ignore_mask.clone()
                    visual_ignore_mask[drop] = True  # mark all K slots as padding

        # ── Conditioning: project to (T_cond, B, D) ─────────────────────────
        cond_seq, cond_kp = self._prepare_crossattn_cond(
            cond, cond_mask, force_mask,
            visual_token=visual_token,
            visual_tokens=visual_tokens_sparse,
            visual_ignore_mask=visual_ignore_mask)

        captured_attn_weights = None
        if self.condition_fusion == 'prefix_self_attn':
            if return_attn_weights:
                raise ValueError("return_attn_weights is only available for condition_fusion='cross_attn'")
            x = self._run_prefix_self_attn_blocks(
                x, cond_seq,
                motion_key_padding_mask=padding_mask,
                cond_key_padding_mask=cond_kp,
            )
        else:
            # ── Cross-attention stack ────────────────────────────────────────
            x, captured_attn_weights = self._run_cross_attn_blocks(
                x, cond_seq,
                motion_key_padding_mask=padding_mask,
                cond_key_padding_mask=cond_kp,
                return_attn_weights=return_attn_weights,
            )   # (S, B, D)

        logits = self.output_process(x)     # (B, num_tokens, S)
        if return_attn_weights:
            return logits, captured_attn_weights
        return logits

    def forward(self, ids, y, m_lens, return_logits=False,
                first_frame_pixels=None,
                sparse_frames=None, visual_indices=None, visual_valid_mask=None):
        '''
        :param ids: (b, n)
        :param y: raw text for text, (b,) for action, LongTensor for id_embedding
        :param m_lens: (b,)
        :param first_frame_pixels: (b, 3, 224, 224) optional first-frame image tensor
        :param sparse_frames: (b, K, 3, 224, 224) optional sparse keyframe images
        :param visual_indices: (b, K) long optional temporal indices (VQ-level)
        :param visual_valid_mask: (b, K) bool optional valid mask
        :param return_logits: if True, also returns logits tensor
        '''
        bs, ntokens = ids.shape
        device = ids.device

        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        ids = torch.where(non_pad_mask, ids, self.pad_id)

        cond_vector, force_mask, cond_mask = self.encode_condition(y, bs, device)

        # BERT-style masking
        rand_time = uniform((bs,), device=device)
        rand_mask_probs = self.noise_schedule(rand_time)
        num_token_masked = (ntokens * rand_mask_probs).round().clamp(min=1)

        batch_randperm = torch.rand((bs, ntokens), device=device).argsort(dim=-1)
        mask = batch_randperm < num_token_masked.unsqueeze(-1)
        mask &= non_pad_mask

        labels = torch.where(mask, ids, self.mask_id)
        x_ids = ids.clone()

        if self.mask_replace_prob > 0.0:
            # BERT-style noise: replace_prob% random token, 88% of rest → mask, ~12% of rest → keep unchanged
            mask_rid = get_mask_subset_prob(mask, self.mask_replace_prob)
            rand_id = torch.randint_like(x_ids, high=self.opt.num_tokens)
            x_ids = torch.where(mask_rid, rand_id, x_ids)
            # 88% of un-replaced masked positions → mask_id
            mask_mid = get_mask_subset_prob(mask & ~mask_rid, 0.88)
            x_ids = torch.where(mask_mid, self.mask_id, x_ids)
        else:
            # Pure masking: all masked positions → mask_id (no replacement, no keep-unchanged trick)
            x_ids = torch.where(mask, self.mask_id, x_ids)

        logits = self.trans_forward(x_ids, cond_vector, ~non_pad_mask, force_mask,
                                    cond_mask=cond_mask,
                                    first_frame_pixels=first_frame_pixels,
                                    sparse_frames=sparse_frames,
                                    visual_indices=visual_indices,
                                    visual_valid_mask=visual_valid_mask)
        ce_loss, pred_id, acc = cal_performance(logits, labels, ignore_index=self.mask_id)

        if return_logits:
            return ce_loss, pred_id, acc, logits
        return ce_loss, pred_id, acc

    def forward_with_cond_scale(self, motion_ids, cond_vector, padding_mask,
                                cond_scale=3, force_mask=False, cond_mask=None,
                                first_frame_pixels=None,
                                sparse_frames=None, visual_indices=None, visual_valid_mask=None):
        if force_mask:
            return self.trans_forward(motion_ids, cond_vector, padding_mask,
                                      force_mask=True, cond_mask=cond_mask)

        logits = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                    cond_mask=cond_mask,
                                    first_frame_pixels=first_frame_pixels,
                                    sparse_frames=sparse_frames,
                                    visual_indices=visual_indices,
                                    visual_valid_mask=visual_valid_mask)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_ids, cond_vector, padding_mask,
                                        force_mask=True, cond_mask=cond_mask)
        scaled_logits = aux_logits + (logits - aux_logits) * cond_scale
        return scaled_logits

    @torch.no_grad()
    @eval_decorator
    def generate(self, conds, m_lens, timesteps: int, cond_scale: int,
                 temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False,
                 first_frame_pixels=None,
                 sparse_frames=None, visual_indices=None, visual_valid_mask=None):

        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        scores = torch.where(padding_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device),
                                            reversed(range(timesteps))):
            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)

            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  cond_mask=cond_mask,
                                                  first_frame_pixels=first_frame_pixels,
                                                  sparse_frames=sparse_frames,
                                                  visual_indices=visual_indices,
                                                  visual_valid_mask=visual_valid_mask)
            logits = logits.permute(0, 2, 1)  # (b, seqlen, ntoken)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def generate_infill(self, conds, m_lens, timesteps, cond_scale,
                        gt_tokens, anchor_mask,
                        temperature=1, topk_filter_thres=0.9,
                        gsample=False, force_mask=False):
        """Generate tokens with fixed GT anchors (oracle infilling).

        The iterative demasking loop proceeds exactly like ``generate()``,
        but anchor positions are **never** overwritten: they keep their GT
        token values and are always treated as "high-confidence / unmasked".

        Args:
            conds:        text conditions (list[str] or LongTensor for id_embedding)
            m_lens:       (B,) token-level lengths
            timesteps:    number of iterative demasking steps
            cond_scale:   classifier-free guidance scale
            gt_tokens:    (B, S) ground-truth VQ token indices
            anchor_mask:  (B, S) bool — True = anchor (immutable GT token)
            temperature, topk_filter_thres, gsample, force_mask: same as generate()

        Returns:
            ids: (B, S) generated token indices (-1 for padding)
        """
        device = next(self.parameters()).device
        seq_len = max(m_lens)
        batch_size = len(m_lens)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)

        # Initialize: anchors get GT values, everything else is masked
        ids = torch.where(padding_mask, self.pad_id, self.mask_id)
        ids = torch.where(anchor_mask, gt_tokens, ids)

        # Anchors start with maximum confidence so they are never re-masked
        scores = torch.where(padding_mask | anchor_mask, 1e5, 0.)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(
                torch.linspace(0, 1, timesteps, device=device),
                reversed(range(timesteps))):

            rand_mask_prob = self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * m_lens).clamp(min=1)

            # Rank by confidence — anchors always have 1e5 so they stay unmasked
            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            # Never mask anchors
            is_mask = is_mask & ~anchor_mask
            ids = torch.where(is_mask, self.mask_id, ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            # Only update non-anchor masked positions
            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            # Keep anchors and padding at max confidence
            scores = scores.masked_fill(~is_mask | anchor_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit(self, conds, tokens, m_lens, timesteps: int, cond_scale: int,
             temperature=1, topk_filter_thres=0.9, gsample=False, force_mask=False,
             edit_mask=None, padding_mask=None):

        assert edit_mask.shape == tokens.shape if edit_mask is not None else True
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        cond_vector, _, cond_mask = self.encode_condition(conds, tokens.shape[0], device)

        if padding_mask is None:
            padding_mask = ~lengths_to_mask(m_lens, seq_len)

        if edit_mask is None:
            mask_free = True
            ids = torch.where(padding_mask, self.pad_id, tokens)
            edit_mask = torch.ones_like(padding_mask)
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            scores = torch.where(edit_mask, 0., 1e5)
        else:
            mask_free = False
            edit_mask = edit_mask & ~padding_mask
            edit_len = edit_mask.sum(dim=-1)
            ids = torch.where(edit_mask, self.mask_id, tokens)
            scores = torch.where(edit_mask, 0., 1e5)
        starting_temperature = temperature

        for timestep, steps_until_x0 in zip(torch.linspace(0, 1, timesteps, device=device),
                                            reversed(range(timesteps))):
            rand_mask_prob = 0.16 if mask_free else self.noise_schedule(timestep)
            num_token_masked = torch.round(rand_mask_prob * edit_len).clamp(min=1)

            sorted_indices = scores.argsort(dim=1)
            ranks = sorted_indices.argsort(dim=1)
            is_mask = (ranks < num_token_masked.unsqueeze(-1))
            ids = torch.where(is_mask, self.mask_id, ids)

            logits = self.forward_with_cond_scale(ids, cond_vector=cond_vector,
                                                  padding_mask=padding_mask,
                                                  cond_scale=cond_scale,
                                                  force_mask=force_mask,
                                                  cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)

            temperature = starting_temperature
            if gsample:
                pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)
            else:
                probs = F.softmax(filtered_logits / temperature, dim=-1)
                pred_ids = Categorical(probs).sample()

            ids = torch.where(is_mask, pred_ids, ids)

            probs_without_temperature = logits.softmax(dim=-1)
            scores = probs_without_temperature.gather(2, pred_ids.unsqueeze(dim=-1)).squeeze(-1)
            scores = scores.masked_fill(~edit_mask, 1e5) if mask_free else scores.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids

    @torch.no_grad()
    @eval_decorator
    def edit_beta(self, conds, conds_og, tokens, m_lens, cond_scale: int, force_mask=False):
        device = next(self.parameters()).device
        seq_len = tokens.shape[1]

        cond_vector, _, cond_mask = self.encode_condition(conds, tokens.shape[0], device)
        if conds_og is not None:
            cond_vector_og, _, _ = self.encode_condition(conds_og, tokens.shape[0], device)
        else:
            cond_vector_og = None

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        ids = torch.where(padding_mask, self.pad_id, tokens)

        # NOTE: cond_vector_neg is passed but forward_with_cond_scale does not accept it.
        # This is preserved from the original code for future extension.
        logits = self.forward_with_cond_scale(ids,
                                              cond_vector=cond_vector,
                                              padding_mask=padding_mask,
                                              cond_scale=cond_scale,
                                              force_mask=force_mask)
        logits = logits.permute(0, 2, 1)

        probs_without_temperature = logits.softmax(dim=-1)
        tokens[tokens == -1] = 0
        og_tokens_scores = probs_without_temperature.gather(2, tokens.unsqueeze(dim=-1)).squeeze(-1)
        return og_tokens_scores


# ──────────────────── Residual Transformer ────────────────────

class ResidualTransformer(BaseCondTransformer):
    def __init__(self, code_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 cond_drop_prob=0.1, num_heads=4, dropout=0.1, clip_dim=512,
                 shared_codebook=False, share_weight=False,
                 clip_version=None, opt=None, finetune_clip=False,
                 finetune_clip_layers=2,
                 conditioning_mode='clip', num_id_samples=50,
                 t5_model_name='t5-base', use_first_frame=False,
                 use_sparse_frames=False, max_sparse_frames=4,
                 visual_drop_prob=0.0, **kargs):
        super().__init__(
            code_dim, cond_mode, latent_dim=latent_dim, ff_size=ff_size,
            num_layers=num_layers, num_heads=num_heads, dropout=dropout,
            clip_dim=clip_dim, cond_drop_prob=cond_drop_prob,
            clip_version=clip_version, opt=opt,
            finetune_clip=finetune_clip, finetune_clip_layers=finetune_clip_layers,
            conditioning_mode=conditioning_mode,
            num_id_samples=num_id_samples,
            t5_model_name=t5_model_name,
            use_first_frame=use_first_frame,
            use_sparse_frames=use_sparse_frames,
            max_sparse_frames=max_sparse_frames,
            visual_drop_prob=visual_drop_prob, **kargs)

        # ── Residual-specific layers ──
        self.encode_quant = partial(F.one_hot, num_classes=self.opt.num_quantizers)
        self.quant_emb = nn.Linear(self.opt.num_quantizers, self.latent_dim)

        _num_tokens = opt.num_tokens + 1  # one pad dummy
        self.pad_id = opt.num_tokens

        self.output_process = OutputProcess(out_feats=code_dim, latent_dim=latent_dim)

        # ── Codebook weight schemes ──
        if shared_codebook:
            token_embed = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
            self.token_embed_weight = token_embed.expand(opt.num_quantizers - 1, _num_tokens, code_dim)
            if share_weight:
                self.output_proj_weight = self.token_embed_weight
                self.output_proj_bias = None
            else:
                output_proj = nn.Parameter(torch.normal(mean=0, std=0.02, size=(_num_tokens, code_dim)))
                output_bias = nn.Parameter(torch.zeros(size=(_num_tokens,)))
                self.output_proj_weight = output_proj.expand(opt.num_quantizers - 1, _num_tokens, code_dim)
                self.output_proj_bias = output_bias.expand(opt.num_quantizers - 1, _num_tokens)
        else:
            if share_weight:
                self.embed_proj_shared_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 2, _num_tokens, code_dim)))
                self.token_embed_weight_ = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_weight_ = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(1, _num_tokens, code_dim)))
                self.output_proj_bias = None
                self.registered = False
            else:
                self.output_proj_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim)))
                self.output_proj_bias = nn.Parameter(
                    torch.zeros(size=(opt.num_quantizers, _num_tokens)))
                self.token_embed_weight = nn.Parameter(
                    torch.normal(mean=0, std=0.02, size=(opt.num_quantizers - 1, _num_tokens, code_dim)))

        self.shared_codebook = shared_codebook
        self.share_weight = share_weight

        self._init_trainable_weights()
        self._init_clip(clip_version)

    def q_schedule(self, bs, low, high):
        noise = uniform((bs,), device=self.opt.device)
        schedule = 1 - cosine_schedule(noise)
        return torch.round(schedule * (high - low)) + low

    def process_embed_proj_weight(self):
        if self.share_weight and (not self.shared_codebook):
            device = next(self.parameters()).device
            self.output_proj_weight = torch.cat(
                [self.embed_proj_shared_weight, self.output_proj_weight_], dim=0).to(device)
            self.token_embed_weight = torch.cat(
                [self.token_embed_weight_, self.embed_proj_shared_weight], dim=0).to(device)

    def output_project(self, logits, qids):
        '''
        :logits: (bs, code_dim, seqlen)
        :qids: (bs)
        :return: logits (bs, ntoken, seqlen)
        '''
        output_proj_weight = self.output_proj_weight[qids]
        output_proj_bias = None if self.output_proj_bias is None else self.output_proj_bias[qids]

        output = torch.einsum('bnc, bcs->bns', output_proj_weight, logits)
        if output_proj_bias is not None:
            output += output_proj_bias.unsqueeze(-1)
        return output

    def trans_forward(self, motion_codes, qids, cond, padding_mask, force_mask=False,
                      cond_mask=None,
                      first_frame_pixels=None,
                      sparse_frames=None, visual_indices=None, visual_valid_mask=None,
                      return_attn_weights=False):
        '''
        Cross-attention forward pass — no prefix tokens.

        :param motion_codes: (B, S, code_dim)  — cumulative VQ code sums
        :param qids:         (B,)              — current quantizer layer index
        :param cond:         (B, raw_cond_dim) or (B, T_text, t5_dim)
        :param padding_mask: (B, S)            — True = padding
        :param force_mask:   bool              — True activates CFG null branch (drops ALL
                             conditioning including visual, for a truly unconditional baseline)
        :param cond_mask:    (B, T_text) bool optional — True = VALID T5 token
        :param first_frame_pixels: (B, 3, 224, 224) optional first-frame image
        :param sparse_frames:    (B, K, 3, 224, 224) optional sparse keyframe images
        :param visual_indices:   (B, K) long optional — VQ-level temporal indices
        :param visual_valid_mask:(B, K) bool optional — True for valid frame slots
        :param return_attn_weights: bool — When True, also return cross-attention weights
                             from the last 2 blocks as a list of (B, S, T_cond) tensors.
                             Default False (zero overhead in normal operation).
        :return: logits (B, code_dim, S)  [normal]
                 or (logits, attn_weights_list) when return_attn_weights=True
        '''
        # ── Motion feature embedding ────────────────────────────────────────
        x = self.input_process(motion_codes)    # (S, B, D)  seq-first
        x = self.position_enc(x)

        # Quantizer-level embedding
        q_onehot = self.encode_quant(qids).float().to(x.device)   # (B, num_q)
        q_emb = self.quant_emb(q_onehot)                          # (B, D)
        x = x + q_emb.unsqueeze(0)                                # (S, B, D)

        # ── Visual tokens (first-frame and/or sparse CLIP) ──────────────────
        # When force_mask=True (CFG null branch) we skip ALL visual encoding so
        # the unconditional baseline is truly unconditioned (no text, no visual).
        visual_token = None
        visual_tokens_sparse = None
        visual_ignore_mask = None

        if not force_mask:
            # ── First-frame conditioning ──
            if self.use_first_frame and first_frame_pixels is not None:
                visual_token = self.encode_first_frame(first_frame_pixels)  # (1, B, D)
                if self.training and self.visual_drop_prob > 0:
                    B = visual_token.shape[1]
                    keep = torch.bernoulli(
                        torch.full((B,), 1.0 - self.visual_drop_prob,
                                   device=visual_token.device)
                    ).view(1, B, 1)
                    visual_token = visual_token * keep

            # ── Sparse keyframe conditioning ──
            if self.use_sparse_frames and sparse_frames is not None:
                visual_tokens_sparse, visual_ignore_mask = self.encode_sparse_frames(
                    sparse_frames, visual_indices, visual_valid_mask)
                if self.training and self.visual_drop_prob > 0:
                    B = visual_tokens_sparse.shape[1]
                    drop = torch.bernoulli(
                        torch.full((B,), self.visual_drop_prob,
                                   device=visual_tokens_sparse.device)
                    ).bool()
                    visual_ignore_mask = visual_ignore_mask.clone()
                    visual_ignore_mask[drop] = True

        # ── Conditioning: project to (T_cond, B, D) ─────────────────────────
        cond_seq, cond_kp = self._prepare_crossattn_cond(
            cond, cond_mask, force_mask,
            visual_token=visual_token,
            visual_tokens=visual_tokens_sparse,
            visual_ignore_mask=visual_ignore_mask)

        captured_attn_weights = None
        if self.condition_fusion == 'prefix_self_attn':
            if return_attn_weights:
                raise ValueError("return_attn_weights is only available for condition_fusion='cross_attn'")
            x = self._run_prefix_self_attn_blocks(
                x, cond_seq,
                motion_key_padding_mask=padding_mask,
                cond_key_padding_mask=cond_kp,
            )
        else:
            # ── Cross-attention stack ────────────────────────────────────────
            x, captured_attn_weights = self._run_cross_attn_blocks(
                x, cond_seq,
                motion_key_padding_mask=padding_mask,
                cond_key_padding_mask=cond_kp,
                return_attn_weights=return_attn_weights,
            )   # (S, B, D)

        logits = self.output_process(x)         # (B, code_dim, S)
        if return_attn_weights:
            return logits, captured_attn_weights
        return logits

    def forward_with_cond_scale(self, motion_codes, q_id, cond_vector, padding_mask,
                                cond_scale=3, force_mask=False, cond_mask=None,
                                first_frame_pixels=None,
                                sparse_frames=None, visual_indices=None, visual_valid_mask=None):
        bs = motion_codes.shape[0]
        qids = torch.full((bs,), q_id, dtype=torch.long, device=motion_codes.device)
        if force_mask:
            logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                        force_mask=True, cond_mask=cond_mask)
            return self.output_project(logits, qids - 1)

        logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                    cond_mask=cond_mask,
                                    first_frame_pixels=first_frame_pixels,
                                    sparse_frames=sparse_frames,
                                    visual_indices=visual_indices,
                                    visual_valid_mask=visual_valid_mask)
        logits = self.output_project(logits, qids - 1)
        if cond_scale == 1:
            return logits

        aux_logits = self.trans_forward(motion_codes, qids, cond_vector, padding_mask,
                                        force_mask=True, cond_mask=cond_mask)
        aux_logits = self.output_project(aux_logits, qids - 1)
        return aux_logits + (logits - aux_logits) * cond_scale

    def forward(self, all_indices, y, m_lens,
                first_frame_pixels=None,
                sparse_frames=None, visual_indices=None, visual_valid_mask=None):
        '''
        :param all_indices: (b, n, q)
        :param y: raw text or action labels
        :param m_lens: (b,)
        :param sparse_frames: (b, 4, 3, 224, 224) optional sparse keyframe images
        :param visual_indices: (b, 4) long optional temporal indices
        :param visual_valid_mask: (b, 4) bool optional valid mask
        '''
        self.process_embed_proj_weight()

        bs, ntokens, num_quant_layers = all_indices.shape
        device = all_indices.device

        non_pad_mask = lengths_to_mask(m_lens, ntokens)
        q_non_pad_mask = repeat(non_pad_mask, 'b n -> b n q', q=num_quant_layers)
        all_indices = torch.where(q_non_pad_mask, all_indices, self.pad_id)

        active_q_layers = q_schedule(bs, low=1, high=num_quant_layers, device=device)

        token_embed = repeat(self.token_embed_weight, 'q c d-> b c d q', b=bs)
        gather_indices = repeat(all_indices[..., :-1], 'b n q -> b n d q', d=token_embed.shape[2])
        all_codes = token_embed.gather(1, gather_indices)
        cumsum_codes = torch.cumsum(all_codes, dim=-1)

        active_indices = all_indices[torch.arange(bs), :, active_q_layers]
        history_sum = cumsum_codes[torch.arange(bs), :, :, active_q_layers - 1]

        cond_vector, force_mask, cond_mask = self.encode_condition(y, bs, device)

        logits = self.trans_forward(history_sum, active_q_layers, cond_vector, ~non_pad_mask,
                                    force_mask, cond_mask=cond_mask,
                                    first_frame_pixels=first_frame_pixels,
                                    sparse_frames=sparse_frames,
                                    visual_indices=visual_indices,
                                    visual_valid_mask=visual_valid_mask)
        logits = self.output_project(logits, active_q_layers - 1)
        ce_loss, pred_id, acc = cal_performance(logits, active_indices, ignore_index=self.pad_id)
        return ce_loss, pred_id, acc

    @torch.no_grad()
    @eval_decorator
    def generate(self, motion_ids, conds, m_lens, temperature=1,
                 topk_filter_thres=0.9, cond_scale=2, num_res_layers=-1,
                 first_frame_pixels=None,
                 sparse_frames=None, visual_indices=None, visual_valid_mask=None):

        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0
        num_quant_layers = self.opt.num_quantizers if num_res_layers == -1 else num_res_layers + 1

        for i in range(1, num_quant_layers):
            token_embed = self.token_embed_weight[i - 1].to(device)
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask,
                                                  cond_scale=cond_scale, cond_mask=cond_mask,
                                                  first_frame_pixels=first_frame_pixels,
                                                  sparse_frames=sparse_frames,
                                                  visual_indices=visual_indices,
                                                  visual_valid_mask=visual_valid_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)
            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
        return all_indices

    @torch.no_grad()
    @eval_decorator
    def edit(self, motion_ids, conds, m_lens, temperature=1,
             topk_filter_thres=0.9, cond_scale=2):

        self.process_embed_proj_weight()

        device = next(self.parameters()).device
        seq_len = motion_ids.shape[1]
        batch_size = len(conds)

        cond_vector, _, cond_mask = self.encode_condition(conds, batch_size, device)

        padding_mask = ~lengths_to_mask(m_lens, seq_len)
        motion_ids = torch.where(padding_mask, self.pad_id, motion_ids)
        all_indices = [motion_ids]
        history_sum = 0

        for i in range(1, self.opt.num_quantizers):
            token_embed = self.token_embed_weight[i - 1]
            token_embed = repeat(token_embed, 'c d -> b c d', b=batch_size)
            gathered_ids = repeat(motion_ids, 'b n -> b n d', d=token_embed.shape[-1])
            history_sum += token_embed.gather(1, gathered_ids)

            logits = self.forward_with_cond_scale(history_sum, i, cond_vector, padding_mask,
                                                  cond_scale=cond_scale, cond_mask=cond_mask)
            logits = logits.permute(0, 2, 1)
            filtered_logits = top_k(logits, topk_filter_thres, dim=-1)
            pred_ids = gumbel_sample(filtered_logits, temperature=temperature, dim=-1)

            ids = torch.where(padding_mask, self.pad_id, pred_ids)
            motion_ids = ids
            all_indices.append(ids)

        all_indices = torch.stack(all_indices, dim=-1)
        all_indices = torch.where(all_indices == self.pad_id, -1, all_indices)
        return all_indices