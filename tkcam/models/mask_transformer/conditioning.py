"""
Modular Conditioning Provider for MoMask Transformer.

Supports three conditioning modes:
  - 'clip':         CLIP ViT-B/32 global text embeddings  (B, 512)
  - 't5':           T5 token-level encoder hidden states   (B, T_text, 768)
  - 'id_embedding': Learnable per-sample embeddings        (B, 512)

Each mode produces a conditioning tensor that is projected to the
Transformer's latent_dim and injected via either:
  Option A  – Prefix tokens (prepended to the motion sequence), or
  Option B  – Cross-attention (motion tokens query conditioning tokens).

Shape legend (used throughout):
  B         = batch size
  S         = motion token sequence length (after VQ downsampling)
  D         = transformer latent_dim  (default 384)
  C         = raw conditioning dim    (512 for CLIP/ID, 768 for T5-base)
  T_text    = number of T5 text tokens (variable per batch, padded)

┌───────────────────────────────────────────────────────────────────────┐
│ Forward-pass shape trace  (Prefix injection, MaskTransformer)       │
│                                                                      │
│  conditioning_mode='clip'                                            │
│    raw_cond:        (B, 512)                                         │
│    projected_cond:  (B, D)     →  unsqueeze(0) →  (1, B, D)  prefix │
│    motion_tokens:   (S, B, D)                                        │
│    xseq:            (1+S, B, D)                                      │
│    output:          (S, B, D)   [strip prefix]                       │
│                                                                      │
│  conditioning_mode='t5'                                              │
│    raw_cond:        (B, T_text, 768)                                 │
│    projected_cond:  (B, T_text, D)  → permute → (T_text, B, D)      │
│    motion_tokens:   (S, B, D)                                        │
│    xseq:            (T_text+S, B, D)                                 │
│    output:          (S, B, D)   [strip T_text prefix tokens]         │
│                                                                      │
│  conditioning_mode='id_embedding'                                    │
│    sample_idx:      (B,)  integers in [0, num_samples)               │
│    raw_cond:        (B, 512)                                         │
│    projected_cond:  (B, D)     →  unsqueeze(0) →  (1, B, D)  prefix │
│    motion_tokens:   (S, B, D)                                        │
│    xseq:            (1+S, B, D)                                      │
│    output:          (S, B, D)   [strip prefix]                       │
│                                                                      │
│  CFG null condition:                                                 │
│    clip             :  zeros (B, D)                                  │
│    id_embedding     :  learned null_idx embedding (B, D)             │
│    t5               :  learned t5_null_proj (1, B, D)  single token  │
│                        Always n_prefix=1, independent of text length │
└───────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple


class ConditioningProvider(nn.Module):
    """Unified conditioning wrapper that toggles between clip / t5 / id_embedding.

    This module owns the raw encoder (CLIP model, T5 model, or nn.Embedding)
    and a projection layer that maps the encoder output to ``latent_dim``.

    Parameters
    ----------
    mode : str
        One of ``'clip'``, ``'t5'``, ``'id_embedding'``.
    latent_dim : int
        Transformer hidden dimension (target projection size).
    clip_dim : int
        Dimension of CLIP / ID embeddings (default 512).
    t5_dim : int
        Hidden size of T5 encoder (768 for t5-base).
    num_samples : int
        Number of learnable sample embeddings for id_embedding mode.
    t5_model_name : str
        HuggingFace model name for T5 encoder.
    clip_version : str
        CLIP model variant (only used when mode='clip').
    cond_drop_prob : float
        Probability of dropping the condition during training (CFG).
    finetune_clip : bool
        Whether to fine-tune CLIP last layers.
    finetune_clip_layers : int
        How many CLIP transformer layers to unfreeze.
    max_t5_length : int
        Max token length for T5 tokenisation.
    """

    SUPPORTED_MODES = ('clip', 't5', 'id_embedding')

    def __init__(
        self,
        mode: str,
        latent_dim: int = 384,
        clip_dim: int = 512,
        t5_dim: int = 768,
        num_samples: int = 50,
        t5_model_name: str = 't5-base',
        clip_version: str = 'ViT-B/32',
        cond_drop_prob: float = 0.1,
        finetune_clip: bool = False,
        finetune_clip_layers: int = 2,
        max_t5_length: int = 64,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        assert mode in self.SUPPORTED_MODES, (
            f"Unknown conditioning mode '{mode}'. Choose from {self.SUPPORTED_MODES}")

        self.mode = mode
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.t5_dim = t5_dim
        self.cond_drop_prob = cond_drop_prob
        self.finetune_clip = finetune_clip
        self.finetune_clip_layers = finetune_clip_layers
        self.max_t5_length = max_t5_length
        self._device = device

        # ----------------------------------------------------------
        # Mode-specific encoder + projection
        # ----------------------------------------------------------
        if mode == 'clip':
            # CLIP encoder is loaded lazily via init_clip() so that
            # model.apply(_init_weights) doesn't clobber pretrained CLIP.
            self.cond_proj = nn.Linear(clip_dim, latent_dim)
            self._raw_dim = clip_dim
            self._seq_cond = False        # single-vector condition

        elif mode == 't5':
            self._init_t5(t5_model_name)
            # Auto-detect actual hidden size from the loaded model so that
            # t5-small (512) and t5-large (1024) are handled correctly even
            # if the caller passes the wrong t5_dim default (768).
            actual_t5_dim = self.t5_encoder.config.d_model
            if actual_t5_dim != t5_dim:
                print(
                    f'[ConditioningProvider] WARNING: t5_dim param ({t5_dim}) '
                    f'!= actual model d_model ({actual_t5_dim}). '
                    f'Using model value.')
                t5_dim = actual_t5_dim
            self.t5_dim = t5_dim           # keep in sync for external access
            self.cond_proj = nn.Linear(t5_dim, latent_dim)
            self._raw_dim = t5_dim
            self._seq_cond = True         # sequence of token embeddings
            # Learned null latent: a single projected vector used as the
            # unconditional prefix during CFG inference (force_mask=True).
            # Kept as a Parameter so it is saved/loaded with the checkpoint
            # and can receive gradient if the null branch is ever trained.
            # Initialised near zero for stable warm-up; model learns to push
            # it toward a useful unconditional direction over training.
            self.t5_null_proj = nn.Parameter(torch.zeros(1, latent_dim))

        elif mode == 'id_embedding':
            self.num_samples = num_samples
            # +1 index reserved for the NULL / uncond embedding
            self.id_emb = nn.Embedding(num_samples + 1, clip_dim)
            self.null_idx = num_samples    # last index = null condition
            self.cond_proj = nn.Linear(clip_dim, latent_dim)
            self._raw_dim = clip_dim
            self._seq_cond = False

        # Null embedding for CFG (used by clip & t5 modes)
        # For id_embedding the null is handled via self.null_idx.
        if mode != 'id_embedding':
            self.register_buffer(
                '_null_cond',
                torch.zeros(1, self._raw_dim if not self._seq_cond else 1))

    # ──────────────────────────────────────────────────────
    # CLIP helpers (mirrors BaseCondTransformer API)
    # ──────────────────────────────────────────────────────

    def init_clip(self, clip_version: str = 'ViT-B/32', device=None):
        """Load and freeze CLIP.  Must be called AFTER model.apply(_init_weights)."""
        if self.mode != 'clip':
            return
        import clip as _clip
        print(f'[ConditioningProvider] Loading CLIP {clip_version} …')
        self.clip_version = clip_version
        clip_model, _ = _clip.load(clip_version, device='cpu', jit=False)

        if self.finetune_clip:
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False
            total = len(clip_model.transformer.resblocks)
            for i in range(total - self.finetune_clip_layers, total):
                for p in clip_model.transformer.resblocks[i].parameters():
                    p.requires_grad = True
            for p in clip_model.ln_final.parameters():
                p.requires_grad = True
            if hasattr(clip_model, 'text_projection') and clip_model.text_projection is not None:
                clip_model.text_projection.requires_grad = True
            print(f'  CLIP fine-tune: last {self.finetune_clip_layers} layers unfrozen')
        else:
            clip_model.eval()
            for p in clip_model.parameters():
                p.requires_grad = False
            # Convert to FP16 on GPU when frozen
            dev = device or self._device
            if dev is not None and str(dev) != 'cpu':
                _clip.model.convert_weights(clip_model)
            print('  CLIP fully frozen')

        self.clip_model = clip_model

    def _encode_clip(self, raw_text: List[str]) -> torch.Tensor:
        """Encode text → (B, clip_dim)."""
        import clip as _clip
        device = next(self.parameters()).device
        tokens = _clip.tokenize(raw_text, truncate=True).to(device)
        if self.finetune_clip:
            return self.clip_model.encode_text(tokens).float()
        with torch.no_grad():
            return self.clip_model.encode_text(tokens).float()

    # ──────────────────────────────────────────────────────
    # T5 helpers
    # ──────────────────────────────────────────────────────

    def _init_t5(self, model_name: str):
        """Load T5 encoder (frozen)."""
        # ── Environment compatibility shim ─────────────────────────────────
        # transformers ≥ 4.50 references torchvision.transforms.functional.
        # InterpolationMode.NEAREST_EXACT (added in torchvision 0.17) inside
        # image_utils.py at module-import time.  Patch the enum to add the
        # missing member when running on older torchvision (e.g. 0.13).
        try:
            from torchvision.transforms.functional import InterpolationMode as _IM
            if not hasattr(_IM, 'NEAREST_EXACT'):
                _IM.NEAREST_EXACT = _IM.NEAREST
        except Exception:
            pass
        # ──────────────────────────────────────────────────────────────────
        from transformers import T5EncoderModel, T5Tokenizer
        print(f'[ConditioningProvider] Loading T5 encoder: {model_name} …')
        self.t5_tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.t5_encoder = T5EncoderModel.from_pretrained(model_name)
        self.t5_encoder.eval()
        for p in self.t5_encoder.parameters():
            p.requires_grad = False
        print(f'  T5 encoder frozen ({model_name}, hidden={self.t5_encoder.config.d_model})')

    def _encode_t5(self, raw_text: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text → (B, T_text, t5_dim), (B, T_text) attention_mask.

        Returns full sequence of hidden states (NOT pooled).
        """
        device = next(self.parameters()).device
        tok = self.t5_tokenizer(
            raw_text,
            padding=True,
            truncation=True,
            max_length=self.max_t5_length,
            return_tensors='pt',
        ).to(device)
        with torch.no_grad():
            out = self.t5_encoder(
                input_ids=tok.input_ids,
                attention_mask=tok.attention_mask,
            )
        hidden = out.last_hidden_state.float()          # (B, T_text, t5_dim)
        attn_mask = tok.attention_mask.bool()            # (B, T_text)
        return hidden, attn_mask

    # ──────────────────────────────────────────────────────
    # Unified public API
    # ──────────────────────────────────────────────────────

    @property
    def is_sequence_condition(self) -> bool:
        """True when the conditioning is a *sequence* of tokens (T5)."""
        return self._seq_cond

    def encode(
        self,
        text_or_ids,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], bool]:
        """Produce the raw conditioning embedding.

        Returns
        -------
        cond : Tensor
            clip / id_embedding → (B, raw_dim)
            t5                  → (B, T_text, t5_dim)
        cond_mask : Optional[Tensor]
            Only for t5 — boolean mask (B, T_text), True = valid token.
            None for clip / id_embedding.
        force_mask : bool
            If True, the condition should be fully masked (uncond mode).
        """
        force_mask = False

        if self.mode == 'clip':
            cond = self._encode_clip(text_or_ids)               # (B, 512)
            return cond, None, force_mask

        elif self.mode == 't5':
            cond, cond_mask = self._encode_t5(text_or_ids)      # (B, T, 768)
            return cond, cond_mask, force_mask

        elif self.mode == 'id_embedding':
            # text_or_ids is expected to be a LongTensor of sample indices (B,)
            ids = text_or_ids.to(device).long()
            cond = self.id_emb(ids)                             # (B, 512)
            return cond, None, force_mask

        raise RuntimeError(f"Unknown mode: {self.mode}")

    # ──────────────────────────────────────────────────────
    # CFG masking
    # ──────────────────────────────────────────────────────

    def mask_cond(
        self,
        cond: torch.Tensor,
        force_mask: bool = False,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply classifier-free-guidance dropout.

        For single-vector modes (clip, id_embedding):
            Zeros out entire condition vectors with probability cond_drop_prob.
        For sequence modes (t5):
            Zeros out all tokens for selected batch elements.

        Parameters
        ----------
        cond : Tensor
            (B, D) or (B, T, D) — *already projected* to latent_dim.
        force_mask : bool
            Force full masking (inference null branch).
        cond_mask : Optional[Tensor]
            (B, T) boolean attention mask for T5 tokens.

        Returns
        -------
        cond : Tensor  (same shape, possibly zeroed)
        cond_mask : Optional[Tensor]  (same or modified)
        """
        if force_mask:
            return torch.zeros_like(cond), cond_mask

        if self.training and self.cond_drop_prob > 0.:
            bs = cond.shape[0]
            drop = torch.bernoulli(
                torch.ones(bs, device=cond.device) * self.cond_drop_prob
            ).bool()

            if cond.dim() == 2:
                # (B, D) — single vector
                cond = cond * (~drop).float().unsqueeze(1)
            else:
                # (B, T, D) — sequence
                cond = cond * (~drop).float().unsqueeze(1).unsqueeze(2)
                if cond_mask is not None:
                    # Zero out attention mask for dropped samples so transformer
                    # sees them as padding.
                    cond_mask = cond_mask & (~drop).unsqueeze(1)

        return cond, cond_mask

    def project_and_prepare(
        self,
        cond: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        force_mask: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """Project raw condition → latent_dim, apply CFG mask, and format
        for prefix injection.

        CFG null-branch behaviour per mode
        -----------------------------------
        clip         : zeros out the projected vector (force_mask / dropout).
        id_embedding : looks up null_idx embedding so the model always receives
                       the *learned* null direction — not arbitrary zeros.
                       During training CFG dropout, dropped samples are ALSO
                       replaced with the null_idx projection so the embedding
                       at null_idx is trained alongside the sample embeddings.
        t5           : force_mask=True → returns a single learned ``t5_null_proj``
                       token (n_prefix=1), always deterministic and text-length-
                       independent.  Training dropout → zeros + fully masks the
                       real-text prefix for dropped samples (n_prefix=T_text).

        Returns
        -------
        prefix : Tensor
            (N_prefix, B, D)  where N_prefix = 1 for clip/id, T_text for t5.
        prefix_key_padding : Optional[Tensor]
            (B, N_prefix) — True where prefix should be masked.
            None when no masking needed.
        n_prefix : int
            Number of prefix tokens to strip from the transformer output.
        """
        if self._seq_cond:
            return self._project_t5_with_cfg(cond, cond_mask, force_mask)
        else:
            # CLIP / ID: (B, raw_dim) → project → (B, D)
            if self.mode == 'id_embedding':
                proj = self._project_id_with_cfg(cond, force_mask)
            else:
                proj = self.cond_proj(cond)                     # (B, D)
                proj, _ = self.mask_cond(proj, force_mask)
            prefix = proj.unsqueeze(0)                          # (1, B, D)
            return prefix, None, 1

    def _project_t5_with_cfg(
        self,
        cond: torch.Tensor,
        cond_mask: Optional[torch.Tensor],
        force_mask: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """T5-specific projection + CFG null handling.

        Fixes the "text-length leak" bug present in the naive zeroing approach:
        when force_mask=True, the old code kept the real text's cond_mask,
        so the null branch had variable prefix length — 'pan left' (2 tokens)
        and 'slowly pan camera to the right while following the subject'
        (8 tokens) produced null representations of *different widths*,
        exposing text-structure information to the unconditional branch.

        This method always returns n_prefix=1 on the null branch, using
        a fixed learned ``t5_null_proj`` vector regardless of input text.

        Behaviour
        ---------
        force_mask=True (CFG inference unconditional branch):
            Returns ``t5_null_proj`` broadcast to ``(1, B, D)`` with
            no key-padding mask (always attended to).  n_prefix=1.

        force_mask=False, training CFG dropout:
            Projects the real T5 sequence ``(B, T, raw_dim)→(B, T, D)``.
            For dropped samples (Bernoulli with cond_drop_prob), zeros the
            full prefix sequence AND sets all their cond_mask positions to
            False (fully key-padded), so the encoder ignores them.  The
            null direction is implicitly learned via the force_mask=True path.
            n_prefix = T_text.

        force_mask=False, evaluation:
            Straight projection, no CFG dropout.  n_prefix = T_text.
        """
        B = cond.shape[0]
        device = cond.device

        # ── Unconditional / null branch ───────────────────────────────────
        if force_mask:
            # Single deterministic null token; independent of input text.
            null = self.t5_null_proj                           # (1, D)
            prefix = null.unsqueeze(1).expand(1, B, -1)        # (1, B, D)
            return prefix, None, 1

        # ── Conditional branch ───────────────────────────────────────────
        proj = self.cond_proj(cond)                            # (B, T, D)

        if self.training and self.cond_drop_prob > 0.:
            drop = torch.bernoulli(
                torch.ones(B, device=device) * self.cond_drop_prob
            ).bool()                                           # (B,)

            if drop.any():
                # Zero-out dropped samples' token sequences so their prefix
                # carries no signal.
                proj = proj * (~drop).float().unsqueeze(1).unsqueeze(2)
                if cond_mask is not None:
                    # Mark all prefix positions of dropped samples as padding
                    # so the encoder fully ignores the (now-zero) tokens.
                    cond_mask = cond_mask & (~drop).unsqueeze(1)

        prefix = proj.permute(1, 0, 2)                        # (T, B, D)
        n_prefix = prefix.shape[0]
        prefix_kp = (~cond_mask) if cond_mask is not None else None  # (B,T) True=masked
        return prefix, prefix_kp, n_prefix

    def _project_id_with_cfg(
        self,
        cond: torch.Tensor,
        force_mask: bool,
    ) -> torch.Tensor:
        """Project an id_embedding condition with correct CFG handling.

        Instead of zeroing the projected vector (which leaves the null_idx
        embedding untrained), this helper:
          • at inference (force_mask=True): looks up self.null_idx so the model
            always sees the *learned* null direction.
          • at training with CFG dropout: substitutes dropped samples with the
            null_idx embedding lookup, training the null embedding alongside
            the sample embeddings.

        Parameters
        ----------
        cond : Tensor (B, clip_dim)
            Raw embedding from id_emb(sample_ids).
        force_mask : bool
            True during the CFG unconditional inference branch.

        Returns
        -------
        proj : Tensor (B, latent_dim)
        """
        bs = cond.shape[0]
        device = cond.device

        null_ids = torch.full((bs,), self.null_idx, dtype=torch.long, device=device)
        null_raw  = self.id_emb(null_ids)           # (B, clip_dim)

        if force_mask:
            # Inference null branch: always use the learned null embedding.
            return self.cond_proj(null_raw)          # (B, D)

        proj      = self.cond_proj(cond)             # (B, D)
        null_proj = self.cond_proj(null_raw)         # (B, D)

        if self.training and self.cond_drop_prob > 0.:
            # Per-sample CFG dropout: replace dropped samples with null_idx
            # projection so the null embedding gets gradient updates.
            drop = torch.bernoulli(
                torch.ones(bs, device=device) * self.cond_drop_prob
            ).bool()                                 # (B,)
            proj = torch.where(drop.unsqueeze(1), null_proj, proj)

        return proj

    # ──────────────────────────────────────────────────────
    # Null condition for CFG inference
    # ──────────────────────────────────────────────────────

    def get_null_cond(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Return a null condition for classifier-free guidance.

        Returns
        -------
        null_cond : Tensor
            clip / id → (B, raw_dim) of zeros
            t5        → (B, 1, t5_dim) of zeros  (single null token)
        null_mask : Optional[Tensor]
            t5 → (B, 1) all True;  others → None
        """
        if self.mode == 'id_embedding':
            # Use the dedicated null index
            ids = torch.full((batch_size,), self.null_idx, dtype=torch.long, device=device)
            return self.id_emb(ids), None

        if self._seq_cond:
            # Return a single null token in *projected* latent space.
            # Consistent with _project_t5_with_cfg(force_mask=True) which
            # also returns n_prefix=1 using t5_null_proj.
            # NOTE: callers that use get_null_cond feed the result back
            # through project_and_prepare, which for T5 expects raw (pre-
            # projection) tensors.  Return the null_proj already in latent
            # space as a (B, 1, latent_dim) tensor; project_and_prepare will
            # not be called again for this null cond.
            null = self.t5_null_proj                            # (1, D)
            null_cond = null.unsqueeze(0).expand(batch_size, 1, -1)  # (B,1,D)
            # All positions are valid (none masked) — single visible null token.
            null_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            return null_cond, null_mask
        else:
            return torch.zeros(batch_size, self._raw_dim, device=device), None

    def parameters_wo_clip(self):
        """Return parameters excluding frozen CLIP (unless fine-tuning)."""
        if self.mode == 'clip' and not self.finetune_clip:
            return [p for n, p in self.named_parameters() if not n.startswith('clip_model.')]
        if self.mode == 't5':
            return [p for n, p in self.named_parameters() if not n.startswith('t5_encoder.')]
        return list(self.parameters())
