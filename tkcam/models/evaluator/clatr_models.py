"""
CLaTr-style Evaluator for 12D Camera Trajectories

Architecture inspired by Courant et al., "Camera Trajectory Representation" (ECCV 2024).
Implements a contrastive text-trajectory evaluator that maps both modalities
into a shared 512-dim latent space using Transformer-based encoders.

The trajectory encoder processes 12D camera data (pos[3], vel[3], rot6d[6])
and the text encoder uses a frozen CLIP backbone with a learned projection.

Once trained, the frozen evaluator provides:
  - FID (Fréchet Inception Distance) on trajectory latents
  - R-Precision (Top-1/2/3) via cosine retrieval
  - Matching Score (avg cosine similarity of paired text-trajectory)
  - Diversity (variance of trajectory latents)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict


# ---------------------------------------------------------------------------
#  Positional Encoding (sinusoidal)
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al., 2017)."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, D)"""
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


# ---------------------------------------------------------------------------
#  Gram-Schmidt Orthonormalization for rot6d
# ---------------------------------------------------------------------------

def gram_schmidt_ortho6d(rot6d: torch.Tensor) -> torch.Tensor:
    """
    Apply Gram-Schmidt orthonormalization to the 6D rotation representation.
    Input:  (*, 6) — first two columns of a rotation matrix flattened.
    Output: (*, 6) — orthonormalized.
    """
    a1 = rot6d[..., :3]
    a2 = rot6d[..., 3:6]

    b1 = F.normalize(a1, dim=-1)
    dot = (b1 * a2).sum(dim=-1, keepdim=True)
    b2 = F.normalize(a2 - dot * b1, dim=-1)

    return torch.cat([b1, b2], dim=-1)


# ---------------------------------------------------------------------------
#  Trajectory Encoder  (Transformer-based, ACTOR-style)
# ---------------------------------------------------------------------------

class TrajectoryEncoder(nn.Module):
    """
    Encode a 12D camera trajectory sequence into a single latent vector.

    Architecture (ACTOR-style, following CLaTr):
        1. Linear projection:  input_dim → latent_dim
        2. Prepend learnable [CLS] token
        3. Add sinusoidal positional encoding
        4. Pass through Transformer encoder layers
        5. Read out the [CLS] token → project to output_dim

    Parameters
    ----------
    input_dim : int
        Dimensionality of each frame (default 12 for pos+vel+rot6d).
    latent_dim : int
        Hidden size of the Transformer (default 256).
    output_dim : int
        Final embedding dimensionality (default 512).
    num_layers : int
        Number of Transformer encoder layers (default 6).
    num_heads : int
        Number of attention heads (default 4).
    ff_size : int
        Feed-forward hidden size (default 1024).
    dropout : float
        Dropout rate (default 0.1).
    activation : str
        Activation function for feed-forward layers (default "gelu").
    ortho_normalize : bool
        If True, apply Gram-Schmidt to the rot6d components (dims 6:12)
        before encoding.
    """

    def __init__(
        self,
        input_dim: int = 12,
        latent_dim: int = 256,
        output_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 4,
        ff_size: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        ortho_normalize: bool = True,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.ortho_normalize = ortho_normalize

        # Input projection
        self.input_proj = nn.Linear(input_dim, latent_dim)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(latent_dim, max_len=5000, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            norm_first=True,  # Pre-norm for better training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(latent_dim),
        )

        # Output projection (CLS → output_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, input_dim)
            Raw 12D camera trajectory.
        lengths : (B,) int tensor, optional
            Actual lengths before padding.  If provided, padded positions are
            masked in the Transformer attention.

        Returns
        -------
        emb : (B, output_dim)
            Trajectory embedding.
        """
        B, T, D = x.shape

        # Optional Gram-Schmidt on the rot6d portion (dims 6:12)
        if self.ortho_normalize and D >= 12:
            rot6d = gram_schmidt_ortho6d(x[..., 6:12])
            x = torch.cat([x[..., :6], rot6d], dim=-1)

        # Input projection
        h = self.input_proj(x)  # (B, T, latent_dim)

        # Prepend [CLS] token
        cls = self.cls_token.expand(B, -1, -1)  # (B, 1, latent_dim)
        h = torch.cat([cls, h], dim=1)  # (B, 1+T, latent_dim)

        # Positional encoding
        h = self.pos_enc(h)

        # Build attention mask: True means *ignore*
        if lengths is not None:
            # +1 for [CLS] token at position 0
            max_len = T + 1
            mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= (lengths.unsqueeze(1) + 1)
            # CLS token should never be masked
            mask[:, 0] = False
        else:
            mask = None

        # Transformer forward
        h = self.transformer(h, src_key_padding_mask=mask)

        # Read [CLS] token
        cls_out = h[:, 0]  # (B, latent_dim)

        # Project to output space
        emb = self.output_proj(cls_out)  # (B, output_dim)
        return emb


# ---------------------------------------------------------------------------
#  Text Encoder (CLIP backbone + learned projection)
# ---------------------------------------------------------------------------

class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP ViT-B/32 text encoder with a trainable linear projection
    from CLIP's 512-d space to the shared evaluator latent space.
    """

    def __init__(self, output_dim: int = 512, clip_model: str = "ViT-B/32"):
        super().__init__()
        import clip as clip_module
        self.clip_model, _ = clip_module.load(clip_model, device="cpu")
        # Freeze CLIP
        for param in self.clip_model.parameters():
            param.requires_grad = False
        self.clip_model.eval()

        clip_dim = self.clip_model.text_projection.shape[1]  # 512
        self.projection = nn.Sequential(
            nn.Linear(clip_dim, clip_dim),
            nn.GELU(),
            nn.Linear(clip_dim, output_dim),
        )

    @torch.no_grad()
    def _encode_clip(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """Run frozen CLIP text encoder."""
        return self.clip_model.encode_text(text_tokens).float()

    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        text_tokens : (B, 77) int tensor from clip.tokenize()

        Returns
        -------
        text_emb : (B, output_dim)
        """
        clip_feat = self._encode_clip(text_tokens)  # (B, 512)
        return self.projection(clip_feat)


# ---------------------------------------------------------------------------
#  InfoNCE Contrastive Loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE contrastive loss.

    For a batch of N matched (text, trajectory) pairs, the positive pair is
    on the diagonal and all off-diagonal entries are negatives.

    Parameters
    ----------
    temperature : float
        Softmax temperature (default 0.07).
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        # CLIP-style learnable logit scale initialized as log(1/temperature).
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1.0 / temperature))

    def forward(
        self, text_emb: torch.Tensor, traj_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        text_emb : (B, D) L2-normalized text embeddings
        traj_emb : (B, D) L2-normalized trajectory embeddings

        Returns
        -------
        loss : scalar
        """
        # Cosine similarity matrix with learnable temperature scaling.
        logits = self.logit_scale.exp().clamp(max=100) * (text_emb @ traj_emb.T)  # (B, B)

        labels = torch.arange(logits.size(0), device=logits.device)

        loss_t2m = F.cross_entropy(logits, labels)
        loss_m2t = F.cross_entropy(logits.T, labels)

        return (loss_t2m + loss_m2t) / 2.0


# ---------------------------------------------------------------------------
#  CLaTr Evaluator Wrapper  (used during generation training/eval)
# ---------------------------------------------------------------------------

class CLaTrEvalWrapper:
    """
    Drop-in replacement for `EvaluatorModelWrapper`.

    After pre-training the contrastive evaluator, load frozen weights and
    use this wrapper to compute FID, R-Precision, Matching Score during the
    main generation training loop.

    Parameters
    ----------
    ckpt_path : str
        Path to the evaluator checkpoint (from ``train_evaluator.py``).
    device : torch.device
        Target device.
    input_dim : int
        Trajectory feature dimension (default 12).
    output_dim : int
        Shared latent space dimension (default 512).
    """

    def __init__(
        self,
        ckpt_path: str,
        device: torch.device,
        input_dim: int = 12,
        output_dim: int = 512,
    ):
        import clip as clip_module
        self.device = device

        # Load checkpoint first so evaluator architecture follows the checkpoint.
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        traj_state = ckpt["traj_encoder"]
        ckpt_input_dim = traj_state["input_proj.weight"].shape[1]
        ckpt_output_dim = traj_state["output_proj.2.weight"].shape[0]

        if input_dim != ckpt_input_dim:
            print(
                f"[CLaTrEvalWrapper] input_dim override: requested {input_dim}, "
                f"checkpoint uses {ckpt_input_dim}. Using checkpoint dim."
            )
        if output_dim != ckpt_output_dim:
            print(
                f"[CLaTrEvalWrapper] output_dim override: requested {output_dim}, "
                f"checkpoint uses {ckpt_output_dim}. Using checkpoint dim."
            )

        # Build sub-networks
        self.traj_encoder = TrajectoryEncoder(
            input_dim=ckpt_input_dim, output_dim=ckpt_output_dim,
        ).to(device)

        self.text_encoder = CLIPTextEncoder(
            output_dim=ckpt_output_dim,
        ).to(device)

        self.traj_encoder.load_state_dict(ckpt["traj_encoder"])
        self.text_encoder.load_state_dict(ckpt["text_encoder"], strict=False)
        print(f"[CLaTrEvalWrapper] Loaded evaluator from {ckpt_path} "
              f"(epoch {ckpt.get('epoch', '?')})")

        # Load normalization stats used during evaluator training
        if "mean" in ckpt and "std" in ckpt:
            self.mean = ckpt["mean"].to(device)
            self.std = ckpt["std"].to(device)
        else:
            self.mean = None
            self.std = None

        # Freeze everything
        self.traj_encoder.eval()
        self.text_encoder.eval()
        for p in self.traj_encoder.parameters():
            p.requires_grad = False
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.clip_tokenize = clip_module.tokenize

        # Pipeline normalization stats — set via set_pipeline_stats() when
        # the generation pipeline uses different mean/std than the evaluator.
        self.pipeline_mean = None
        self.pipeline_std = None
        self.pipeline_dataset_name = ""
        self._warned_pipeline_dim_mismatch = False
        self._warned_eval_stats_dim_mismatch = False

    def set_pipeline_stats(self, mean: torch.Tensor, std: torch.Tensor, dataset_name: str = ""):
        """Register the generation pipeline's Z-normalization stats.

        If these differ from the evaluator's own mean/std (stored in the
        checkpoint), ``get_trajectory_embeddings`` will de-normalize with
        the pipeline stats and re-normalize with the evaluator stats so
        the trajectory encoder sees the distribution it was trained on.
        """
        self.pipeline_mean = mean.to(self.device).float()
        self.pipeline_std = std.to(self.device).float()
        self.pipeline_dataset_name = dataset_name

    def _convert_12d_to_9d_with_integrated_position(self, motions_raw: torch.Tensor) -> torch.Tensor:
        """
        Convert raw 12D [x,y,z,dx,dy,dz,rot6d] to raw 9D [rot6d,x,y,z].

        Position is reconstructed by integrating velocity channels:
            pos[0] = x0,y0,z0
            pos[t] = pos[t-1] + vel[t], t>=1
        """
        B, T, D = motions_raw.shape
        if D != 12:
            raise ValueError(f"Expected 12D motions for conversion, got {D}D")

        pos0 = motions_raw[:, 0:1, 0:3]              # (B,1,3)
        vel = motions_raw[:, :, 3:6]                 # (B,T,3)
        rot6d = motions_raw[:, :, 6:12]              # (B,T,6)

        pos = torch.zeros((B, T, 3), device=motions_raw.device, dtype=motions_raw.dtype)
        pos[:, 0:1, :] = pos0
        if T > 1:
            # integrate using vel[t] for transition (t-1 -> t), matching camera_geometry logic
            pos[:, 1:, :] = pos0 + torch.cumsum(vel[:, 1:, :], dim=1)

        return torch.cat([rot6d, pos], dim=-1)       # (B,T,9)

    def _convert_tkcam9_to_clatr9(self, motions_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw TKCAM 9D [x,y,z,rot6d] to evaluator 9D [rot6d,x,y,z]."""
        B, T, D = motions_raw.shape
        if D != 9:
            raise ValueError(f"Expected 9D motions for conversion, got {D}D")
        return torch.cat([motions_raw[:, :, 3:9], motions_raw[:, :, :3]], dim=-1)

    def _prepare_motion_for_evaluator(self, motions_raw: torch.Tensor) -> torch.Tensor:
        """Convert raw pipeline motion into the evaluator's expected input dim."""
        expected_dim = self.traj_encoder.input_dim
        current_dim = motions_raw.shape[-1]

        if current_dim == expected_dim:
            dataset_name = getattr(self, "pipeline_dataset_name", "")
            if current_dim == 9 and isinstance(dataset_name, str) and "rotmat9" in dataset_name.lower():
                return self._convert_tkcam9_to_clatr9(motions_raw)
            return motions_raw

        # Main migration case: pipeline is 12D, evaluator is 9D.
        if current_dim == 12 and expected_dim == 9:
            return self._convert_12d_to_9d_with_integrated_position(motions_raw)

        raise ValueError(
            f"Unsupported evaluator motion conversion: pipeline dim={current_dim}, "
            f"evaluator expects dim={expected_dim}."
        )

    # ── Core embedding methods ─────────────────────────────

    @torch.no_grad()
    def get_trajectory_embeddings(
        self, motions: torch.Tensor, m_lens: torch.Tensor
    ) -> np.ndarray:
        """
        Get trajectory latent embeddings.

        Parameters
        ----------
        motions : (B, T, D) — Z-normalized camera trajectory (pipeline stats)
        m_lens  : (B,) — actual frame counts

        Returns
        -------
        emb : (B, output_dim) numpy array

        Notes
        -----
        If ``set_pipeline_stats`` was called and the evaluator checkpoint
        contains its own mean/std, motions are de-normalized from the
        pipeline space and re-normalized into the evaluator's space.
        """
        motions = motions.to(self.device).float()
        m_lens = m_lens.to(self.device).long()

        # 1) De-normalize from pipeline space (if matching stats are available).
        motions_raw = motions
        if self.pipeline_mean is not None and self.pipeline_std is not None:
            if self.pipeline_mean.shape[-1] == motions.shape[-1]:
                motions_raw = motions * self.pipeline_std + self.pipeline_mean
            elif not self._warned_pipeline_dim_mismatch:
                print(
                    f"[CLaTrEvalWrapper] Warning: pipeline stats dim "
                    f"{self.pipeline_mean.shape[-1]} != motion dim {motions.shape[-1]}; "
                    "skipping pipeline de-normalization."
                )
                self._warned_pipeline_dim_mismatch = True

        # 2) Convert raw trajectory to evaluator expected format/dim.
        motions_eval = self._prepare_motion_for_evaluator(motions_raw)

        # 3) Re-normalize into evaluator training space when stats are available.
        if self.mean is not None and self.std is not None:
            if self.mean.shape[-1] == motions_eval.shape[-1]:
                motions_eval = (motions_eval - self.mean) / self.std
            elif not self._warned_eval_stats_dim_mismatch:
                print(
                    f"[CLaTrEvalWrapper] Warning: evaluator stats dim "
                    f"{self.mean.shape[-1]} != evaluator motion dim {motions_eval.shape[-1]}; "
                    "skipping evaluator re-normalization."
                )
                self._warned_eval_stats_dim_mismatch = True

        emb = self.traj_encoder(motions_eval, m_lens)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_text_embeddings(self, captions: list) -> np.ndarray:
        """
        Get text latent embeddings from raw caption strings.

        Parameters
        ----------
        captions : list[str]

        Returns
        -------
        emb : (B, output_dim) numpy array
        """
        tokens = self.clip_tokenize(captions, truncate=True).to(self.device)
        emb = self.text_encoder(tokens)
        emb = F.normalize(emb, dim=-1)
        return emb.cpu().numpy()

    @torch.no_grad()
    def get_co_embeddings(
        self,
        captions: list,
        motions: torch.Tensor,
        m_lens: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get paired text and trajectory embeddings.

        Parameters
        ----------
        captions : list[str]
        motions  : (B, T, D) tensor
        m_lens   : (B,) tensor

        Returns
        -------
        text_emb : (B, output_dim) numpy
        traj_emb : (B, output_dim) numpy
        """
        text_emb = self.get_text_embeddings(captions)
        traj_emb = self.get_trajectory_embeddings(motions, m_lens)
        return text_emb, traj_emb


# ---------------------------------------------------------------------------
#  Metric Functions
# ---------------------------------------------------------------------------

def compute_fid(gt_emb: np.ndarray, gen_emb: np.ndarray) -> float:
    """
    Compute Fréchet Inception Distance between two sets of embeddings.

    Parameters
    ----------
    gt_emb  : (N, D) ground truth trajectory embeddings
    gen_emb : (M, D) generated trajectory embeddings

    Returns
    -------
    fid : float
    """
    from scipy import linalg

    mu1 = np.mean(gt_emb, axis=0)
    sigma1 = np.cov(gt_emb, rowvar=False)

    mu2 = np.mean(gen_emb, axis=0)
    sigma2 = np.cov(gen_emb, rowvar=False)

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


def compute_r_precision(
    text_emb: np.ndarray,
    traj_emb: np.ndarray,
    top_k: int = 3,
    pool_size: int = 32,
    seed: int = 0,
    batch_size: int = None,  # deprecated alias for pool_size
) -> np.ndarray:
    """
    Compute R-Precision: for each sample, rank the matching caption against
    (pool_size - 1) randomly-chosen distractors using cosine similarity.
    Report Top-1, Top-2, …, Top-k accuracy.

    Unlike the old implementation that silently drops tail samples when
    N % pool_size != 0, this version evaluates every sample by
    sampling distractors from the full set.

    Parameters
    ----------
    text_emb  : (N, D) text embeddings (L2-normalized)
    traj_emb  : (N, D) trajectory embeddings (L2-normalized)
    top_k     : int (default 3)
    pool_size : int — number of candidates per query (default 32).
                A larger pool (e.g. 64 or 256) gives more stable metrics.
    seed      : int — RNG seed for reproducibility across epochs.
    batch_size: deprecated, use pool_size instead.

    Returns
    -------
    r_precision : (top_k,) array of accuracies
    """
    if batch_size is not None:
        pool_size = batch_size  # backward compat

    N = text_emb.shape[0]
    assert text_emb.shape[0] == traj_emb.shape[0]

    if N < pool_size:
        pool_size = N
    if N < 2:
        return np.zeros(top_k)

    rng = np.random.RandomState(seed)
    correct = np.zeros(top_k)

    for i in range(N):
        # Build distractor pool: sample (pool_size - 1) indices != i
        candidates = np.delete(np.arange(N), i)
        distractor_idx = rng.choice(candidates, size=min(pool_size - 1, len(candidates)), replace=False)
        pool_idx = np.concatenate([[i], distractor_idx])  # GT at position 0

        t_pool = text_emb[pool_idx]   # (pool, D)
        m_query = traj_emb[i:i+1]     # (1, D)

        # Cosine similarity: how well does each text match this trajectory?
        sim = (t_pool @ m_query.T).squeeze(-1)  # (pool,)
        ranking = np.argsort(-sim)  # descending

        gt_rank = np.where(ranking == 0)[0][0]  # rank of GT (index 0 in pool)
        for k in range(top_k):
            if gt_rank <= k:
                correct[k] += 1

    r_precision = correct / N
    return r_precision


def compute_matching_score(
    text_emb: np.ndarray, traj_emb: np.ndarray
) -> float:
    """
    Average cosine similarity between paired text and trajectory embeddings.

    Parameters
    ----------
    text_emb : (N, D)
    traj_emb : (N, D)

    Returns
    -------
    score : float  (higher is better)
    """
    # Both should already be L2-normalized
    cos_sim = np.sum(text_emb * traj_emb, axis=-1)  # (N,)
    return float(np.mean(cos_sim))


def compute_diversity(emb: np.ndarray, num_pairs: int = 300, seed: int = 0) -> float:
    """
    Compute diversity as the average pairwise L2 distance in latent space.

    Uses a fixed RNG seed so the *same* validation set yields identical
    diversity numbers across epochs (stability).

    Parameters
    ----------
    emb : (N, D) embeddings
    num_pairs : int — number of random pairs to sample
    seed : int — RNG seed for reproducibility

    Returns
    -------
    diversity : float
    """
    N = emb.shape[0]
    if N < 2:
        return 0.0
    num_pairs = min(num_pairs, N * (N - 1) // 2)
    rng = np.random.RandomState(seed)
    idx1 = rng.randint(0, N, num_pairs * 2)
    idx2 = rng.randint(0, N, num_pairs * 2)
    # Remove same-index pairs and take up to num_pairs
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask][:num_pairs], idx2[mask][:num_pairs]
    if len(idx1) == 0:
        return 0.0
    diffs = emb[idx1] - emb[idx2]
    dists = np.linalg.norm(diffs, axis=-1)
    return float(np.mean(dists))


# ---------------------------------------------------------------------------
#  All-in-one evaluation function
# ---------------------------------------------------------------------------

def evaluate_clatr_metrics(
    eval_wrapper: CLaTrEvalWrapper,
    captions: list,
    gt_motions: torch.Tensor,
    gt_lengths: torch.Tensor,
    gen_motions: torch.Tensor,
    gen_lengths: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute all CLaTr-style metrics for a set of ground truth and generated
    trajectories.

    Parameters
    ----------
    eval_wrapper : CLaTrEvalWrapper
    captions     : list[str]
    gt_motions   : (N, T, D)
    gt_lengths   : (N,)
    gen_motions  : (N, T, D)
    gen_lengths  : (N,)

    Returns
    -------
    metrics : dict with keys:
        fid, r_precision_top1, r_precision_top2, r_precision_top3,
        matching_score, gt_diversity, gen_diversity
    """
    text_emb = eval_wrapper.get_text_embeddings(captions)
    gt_traj_emb = eval_wrapper.get_trajectory_embeddings(gt_motions, gt_lengths)
    gen_traj_emb = eval_wrapper.get_trajectory_embeddings(gen_motions, gen_lengths)

    # FID
    fid = compute_fid(gt_traj_emb, gen_traj_emb)

    # R-Precision (on generated trajectories vs. their captions)
    r_prec = compute_r_precision(text_emb, gen_traj_emb, top_k=3)

    # Matching Score
    match_score = compute_matching_score(text_emb, gen_traj_emb)

    # Diversity
    gt_div = compute_diversity(gt_traj_emb)
    gen_div = compute_diversity(gen_traj_emb)

    return {
        "fid": fid,
        "r_precision_top1": r_prec[0],
        "r_precision_top2": r_prec[1],
        "r_precision_top3": r_prec[2],
        "matching_score": match_score,
        "gt_diversity": gt_div,
        "gen_diversity": gen_div,
    }
