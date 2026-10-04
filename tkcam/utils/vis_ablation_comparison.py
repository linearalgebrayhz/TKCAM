"""
Ablation Study Visualization: 2×2 Panel Comparison of RVQ + Transformer Contributions

Generates a side-by-side (2×2 grid) animated comparison of four trajectory paths:

  ┌──────────────────────────┬──────────────────────────┐
  │  Path A: GT Base-Only    │  Path B: Full GT Recon   │
  │  (Level-0 GT tokens)     │  (All GT RVQ levels)     │
  ├──────────────────────────┼──────────────────────────┤
  │  Path C: Pred Base-Only  │  Path D: Full Model Pred │
  │  (MTransformer only)     │  (MTrans + RTrans)       │
  └──────────────────────────┴──────────────────────────┘

Path A: Theoretical upper bound of the MTransformer — GT tokens, only Level 0.
Path B: RVQ reconstruction ceiling — GT tokens, all quantizer levels.
Path C: MTransformer predicted base token only (Level 0), decoded through RVQ.
Path D: Standard full-model inference (MTransformer base + RTransformer residuals).

All panels share axis limits, camera viewing angle, and OpenGL coordinate convention.
Positions are velocity-integrated (from dx, dy, dz channels) for plotting, consistent
with plot_camera_trajectory_animation_vel_integrated. Optional 1D Gaussian smoothing
(--smooth / --sigma) applied to raw paths before processing; integration uses its own
smoothing (smooth_sigma) on the integrated positions.
L2 position error (vs Path B) overlaid on Path A and Path C panels.

Usage:
    python vis_ablation_comparison.py \\
        --dataset_name realestate10k_rotmat \\
        --name mtrans_3k_clip_crossattn_reduce \\
        --res_name rtrans_3k_clip_crossattn_reduce \\
        --conditioning_mode clip \\
        --gpu_id 0 \\
        --text_path camera_prompts.txt \\
        --ext ablation_study \\
        --time_steps 10 --cond_scale 3 --temperature 0.3 --topkr 0.9 \\
        --smooth --sigma 1.0 \\
        --repeat_times 1

python utils/vis_ablation_comparison.py \
    --name mtrans_5k_newdata_xframe_r4_latest \
    --res_name rtrans_5k_newdata_xframe_r4_latest \
    --dataset_name realestate10k_rotmat \
    --conditioning_mode t5 \
    --use_keyframes \
    --time_steps 20 --cond_scale 1.0 --temperature 0.2 --topkr 0.9 \
    --smooth --sigma 1.0 \
    --repeat_times 2 \
    --split test --num_samples 100 \
    --ext ablation_study_xframe_r4
"""

import json
import os
import sys
import textwrap
from pathlib import Path
from os.path import join as pjoin

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Ensure the project root is first on sys.path and remove the utils/ dir
# to prevent circular imports (this script lives inside the utils package).
_utils_dir = os.path.dirname(os.path.abspath(__file__))
sys.path = [p for p in sys.path if os.path.abspath(p) != _utils_dir]
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.ndimage import gaussian_filter1d
from PIL import Image
from torchvision import transforms

from tkcam.models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from tkcam.models.vq.model import RVQVAE, LengthEstimator

from tkcam.options.eval_option import EvalT2MOptions
from tkcam.utils.get_opt import get_opt
from tkcam.utils.fixseed import fixseed
from tkcam.utils.unified_data_format import (
    UnifiedCameraData,
    CameraDataFormat,
    detect_format_from_dataset_name,
)
from tkcam.utils.camera_geometry import (
    sixd_to_matrix,
    forward_from_sixd,
    to_mpl,
    matrix_to_sixd,
    integrate_velocity_to_positions,
)

clip_version = 'ViT-B/32'


def _resolve_text_conditioning(ckpt_opt, cli_opt, role='model'):
    """Prefer conditioning_mode / T5 / id counts from checkpoint opt.txt (see gen_camera.py)."""
    cm_saved = getattr(ckpt_opt, 'conditioning_mode', None)
    cm_cli = getattr(cli_opt, 'conditioning_mode', 'clip')
    if cm_saved is not None and cm_saved != cm_cli:
        print(
            f'[vis_ablation] {role}: using conditioning_mode={cm_saved!r} from opt.txt '
            f'(CLI was {cm_cli!r}).',
        )
    cm = cm_saved if cm_saved is not None else cm_cli
    num_id = getattr(ckpt_opt, 'num_id_samples', getattr(cli_opt, 'num_id_samples', 50))
    t5_name = getattr(ckpt_opt, 't5_model_name', getattr(cli_opt, 't5_model_name', 't5-base'))
    return cm, num_id, t5_name


def _build_clip_preprocess(use_bicubic_resize: bool):
    resize = (
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC)
        if use_bicubic_resize else
        transforms.Resize(224)
    )
    return transforms.Compose([
        resize,
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def prepare_visual_conditioning_for_ablation(
    sample_names,
    gt_frame_lengths,
    scene_id_mapping,
    frame_dir,
    model_opt,
    device,
    clip_preprocess,
):
    """Build per-sample CLIP visual conditioning tensors for the ablation batch.

    Each sample is looked up via *scene_id_mapping* -> hash_id ->
    ``frame_dir/{hash_id}/`` and up to ``max_sparse_frames`` images are loaded
    with uniform temporal sampling.

    Returns (first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask),
    all ``None`` when the model does not use visual conditioning.
    """
    use_first_frame = getattr(model_opt, 'use_first_frame', False)
    use_sparse_frames = getattr(model_opt, 'use_sparse_frames', False)
    if not use_first_frame and not use_sparse_frames:
        return None, None, None, None

    max_k = getattr(model_opt, 'max_sparse_frames', 4)
    B = len(sample_names)
    frame_dir = Path(frame_dir)

    if use_first_frame:
        pixels = torch.zeros(B, 3, 224, 224)
        for b, name in enumerate(sample_names):
            hash_id = scene_id_mapping.get(name, name)
            scene_dir = frame_dir / hash_id
            if not scene_dir.exists():
                continue
            files = sorted(scene_dir.glob('frame_*.jpg'))
            if not files:
                continue
            img = Image.open(files[0]).convert('RGB')
            pixels[b] = clip_preprocess(img)
        return pixels.to(device), None, None, None

    # use_sparse_frames
    sparse = torch.zeros(B, max_k, 3, 224, 224)
    indices = torch.zeros(B, max_k, dtype=torch.long)
    valid = torch.zeros(B, max_k, dtype=torch.bool)

    for b, (name, gt_len) in enumerate(zip(sample_names, gt_frame_lengths)):
        hash_id = scene_id_mapping.get(name, name)
        scene_dir = frame_dir / hash_id
        if not scene_dir.exists():
            continue
        files = sorted(scene_dir.glob('frame_*.jpg'))
        if not files:
            continue
        n_avail = len(files)
        n_use = min(n_avail, max_k, max(1, gt_len))
        chosen_file_idxs = np.linspace(0, n_avail - 1, n_use, dtype=int)
        raw_max = max(gt_len - 1, 1)
        for slot, fi in enumerate(chosen_file_idxs):
            raw_frame_idx = round(fi * raw_max / max(n_avail - 1, 1))
            vq_idx = raw_frame_idx // 4
            img = Image.open(files[fi]).convert('RGB')
            sparse[b, slot] = clip_preprocess(img)
            indices[b, slot] = vq_idx
            valid[b, slot] = True

    return None, sparse.to(device), indices.to(device), valid.to(device)


# ──────────────────────────────────────────────────────────────────────────────
# Model loaders (reused from gen_camera.py)
# ──────────────────────────────────────────────────────────────────────────────

def load_vq_model(vq_opt):
    vq_model = RVQVAE(
        vq_opt, vq_opt.dim_pose, vq_opt.nb_code, vq_opt.code_dim,
        vq_opt.output_emb_width, vq_opt.down_t, vq_opt.stride_t,
        vq_opt.width, vq_opt.depth, vq_opt.dilation_growth_rate,
        vq_opt.vq_act, vq_opt.vq_norm,
    )
    is_camera = any(n in vq_opt.dataset_name.lower() for n in ("cam", "estate", "realestate"))
    if is_camera:
        ckpt_files = ['net_best_recon.tar', 'net_best_position.tar',
                      'net_best_smoothness.tar', 'latest.tar']
        loaded = False
        for cf in ckpt_files:
            path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', cf)
            if os.path.exists(path):
                ckpt = torch.load(path, map_location='cpu')
                key = 'vq_model' if 'vq_model' in ckpt else 'net'
                vq_model.load_state_dict(ckpt[key])
                print(f'Loading VQ Model {vq_opt.name} from {cf}')
                loaded = True
                break
        if not loaded:
            raise FileNotFoundError(f"No VQ checkpoint found for {vq_opt.name}")
    else:
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name,
                                'model', 'net_best_fid.tar'), map_location='cpu')
        key = 'vq_model' if 'vq_model' in ckpt else 'net'
        vq_model.load_state_dict(ckpt[key])
        print(f'Loading VQ Model {vq_opt.name} Completed!')
    return vq_model, vq_opt


def load_trans_model(model_opt, opt, which_model, text_cond=None, allow_root_fallback=True):
    if text_cond is None:
        text_cond = _resolve_text_conditioning(model_opt, opt, role='mask_transformer')
    conditioning_mode, num_id_samples, t5_model_name = text_cond
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim, cond_mode='text',
        latent_dim=model_opt.latent_dim, ff_size=model_opt.ff_size,
        num_layers=model_opt.n_layers, num_heads=model_opt.n_heads,
        dropout=model_opt.dropout, clip_dim=512,
        cond_drop_prob=model_opt.cond_drop_prob, clip_version=clip_version,
        conditioning_mode=conditioning_mode,
        num_id_samples=num_id_samples,
        t5_model_name=t5_model_name,
        use_first_frame=getattr(model_opt, 'use_first_frame', False),
        use_sparse_frames=getattr(model_opt, 'use_sparse_frames', False),
        max_sparse_frames=getattr(model_opt, 'max_sparse_frames', 4),
        visual_drop_prob=getattr(model_opt, 'visual_drop_prob', 0.0),
        opt=model_opt,
    )
    model_path = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model)
    root_path = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, which_model)
    if os.path.exists(model_path):
        ckpt_path = model_path
    elif allow_root_fallback and os.path.exists(root_path):
        ckpt_path = root_path
        print(f'  Loading from root (CLaTr eval saves net_best_fid to root): {which_model}')
    else:
        if allow_root_fallback:
            raise FileNotFoundError(f'Checkpoint not found: {model_path} or {root_path}')
        raise FileNotFoundError(f'Checkpoint not found: {model_path}')

    ckpt = torch.load(ckpt_path, map_location='cpu')
    key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing, unexpected = t2m_transformer.load_state_dict(ckpt[key], strict=False)
    _allowed = ('clip_model.', 'cond_provider.', 'clip_image_encoder.')
    assert len(unexpected) == 0, f'Unexpected keys in MaskTransformer: {unexpected}'
    assert all(any(k.startswith(p) for p in _allowed) for k in missing), \
        f'Missing trainable keys in MaskTransformer: {missing}'
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_res_model(res_opt, vq_opt, opt, text_cond=None):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    if text_cond is None:
        text_cond = _resolve_text_conditioning(res_opt, opt, role='res_transformer')
    conditioning_mode, num_id_samples, t5_model_name = text_cond
    res_transformer = ResidualTransformer(
        code_dim=vq_opt.code_dim, cond_mode='text',
        latent_dim=res_opt.latent_dim, ff_size=res_opt.ff_size,
        num_layers=res_opt.n_layers, num_heads=res_opt.n_heads,
        dropout=res_opt.dropout, clip_dim=512,
        shared_codebook=vq_opt.shared_codebook,
        cond_drop_prob=res_opt.cond_drop_prob,
        share_weight=res_opt.share_weight, clip_version=clip_version,
        conditioning_mode=conditioning_mode,
        num_id_samples=num_id_samples,
        t5_model_name=t5_model_name,
        use_first_frame=getattr(res_opt, 'use_first_frame', False),
        use_sparse_frames=getattr(res_opt, 'use_sparse_frames', False),
        max_sparse_frames=getattr(res_opt, 'max_sparse_frames', 4),
        visual_drop_prob=getattr(res_opt, 'visual_drop_prob', 0.0),
        opt=res_opt,
    )
    is_camera = any(n in res_opt.dataset_name.lower() for n in ("cam", "estate", "realestate"))
    res_which = getattr(opt, 'res_which_epoch', None)

    def _res_ckpt_full_path(basename):
        mp = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', basename)
        rp = pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, basename)
        if os.path.exists(mp):
            return mp
        if os.path.exists(rp):
            print(f'  Loading residual ckpt from run root: {basename}')
            return rp
        return None

    if res_which:
        ckpt_bn = res_which if str(res_which).endswith('.tar') else f'{res_which}.tar'
        ckpt_path = _res_ckpt_full_path(ckpt_bn)
        if ckpt_path is None:
            raise FileNotFoundError(
                f'res_which_epoch={res_which!r}: no {ckpt_bn} for {res_opt.dataset_name}/{res_opt.name}',
            )
        ckpt = torch.load(ckpt_path, map_location=opt.device)
    elif is_camera:
        ckpt_files = ['net_best_fid.tar', 'net_best_acc.tar', 'net_best_loss.tar']
        loaded = False
        for cf in ckpt_files:
            ckpt_path = _res_ckpt_full_path(cf)
            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path, map_location=opt.device)
                loaded = True
                print(f'Loading Residual Transformer {res_opt.name} from {cf}')
                break
        if not loaded:
            raise FileNotFoundError(f"No Res-Transformer checkpoint found for {res_opt.name}")
    else:
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name,
                                'model', 'net_best_fid.tar'), map_location=opt.device)
    missing, unexpected = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    _allowed = ('clip_model.', 'cond_provider.', 'clip_image_encoder.')
    assert len(unexpected) == 0, f'Unexpected keys in ResidualTransformer: {unexpected}'
    assert all(any(k.startswith(p) for p in _allowed) for k in missing), \
        f'Missing trainable keys in ResidualTransformer: {missing}'
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def apply_gram_schmidt_to_rot6d(data: np.ndarray, format_type: CameraDataFormat) -> np.ndarray:
    """Apply Gram-Schmidt orthonormalization to the 6D rotation columns.

    For FULL_12_ROTMAT format, rot6d lives in columns [6:12].
    For other formats, just return as-is (no raw rot6d columns).
    """
    if format_type != CameraDataFormat.FULL_12_ROTMAT:
        return data
    data = data.copy()
    rot6d = data[:, 6:12]
    R = sixd_to_matrix(rot6d)          # (..., 3, 3)
    clean_6d = matrix_to_sixd(R)       # (..., 6) — already ortho-normalised
    data[:, 6:12] = clean_6d
    return data


def smooth_trajectory(data: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply 1-D Gaussian filter along the time axis to each feature."""
    smoothed = np.empty_like(data)
    for d in range(data.shape[1]):
        smoothed[:, d] = gaussian_filter1d(data[:, d], sigma=sigma)
    return smoothed


def compute_l2_position_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """Mean L2 position error (first 3 dims) between pred and ref."""
    min_len = min(len(pred), len(ref))
    diff = pred[:min_len, :3] - ref[:min_len, :3]
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def _orientation_vector(orientations_row, format_type):
    """Compute a forward direction vector from one frame's orientation data."""
    if format_type == CameraDataFormat.FULL_12_ROTMAT:
        rot6d = orientations_row[:6]
        fwd = forward_from_sixd(rot6d)
        return fwd
    elif format_type == CameraDataFormat.FULL_12_EULER:
        pitch, yaw, roll = orientations_row[:3]
        fwd = np.array([
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return fwd / (np.linalg.norm(fwd) + 1e-8)
    elif format_type == CameraDataFormat.POSITION_ORIENTATION_6:
        pitch, yaw, roll = orientations_row[:3]
        fwd = np.array([
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return fwd / (np.linalg.norm(fwd) + 1e-8)
    elif format_type == CameraDataFormat.LEGACY_5:
        pitch, yaw = orientations_row[:2]
        fwd = np.array([
            -np.sin(yaw) * np.cos(pitch),
            np.sin(pitch),
            -np.cos(yaw) * np.cos(pitch),
        ])
        return fwd / (np.linalg.norm(fwd) + 1e-8)
    else:
        return np.array([0, 0, -1.0])


# ──────────────────────────────────────────────────────────────────────────────
# 2×2 Panel Animated Visualization
# ──────────────────────────────────────────────────────────────────────────────

def render_ablation_comparison(
    paths: dict,           # {'A': (N,D), 'B': (N,D), 'C': (N,D), 'D': (N,D)}
    save_path: str,
    text_prompt: str = "",
    format_type: CameraDataFormat = None,
    fps: int = 30,
    stride: int = 1,
    figsize: tuple = (16, 14),
    arrow_scale_factor: float = 0.06,
    show_trail: bool = True,
    trail_length: int = 30,
    smooth: bool = True,
    smooth_sigma: float = 1.5,
):
    """Render a 2×2 animated comparison of four trajectory paths.

    Uses velocity-integrated positions (from dx, dy, dz channels) for plotting,
    consistent with plot_camera_trajectory_animation_vel_integrated.

    Layout:
        top-left:  Path A (GT Base Only)       top-right: Path B (Full GT Recon)
        bot-left:  Path C (Pred Base Only)      bot-right: Path D (Full Model Pred)
    """
    labels = {
        'A': 'Path A: GT Base-Only (L0)',
        'B': 'Path B: Full GT Recon (All Levels)',
        'C': 'Path C: Pred Base-Only (MTransformer)',
        'D': 'Path D: Full Model Prediction',
    }
    panel_positions = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    # Parse all paths: velocity-integrated positions + orientations
    parsed = {}
    for key, raw in paths.items():
        ud = UnifiedCameraData(raw, format_type=format_type)
        orientations = ud.orientations.numpy()
        # Velocity-integrated positions (from dx, dy, dz channels)
        try:
            raw_pos = integrate_velocity_to_positions(
                raw, smooth=smooth, smooth_sigma=smooth_sigma
            )
        except ValueError as exc:
            print(f"[ablation] Path {key}: velocity integration failed ({exc}), using direct positions")
            raw_pos = ud.positions.numpy()
        mpl_pos = to_mpl(raw_pos)
        parsed[key] = {
            'positions': mpl_pos,
            'raw_positions': raw_pos,
            'orientations': orientations,
            'format_type': ud.format_type,
        }

    # Compute shared axis limits across all four paths
    all_positions = np.concatenate([p['positions'] for p in parsed.values()], axis=0)
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)
    extent = np.max(pos_max - pos_min)
    padding = max(extent * 0.12, 0.05)
    xlim = (pos_min[0] - padding, pos_max[0] + padding)
    ylim = (pos_min[1] - padding, pos_max[1] + padding)
    zlim = (pos_min[2] - padding, pos_max[2] + padding)

    # Arrow scale
    arrow_len = max(arrow_scale_factor * extent, 0.01)
    arrow_len = min(arrow_len, 0.25)

    # Pre-compute orientation vectors for all paths
    for key in parsed:
        vecs = []
        p = parsed[key]
        for i in range(len(p['positions'])):
            fwd_gl = _orientation_vector(p['orientations'][i], p['format_type'])
            fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).flatten()
            vecs.append(fwd_mpl)
        p['fwd_vectors'] = np.array(vecs)

    # Compute L2 errors relative to Path B
    ref_positions = parsed['B']['raw_positions']
    errors = {}
    for key in ('A', 'C'):
        errors[key] = compute_l2_position_error(
            np.column_stack([parsed[key]['raw_positions']]),  # just positions from raw
            np.column_stack([ref_positions]),
        )
    # Recompute properly (the above was wrong — just use raw positions directly)
    for key in ('A', 'C'):
        pred_pos = parsed[key]['raw_positions']
        ref_pos = parsed['B']['raw_positions']
        min_len = min(len(pred_pos), len(ref_pos))
        diff = pred_pos[:min_len] - ref_pos[:min_len]
        errors[key] = float(np.mean(np.linalg.norm(diff, axis=1)))

    # Determine max frames across all paths
    max_frames = max(len(p['positions']) for p in parsed.values())

    # Build figure
    fig = plt.figure(figsize=figsize)

    # Wrap prompt
    wrap_width = 90
    wrapped_prompt = '\n'.join(textwrap.wrap(text_prompt, width=wrap_width)) if text_prompt else ''
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.98, va='top')

    axes = {}
    artists = {}
    for key in ('A', 'B', 'C', 'D'):
        ax = fig.add_subplot(2, 2, panel_positions[key], projection='3d')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        ax.set_xlabel('X (Right)', fontsize=8)
        ax.set_ylabel('Depth (Forward)', fontsize=8)
        ax.set_zlabel('Y (Up)', fontsize=8)
        ax.view_init(elev=25, azim=-60)

        title_text = labels[key]
        if key in errors:
            title_text += f'\nL2 pos error vs B: {errors[key]:.4f}'
        ax.set_title(title_text, fontsize=9, pad=8)

        pos = parsed[key]['positions']

        # Full trajectory (light)
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], '-', color='cornflowerblue',
                linewidth=1.2, alpha=0.4, label='Full Path')
        # Start / End markers
        ax.scatter(*pos[0], c='green', s=80, marker='^', label='Start', zorder=5)
        ax.scatter(*pos[-1], c='red', s=80, marker='v', label='End', zorder=5)

        # Animated artists
        trail_line, = ax.plot([], [], [], '-', color='orange', linewidth=2.5, alpha=0.8)
        current_pt = ax.scatter([], [], [], c='red', s=120, zorder=10)

        ax.legend(fontsize=6, loc='upper right')
        axes[key] = ax
        artists[key] = {'trail': trail_line, 'point': current_pt, 'arrow': None}

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    def animate(frame_idx):
        changed = []
        for key in ('A', 'B', 'C', 'D'):
            pos = parsed[key]['positions']
            fwd = parsed[key]['fwd_vectors']
            ax = axes[key]
            n = len(pos)
            # Clamp frame index
            fi = min(frame_idx, n - 1)

            # Update trail
            t_start = max(0, fi - trail_length) if show_trail else fi
            trail_pos = pos[t_start:fi + 1]
            artists[key]['trail'].set_data(trail_pos[:, 0], trail_pos[:, 1])
            artists[key]['trail'].set_3d_properties(trail_pos[:, 2])

            # Update current point
            artists[key]['point']._offsets3d = ([pos[fi, 0]], [pos[fi, 1]], [pos[fi, 2]])

            # Remove previous arrow
            if artists[key]['arrow'] is not None:
                try:
                    artists[key]['arrow'].remove()
                except Exception:
                    pass

            # Draw orientation arrow
            artists[key]['arrow'] = ax.quiver(
                pos[fi, 0], pos[fi, 1], pos[fi, 2],
                fwd[fi, 0] * arrow_len, fwd[fi, 1] * arrow_len, fwd[fi, 2] * arrow_len,
                color='red', arrow_length_ratio=0.3, linewidth=1.5,
            )

            changed.extend([artists[key]['trail'], artists[key]['point']])
        return changed

    # Build frame list with stride
    frame_indices = list(range(0, max_frames, stride))
    if frame_indices[-1] != max_frames - 1:
        frame_indices.append(max_frames - 1)

    interval = 1000 / fps
    anim = FuncAnimation(fig, animate, frames=frame_indices,
                         interval=interval, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)
        anim.save(save_path, writer=writer)
    else:
        save_path += '.mp4'
        writer = FFMpegWriter(fps=fps, codec='libx264', bitrate=2000)
        anim.save(save_path, writer=writer)

    plt.close(fig)
    print(f"Ablation comparison animation saved to {save_path}")


def render_ablation_static(
    paths: dict,
    save_path: str,
    text_prompt: str = "",
    format_type: CameraDataFormat = None,
    arrow_scale_factor: float = 0.06,
    figsize: tuple = (16, 14),
    smooth: bool = True,
    smooth_sigma: float = 1.5,
):
    """Render a static 2×2 comparison plot (PNG).

    Uses velocity-integrated positions (from dx, dy, dz channels) for plotting,
    consistent with plot_camera_trajectory_animation_vel_integrated.
    """
    labels = {
        'A': 'Path A: GT Base-Only (L0)',
        'B': 'Path B: Full GT Recon (All Levels)',
        'C': 'Path C: Pred Base-Only (MTransformer)',
        'D': 'Path D: Full Model Prediction',
    }
    panel_positions = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

    parsed = {}
    for key, raw in paths.items():
        ud = UnifiedCameraData(raw, format_type=format_type)
        orientations = ud.orientations.numpy()
        # Velocity-integrated positions (from dx, dy, dz channels)
        try:
            raw_pos = integrate_velocity_to_positions(
                raw, smooth=smooth, smooth_sigma=smooth_sigma
            )
        except ValueError as exc:
            print(f"[ablation] Path {key}: velocity integration failed ({exc}), using direct positions")
            raw_pos = ud.positions.numpy()
        mpl_pos = to_mpl(raw_pos)
        parsed[key] = {
            'positions': mpl_pos,
            'raw_positions': raw_pos,
            'orientations': orientations,
            'format_type': ud.format_type,
        }

    # Pre-compute orientation vectors
    for key in parsed:
        vecs = []
        p = parsed[key]
        for i in range(len(p['positions'])):
            fwd_gl = _orientation_vector(p['orientations'][i], p['format_type'])
            fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).flatten()
            vecs.append(fwd_mpl)
        p['fwd_vectors'] = np.array(vecs)

    # Shared limits
    all_positions = np.concatenate([p['positions'] for p in parsed.values()], axis=0)
    pos_min = all_positions.min(axis=0)
    pos_max = all_positions.max(axis=0)
    extent = np.max(pos_max - pos_min)
    padding = max(extent * 0.12, 0.05)
    xlim = (pos_min[0] - padding, pos_max[0] + padding)
    ylim = (pos_min[1] - padding, pos_max[1] + padding)
    zlim = (pos_min[2] - padding, pos_max[2] + padding)

    arrow_len = max(arrow_scale_factor * extent, 0.01)
    arrow_len = min(arrow_len, 0.25)

    # L2 errors
    ref_pos = parsed['B']['raw_positions']
    errors = {}
    for key in ('A', 'C'):
        pred_pos = parsed[key]['raw_positions']
        min_len = min(len(pred_pos), len(ref_pos))
        diff = pred_pos[:min_len] - ref_pos[:min_len]
        errors[key] = float(np.mean(np.linalg.norm(diff, axis=1)))

    fig = plt.figure(figsize=figsize)
    wrapped_prompt = '\n'.join(textwrap.wrap(text_prompt, width=90)) if text_prompt else ''
    fig.suptitle(wrapped_prompt, fontsize=11, y=0.98, va='top')

    for key in ('A', 'B', 'C', 'D'):
        ax = fig.add_subplot(2, 2, panel_positions[key], projection='3d')
        ax.set_xlim(xlim); ax.set_ylim(ylim); ax.set_zlim(zlim)
        ax.set_xlabel('X (Right)', fontsize=8)
        ax.set_ylabel('Depth (Forward)', fontsize=8)
        ax.set_zlabel('Y (Up)', fontsize=8)
        ax.view_init(elev=25, azim=-60)

        title_text = labels[key]
        if key in errors:
            title_text += f'\nL2 pos error vs B: {errors[key]:.4f}'
        ax.set_title(title_text, fontsize=9, pad=8)

        pos = parsed[key]['positions']
        fwd = parsed[key]['fwd_vectors']

        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], 'b-', linewidth=2, alpha=0.7, label='Path')
        ax.scatter(*pos[0], c='green', s=100, marker='^', label='Start')
        ax.scatter(*pos[-1], c='red', s=100, marker='v', label='End')

        # Orientation arrows at sampled frames
        step = max(1, len(pos) // 10)
        for i in range(0, len(pos), step):
            ax.quiver(pos[i, 0], pos[i, 1], pos[i, 2],
                      fwd[i, 0] * arrow_len, fwd[i, 1] * arrow_len, fwd[i, 2] * arrow_len,
                      color='red', arrow_length_ratio=0.3, linewidth=1.0)

        ax.legend(fontsize=6, loc='upper right')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Ablation comparison static plot saved to {save_path}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI & Main
# ──────────────────────────────────────────────────────────────────────────────

def add_ablation_args(parser):
    """Add ablation-specific CLI arguments to an existing EvalT2MOptions parser."""
    parser.add_argument('--smooth', action='store_true',
                        help='Apply 1D Gaussian smoothing to all paths before rendering')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Sigma for Gaussian smoothing (default: 1.0)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Render every Nth frame for animation (stride=2 halves render time)')
    parser.add_argument('--anim_fps', type=int, default=30,
                        help='FPS for the output animation')
    parser.add_argument('--output_format', type=str, default='mp4', choices=['mp4', 'gif'],
                        help='Output animation format')
    parser.add_argument('--no_static', action='store_true',
                        help='Skip static PNG output')
    parser.add_argument('--no_animation', action='store_true',
                        help='Skip animated MP4/GIF output')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to draw from the dataset split (default: 10)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which dataset split to draw GT samples from (default: test)')
    parser.add_argument('--frame_dir', type=str,
                        default='/data4/haozhe/CamTraj/data/processed_estate/train_frames',
                        help='Root directory containing per-scene frame subdirectories '
                             '(default: /data4/haozhe/CamTraj/data/processed_estate/train_frames)')
    # Temporary factor-isolation flags (remove after diagnosis is complete).
    parser.add_argument('--legacy_single_batch_generation', action='store_true',
                        help='Disable mini-batch generation and run all prompts in one giant batch (legacy behavior).')
    parser.add_argument('--disable_mask_root_fallback', action='store_true',
                        help='Force mask checkpoint loading from model/ only (disable run-root fallback).')
    parser.add_argument('--legacy_clip_resize', action='store_true',
                        help='Use legacy CLIP preprocess Resize(224) instead of BICUBIC resize.')
    return parser


if __name__ == '__main__':
    # ── Parse arguments ──
    base_parser = EvalT2MOptions()
    add_ablation_args(base_parser.parser)
    opt = base_parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else f"cuda:{opt.gpu_id}")
    torch.autograd.set_detect_anomaly(True)

    # Format detection
    viz_format_type = detect_format_from_dataset_name(opt.dataset_name)
    print(f"Dataset: {opt.dataset_name}")
    print(f"Visualization format: {viz_format_type}")

    # ── Directories ──
    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    result_dir = pjoin('./generation', opt.ext)
    os.makedirs(result_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    # ── Load RVQ ──
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    print(f'VQ model dim_pose = {vq_opt.dim_pose}')
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # ── Load R-Transformer ──
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)
    text_cond_m = _resolve_text_conditioning(model_opt, opt, role='mask_transformer')
    text_cond_r = _resolve_text_conditioning(res_opt, opt, role='res_transformer')
    if text_cond_m[0] != text_cond_r[0]:
        print(
            f'[vis_ablation] WARNING: mask conditioning_mode={text_cond_m[0]!r} != '
            f'res {text_cond_r[0]!r}',
        )
    res_model = load_res_model(res_opt, vq_opt, opt, text_cond=text_cond_r)
    assert res_opt.vq_name == model_opt.vq_name

    # ── Load M-Transformer ──
    which_m = getattr(opt, 'which_epoch', 'net_best_fid')
    ckpt_m = which_m if str(which_m).endswith('.tar') else f'{which_m}.tar'
    t2m_transformer = load_trans_model(
        model_opt, opt, ckpt_m, text_cond=text_cond_m,
        allow_root_fallback=(not getattr(opt, 'disable_mask_root_fallback', False)),
    )

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()
    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    # ── Normalisation stats ──
    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))

    def inv_transform(data):
        return data * std + mean

    # ── Resolve data_root ──
    _data_root_defaults = {
        't2m': './dataset/HumanML3D/',
        'kit': './dataset/KIT-ML/',
        'cam': './dataset/CameraTraj/',
        'realestate10k_6': './dataset/RealEstate10K_6feat/',
        'realestate10k_12': './dataset/RealEstate10K_12feat/',
        'realestate10k_quat': './dataset/RealEstate10K_quat/',
        'realestate10k_rotmat': './dataset/RealEstate10K_rotmat/',
    }
    if opt.data_root:
        data_root = opt.data_root
    elif hasattr(model_opt, 'data_root') and model_opt.data_root:
        data_root = model_opt.data_root
    else:
        data_root = _data_root_defaults.get(opt.dataset_name, f'./dataset/{opt.dataset_name}/')
    print(f"Data root: {data_root}")

    # ── Load samples directly from dataset split ──
    # This ensures every sample has GT data for Path A & B
    conditioning_mode = text_cond_m[0]
    motion_dir = pjoin(data_root, 'new_joint_vecs')
    texts_dir = pjoin(data_root, 'texts')

    split_file = pjoin(data_root, f'{opt.split}.txt')
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}")

    with open(split_file, 'r') as f:
        split_sample_names = [line.strip() for line in f if line.strip()]

    # Limit to --num_samples
    num_samples = min(opt.num_samples, len(split_sample_names))
    split_sample_names = split_sample_names[:num_samples]

    # Load GT motion + text prompt for each sample
    gt_sample_names = []
    gt_motions = []
    captions = []
    for sample_name in split_sample_names:
        gt_path = pjoin(motion_dir, f"{sample_name}.npy")
        text_path = pjoin(texts_dir, f"{sample_name}.txt")
        if not os.path.exists(gt_path):
            print(f"  Skipping {sample_name}: GT motion not found at {gt_path}")
            continue
        gt_motion = np.load(gt_path)
        # Read text prompt
        if os.path.exists(text_path):
            with open(text_path, 'r') as tf:
                raw_text = tf.readline().strip()
                caption = raw_text.split('#')[0].strip()
        else:
            caption = f"Sample {sample_name}"
        gt_sample_names.append(sample_name)
        gt_motions.append(gt_motion)
        captions.append(caption)

    if len(captions) == 0:
        raise RuntimeError(f"No valid samples found in {split_file}. "
                           f"Check that {motion_dir} and {texts_dir} contain matching files.")

    print(f"\nLoaded {len(captions)} samples from {opt.split} split")
    print(f"Processing {len(captions)} samples × {opt.repeat_times} repeats\n")

    # Build token lengths from GT motion lengths or default
    if opt.motion_length > 0:
        token_lens = torch.LongTensor([opt.motion_length // 4] * len(captions)).to(opt.device)
    else:
        # Use actual GT lengths (rounded to unit_length)
        raw_lengths = [len(m) for m in gt_motions]
        # Round each length down to nearest multiple of 4 (unit_length)
        rounded_lengths = [(l // 4) for l in raw_lengths]
        token_lens = torch.LongTensor(rounded_lengths).to(opt.device)
    m_length = token_lens * 4

    # Set generation conditions based on conditioning mode
    if conditioning_mode == 'id_embedding':
        # For id_embedding, map sample names to train-split indices
        train_split_file = pjoin(data_root, 'train.txt')
        if os.path.exists(train_split_file):
            with open(train_split_file, 'r') as f:
                train_names = [l.strip() for l in f if l.strip()]
            name_to_id = {n: i for i, n in enumerate(train_names)}
        else:
            name_to_id = {}
        sample_id_list = [name_to_id.get(n, 0) for n in gt_sample_names]
        gen_conds = torch.LongTensor(sample_id_list).to(opt.device)
    else:
        gen_conds = captions

    # ── Visual conditioning (per-sample, from dataset frames) ──────────────
    vis_first_frame = None
    vis_sparse_frames = None
    vis_indices = None
    vis_valid_mask = None
    clip_preprocess = _build_clip_preprocess(
        use_bicubic_resize=(not getattr(opt, 'legacy_clip_resize', False)))
    print(
        "[vis_ablation] Factor toggles: "
        f"mini_batch={'OFF' if getattr(opt, 'legacy_single_batch_generation', False) else 'ON'}, "
        f"mask_root_fallback={'OFF' if getattr(opt, 'disable_mask_root_fallback', False) else 'ON'}, "
        f"clip_bicubic_resize={'OFF' if getattr(opt, 'legacy_clip_resize', False) else 'ON'}"
    )

    _model_has_visual = (
        getattr(model_opt, 'use_first_frame', False)
        or getattr(model_opt, 'use_sparse_frames', False)
    )
    if opt.use_keyframes and _model_has_visual:
        mapping_path = pjoin(data_root, 'scene_id_mapping.json')
        if os.path.exists(mapping_path):
            with open(mapping_path) as _f:
                scene_id_mapping = json.load(_f)
        else:
            scene_id_mapping = {}
            print(f"Warning: scene_id_mapping.json not found at {mapping_path}")

        gt_frame_lengths = [len(m) for m in gt_motions]
        (vis_first_frame,
         vis_sparse_frames,
         vis_indices,
         vis_valid_mask) = prepare_visual_conditioning_for_ablation(
            gt_sample_names, gt_frame_lengths, scene_id_mapping,
            opt.frame_dir, model_opt, opt.device, clip_preprocess,
        )
        vis_mode = 'first_frame' if vis_first_frame is not None else 'sparse_frames'
        print(f"Visual conditioning: {vis_mode} "
              f"(model use_first_frame={getattr(model_opt, 'use_first_frame', False)}, "
              f"use_sparse_frames={getattr(model_opt, 'use_sparse_frames', False)})")
    elif opt.use_keyframes and not _model_has_visual:
        print("Warning: --use_keyframes set but model has no visual conditioning. Ignoring.")

    # ── Main generation loop ──
    for r in range(opt.repeat_times):
        print(f"{'='*70}")
        print(f"  Repeat {r}")
        print(f"{'='*70}")

        with torch.no_grad():
            # Generate in mini-batches (default) to avoid prompt-collapse on large
            # T5 batches. Legacy factor-isolation mode can force one giant batch.
            if getattr(opt, 'legacy_single_batch_generation', False):
                gen_batch_size = len(captions)
            else:
                gen_batch_size = getattr(opt, 'batch_size', 32)
            n_samples = len(captions)
            print(f"[vis_ablation] Effective generation batch_size={gen_batch_size}, n_samples={n_samples}")
            path_c_chunks = []
            path_d_chunks = []

            for batch_start in range(0, n_samples, gen_batch_size):
                batch_end = min(batch_start + gen_batch_size, n_samples)
                b_token_lens = token_lens[batch_start:batch_end]

                if conditioning_mode == 'id_embedding':
                    b_conds = gen_conds[batch_start:batch_end]
                else:
                    b_conds = captions[batch_start:batch_end]

                b_vis_first = vis_first_frame[batch_start:batch_end] if vis_first_frame is not None else None
                b_vis_sparse = vis_sparse_frames[batch_start:batch_end] if vis_sparse_frames is not None else None
                b_vis_idx = vis_indices[batch_start:batch_end] if vis_indices is not None else None
                b_vis_mask = vis_valid_mask[batch_start:batch_end] if vis_valid_mask is not None else None

                # ── Path C: base-only prediction ──
                mids_base = t2m_transformer.generate(
                    b_conds, b_token_lens,
                    timesteps=opt.time_steps,
                    cond_scale=opt.cond_scale,
                    temperature=opt.temperature,
                    topk_filter_thres=opt.topkr,
                    gsample=opt.gumbel_sample,
                    first_frame_pixels=b_vis_first,
                    sparse_frames=b_vis_sparse,
                    visual_indices=b_vis_idx,
                    visual_valid_mask=b_vis_mask,
                )
                mids_base_only = mids_base.unsqueeze(-1)
                pred_base_only = vq_model.forward_decoder(mids_base_only)
                path_c_chunks.append(inv_transform(pred_base_only.detach().cpu().numpy()))

                # ── Path D: full prediction (base + residual) ──
                mids_full = res_model.generate(
                    mids_base, b_conds, b_token_lens,
                    temperature=1, cond_scale=getattr(opt, 'res_cond_scale', 5),
                    first_frame_pixels=b_vis_first,
                    sparse_frames=b_vis_sparse,
                    visual_indices=b_vis_idx,
                    visual_valid_mask=b_vis_mask,
                )
                pred_full = vq_model.forward_decoder(mids_full)
                path_d_chunks.append(inv_transform(pred_full.detach().cpu().numpy()))

            # Different mini-batches can decode to different temporal lengths.
            # Pad to a common T before concatenation (same strategy as gen_camera.py).
            max_t_c = max(x.shape[1] for x in path_c_chunks)
            padded_c = []
            for x in path_c_chunks:
                if x.shape[1] < max_t_c:
                    pad = np.zeros((x.shape[0], max_t_c - x.shape[1], x.shape[2]), dtype=x.dtype)
                    x = np.concatenate([x, pad], axis=1)
                padded_c.append(x)
            path_C_all = np.concatenate(padded_c, axis=0)

            max_t_d = max(x.shape[1] for x in path_d_chunks)
            padded_d = []
            for x in path_d_chunks:
                if x.shape[1] < max_t_d:
                    pad = np.zeros((x.shape[0], max_t_d - x.shape[1], x.shape[2]), dtype=x.dtype)
                    x = np.concatenate([x, pad], axis=1)
                padded_d.append(x)
            path_D_all = np.concatenate(padded_d, axis=0)

        # Process each sample
        for k, caption in enumerate(captions):
            length = m_length[k].item()
            sample_name = gt_sample_names[k]
            gt_motion = gt_motions[k]
            print(f"\n---->Sample {k} [{sample_name}]: \"{caption}\" (length={length}, GT frames={len(gt_motion)})")

            # Encode GT through RVQ to get code indices
            gt_tensor = torch.tensor(gt_motion, dtype=torch.float32).unsqueeze(0)  # (1, T, D)
            # Normalize GT data (stored unnormalized — raw features)
            gt_normalized = (gt_tensor.numpy() - mean) / std
            gt_tensor = torch.tensor(gt_normalized, dtype=torch.float32).to(opt.device)

            with torch.no_grad():
                gt_code_idx, gt_all_codes = vq_model.encode(gt_tensor)
                # gt_code_idx: (1, T', Q), where T' = T // stride

                # ── Path A: GT Base-Only (Level 0 only) ──
                gt_base_only_idx = gt_code_idx[:, :, 0:1]   # (1, T', 1)
                path_A_recon = vq_model.forward_decoder(gt_base_only_idx)
                path_A_recon = path_A_recon.detach().cpu().numpy()
                path_A = inv_transform(path_A_recon)[0]

                # ── Path B: Full GT Reconstruction (all levels) ──
                path_B_recon = vq_model.forward_decoder(gt_code_idx)
                path_B_recon = path_B_recon.detach().cpu().numpy()
                path_B = inv_transform(path_B_recon)[0]

            path_C = path_C_all[k][:length]
            path_D = path_D_all[k][:length]

            # Trim all paths to the same length for fair comparison
            min_len = min(len(path_A), len(path_B), len(path_C), len(path_D))
            path_A = path_A[:min_len]
            path_B = path_B[:min_len]
            path_C = path_C[:min_len]
            path_D = path_D[:min_len]

            # Apply Gram-Schmidt orthonormalization to 6D rotation components
            path_A = apply_gram_schmidt_to_rot6d(path_A, viz_format_type)
            path_B = apply_gram_schmidt_to_rot6d(path_B, viz_format_type)
            path_C = apply_gram_schmidt_to_rot6d(path_C, viz_format_type)
            path_D = apply_gram_schmidt_to_rot6d(path_D, viz_format_type)

            # Optional Gaussian smoothing
            if opt.smooth:
                sigma = opt.sigma
                print(f"  Applying Gaussian smoothing (sigma={sigma})")
                path_A = smooth_trajectory(path_A, sigma=sigma)
                path_B = smooth_trajectory(path_B, sigma=sigma)
                path_C = smooth_trajectory(path_C, sigma=sigma)
                path_D = smooth_trajectory(path_D, sigma=sigma)
                # Re-apply Gram-Schmidt after smoothing to keep rotations valid
                path_A = apply_gram_schmidt_to_rot6d(path_A, viz_format_type)
                path_B = apply_gram_schmidt_to_rot6d(path_B, viz_format_type)
                path_C = apply_gram_schmidt_to_rot6d(path_C, viz_format_type)
                path_D = apply_gram_schmidt_to_rot6d(path_D, viz_format_type)

            paths_dict = {'A': path_A, 'B': path_B, 'C': path_C, 'D': path_D}

            # ── Output directory ──
            sample_dir = pjoin(result_dir, str(k))
            os.makedirs(sample_dir, exist_ok=True)

            # ── Save raw .npy for each path ──
            for pkey, pdata in paths_dict.items():
                np.save(pjoin(sample_dir, f"sample{k}_repeat{r}_path{pkey}.npy"), pdata)

            # ── Render static PNG ──
            if not opt.no_static:
                static_path = pjoin(sample_dir,
                                    f"sample{k}_repeat{r}_len{min_len}_ablation.png")
                render_ablation_static(
                    paths_dict, static_path, text_prompt=caption,
                    format_type=viz_format_type,
                    smooth=True,
                    smooth_sigma=opt.sigma,
                )

            # ── Render animation ──
            if not opt.no_animation:
                ext = getattr(opt, 'output_format', 'mp4')
                anim_path = pjoin(sample_dir,
                                  f"sample{k}_repeat{r}_len{min_len}_ablation.{ext}")
                render_ablation_comparison(
                    paths_dict, anim_path, text_prompt=caption,
                    format_type=viz_format_type,
                    fps=opt.anim_fps,
                    stride=opt.stride,
                    smooth=True,
                    smooth_sigma=opt.sigma,
                )

            # ── Print L2 errors ──
            ref_pos = paths_dict['B'][:, :3]
            for pkey in ('A', 'C', 'D'):
                pred_pos = paths_dict[pkey][:, :3]
                l2 = float(np.mean(np.linalg.norm(pred_pos - ref_pos, axis=1)))
                print(f"  L2 pos error (Path {pkey} vs B): {l2:.6f}")

            print(f"  Outputs saved to {sample_dir}")

    print(f"\n{'='*70}")
    print(f"Ablation comparison complete. Results in {result_dir}")
    print(f"{'='*70}")
