import os
from os.path import join as pjoin
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before any pyplot import

import torch

from tkcam.models.mask_transformer.transformer import MaskTransformer, ResidualTransformer
from tkcam.models.vq.model import RVQVAE

from tkcam.options.eval_option import EvalT2MOptions
from tkcam.utils.get_opt import get_opt
from tkcam.utils.dataset_layout import traj_dir

from tkcam.utils.fixseed import fixseed
from tkcam.utils.unified_data_format import UnifiedCameraData, CameraDataFormat
from tkcam.utils.dataset_config import get_unified_dataset_config
from tkcam.utils.camera_geometry import (
    forward_from_sixd, to_mpl,
)

import numpy as np
from torch.distributions.categorical import Categorical
from PIL import Image
clip_version = 'ViT-B/32'


def _resolve_text_conditioning(ckpt_opt, cli_opt, role='model'):
    """Use text-conditioning settings from checkpoint opt.txt, not only CLI defaults.

    Training often uses ``--conditioning_mode t5`` while EvalT2MOptions defaults to
    ``clip``. Building the wrong module layout makes ``load_state_dict`` look OK
    (strict=False) but encodes prompts with the wrong pathway — a common source of
    collapsed / text-agnostic generations at inference.
    """
    cm_saved = getattr(ckpt_opt, 'conditioning_mode', None)
    cm_cli = getattr(cli_opt, 'conditioning_mode', 'clip')
    if cm_saved is not None and cm_saved != cm_cli:
        print(
            f'[gen_camera] {role}: using conditioning_mode={cm_saved!r} from opt.txt '
            f'(CLI was {cm_cli!r}).',
        )
    cm = cm_saved if cm_saved is not None else cm_cli
    num_id = getattr(ckpt_opt, 'num_id_samples', getattr(cli_opt, 'num_id_samples', 50))
    t5_name = getattr(ckpt_opt, 't5_model_name', getattr(cli_opt, 't5_model_name', 't5-base'))
    return cm, num_id, t5_name


def load_vq_model(vq_opt):
    vq_model = RVQVAE(vq_opt,
                vq_opt.dim_pose,
                vq_opt.nb_code,
                vq_opt.code_dim,
                vq_opt.output_emb_width,
                vq_opt.down_t,
                vq_opt.stride_t,
                vq_opt.width,
                vq_opt.depth,
                vq_opt.dilation_growth_rate,
                vq_opt.vq_act,
                vq_opt.vq_norm)
    
    # Choose checkpoint file based on dataset type
    is_camera_dataset = any(name in vq_opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
    if is_camera_dataset:
        # For camera datasets, try different checkpoint files in order of preference
        checkpoint_files = [
            'net_best_recon.tar',      # Best reconstruction loss
            'net_best_position.tar',   # Best position accuracy
            'net_best_smoothness.tar', # Best smoothness
            'latest.tar'               # Latest checkpoint
        ]
        
        checkpoint_loaded = False
        for checkpoint_file in checkpoint_files:
            checkpoint_path = pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', checkpoint_file)
            if os.path.exists(checkpoint_path):
                ckpt = torch.load(checkpoint_path, map_location='cpu')
                model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
                vq_model.load_state_dict(ckpt[model_key])
                print(f'Loading VQ Model {vq_opt.name} from {checkpoint_file} Completed!')
                if checkpoint_file != 'net_best_recon.tar':
                    print(f'[Warning] VQ {vq_opt.name}: no net_best_recon.tar, fell back to {checkpoint_file} (epoch {ckpt.get("ep")})')
                checkpoint_loaded = True
                break
        
        if not checkpoint_loaded:
            raise FileNotFoundError(f"No VQ checkpoint found in {pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'net_best_fid.tar'),
                                map_location='cpu')
        model_key = 'vq_model' if 'vq_model' in ckpt else 'net'
        vq_model.load_state_dict(ckpt[model_key])
        print(f'Loading VQ Model {vq_opt.name} Completed!')
    
    return vq_model, vq_opt

def load_trans_model(model_opt, opt, which_model, text_cond=None):
    if text_cond is None:
        text_cond = _resolve_text_conditioning(model_opt, opt, role='mask_transformer')
    conditioning_mode, num_id_samples, t5_model_name = text_cond
    t2m_transformer = MaskTransformer(code_dim=model_opt.code_dim,
                                      cond_mode='text',
                                      latent_dim=model_opt.latent_dim,
                                      ff_size=model_opt.ff_size,
                                      num_layers=model_opt.n_layers,
                                      num_heads=model_opt.n_heads,
                                      dropout=model_opt.dropout,
                                      clip_dim=512,
                                      cond_drop_prob=model_opt.cond_drop_prob,
                                      clip_version=clip_version,
                                      conditioning_mode=conditioning_mode,
                                      num_id_samples=num_id_samples,
                                      t5_model_name=t5_model_name,
                                      use_first_frame=getattr(model_opt, 'use_first_frame', False),
                                      use_sparse_frames=getattr(model_opt, 'use_sparse_frames', False),
                                      max_sparse_frames=getattr(model_opt, 'max_sparse_frames', 4),
                                      visual_drop_prob=getattr(model_opt, 'visual_drop_prob', 0.0),
                                      opt=model_opt)
    model_path = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, 'model', which_model)
    root_path = pjoin(model_opt.checkpoints_dir, model_opt.dataset_name, model_opt.name, which_model)
    if os.path.exists(model_path):
        ckpt_path = model_path
    elif os.path.exists(root_path):
        ckpt_path = root_path
        print(f'  Loading from root (CLaTr eval saves net_best_fid to root): {which_model}')
    else:
        raise FileNotFoundError(f'Checkpoint not found: {model_path} or {root_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing_keys, unexpected_keys = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0, f'Unexpected keys in MaskTransformer: {unexpected_keys}'
    allowed = ('clip_model.', 'cond_provider.t5_encoder.', 'clip_image_encoder.')
    assert all(any(k.startswith(p) for p in allowed) for k in missing_keys), \
        f'Missing trainable keys in MaskTransformer: {missing_keys}'
    print(f'Loading Transformer {opt.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer

def load_res_model(res_opt, vq_opt, opt, text_cond=None):
    res_opt.num_quantizers = vq_opt.num_quantizers
    res_opt.num_tokens = vq_opt.nb_code
    if text_cond is None:
        text_cond = _resolve_text_conditioning(res_opt, opt, role='res_transformer')
    conditioning_mode, num_id_samples, t5_model_name = text_cond
    res_transformer = ResidualTransformer(code_dim=vq_opt.code_dim,
                                            cond_mode='text',
                                            latent_dim=res_opt.latent_dim,
                                            ff_size=res_opt.ff_size,
                                            num_layers=res_opt.n_layers,
                                            num_heads=res_opt.n_heads,
                                            dropout=res_opt.dropout,
                                            clip_dim=512,
                                            shared_codebook=vq_opt.shared_codebook,
                                            cond_drop_prob=res_opt.cond_drop_prob,
                                            share_weight=res_opt.share_weight,
                                            clip_version=clip_version,
                                            conditioning_mode=conditioning_mode,
                                            num_id_samples=num_id_samples,
                                            t5_model_name=t5_model_name,
                                            use_first_frame=getattr(res_opt, 'use_first_frame', False),
                                            use_sparse_frames=getattr(res_opt, 'use_sparse_frames', False),
                                            max_sparse_frames=getattr(res_opt, 'max_sparse_frames', 4),
                                            visual_drop_prob=getattr(res_opt, 'visual_drop_prob', 0.0),
                                            opt=res_opt)

    # Choose checkpoint file based on dataset type
    is_camera_dataset = any(name in res_opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
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
                f'res_which_epoch={res_which!r}: no file {ckpt_bn} under model/ or run root '
                f'for {res_opt.dataset_name}/{res_opt.name}',
            )
        ckpt = torch.load(ckpt_path, map_location=opt.device)
    elif is_camera_dataset:
        # For camera datasets, prefer validation-selected checkpoints.
        checkpoint_files = [
            'net_best_fid.tar',        # Best CLaTr FID when available
            'net_best_acc.tar',        # Best accuracy
            'net_best_loss.tar',       # Best loss
        ]

        checkpoint_loaded = False
        for checkpoint_file in checkpoint_files:
            ckpt_path = _res_ckpt_full_path(checkpoint_file)
            if ckpt_path is not None:
                ckpt = torch.load(ckpt_path, map_location=opt.device)
                checkpoint_loaded = True
                break

        if not checkpoint_loaded:
            raise FileNotFoundError(f"No residual transformer checkpoint found in {pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model')}")
    else:
        # For human motion datasets, use the original logic
        ckpt = torch.load(pjoin(res_opt.checkpoints_dir, res_opt.dataset_name, res_opt.name, 'model', 'net_best_fid.tar'),
                          map_location=opt.device)
    missing_keys, unexpected_keys = res_transformer.load_state_dict(ckpt['res_transformer'], strict=False)
    assert len(unexpected_keys) == 0, f'Unexpected keys in ResidualTransformer: {unexpected_keys}'
    allowed = ('clip_model.', 'cond_provider.t5_encoder.', 'clip_image_encoder.')
    assert all(any(k.startswith(p) for p in allowed) for k in missing_keys), \
        f'Missing trainable keys in ResidualTransformer: {missing_keys}'
    print(f'Loading Residual Transformer {res_opt.name} from epoch {ckpt["ep"]}!')
    return res_transformer

def prepare_visual_conditioning_for_inference(
    keyframe_dir: str,
    keyframe_indices_str: str,
    model_opt,
    batch_size: int,
    token_seq_len: int,
    device,
):
    """Build CLIP-based visual conditioning tensors for inference.

    Mirrors the tensor layout produced by
    ``transformer_trainer.MaskTransformerTrainer._prepare_batch()``.

    Supports both conditioning modes stored in *model_opt*:

    ``use_first_frame=True``
        Takes the **first** image file in *keyframe_dir* as the single
        first-frame anchor.  Returns ``first_frame_pixels (B, 3, 224, 224)``.

    ``use_sparse_frames=True``
        Loads up to ``max_sparse_frames`` images from *keyframe_dir*
        (sorted alphabetically) and pairs them with raw frame indices
        from *keyframe_indices_str* (comma-separated).
        Raw indices are divided by 4 → VQ-level, matching the downsampling
        applied in the trainer (``visual_indices //= 4``).
        Returns ``sparse_frames (B,K,3,224,224)``,
        ``visual_indices (B,K)`` long, ``visual_valid_mask (B,K)`` bool.

    Parameters
    ----------
    keyframe_dir : str
        Directory containing jpg/png frame images (sorted = chronological).
    keyframe_indices_str : str
        Comma-separated **raw** frame indices (e.g. ``"0,60,120,180"``).
        If empty, indices are spread uniformly across the timeline.
    model_opt :
        Loaded opt namespace for the MaskTransformer checkpoint.
    batch_size : int
    token_seq_len : int
        VQ token sequence length (= motion_length // 4).
    device : torch.device

    Returns
    -------
    first_frame_pixels : Tensor (B, 3, 224, 224) or None
    sparse_frames      : Tensor (B, K, 3, 224, 224) or None
    visual_indices     : Tensor (B, K) long or None  — already VQ-level
    visual_valid_mask  : Tensor (B, K) bool or None
    """
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    frame_dir = Path(keyframe_dir)
    all_files = sorted([
        f for f in frame_dir.iterdir()
        if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
    ])
    if not all_files:
        raise ValueError(f'No image files found in {frame_dir}')

    use_first_frame   = getattr(model_opt, 'use_first_frame',   False)
    use_sparse_frames = getattr(model_opt, 'use_sparse_frames', False)
    max_sparse_frames = getattr(model_opt, 'max_sparse_frames', 4)

    # ── use_first_frame: one image → first_frame_pixels ───────────────────────
    if use_first_frame:
        img_path = all_files[0]
        print(f'  [visual] First-frame conditioning: {img_path.name}')
        img = Image.open(img_path).convert('RGB')
        tensor = preprocess(img).unsqueeze(0)                      # (1,3,224,224)
        first_frame_pixels = tensor.expand(batch_size, -1, -1, -1).to(device)
        return first_frame_pixels, None, None, None

    # ── use_sparse_frames: up to K images → sparse conditioning ───────────────
    if use_sparse_frames:
        K = max_sparse_frames

        # Resolve raw frame indices
        if keyframe_indices_str:
            raw_indices = [int(x.strip()) for x in keyframe_indices_str.split(',')]
        else:
            # Auto-spread uniformly across raw frame timeline
            n      = len(all_files)
            raw_max = token_seq_len * 4 - 1
            raw_indices = [round(i * raw_max / max(1, n - 1)) for i in range(n)]

        n_use = min(len(all_files), len(raw_indices), K)

        frames     = torch.zeros(1, K, 3, 224, 224)
        vq_indices = torch.zeros(1, K, dtype=torch.long)
        valid_mask = torch.zeros(1, K, dtype=torch.bool)

        print(f'  [visual] Sparse keyframe conditioning ({n_use}/{K} slots):')
        for slot, (img_path, raw_idx) in enumerate(
                zip(all_files[:n_use], raw_indices[:n_use])):
            vq_idx = raw_idx // 4          # matches transformer_trainer line 165
            img = Image.open(img_path).convert('RGB')
            frames[0, slot]     = preprocess(img)
            vq_indices[0, slot] = vq_idx
            valid_mask[0, slot] = True
            print(f'    slot {slot}: {img_path.name}'
                  f'  raw={raw_idx}  VQ_token={vq_idx}')

        sparse_frames_t  = frames.expand(batch_size, -1, -1, -1, -1).to(device)
        visual_indices_t = vq_indices.expand(batch_size, -1).to(device)
        visual_valid_t   = valid_mask.expand(batch_size, -1).to(device)
        return None, sparse_frames_t, visual_indices_t, visual_valid_t

    # Model uses neither mode — nothing to do
    print('  [visual] Model has no visual conditioning (use_first_frame=False, '
          'use_sparse_frames=False). Keyframe dir ignored.')
    return None, None, None, None

# matches training processing
def _clip_preprocess():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])


def _resolve_frame_source_to_keyframe_dir(
    frame_source: str,
    frame_root: Path,
    scene_id_mapping: dict,
):
    """Resolve text#length#frame_source field to a concrete keyframe directory.

    Supports:
    - Absolute keyframe directory path
    - Relative directory under frame_root
    - Scene ID / hash ID (resolved via scene_id_mapping when available)
    """
    if frame_source is None:
        return None
    src = frame_source.strip()
    if not src:
        return None

    p = Path(src).expanduser()
    if p.exists() and p.is_dir():
        return str(p)

    # Relative path under --frame_dir root
    rel = (frame_root / src).expanduser()
    if rel.exists() and rel.is_dir():
        return str(rel)

    # scene_id -> hash_id via mapping, then hash_id under frame_root
    mapped = scene_id_mapping.get(src, src)
    mapped_dir = (frame_root / mapped).expanduser()
    if mapped_dir.exists() and mapped_dir.is_dir():
        return str(mapped_dir)

    return None


def prepare_visual_conditioning_per_sample(
    keyframe_dirs,
    token_seq_lens,
    model_opt,
    device,
):
    """Build CLIP-based visual conditioning with **per-sample** keyframe dirs.

    Unlike ``prepare_visual_conditioning_for_inference`` (single dir broadcast),
    this function accepts a list of directories (one per batch item) and builds
    independent visual tensors for each.

    Parameters
    ----------
    keyframe_dirs : list[str | None]
        One directory path per batch item. ``None`` entries yield zero tensors
        (visual tokens masked out for that sample).
    token_seq_lens : list[int] | Tensor
        VQ-level token lengths per batch item (used to auto-spread indices).
    model_opt : Namespace
    device : torch.device

    Returns
    -------
    first_frame_pixels, sparse_frames, visual_indices, visual_valid_mask
    """
    use_first_frame = getattr(model_opt, 'use_first_frame', False)
    use_sparse_frames = getattr(model_opt, 'use_sparse_frames', False)
    if not use_first_frame and not use_sparse_frames:
        return None, None, None, None

    preprocess = _clip_preprocess()
    B = len(keyframe_dirs)
    max_k = getattr(model_opt, 'max_sparse_frames', 4)

    if use_first_frame:
        pixels = torch.zeros(B, 3, 224, 224)
        for b, kd in enumerate(keyframe_dirs):
            if kd is None:
                continue
            frame_dir = Path(kd)
            files = sorted([
                f for f in frame_dir.iterdir()
                if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
            ]) if frame_dir.exists() else []
            if files:
                img = Image.open(files[0]).convert('RGB')
                pixels[b] = preprocess(img)
        return pixels.to(device), None, None, None

    # use_sparse_frames
    sparse = torch.zeros(B, max_k, 3, 224, 224)
    indices = torch.zeros(B, max_k, dtype=torch.long)
    valid = torch.zeros(B, max_k, dtype=torch.bool)

    if hasattr(token_seq_lens, 'tolist'):
        token_seq_lens = token_seq_lens.tolist()

    for b, kd in enumerate(keyframe_dirs):
        if kd is None:
            continue
        frame_dir = Path(kd)
        if not frame_dir.exists():
            continue
        files = sorted([
            f for f in frame_dir.iterdir()
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')
        ])
        if not files:
            continue
        n_avail = len(files)
        n_use = min(n_avail, max_k)
        tsl = token_seq_lens[b]
        raw_max = max(tsl * 4 - 1, 1)

        chosen = np.linspace(0, n_avail - 1, n_use, dtype=int) if n_avail > 1 else [0]
        for slot, fi in enumerate(chosen[:max_k]):
            raw_idx = round(int(fi) * raw_max / max(n_avail - 1, 1))
            img = Image.open(files[fi]).convert('RGB')
            sparse[b, slot] = preprocess(img)
            indices[b, slot] = raw_idx // 4
            valid[b, slot] = True

    return None, sparse.to(device), indices.to(device), valid.to(device)


@torch.no_grad()
def debug_trace_single_sample(
    *,
    t2m_transformer,
    conds,
    token_lens,
    cond_scale,
    norm_mean,
    norm_std,
    vis_first_frame=None,
    vis_sparse_frames=None,
    vis_indices=None,
    vis_valid_mask=None,
):
    """Print one-sample debug trace for conditioning + first-step logits.

    Enable once per run with:
        GEN_CAMERA_DEBUG_TRACE=1 python gen_camera.py ...
    """
    device = next(t2m_transformer.parameters()).device
    m_lens_1 = token_lens[:1].to(device)
    seq_len = int(m_lens_1.max().item())
    pos = torch.arange(seq_len, device=device).unsqueeze(0)
    padding_mask = pos >= m_lens_1.unsqueeze(1)  # (1, S), True = pad

    if torch.is_tensor(conds):
        conds_1 = conds[:1]
    else:
        conds_1 = [conds[0]]

    ff_1 = vis_first_frame[:1] if vis_first_frame is not None else None
    sp_1 = vis_sparse_frames[:1] if vis_sparse_frames is not None else None
    vi_1 = vis_indices[:1] if vis_indices is not None else None
    vm_1 = vis_valid_mask[:1] if vis_valid_mask is not None else None

    cond_vector, _, cond_mask = t2m_transformer.encode_condition(conds_1, 1, device)
    if cond_vector.dim() == 2:
        emb_norm = cond_vector.norm(dim=1)
        emb_msg = f"shape={tuple(cond_vector.shape)}, L2={emb_norm.detach().cpu().numpy().tolist()}"
    else:
        flat_norm = cond_vector.flatten(1).norm(dim=1)
        token_norm = cond_vector.norm(dim=-1).mean(dim=1)
        emb_msg = (
            f"shape={tuple(cond_vector.shape)}, "
            f"flatten_L2={flat_norm.detach().cpu().numpy().tolist()}, "
            f"mean_token_L2={token_norm.detach().cpu().numpy().tolist()}"
        )

    if ff_1 is not None:
        img_msg = (
            f"first_frame_pixels shape={tuple(ff_1.shape)}, "
            f"min={ff_1.min().item():.6f}, max={ff_1.max().item():.6f}"
        )
    elif sp_1 is not None:
        img_msg = (
            f"sparse_frames shape={tuple(sp_1.shape)}, "
            f"min={sp_1.min().item():.6f}, max={sp_1.max().item():.6f}, "
            f"valid_mask={vm_1.detach().cpu().numpy().astype(int).tolist() if vm_1 is not None else None}, "
            f"indices={vi_1.detach().cpu().numpy().tolist() if vi_1 is not None else None}"
        )
    else:
        img_msg = "no visual tensor (text-only conditioning)"

    z_mean = float(norm_mean[2]) if norm_mean is not None and len(norm_mean) > 2 else float("nan")
    z_std = float(norm_std[2]) if norm_std is not None and len(norm_std) > 2 else float("nan")
    vz_mean = float(norm_mean[5]) if norm_mean is not None and len(norm_mean) > 5 else float("nan")
    vz_std = float(norm_std[5]) if norm_std is not None and len(norm_std) > 5 else float("nan")

    ids = torch.where(
        padding_mask,
        torch.full_like(padding_mask, t2m_transformer.pad_id, dtype=torch.long),
        torch.full_like(padding_mask, t2m_transformer.mask_id, dtype=torch.long),
    )

    logits = t2m_transformer.forward_with_cond_scale(
        ids,
        cond_vector=cond_vector,
        padding_mask=padding_mask,
        cond_scale=cond_scale,
        cond_mask=cond_mask,
        first_frame_pixels=ff_1,
        sparse_frames=sp_1,
        visual_indices=vi_1,
        visual_valid_mask=vm_1,
    )  # (1, vocab, S)

    logits_step = logits.permute(0, 2, 1)  # (1, S, vocab)
    argmax_ids = logits_step.argmax(dim=-1)
    first_valid_pos = int((~padding_mask[0]).nonzero(as_tuple=False)[0].item())
    first_argmax = int(argmax_ids[0, first_valid_pos].item())

    print("\n" + "=" * 80)
    print("[Debug Trace] Single-sample conditioning + first-step logits")
    print(f"text_embedding: {emb_msg}")
    print(f"image_tensor(pre-CLIP): {img_msg}")
    print(
        "norm_stats: "
        f"z_mean={z_mean:.6f}, z_std={z_std:.6f}, "
        f"vz_mean={vz_mean:.6f}, vz_std={vz_std:.6f}"
    )
    print(
        "timestep1: "
        f"logits_shape={tuple(logits_step.shape)}, "
        f"first_valid_pos={first_valid_pos}, first_argmax_token={first_argmax}"
    )
    print("=" * 80 + "\n")

def plot_camera_trajectory_animation(data, save_path, title="Camera Trajectory", 
                                   fps=30, arrow_scale_factor=0.05, 
                                   min_arrow_length=0.01, max_arrow_length=0.2,
                                   show_trail=True, trail_length=30, figsize=(12, 10),
                                   format_type=None, stride=1, rotate_view=True):
    """
    Create an animated 3D visualization of camera trajectory with smooth movement
    Supports multiple camera data formats (5D, 6D, 12D) with automatic detection
    
    Args:
        data: Camera trajectory data (seq_len, features) - supports 5D, 6D, or 12D formats
        save_path: Path to save the animation (supports .gif, .mp4)
        title: Title for the animation
        fps: Frames per second for the animation
        arrow_scale_factor: Factor to scale arrows relative to trajectory extent
        min_arrow_length: Minimum arrow length to ensure visibility
        max_arrow_length: Maximum arrow length to prevent overly long arrows
        show_trail: Whether to show a trail behind the camera
        trail_length: Number of previous positions to show in trail
        figsize: Figure size tuple
        format_type: Explicit format type (CameraDataFormat), if None will auto-detect
        stride: Render every Nth frame (stride=2 halves render time, stride=3 thirds it, etc.)
        rotate_view: Whether to rotate camera view each frame (False = fixed view, much faster)
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Use unified data format for automatic handling of different dimensions
    unified_data = UnifiedCameraData(data, format_type=format_type)
    raw_positions = unified_data.positions.numpy()  # Always [x, y, z]
    orientations = unified_data.orientations.numpy()  # Depends on format
    
    # OpenGL convention: X=right, Y=up, -Z=forward
    # Visualization: X=right, Y=depth, Z=up
    # Transform via camera_geometry.to_mpl: [x, y, z] -> [x, -z, y]
    positions = to_mpl(raw_positions)
    
    # Create figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate dynamic arrow length based on trajectory extent
    pos_ranges = np.ptp(positions, axis=0)
    trajectory_extent = np.max(pos_ranges)
    
    if len(positions) > 1:
        step_distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_step_size = np.mean(step_distances)
        scale_reference = max(trajectory_extent, avg_step_size * 10)
    else:
        scale_reference = trajectory_extent
    
    base_arrow_length = max(arrow_scale_factor * scale_reference, min_arrow_length)
    base_arrow_length = min(base_arrow_length, max_arrow_length)
    
    # Set up plot limits with some padding
    padding = trajectory_extent * 0.1
    ax.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
    ax.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
    ax.set_zlim(positions[:, 2].min() - padding, positions[:, 2].max() + padding)
    
    # Format title
    title_length = len(title)
    if title_length > 100:
        title_fontsize = 8
        title_wrap_width = 80
    elif title_length > 60:
        title_fontsize = 10
        title_wrap_width = 60
    elif title_length > 30:
        title_fontsize = 12
        title_wrap_width = 40
    else:
        title_fontsize = 14
        title_wrap_width = 30
    
    import textwrap
    wrapped_title = '\n'.join(textwrap.wrap(title, width=title_wrap_width))
    
    # Format info with descriptive names
    format_display_names = {
        "LEGACY_5": "5D Legacy",
        "POSITION_ORIENTATION_6": "6D Euler",
        "QUATERNION_10": "10D Quat",
        "FULL_12_EULER": "12D Euler",
        "FULL_12_ROTMAT": "12D RotMat"
    }
    format_name = format_display_names.get(unified_data.format_type.name, unified_data.format_type.name)
    format_info = f"[{format_name}: {unified_data.num_features}D]"
    display_title = f"{wrapped_title}\n{format_info}"
    
    ax.set_title(display_title, fontsize=title_fontsize, pad=20)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Depth (Forward)')  # This is now -Z from original data
    ax.set_zlabel('Y (Up)')           # This is now Y from original data
    
    # Initialize empty line and point objects for animation
    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Full Path')
    trail_line, = ax.plot([], [], [], 'orange', linewidth=3, alpha=0.8, label='Recent Trail')
    current_point = ax.scatter([], [], [], c='red', s=200, label='Current Position')
    orientation_arrow = None
    
    # Pre-compute all orientation vectors (expensive operations done once)
    orientation_vectors = []
    for frame_idx in range(len(positions)):
        ori = orientations[frame_idx]
        
        # Compute forward vector in OpenGL world space, then map to MPL
        if unified_data.format_type == CameraDataFormat.QUATERNION_10:
            from tkcam.common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            # Use camera_geometry: forward = -(col0 x col1)
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            # Euler-based formats (5D, 6D, 12D Euler)
            pitch, yaw = ori[0], ori[1]
            # OpenGL forward from Euler: direction camera looks
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),   # X
                -np.sin(pitch),                  # Y
                -np.cos(pitch) * np.cos(yaw)     # -Z (forward)
            ])

        # Deprecated. This may not be correct now.
        
        # Apply to_mpl mapping identically to position transform: [x,y,z] -> [x,-z,y]
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        
        # Normalize
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        
        orientation_vectors.append(tuple(fwd_mpl))
    
    # Add start and end markers
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], 
              c='green', s=150, label='Start', marker='^')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], 
              c='red', s=150, label='End', marker='v')
    
    ax.legend()
    
    def animate(frame):
        nonlocal orientation_arrow
        
        # Clear previous orientation arrow
        if orientation_arrow is not None:
            orientation_arrow.remove()
        
        # Update full trajectory (fade in effect)
        alpha = min(1.0, frame / 20)  # Fade in over first 20 frames
        trajectory_line.set_data_3d(positions[:frame+1, 0], 
                                   positions[:frame+1, 1], 
                                   positions[:frame+1, 2])
        trajectory_line.set_alpha(alpha * 0.6)
        
        # Update trail
        if show_trail and frame > 0:
            trail_start = max(0, frame - trail_length)
            trail_positions = positions[trail_start:frame+1]
            trail_line.set_data_3d(trail_positions[:, 0], 
                                  trail_positions[:, 1], 
                                  trail_positions[:, 2])
        
        # Update current position
        current_pos = positions[frame]
        current_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Get pre-computed orientation
        dx, dy, dz = orientation_vectors[frame]
        
        # Draw camera forward direction arrow (where camera looks)
        orientation_arrow = ax.quiver(current_pos[0], current_pos[1], current_pos[2], 
                                     dx, dy, dz, 
                                     length=1.5*base_arrow_length, 
                                     color='purple', alpha=0.8, 
                                     arrow_length_ratio=0.3,
                                     linewidth=2,
                                     label='Camera Forward' if frame == 0 else '')
        
        # Update view angle for dynamic perspective (optional, expensive)
        if rotate_view:
            ax.view_init(elev=20, azim=frame * 0.5 % 360)
        
        return trajectory_line, trail_line, current_point, orientation_arrow
    
    # Create animation with optional frame subsampling
    frame_indices = list(range(0, len(positions), stride))
    # Always include last frame
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)
    
    interval = 1000 / fps  # Convert fps to interval in milliseconds
    anim = FuncAnimation(fig, animate, frames=frame_indices, 
                        interval=interval, blit=False, repeat=True)
    
    # Save animation
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=100)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=100)
    else:
        # Default to gif
        writer = PillowWriter(fps=fps)
        anim.save(save_path + '.gif', writer=writer, dpi=100)

    # Release FuncAnimation + its closure (holds positions/orientations arrays) before GC
    del anim
    plt.close(fig)
    print(f"Camera trajectory animation saved to {save_path}")

def plot_camera_trajectory_debug(data, save_path, title="Camera Trajectory Debug", 
                                fps=20, show_velocity=True, show_topdown=True,
                                show_statistics=True, figsize=(15, 12), text_prompt=None,
                                format_type=None, stride=1):
    """
    Create comprehensive debugging visualization with multiple views and analysis
    
    Args:
        data: Camera trajectory data (seq_len, features)
        save_path: Path to save the debug animation
        title: Title for the debug view
        fps: Frames per second
        show_velocity: Whether to show velocity vectors
        show_topdown: Whether to show top-down view (X-Depth plane)
        show_statistics: Whether to show statistical information
        figsize: Figure size tuple
        text_prompt: Optional text prompt to display in statistics panel for comparison
        format_type: Explicit format type (CameraDataFormat), if None will auto-detect
        stride: Render every Nth frame (stride=2 halves render time, etc.)
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np
    
    # Use unified data format
    unified_data = UnifiedCameraData(data, format_type=format_type)
    raw_positions = unified_data.positions.numpy()
    orientations = unified_data.orientations.numpy()
    
    # OpenGL convention: X=right, Y=up, -Z=forward
    # Transform via camera_geometry.to_mpl: [x, y, z] -> [x, -z, y]
    positions = to_mpl(raw_positions)
    
    # Calculate derivatives for analysis
    velocities = np.zeros_like(positions)
    if len(positions) > 1:
        velocities[1:] = np.diff(positions, axis=0)
    
    # Calculate statistics
    velocity_magnitudes = np.linalg.norm(velocities, axis=1)
    
    # Pre-compute all orientation vectors to avoid repeated calculations
    orientation_vectors_debug = []
    for frame_idx in range(len(positions)):
        ori = orientations[frame_idx]
        
        # Compute forward vector in OpenGL world space
        if unified_data.format_type == CameraDataFormat.QUATERNION_10:
            from tkcam.common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            # Euler-based formats
            pitch, yaw = ori[0], ori[1]
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),
                -np.sin(pitch),
                -np.cos(pitch) * np.cos(yaw)
            ])
        
        # Apply to_mpl mapping: [x,y,z] -> [x,-z,y]
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        
        orientation_vectors_debug.append((fwd_mpl[0], fwd_mpl[1], fwd_mpl[2], ori))
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    
    # Main 3D view
    ax_3d = fig.add_subplot(2, 2, 1, projection='3d')
    
    # Velocity plot
    ax_vel = fig.add_subplot(2, 2, 2)
    
    # Top-down view (X-Depth plane)
    ax_topdown = fig.add_subplot(2, 2, 3)
    
    # Statistics text
    ax_stats = fig.add_subplot(2, 2, 4)
    ax_stats.axis('off')
    
    # Set up 3D plot
    padding = np.max(np.ptp(positions, axis=0)) * 0.1
    ax_3d.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
    ax_3d.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
    ax_3d.set_zlim(positions[:, 2].min() - padding, positions[:, 2].max() + padding)
    ax_3d.set_title('3D Camera Trajectory')
    ax_3d.set_xlabel('X (Right)')
    ax_3d.set_ylabel('Depth (Forward)')
    ax_3d.set_zlabel('Y (Up)')
    
    # Set up velocity plot
    ax_vel.set_title('Velocity Magnitude Over Time')
    ax_vel.set_xlabel('Frame')
    ax_vel.set_ylabel('Velocity')
    ax_vel.grid(True)
    
    # Set up top-down view (calculate limits once)
    x_range = positions[:, 0].max() - positions[:, 0].min()
    y_range = positions[:, 1].max() - positions[:, 1].min()
    max_range = max(x_range, y_range)
    center_x = (positions[:, 0].max() + positions[:, 0].min()) / 2
    center_y = (positions[:, 1].max() + positions[:, 1].min()) / 2
    margin = max_range * 0.1
    topdown_xlim = (center_x - max_range/2 - margin, center_x + max_range/2 + margin)
    topdown_ylim = (center_y - max_range/2 - margin, center_y + max_range/2 + margin)
    arrow_scale_3d = np.max(np.ptp(positions, axis=0)) * 0.1
    arrow_scale_2d = np.max(np.ptp(positions[:, :2], axis=0)) * 0.1
    
    # Initialize plot artists (create once, update data in animation)
    traj_3d_line, = ax_3d.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
    curr_3d_point = ax_3d.scatter([], [], [], c='red', s=100)
    
    vel_line, = ax_vel.plot([], [], 'g-', linewidth=2)
    vel_point = ax_vel.scatter([], [], c='red', s=50, zorder=5)
    
    traj_topdown_line, = ax_topdown.plot([], [], 'b-', linewidth=2, alpha=0.7)
    curr_topdown_point = ax_topdown.scatter([], [], c='red', s=100, zorder=5)
    ax_topdown.set_xlim(topdown_xlim)
    ax_topdown.set_ylim(topdown_ylim)
    
    def animate_debug(frame):
        # Get current data
        current_pos = positions[frame]
        dx, dy, dz, ori = orientation_vectors_debug[frame]
        
        # Update 3D trajectory
        if frame > 0:
            traj_3d_line.set_data_3d(positions[:frame+1, 0], positions[:frame+1, 1], positions[:frame+1, 2])
        curr_3d_point._offsets3d = ([current_pos[0]], [current_pos[1]], [current_pos[2]])
        
        # Remove previous arrows if they exist (cannot reuse quiver/arrow objects)
        for artist in list(ax_3d.collections) + list(ax_topdown.patches):
            if artist not in [curr_3d_point, curr_topdown_point]:
                artist.remove()
        
        # Draw orientation arrow in 3D
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2], 
                        dx, dy, dz, length=arrow_scale_3d, color='purple', alpha=0.8)
        
        # Draw velocity vector in 3D
        if show_velocity and frame > 0:
            vel = velocities[frame]
            vel_norm = np.linalg.norm(vel)
            if vel_norm > 1e-6:
                vel_normalized = vel / vel_norm
                vel_length = min(vel_norm * 10, arrow_scale_3d)
                ax_3d.quiver(current_pos[0], current_pos[1], current_pos[2], 
                           vel_normalized[0], vel_normalized[1], vel_normalized[2], 
                           length=vel_length, color='green', alpha=0.6)
        
        # Update velocity plot
        vel_line.set_data(range(frame + 1), velocity_magnitudes[:frame+1])
        vel_point.set_offsets([[frame, velocity_magnitudes[frame]]])
        if frame > 0:
            ax_vel.set_xlim(0, max(10, frame + 1))
            max_vel = max(velocity_magnitudes[:frame+1])
            ax_vel.set_ylim(0, max_vel * 1.1 if max_vel > 0 else 1)
        
        # Update top-down view
        if frame > 0:
            traj_topdown_line.set_data(positions[:frame+1, 0], positions[:frame+1, 1])
        curr_topdown_point.set_offsets([[current_pos[0], current_pos[1]]])
        
        # Draw orientation arrow in top-down view
        if np.sqrt(dx**2 + dy**2 + dz**2) > 1e-6:
            ax_topdown.arrow(current_pos[0], current_pos[1], 
                           dx * arrow_scale_2d, dy * arrow_scale_2d,
                           head_width=arrow_scale_2d*0.3, head_length=arrow_scale_2d*0.2,
                           fc='purple', ec='purple', alpha=0.8)
        
        # Update statistics text
        ax_stats.clear()
        ax_stats.axis('off')
        
        if show_statistics:
            stats_text = f"""Frame: {frame}/{len(positions)-1}

Current Position:
X: {current_pos[0]:.3f}
Y: {current_pos[1]:.3f}  
Z: {current_pos[2]:.3f}

Current Orientation:
Pitch: {ori[0]:.3f}
Yaw: {ori[1]:.3f}
"""
            
            if text_prompt:
                import textwrap
                wrapped_prompt = textwrap.fill(text_prompt, width=35)
                stats_text = f"""TEXT PROMPT:
"{wrapped_prompt}"
{'─' * 40}

{stats_text}"""
            if len(ori) > 2:
                stats_text += f"Roll: {ori[2]:.3f}\n"
                
            if frame > 0:
                stats_text += f"""
Current Velocity: {velocity_magnitudes[frame]:.3f}
Avg Velocity: {np.mean(velocity_magnitudes[1:frame+1]):.3f}
"""
            
            stats_text += f"""
Orientation Vector:
X-component: {dx:.3f}
Y-component: {dy:.3f}
Z-component: {dz:.3f}
"""
            
            if frame > 5:
                recent_positions = positions[max(0, frame-5):frame+1]
                path_length = np.sum(np.linalg.norm(np.diff(recent_positions, axis=0), axis=1))
                smoothness = 1.0 / (1.0 + np.std(velocity_magnitudes[max(1, frame-5):frame+1]))
                
                recent_displacement = recent_positions[-1] - recent_positions[0]
                dominant_axis = np.argmax(np.abs(recent_displacement))
                axis_names = ['X (Right)', 'Depth (Forward)', 'Y (Up)']
                direction = 'positive' if recent_displacement[dominant_axis] > 0 else 'negative'
                
                stats_text += f"""
MOTION ANALYSIS:
Recent Path Length: {path_length:.3f}
Smoothness: {smoothness:.3f}
Dominant Motion: {direction} {axis_names[dominant_axis]}
Speed: {velocity_magnitudes[frame]:.3f}
"""
            
            ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes, 
                         fontsize=10, verticalalignment='top', fontfamily='monospace',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    
    # Create animation with optional frame subsampling
    frame_indices = list(range(0, len(positions), stride))
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)
    
    interval = 1000 / fps
    anim = FuncAnimation(fig, animate_debug, frames=frame_indices, 
                        interval=interval, blit=False, repeat=True)
    
    # Save animation
    if save_path.endswith('.gif'):
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer, dpi=80)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=80)
    else:
        # Default to mp4 for better quality
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path + '.mp4', writer=writer, dpi=100)

    del anim
    plt.close(fig)
    print(f"Debug animation saved to {save_path}")

def plot_camera_trajectory(data, save_path, title="Camera Trajectory", arrow_scale_factor=0.05, 
                          min_arrow_length=0.01, max_arrow_length=0.2, format_type=None):
    """
    Plot camera trajectory as a 3D path with dynamically scaled orientation arrows
    Supports multiple camera data formats (5D, 6D, 12D) with automatic detection
    
    Args:
        data: Camera trajectory data (seq_len, features) - supports 5D, 6D, or 12D formats
        save_path: Path to save the plot
        title: Title for the plot
        arrow_scale_factor: Factor to scale arrows relative to trajectory extent (default: 0.05 = 5%)
        min_arrow_length: Minimum arrow length to ensure visibility (default: 0.01)
        max_arrow_length: Maximum arrow length to prevent overly long arrows (default: 0.2)
        format_type: Explicit format type (CameraDataFormat), if None will auto-detect
    """
    # Use unified data format for automatic handling of different dimensions
    unified_data = UnifiedCameraData(data, format_type=format_type)
    raw_positions = unified_data.positions.numpy()
    orientations = unified_data.orientations.numpy()
    
    # OpenGL: X=right, Y=up, -Z=forward -> Viz: X=right, Y=depth, Z=up
    # Transform via camera_geometry.to_mpl: [x, y, z] -> [x, -z, y]
    positions = to_mpl(raw_positions)
    
    # Create a simple 3D plot
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Adaptive font size based on title length
    title_length = len(title)
    if title_length > 100:
        title_fontsize = 8
        title_wrap_width = 80
    elif title_length > 60:
        title_fontsize = 10
        title_wrap_width = 60
    elif title_length > 30:
        title_fontsize = 12
        title_wrap_width = 40
    else:
        title_fontsize = 14
        title_wrap_width = 30
    
    # Wrap long titles
    import textwrap
    wrapped_title = '\n'.join(textwrap.wrap(title, width=title_wrap_width))
    
    # Add format information to title with friendly display names
    format_display_names = {
        "LEGACY_5": "5D Legacy",
        "POSITION_ORIENTATION_6": "6D Euler",
        "QUATERNION_10": "10D Quat",
        "FULL_12_EULER": "12D Euler",
        "FULL_12_ROTMAT": "12D RotMat"
    }
    format_name = format_display_names.get(unified_data.format_type.name, unified_data.format_type.name)
    format_info = f"[{format_name}: {unified_data.num_features}D]"
    display_title = f"{wrapped_title}\n{format_info}"
    
    # Plot camera positions
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2, label='Camera Path')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='g', s=100, label='Start')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='r', s=100, label='End')
    
    # Calculate dynamic arrow length based on trajectory extent
    pos_ranges = np.ptp(positions, axis=0)  # Range (max - min) for each axis
    trajectory_extent = np.max(pos_ranges)  # Maximum extent across all axes
    
    # Also consider average step size for more refined scaling
    if len(positions) > 1:
        step_distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        avg_step_size = np.mean(step_distances)
        # Use the larger of trajectory extent or a multiple of average step size
        # This helps when trajectories are very dense or very sparse
        scale_reference = max(trajectory_extent, avg_step_size * 10)
    else:
        scale_reference = trajectory_extent
    
    # Adaptive arrow length: scale based on trajectory characteristics with user-configurable parameters
    base_arrow_length = max(arrow_scale_factor * scale_reference, min_arrow_length)
    base_arrow_length = min(base_arrow_length, max_arrow_length)
    
    # Plot camera orientations as arrows at key points
    step = max(1, len(positions) // 10)
    for i in range(0, len(positions), step):
        pos = positions[i]
        ori = orientations[i]
        
        # Compute forward vector in OpenGL world space
        if unified_data.format_type == CameraDataFormat.QUATERNION_10:
            from tkcam.common.quaternion import qrot
            quat = torch.tensor(ori, dtype=torch.float32)
            forward_local = torch.tensor([0.0, 0.0, -1.0])
            fwd_gl = qrot(quat.unsqueeze(0), forward_local.unsqueeze(0)).squeeze(0).numpy()
        elif unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            # Euler-based formats (5D, 6D, 12D Euler)
            pitch, yaw = ori[0], ori[1]
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),
                -np.sin(pitch),
                -np.cos(pitch) * np.cos(yaw)
            ])
        
        # Apply to_mpl mapping: [x,y,z] -> [x,-z,y]
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        
        # Normalize
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        
        dx, dy, dz = fwd_mpl[0], fwd_mpl[1], fwd_mpl[2]
        
        # Draw arrow with adaptive length (orange for camera forward direction)
        ax.quiver(pos[0], pos[1], pos[2], dx, dy, dz, 
                 length=1.5*base_arrow_length, color='orange', alpha=0.7, 
                 arrow_length_ratio=0.3)  # Make arrowhead proportional
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(display_title, fontsize=title_fontsize, pad=20)
    ax.legend()
    
    # Add text annotation showing the arrow scale for reference
    # info_text = f'Arrow scale: {base_arrow_length:.3f}\nTrajectory extent: {trajectory_extent:.3f}'
    # if len(positions) > 1:
    #     info_text += f'\nAvg step size: {avg_step_size:.3f}'
    # ax.text2D(0.02, 0.98, info_text, 
    #           transform=ax.transAxes, fontsize=8, verticalalignment='top',
    #           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_camera_trajectory_animation_vel_integrated(
    data, save_path, title="Camera Trajectory (Vel. Integrated)",
    fps=30, show_trail=True, trail_length=30, figsize=(12, 10),
    format_type=None, stride=1, rotate_view=True,
    smooth=True, smooth_sigma=1.5,
):
    """Animate camera trajectory reconstructed by integrating the velocity channels.

    Uses the  [dx, dy, dz]  channels (indices 3-5) of the 12-D feature vector
    ``[x, y, z, dx, dy, dz, rot6d(6)]`` to reconstruct positions via cumulative
    summation, anchored at the first frame's direct position.  Gaussian smoothing
    is applied to the integrated positions before rendering.

    The current position-based visualization does **not** apply any smoothing;
    this function adds smoothing only to the velocity-integrated trajectory.

    Args:
        data:         (N, D) camera features.  D must be >= 9 (position + velocity +
                      at least one orientation channel).  If D < 6 the function logs
                      a warning and returns without writing any file.
        save_path:    Output path (.mp4 or .gif).
        title:        Plot title.
        fps:          Frames per second.
        show_trail:   Draw a coloured trail behind the current camera position.
        trail_length: Number of frames in the trail.
        figsize:      Matplotlib figure size.
        format_type:  CameraDataFormat override for orientation extraction.
        stride:       Render every N-th frame.
        rotate_view:  Slowly rotate the 3-D view each frame.
        smooth:       Apply Gaussian smoothing to the integrated positions.
        smooth_sigma: Gaussian sigma (frames) for smoothing.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
    import matplotlib.pyplot as plt
    from tkcam.utils.camera_geometry import integrate_velocity_to_positions

    if data.shape[-1] < 6:
        print(
            f"[vel_integrated] Skipping: data has only {data.shape[-1]} channels "
            f"(need >= 6 for velocity integration)."
        )
        return

    # Integrate velocity channels → positions in OpenGL frame
    try:
        raw_positions = integrate_velocity_to_positions(
            data, smooth=smooth, smooth_sigma=smooth_sigma
        )
    except ValueError as exc:
        print(f"[vel_integrated] Skipping: {exc}")
        return

    # Map to Matplotlib Z-up convention: [x, y, z] → [x, -z, y]
    positions = to_mpl(raw_positions)

    # Extract orientations from original data via UnifiedCameraData
    unified_data = UnifiedCameraData(data, format_type=format_type)
    orientations = unified_data.orientations.numpy()

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    pos_ranges = np.ptp(positions, axis=0)
    trajectory_extent = np.max(pos_ranges) if np.max(pos_ranges) > 0 else 1.0

    if len(positions) > 1:
        step_distances = np.sqrt(np.sum(np.diff(positions, axis=0) ** 2, axis=1))
        avg_step_size = np.mean(step_distances)
        scale_reference = max(trajectory_extent, avg_step_size * 10)
    else:
        scale_reference = trajectory_extent

    arrow_scale_factor, min_arrow_length, max_arrow_length = 0.05, 0.01, 0.2
    base_arrow_length = max(arrow_scale_factor * scale_reference, min_arrow_length)
    base_arrow_length = min(base_arrow_length, max_arrow_length)

    padding = trajectory_extent * 0.1
    ax.set_xlim(positions[:, 0].min() - padding, positions[:, 0].max() + padding)
    ax.set_ylim(positions[:, 1].min() - padding, positions[:, 1].max() + padding)
    ax.set_zlim(positions[:, 2].min() - padding, positions[:, 2].max() + padding)

    import textwrap
    wrapped_caption = '\n'.join(textwrap.wrap(title, width=60))
    smooth_tag = f"σ={smooth_sigma}" if smooth else "no-smooth"
    # Use suptitle for the (potentially long) caption so it never clips against
    # the 3D axes bounding box; reserve ax.set_title for the concise technical tag.
    fig.suptitle(wrapped_caption, fontsize=10, y=0.98, wrap=True)
    ax.set_title(f"[vel-integrated · {smooth_tag}]", fontsize=9, pad=8)
    fig.subplots_adjust(top=0.88)
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Depth (Forward)')
    ax.set_zlabel('Y (Up)')

    trajectory_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.6, label='Full Path')
    trail_line, = ax.plot([], [], [], 'orange', linewidth=3, alpha=0.8, label='Trail')
    current_point = ax.scatter([], [], [], c='red', s=200, label='Current')

    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2],
               c='green', s=150, label='Start', marker='^')
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2],
               c='red', s=150, label='End', marker='v')
    ax.legend()

    # Pre-compute orientation vectors
    orientation_vectors = []
    for frame_idx in range(len(positions)):
        ori = orientations[frame_idx]
        if unified_data.format_type == CameraDataFormat.FULL_12_ROTMAT:
            from tkcam.utils.camera_geometry import forward_from_sixd
            fwd_gl = forward_from_sixd(ori.reshape(1, 6)).squeeze(0)
        else:
            pitch, yaw = ori[0], ori[1]
            fwd_gl = np.array([
                np.cos(pitch) * np.sin(yaw),
                -np.sin(pitch),
                -np.cos(pitch) * np.cos(yaw),
            ])
        fwd_mpl = to_mpl(fwd_gl.reshape(1, 3)).squeeze(0)
        norm = np.linalg.norm(fwd_mpl)
        if norm > 1e-6:
            fwd_mpl = fwd_mpl / norm
        orientation_vectors.append(tuple(fwd_mpl))

    orientation_arrow = [None]

    def animate(frame):
        if orientation_arrow[0] is not None:
            orientation_arrow[0].remove()

        trajectory_line.set_data_3d(
            positions[:frame + 1, 0],
            positions[:frame + 1, 1],
            positions[:frame + 1, 2],
        )
        trajectory_line.set_alpha(min(1.0, frame / 20) * 0.6)

        if show_trail and frame > 0:
            ts = max(0, frame - trail_length)
            seg = positions[ts:frame + 1]
            trail_line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])

        cp = positions[frame]
        current_point._offsets3d = ([cp[0]], [cp[1]], [cp[2]])

        dx, dy, dz = orientation_vectors[frame]
        orientation_arrow[0] = ax.quiver(
            cp[0], cp[1], cp[2], dx, dy, dz,
            length=1.5 * base_arrow_length,
            color='purple', alpha=0.8, arrow_length_ratio=0.3, linewidth=2,
        )

        if rotate_view:
            ax.view_init(elev=20, azim=frame * 0.5 % 360)

        return trajectory_line, trail_line, current_point, orientation_arrow[0]

    frame_indices = list(range(0, len(positions), stride))
    if frame_indices[-1] != len(positions) - 1:
        frame_indices.append(len(positions) - 1)

    anim = FuncAnimation(fig, animate, frames=frame_indices,
                         interval=1000 / fps, blit=False, repeat=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_path.endswith('.gif'):
        anim.save(save_path, writer=PillowWriter(fps=fps), dpi=100)
    elif save_path.endswith('.mp4'):
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path, writer=writer, dpi=100)
    else:
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='CamTraj'), bitrate=1800)
        anim.save(save_path + '.mp4', writer=writer, dpi=100)

    del anim
    plt.close(fig)
    print(f"Velocity-integrated trajectory animation saved to {save_path}")


if __name__ == '__main__':
    parser = EvalT2MOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    torch.autograd.set_detect_anomaly(True)

    # Get dataset configuration with automatic format detection
    dataset_config = get_unified_dataset_config(opt)
    dim_pose = dataset_config['dim_pose']
    detected_format = dataset_config.get('detected_format', 'Unknown')
    
    # Get format type for correct visualization
    from tkcam.utils.unified_data_format import detect_format_from_dataset_name
    viz_format_type = detect_format_from_dataset_name(opt.dataset_name)
    
    print(f"Dataset: {opt.dataset_name}")
    print(f"Detected camera format: {detected_format}")
    print(f"Feature dimensions: {dim_pose}")
    print(f"Visualization format: {viz_format_type}")
    print("Vel-integration visualization: ENABLED (always using velocity-integrated plotting)")

    root_dir = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    result_dir = pjoin('./generation', opt.ext)
    joints_dir = pjoin(result_dir, 'joints')
    animation_dir = pjoin(result_dir, 'animations')
    os.makedirs(joints_dir, exist_ok=True)
    os.makedirs(animation_dir, exist_ok=True)

    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=opt.device)

    #######################
    ######Loading RVQ######
    #######################
    vq_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=opt.device)
    # get_opt() already sets dim_pose correctly from its internal dataset defaults table.
    # Do NOT override with auto-detected dim_pose — auto-detection may fail when data
    # directory doesn't exist (e.g., running inference on a different machine),
    # causing it to fall back to 6 even for 12D datasets like realestate10k_rotmat.
    print(f'VQ model dim_pose = {vq_opt.dim_pose}')
    vq_model, vq_opt = load_vq_model(vq_opt)

    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    #################################
    ######Loading R-Transformer######
    #################################
    res_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.res_name, 'opt.txt')
    res_opt = get_opt(res_opt_path, device=opt.device)

    text_cond_m = _resolve_text_conditioning(model_opt, opt, role='mask_transformer')
    text_cond_r = _resolve_text_conditioning(res_opt, opt, role='res_transformer')
    if text_cond_m[0] != text_cond_r[0]:
        print(
            f'[gen_camera] WARNING: mask conditioning_mode={text_cond_m[0]!r} != '
            f'res {text_cond_r[0]!r} — encoders may disagree.',
        )

    res_model = load_res_model(res_opt, vq_opt, opt, text_cond=text_cond_r)

    assert res_opt.vq_name == model_opt.vq_name

    #################################
    ######Loading M-Transformer######
    #################################
    which_ckpt = getattr(opt, 'which_epoch', 'net_best_fid')
    ckpt_name = which_ckpt if which_ckpt.endswith('.tar') else f'{which_ckpt}.tar'
    t2m_transformer = load_trans_model(model_opt, opt, ckpt_name, text_cond=text_cond_m)

    t2m_transformer.eval()
    vq_model.eval()
    res_model.eval()


    res_model.to(opt.device)
    t2m_transformer.to(opt.device)
    vq_model.to(opt.device)

    ##### ---- Dataloader ---- #####
    opt.nb_joints = 1  # Camera has no joints

    mean = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'mean.npy'))
    std = np.load(pjoin(opt.checkpoints_dir, opt.dataset_name, model_opt.vq_name, 'meta', 'std.npy'))
    def inv_transform(data):
        return data * std + mean

    prompt_list = []
    length_list = []
    keyframe_dirs_per_prompt = []
    # Must match MaskTransformer (from checkpoint opt.txt via text_cond_m).
    conditioning_mode = text_cond_m[0]

    # ── id_embedding mode: conditions are sample IDs, not text ──
    if conditioning_mode == 'id_embedding':
        sample_ids_str = getattr(opt, 'sample_ids', '')
        num_id_samples = text_cond_m[1]
        
        if sample_ids_str:
            sample_id_list = [int(x.strip()) for x in sample_ids_str.split(',')]
        else:
            # Generate all IDs
            sample_id_list = list(range(num_id_samples))
        
        # Validate IDs
        for sid in sample_id_list:
            if sid < 0 or sid >= num_id_samples:
                raise ValueError(f"Sample ID {sid} out of range [0, {num_id_samples})")
        
        # Create caption labels for display
        captions = [f"Sample ID: {sid}" for sid in sample_id_list]
        # Create condition tensor (LongTensor of sample IDs)
        cond_ids = torch.LongTensor(sample_id_list).to(opt.device)
        
        # Set lengths 
        if opt.motion_length > 0:
            token_lens = torch.LongTensor([opt.motion_length // 4] * len(sample_id_list)).to(opt.device)
        else:
            default_camera_length = getattr(opt, 'default_camera_length', 200)
            print(f"id_embedding mode: Using FIXED length of {default_camera_length} frames")
            token_lens = torch.LongTensor([default_camera_length // 4] * len(sample_id_list)).to(opt.device)
        
        m_length = token_lens * 4
        print(f"id_embedding mode: generating {len(sample_id_list)} samples (IDs: {sample_id_list})")
        
        # Load ground truth data and text descriptions for comparison
        gt_data_list = []
        gt_text_descriptions = []  # Actual text prompts for each sample
        train_split_file = pjoin(opt.data_root, 'train.txt')
        if os.path.exists(train_split_file):
            with open(train_split_file, 'r') as f:
                all_sample_names = [line.strip() for line in f.readlines()]
            
            motion_dir = traj_dir(opt.data_root)
            texts_dir = pjoin(opt.data_root, 'texts')
            for sid in sample_id_list:
                if sid < len(all_sample_names):
                    sample_name = all_sample_names[sid]
                    gt_path = pjoin(motion_dir, f"{sample_name}.npy")
                    if os.path.exists(gt_path):
                        gt_motion = np.load(gt_path)
                        gt_data_list.append(gt_motion)
                    else:
                        gt_data_list.append(None)
                        print(f"  Warning: GT not found for sample ID {sid} at {gt_path}")
                    
                    # Load text description
                    text_path = pjoin(texts_dir, f"{sample_name}.txt")
                    if os.path.exists(text_path):
                        with open(text_path, 'r') as tf:
                            raw_text = tf.readline().strip()
                            # Text format: "description#POS-tagged version" — take first part
                            gt_text_descriptions.append(raw_text.split('#')[0].strip())
                    else:
                        gt_text_descriptions.append(f"Sample ID: {sid}")
                else:
                    gt_data_list.append(None)
                    gt_text_descriptions.append(f"Sample ID: {sid}")
            print(f"  Loaded {sum(1 for g in gt_data_list if g is not None)}/{len(sample_id_list)} ground truth trajectories")
            print(f"  Loaded {sum(1 for t in gt_text_descriptions if not t.startswith('Sample ID'))} text descriptions")
        else:
            print(f"  Warning: train.txt not found at {train_split_file}, GT comparison disabled")
            gt_data_list = [None] * len(sample_id_list)
            gt_text_descriptions = [f"Sample ID: {sid}" for sid in sample_id_list]

    # ── text-based modes (clip / t5) ──
    else:
        est_length = False
        keyframe_dirs_per_prompt = []
        frame_root = Path(getattr(
            opt, 'frame_dir', '/data4/haozhe/CamTraj/data/processed_estate/train_frames'))
        scene_id_mapping = {}
        mapping_path = Path(opt.data_root) / 'scene_id_mapping.json'
        if mapping_path.exists():
            try:
                with open(mapping_path, 'r') as mf:
                    scene_id_mapping = json.load(mf)
            except Exception as e:
                print(f"Warning: failed to load scene_id_mapping at {mapping_path}: {e}")

        if opt.text_prompt != "":
            prompt_list.append(opt.text_prompt)
            keyframe_dirs_per_prompt.append(None)
            if opt.motion_length == 0:
                est_length = True
            else:
                length_list.append(opt.motion_length)
        elif opt.text_path != "":
            with open(opt.text_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    infos = line.split('#', 2)
                    prompt_list.append(infos[0])
                    if len(infos) >= 2 and infos[1].strip().isdigit():
                        length_list.append(int(infos[1].strip()))
                    else:
                        est_length = True
                        length_list = []
                    if len(infos) >= 3 and infos[2].strip():
                        frame_source = infos[2].strip()
                        resolved_dir = _resolve_frame_source_to_keyframe_dir(
                            frame_source=frame_source,
                            frame_root=frame_root,
                            scene_id_mapping=scene_id_mapping,
                        )
                        if resolved_dir is None:
                            print(
                                f"Warning: cannot resolve frame_source '{frame_source}' "
                                f"to a keyframe directory. This prompt will use no keyframes."
                            )
                        keyframe_dirs_per_prompt.append(resolved_dir)
                    else:
                        keyframe_dirs_per_prompt.append(None)
        else:
            raise ValueError("A text prompt, or a file of text prompts are required!!!")

        if est_length:
            default_camera_length = getattr(opt, 'default_camera_length', 200)
            print(f"Camera dataset: Using FIXED length of {default_camera_length} frames (~{default_camera_length/30:.1f}s)")
            print(f"  (Length estimator disabled - not trained on camera data)")
            token_lens = torch.LongTensor([default_camera_length // 4] * len(prompt_list))
            token_lens = token_lens.to(opt.device).long()
        else:
            token_lens = torch.LongTensor(length_list) // 4
            token_lens = token_lens.to(opt.device).long()

        m_length = token_lens * 4
        captions = prompt_list
        cond_ids = None  # Not used in text modes
        gt_data_list = None  # No GT comparison for text modes
        gt_text_descriptions = None
    
    # ── Visual conditioning (CLIP-based) ─────────────────────────────────────
    # Prepare once before the repeat loop; tensors are reused across repeats.
    vis_first_frame   = None   # (B, 3, 224, 224)  or None
    vis_sparse_frames = None   # (B, K, 3, 224, 224) or None
    vis_indices       = None   # (B, K) long  — VQ-level
    vis_valid_mask    = None   # (B, K) bool

    _model_has_visual = (
        getattr(model_opt, 'use_first_frame', False)
        or getattr(model_opt, 'use_sparse_frames', False)
    )

    _has_per_prompt_dirs = any(d is not None for d in keyframe_dirs_per_prompt)
    _has_global_dir = bool(opt.keyframe_dir)

    # --use_keyframes is the single runtime gate.
    # If disabled, ignore both per-prompt frame_source and global keyframe_dir.
    if _model_has_visual and opt.use_keyframes and (_has_per_prompt_dirs or _has_global_dir):
        print(f"\n{'='*70}")
        print("Visual keyframe conditioning enabled for inference")
        print(f"{'='*70}")

        if _has_per_prompt_dirs:
            # Per-sample mode: fill in global fallback for None entries
            fallback = opt.keyframe_dir if opt.keyframe_dir else None
            resolved_dirs = [d if d is not None else fallback for d in keyframe_dirs_per_prompt]
            n_with_kf = sum(1 for d in resolved_dirs if d is not None)
            print(f"  Per-prompt keyframe dirs: {n_with_kf}/{len(resolved_dirs)} prompts have keyframes")
            (vis_first_frame,
             vis_sparse_frames,
             vis_indices,
             vis_valid_mask) = prepare_visual_conditioning_per_sample(
                keyframe_dirs=resolved_dirs,
                token_seq_lens=token_lens,
                model_opt=model_opt,
                device=opt.device,
            )
        else:
            # Single-dir broadcast mode (original behavior)
            batch_size_inf = len(captions) if conditioning_mode != 'id_embedding' \
                             else len(cond_ids)
            (vis_first_frame,
             vis_sparse_frames,
             vis_indices,
             vis_valid_mask) = prepare_visual_conditioning_for_inference(
                keyframe_dir=opt.keyframe_dir,
                keyframe_indices_str=opt.keyframe_indices or '',
                model_opt=model_opt,
                batch_size=batch_size_inf,
                token_seq_len=token_lens[0].item(),
                device=opt.device,
            )
        print(f"{'='*70}\n")
    elif (opt.use_keyframes and (_has_global_dir or _has_per_prompt_dirs)) and not _model_has_visual:
        print("Warning: keyframes provided but model has no visual conditioning. "
              "Keyframe dirs will be ignored.")
    elif (not opt.use_keyframes) and (_has_global_dir or _has_per_prompt_dirs):
        print("Keyframe inputs detected but --use_keyframes is OFF. Ignoring all keyframes.")

    debug_trace_enabled = os.environ.get("GEN_CAMERA_DEBUG_TRACE", "0") == "1"
    debug_trace_done = False

    # Process samples in mini-batches matching the batch size used during training
    # evaluation.  Sending all prompts at once (e.g. 251) causes T5/attention to
    # degrade — producing near-identical degenerate outputs for every prompt.
    gen_batch_size = getattr(opt, 'batch_size', 32)
    n_samples = len(captions)
    print(f"[gen_camera] Effective generation batch_size={gen_batch_size}, n_samples={n_samples}")

    for r in range(opt.repeat_times):
        print("-->Repeat %d"%r)
        all_data_batches = []

        for batch_start in range(0, n_samples, gen_batch_size):
            batch_end = min(batch_start + gen_batch_size, n_samples)
            print(f"  Generating samples {batch_start}–{batch_end - 1} / {n_samples - 1}")

            b_token_lens = token_lens[batch_start:batch_end]

            if conditioning_mode == 'id_embedding':
                b_gen_conds = cond_ids[batch_start:batch_end]
            else:
                b_gen_conds = captions[batch_start:batch_end]

            # Slice visual conditioning tensors if present
            b_vis_first  = vis_first_frame[batch_start:batch_end]   if vis_first_frame   is not None else None
            b_vis_sparse = vis_sparse_frames[batch_start:batch_end] if vis_sparse_frames is not None else None
            b_vis_idx    = vis_indices[batch_start:batch_end]       if vis_indices       is not None else None
            b_vis_mask   = vis_valid_mask[batch_start:batch_end]    if vis_valid_mask    is not None else None

            if debug_trace_enabled and not debug_trace_done:
                try:
                    debug_trace_single_sample(
                        t2m_transformer=t2m_transformer,
                        conds=b_gen_conds,
                        token_lens=b_token_lens,
                        cond_scale=opt.cond_scale,
                        norm_mean=mean,
                        norm_std=std,
                        vis_first_frame=b_vis_first,
                        vis_sparse_frames=b_vis_sparse,
                        vis_indices=b_vis_idx,
                        vis_valid_mask=b_vis_mask,
                    )
                    debug_trace_done = True
                except Exception as _e:
                    print(f"[Debug Trace] failed: {_e}")

            with torch.no_grad():
                mids = t2m_transformer.generate(
                    b_gen_conds, b_token_lens,
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
                mids = res_model.generate(
                    mids, b_gen_conds, b_token_lens,
                    temperature=1, cond_scale=getattr(opt, 'res_cond_scale', 5),
                    first_frame_pixels=b_vis_first,
                    sparse_frames=b_vis_sparse,
                    visual_indices=b_vis_idx,
                    visual_valid_mask=b_vis_mask,
                )
                pred_motions = vq_model.forward_decoder(mids)
                pred_motions = pred_motions.detach().cpu().numpy()
                all_data_batches.append(inv_transform(pred_motions))

        # Merge all mini-batch outputs; each entry is (B, T, D) — pad to common T
        max_t = max(x.shape[1] for x in all_data_batches)
        padded = []
        for x in all_data_batches:
            if x.shape[1] < max_t:
                pad = np.zeros((x.shape[0], max_t - x.shape[1], x.shape[2]), dtype=x.dtype)
                x = np.concatenate([x, pad], axis=1)
            padded.append(x)
        data = np.concatenate(padded, axis=0)

        for k, (caption, joint_data) in enumerate(zip(captions, data)):
            print("---->Sample %d: %s %d"%(k, caption, m_length[k]))
            animation_path = pjoin(animation_dir, str(k))
            joint_path = pjoin(joints_dir, str(k))

            os.makedirs(animation_path, exist_ok=True)
            os.makedirs(joint_path, exist_ok=True)

            joint_data = joint_data[:m_length[k]]
            
            # Save raw camera data
            np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_pred.npy"%(k, r, m_length[k])), joint_data)
            
            # Always use velocity-integrated visualization (no direct xyz plotting).
            pred_velint_path = pjoin(animation_path, "sample%d_repeat%d_len%d_pred_velint.mp4"%(k, r, m_length[k]))
            try:
                plot_camera_trajectory_animation_vel_integrated(
                    joint_data, pred_velint_path, title=f"[Pred / Vel-Int] {caption}",
                    fps=30, show_trail=True, trail_length=20,
                    format_type=viz_format_type, smooth=True, smooth_sigma=1.5,
                )
                print(f"Pred vel-integrated animation saved to {pred_velint_path}")
            except Exception as e:
                import traceback
                print(f"[Warning] Vel-integrated pred animation failed: {e}")
                traceback.print_exc()

            # ── Ground truth comparison for id_embedding mode ──
            if conditioning_mode == 'id_embedding' and gt_data_list is not None and gt_data_list[k] is not None:
                gt_raw = gt_data_list[k]  # Already in raw (unnormalized) feature space
                gt_len = min(len(gt_raw), m_length[k].item())
                gt_trimmed = gt_raw[:gt_len]
                
                # Save GT raw data
                np.save(pjoin(joint_path, "sample%d_repeat%d_len%d_gt.npy"%(k, r, gt_len)), gt_trimmed)
                
                # Always use velocity-integrated GT visualization + GT-vs-Pred comparison.
                gt_velint_path = pjoin(animation_path, "sample%d_repeat%d_len%d_gt_velint.mp4"%(k, r, gt_len))
                try:
                    plot_camera_trajectory_animation_vel_integrated(
                        gt_trimmed, gt_velint_path, title=f"[GT / Vel-Int] {caption}",
                        fps=30, show_trail=True, trail_length=20,
                        format_type=viz_format_type, smooth=True, smooth_sigma=1.5,
                    )
                    print(f"  GT vel-integrated animation saved to {gt_velint_path}")
                except Exception as e:
                    import traceback
                    print(f"[Warning] Vel-integrated GT animation failed: {e}")
                    traceback.print_exc()

                # Side-by-side comparison (GT vel-int vs Pred vel-int)
                try:
                    from tkcam.utils.clatr_camera_eval import plot_trajectory_comparison_animation_vel_integrated
                    cmp_velint_path = pjoin(animation_path, "sample%d_repeat%d_len%d_velint_cmp.mp4"%(k, r, m_length[k]))
                    plot_trajectory_comparison_animation_vel_integrated(
                        gt_trimmed, joint_data, caption,
                        cmp_velint_path,
                        fps=20, stride=2,
                        format_type=viz_format_type,
                    )
                    print(f"  GT vs Pred vel-integrated comparison saved to {cmp_velint_path}")
                except Exception as e:
                    import traceback
                    print(f"[Warning] Vel-integrated comparison animation failed: {e}")
                    traceback.print_exc()

                print(f"  GT trajectory saved ({gt_len} frames) with vel-integrated views")
            
            # Save camera data as text file for easy inspection with format-aware headers
            unified_data = UnifiedCameraData(joint_data, format_type=viz_format_type)
            positions = unified_data.positions.numpy()
            orientations = unified_data.orientations.numpy()
            
            # Create format-appropriate header
            if unified_data.format_type == CameraDataFormat.LEGACY_5:
                header = 'x y z pitch yaw'
            elif unified_data.format_type == CameraDataFormat.POSITION_ORIENTATION_6:
                header = 'x y z pitch yaw roll'
            else:  # FULL_12
                header = 'x y z pitch yaw roll'  # Only save position + orientation for readability
            
            camera_data = np.column_stack([positions, orientations])
            np.savetxt(pjoin(joint_path, "sample%d_repeat%d_len%d_pred.txt"%(k, r, m_length[k])), 
                      camera_data, fmt='%.6f', 
                      header=header, comments='')
            
            # Also save format information
            format_info_path = pjoin(joint_path, "sample%d_repeat%d_len%d_pred_format.txt"%(k, r, m_length[k]))
            with open(format_info_path, 'w') as f:
                f.write(f"Original format: {unified_data.format_type.name}\n")
                f.write(f"Original dimensions: {unified_data.num_features}\n")
                f.write(f"Raw data shape: {joint_data.shape}\n")
                f.write(f"Position shape: {positions.shape}\n")
                f.write(f"Orientation shape: {orientations.shape}\n")
                f.write(f"Caption: {caption}\n")

            print(f"Pred vel-integrated animation saved to {pred_velint_path}")
            print(f"Raw data saved to {pjoin(joint_path, 'sample%d_repeat%d_len%d.npy'%(k, r, m_length[k]))}")

            # Force cyclic GC after each sample — matplotlib FuncAnimation objects
            # create reference cycles (fig ↔ closure ↔ event handlers) that
            # CPython's reference-counting GC alone cannot break promptly.
            import gc
            gc.collect() 


"""
Usage (module CLI; for everyday inference prefer scripts/generate.py, which takes durations in
seconds, a first-frame image, or keyframes at given times):

python -m tkcam.gen_camera --dataset_name realestate10k_rotmat \
    --name mtrans_25kfixed_noframe_r4 --res_name rtrans_25kfixed_noframe_r4 --conditioning_mode t5 \
    --text_prompt "The camera dollies forward then arcs left around the living room." \
    --motion_length 180 --gpu_id 0 --ext demo

--name / --res_name select the mask / residual transformer runs under
checkpoints/{dataset_name}/; --motion_length is in frames (default 200, i.e. ~6.7 s at 30 fps).
Visual conditioning (first-frame or sparse-keyframe checkpoints) needs --use_keyframes plus
--keyframe_dir <dir with jpg/png frames> and, for sparse keyframes, --keyframe_indices "0,60,120"
(frame indices, matched to the images in sorted order).
--text_path <file> runs one prompt per line in the form "text#length#frame_source", where length is
in frames and frame_source is an optional frame directory (or a scene id, looked up under
./data/processed_estate/train_frames).
Outputs go to generation/<ext>/: joints/<k>/*.npy (12D trajectories) and animations/<k>/*.mp4.
"""
