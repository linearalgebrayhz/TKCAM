"""
vis_attention_maps.py

Diagnostic tool: extract and visualize cross-attention weights from the last
2 layers of the MaskTransformer to detect modality collapse.

What it reveals
---------------
- If the model attends *only* to the IMG token  → modality collapse (ignores text)
- If attention is distributed across text tokens → healthy text conditioning
- The printed "Attention Ratio" gives a numerical modality-bias metric

Key axis
    [IMG]  [token_0]  [token_1]  ...  [token_T]    (keys: visual + T5 text)
Query axis
    Frame 0 .. Frame S-1                            (queries: motion tokens)

Usage examples
--------------
# Pure text attention (no visual token)
python vis_attention_maps.py \
    --dataset_name realestate10k_rotmat \
    --name mtrans_5k_newdata_1frame2 \
    --text_prompt "the camera slowly pans left" \
    --conditioning_mode t5 \
    --gpu_id 0

# With a first-frame image (adds IMG key token)
python vis_attention_maps.py \
    --dataset_name realestate10k_rotmat \
    --name mtrans_5k_newdata_1frame2 \
    --text_prompt "the camera slowly pans left" \
    --conditioning_mode t5 \
    --first_frame_path /path/to/frame.jpg \
    --gpu_id 0

# Batch from a text file (one prompt per line)
python vis_attention_maps.py \
    --dataset_name realestate10k_rotmat \
    --name mtrans_5k_newdata_1frame2 \
    --text_path camera_prompts.txt \
    --conditioning_mode t5 \
    --gpu_id 0
"""

import os
import sys
import argparse
import textwrap
from os.path import join as pjoin
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from PIL import Image

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tkcam.models.mask_transformer.transformer import MaskTransformer
from tkcam.utils.get_opt import get_opt

clip_version = 'ViT-B/32'


# ─────────────────────── Model loading ───────────────────────────────────────

def load_trans_model(model_opt, opt, which_model: str) -> MaskTransformer:
    """Load MaskTransformer from checkpoint (mirrors gen_camera.load_trans_model)."""
    conditioning_mode = getattr(opt, 'conditioning_mode', 'clip')
    t2m_transformer = MaskTransformer(
        code_dim=model_opt.code_dim,
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
        num_id_samples=getattr(opt, 'num_id_samples', 50),
        t5_model_name=getattr(opt, 't5_model_name', 't5-base'),
        # ── Visual conditioning flags — must match the training config ───────
        use_first_frame=getattr(model_opt, 'use_first_frame', False),
        use_sparse_frames=getattr(model_opt, 'use_sparse_frames', False),
        max_sparse_frames=getattr(model_opt, 'max_sparse_frames', 4),
        visual_drop_prob=getattr(model_opt, 'visual_drop_prob', 0.0),
        opt=model_opt,
    )
    ckpt_path = pjoin(
        model_opt.checkpoints_dir, model_opt.dataset_name,
        model_opt.name, 'model', which_model,
    )
    print(f'Loading MaskTransformer from: {ckpt_path}')
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_key = 't2m_transformer' if 't2m_transformer' in ckpt else 'trans'
    missing, unexpected = t2m_transformer.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected) == 0, f'Unexpected keys: {unexpected}'
    # Missing keys are expected for frozen encoders (CLIP / T5) that are
    # re-initialised from their own pretrained weights, not stored in the ckpt.
    allowed_missing_prefixes = ('clip_model.', 'cond_provider.',
                                'clip_image_encoder.')
    truly_missing = [k for k in missing
                     if not any(k.startswith(p) for p in allowed_missing_prefixes)]
    assert len(truly_missing) == 0, f'Missing trainable keys: {truly_missing}'
    print(f'Loaded from epoch {ckpt.get("ep", "?")}')
    return t2m_transformer


# ─────────────────────── Token-label helpers ─────────────────────────────────

def _clean_t5_token(tok: str) -> str:
    """Strip SentencePiece artefacts and shorten special tokens."""
    tok = tok.replace('\u2581', '')   # remove ▁ prefix
    if tok in ('</s>', '<pad>', '<unk>'):
        return tok
    return tok if tok else '?'


def get_t5_token_labels(text: str, tokenizer, max_length: int = 64):
    """Return list of decoded T5 token strings for *text* (incl. </s>)."""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors='pt',
    )
    ids = enc.input_ids[0].tolist()
    raw_tokens = tokenizer.convert_ids_to_tokens(ids)
    return [_clean_t5_token(t) for t in raw_tokens]


def build_key_labels(
    text: str,
    model: MaskTransformer,
    num_visual_tokens: int,
    T_cond: int,
    tokenizer=None,
    visual_labels: list = None,
) -> list:
    """
    Build key-axis labels for the attention heatmap.

    Parameters
    ----------
    text : str
        The text prompt used for this sample.
    model : MaskTransformer
        The loaded transformer (needed to detect conditioning mode).
    num_visual_tokens : int
        Number of visual (image) tokens prepended to the key sequence.
    T_cond : int
        Total key length from the attention weight tensor.
    tokenizer : optional
        T5 tokenizer (if available) for pretty token labels.

    Returns
    -------
    labels : List[str] of length T_cond
    """
    labels: list = []

    # Visual token labels
    if num_visual_tokens == 1:
        labels.append('IMG')
    elif num_visual_tokens > 1:
        # If visual_labels supplied (e.g. short filenames), use them; else generic
        if visual_labels and len(visual_labels) >= num_visual_tokens:
            labels.extend(visual_labels[:num_visual_tokens])
        else:
            labels.extend([f'IMG_{i}' for i in range(num_visual_tokens)])

    # Text token labels
    num_text_tokens = T_cond - num_visual_tokens
    if tokenizer is not None:
        t5_labels = get_t5_token_labels(text, tokenizer)
        # Pad or truncate to match the actual tensor width
        if len(t5_labels) < num_text_tokens:
            t5_labels += ['<pad>'] * (num_text_tokens - len(t5_labels))
        else:
            t5_labels = t5_labels[:num_text_tokens]
        labels.extend(t5_labels)
    else:
        # Fallback: numeric indices
        labels.extend([str(i) for i in range(num_text_tokens)])

    return labels


# ─────────────────────── Image helpers ───────────────────────────────────────

def load_first_frame(image_path: str, device) -> torch.Tensor:
    """Load and preprocess a single image → (1, 3, 224, 224) float tensor."""
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])
    img = Image.open(image_path).convert('RGB')
    tensor = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    return tensor


def load_sparse_frames_from_dir(
    frame_dir: str,
    K: int,
    seq_len: int,
    count: int = 4,
    strategy: str = 'uniform',
    device=None,
) -> tuple:
    """Sample frames from a directory and build sparse-keyframe tensors.

    Parameters
    ----------
    frame_dir : str
        Directory containing jpg/png images, sorted alphabetically
        (which matches chronological order for standard naming like
        ``frame_000001_*.jpg``).
    K : int
        Model's ``max_sparse_frames`` — total key-sequence slots.
    seq_len : int
        VQ token sequence length (= motion_length // 4).
        Used to spread temporal indices across [0, seq_len).
    count : int
        How many frames to actually sample (≤ K).
    strategy : str
        ``'uniform'``   — evenly spaced across all files in the dir.
        ``'first'``     — the first *count* files.
        ``'endpoints'`` — first file + last file + *count-2* middle files.
    device : torch.device or None

    Returns
    -------
    frames_tensor      : (1, K, 3, 224, 224)
    visual_indices     : (1, K)  long  — VQ-level temporal positions
    visual_valid_mask  : (1, K)  bool  — True = valid slot
    """
    from torchvision import transforms

    # ── Collect and sort all image files ─────────────────────────────────────
    frame_dir = Path(frame_dir)
    all_files = sorted(
        [f for f in frame_dir.iterdir()
         if f.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp')]
    )
    if len(all_files) == 0:
        raise ValueError(f'No image files found in {frame_dir}')

    n_files = len(all_files)
    count = min(count, K, n_files)   # can't exceed K slots or available files

    # ── Select file indices according to strategy ─────────────────────────────
    if strategy == 'first':
        selected_file_indices = list(range(count))

    elif strategy == 'endpoints':
        if count <= 1:
            selected_file_indices = [0]
        elif count == 2:
            selected_file_indices = [0, n_files - 1]
        else:
            mid_count = count - 2
            mid_indices = [
                round(1 + i * (n_files - 2) / (mid_count + 1))
                for i in range(1, mid_count + 1)
            ]
            selected_file_indices = [0] + mid_indices + [n_files - 1]

    else:  # 'uniform' (default)
        if count == 1:
            selected_file_indices = [0]
        else:
            selected_file_indices = [
                round(i * (n_files - 1) / (count - 1))
                for i in range(count)
            ]

    selected_files = [all_files[i] for i in selected_file_indices]

    # ── Map file positions → VQ temporal indices ──────────────────────────────
    # Spread sampled frames uniformly over [0, seq_len) so the model
    # receives temporally meaningful positional embeddings.
    if count == 1:
        vq_positions = [0]
    else:
        vq_positions = [
            round(rank * (seq_len - 1) / (count - 1))
            for rank in range(count)
        ]

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f'  Sparse frames ({strategy}, {count}/{n_files} selected from {frame_dir}):')
    for rank, (fpath, vq_pos) in enumerate(zip(selected_files, vq_positions)):
        print(f'    slot {rank}: {fpath.name}  →  VQ token {vq_pos}')

    # ── Build tensors ─────────────────────────────────────────────────────────
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    frames      = torch.zeros(1, K, 3, 224, 224)
    vis_indices = torch.zeros(1, K, dtype=torch.long)
    valid_mask  = torch.zeros(1, K, dtype=torch.bool)

    for slot, (fpath, vq_pos) in enumerate(zip(selected_files, vq_positions)):
        img = Image.open(fpath).convert('RGB')
        frames[0, slot]      = preprocess(img)
        vis_indices[0, slot] = vq_pos
        valid_mask[0, slot]  = True

    if device is not None:
        frames      = frames.to(device)
        vis_indices = vis_indices.to(device)
        valid_mask  = valid_mask.to(device)

    return frames, vis_indices, valid_mask


# ─────────────────────── Core visualization ──────────────────────────────────

def compute_modality_ratio(
    attn: np.ndarray,
    num_visual_tokens: int,
) -> dict:
    """
    Compute the attention ratio between visual and text tokens.

    Parameters
    ----------
    attn : ndarray (S, T_cond)
        Attention weight map for a single sample, averaged over heads.
    num_visual_tokens : int
        How many leading key positions are visual (image) tokens.

    Returns
    -------
    dict with keys: 'img_ratio', 'text_ratio', 'img_sum', 'text_sum', 'total'
    """
    total = float(attn.sum())
    if total < 1e-9:
        return dict(img_ratio=0.0, text_ratio=0.0, img_sum=0.0, text_sum=0.0, total=total)

    img_sum = float(attn[:, :num_visual_tokens].sum()) if num_visual_tokens > 0 else 0.0
    text_sum = float(attn[:, num_visual_tokens:].sum())

    return dict(
        img_ratio=img_sum / total,
        text_ratio=text_sum / total,
        img_sum=img_sum,
        text_sum=text_sum,
        total=total,
    )


def _apply_yticks(ax, S: int, max_labels: int = 32):
    stride = max(1, S // max_labels)
    ticks = list(range(0, S, stride))
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(f) for f in ticks], fontsize=7)
    ax.set_ylabel('Motion Frame (Query)', fontsize=10)


def save_heatmap(
    attn: np.ndarray,
    key_labels: list,
    save_path: str,
    title: str,
    num_visual_tokens: int,
    max_query_labels: int = 32,
    colormap: str = 'magma',
):
    """
    Save TWO heatmap panels as a single PNG side-by-side:

    Left  — Full map (IMG + text) with global-max scale.
            Shows the true modality ratio; IMG will dominate.
    Right — Text-only zoom: strips the visual columns and rescales
            the colormap to the text region's own max.
            Reveals which text tokens are actually important.

    Parameters
    ----------
    attn : ndarray (S, T_cond)
    key_labels : list[str]
    save_path : str
    title : str
    num_visual_tokens : int
    max_query_labels : int
    colormap : str
    """
    S, T_cond = attn.shape
    has_visual = num_visual_tokens > 0
    text_attn  = attn[:, num_visual_tokens:]   # (S, T_text) — may be full if no visual
    T_text     = text_attn.shape[1]
    text_labels = key_labels[num_visual_tokens:]

    # ── figure layout ────────────────────────────────────────────────────────
    # Always produce two panels; if no visual tokens both show the same data
    # from different scale perspectives (global vs 99th-percentile clipped).
    n_panels = 2
    col_width_full = max(8, T_cond  * 0.30)
    col_width_text = max(6, T_text  * 0.30)
    fig_h          = max(5, S * 0.10)
    fig, axes = plt.subplots(
        1, n_panels,
        figsize=(col_width_full + col_width_text + 1.5, fig_h),
        gridspec_kw={'width_ratios': [T_cond, T_text]},
    )

    def _xticks(ax, labels, fontsize=8):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=fontsize)
        ax.set_xlabel('Conditioning Token (Key)', fontsize=9)

    def _separator(ax, pos):
        """Draw visual|text boundary line."""
        if 0 < pos < T_cond:
            ax.axvline(x=pos - 0.5, color='cyan', linewidth=1.5, linestyle='--',
                       label='visual | text')
            ax.legend(loc='upper right', fontsize=6)

    # ── Panel A: full map, global scale ──────────────────────────────────────
    ax_full = axes[0]
    vmax_full = attn.max() + 1e-9
    im_full = ax_full.imshow(attn, aspect='auto', cmap=colormap,
                             interpolation='nearest', vmin=0.0, vmax=vmax_full)
    plt.colorbar(im_full, ax=ax_full, fraction=0.03, pad=0.02,
                 label='Attention Weight')
    _xticks(ax_full, key_labels)
    _separator(ax_full, num_visual_tokens)
    _apply_yticks(ax_full, S, max_query_labels)
    ax_full.set_title(
        f'Full map  (global scale)\n'
        + (f'IMG={attn[:, :num_visual_tokens].mean():.3f}/frame  '
           if has_visual else '')
        + f'Text avg={text_attn.mean():.4f}/token',
        fontsize=8,
    )

    # ── Panel B: text-only, rescaled to text range ───────────────────────────
    ax_text = axes[1]
    # Clip at 99th percentile of text region to prevent one dominant token
    # from crushing the rest (same problem as the full map, just within text).
    vmax_text = float(np.percentile(text_attn, 99)) if text_attn.size > 0 else 1e-9
    vmax_text = max(vmax_text, 1e-9)
    im_text = ax_text.imshow(text_attn, aspect='auto', cmap=colormap,
                             interpolation='nearest', vmin=0.0, vmax=vmax_text)
    plt.colorbar(im_text, ax=ax_text, fraction=0.03, pad=0.02,
                 label='Attention Weight (text scale)')
    _xticks(ax_text, text_labels)
    _apply_yticks(ax_text, S, max_query_labels)
    ax_text.set_title(
        f'Text tokens only  (99th-pct scale, vmax={vmax_text:.4f})\n'
        f'Reveals relative importance within the text prompt',
        fontsize=8,
    )

    # ── suptitle ─────────────────────────────────────────────────────────────
    wrapped = '\n'.join(textwrap.wrap(title, width=100))
    fig.suptitle(wrapped, fontsize=8, y=1.01)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Heatmap saved → {save_path}')


def visualize_sample(
    sample_idx: int,
    text: str,
    attn_weights_list: list,   # list of (B, S, T_cond) tensors
    key_labels: list,
    num_visual_tokens: int,
    save_dir: str,
    colormap: str = 'magma',
):
    """
    Generate and save heatmaps + print modality ratio for one text prompt.

    Parameters
    ----------
    attn_weights_list : list of Tensor (B, S, T_cond)
        Captured from the last 2 cross-attn blocks (index 0 = earlier layer).
    """
    n_layers = len(attn_weights_list)
    print(f'\n{"─"*60}')
    print(f'Sample {sample_idx}: "{text}"')
    print(f'  Attention maps: {n_layers} layer(s), '
          f'seq_len={attn_weights_list[0].shape[1]}, '
          f'T_cond={attn_weights_list[0].shape[2]}')

    for layer_rel, w_tensor in enumerate(attn_weights_list):
        # w_tensor: (B, S, T_cond)  — already head-averaged
        layer_abs = -(n_layers - layer_rel)   # e.g. -2, -1
        layer_tag = f'layer_last{abs(layer_abs)}'

        # Extract this sample's weights (index into batch dim)
        attn = w_tensor[sample_idx].float().cpu().numpy()  # (S, T_cond)

        # ── Modality ratio ──────────────────────────────────────────────────
        ratio = compute_modality_ratio(attn, num_visual_tokens)
        print(f'\n  [{layer_tag}] Attention Ratio:')
        if num_visual_tokens > 0:
            print(f'    IMG  tokens  : {ratio["img_ratio"]*100:6.2f}%  '
                  f'(sum={ratio["img_sum"]:.4f})')
        print(f'    Text tokens  : {ratio["text_ratio"]*100:6.2f}%  '
              f'(sum={ratio["text_sum"]:.4f})')
        if num_visual_tokens > 0:
            if ratio['img_ratio'] > 0.8:
                print('    ⚠  MODALITY COLLAPSE detected — model over-attends to image!')
            elif ratio['img_ratio'] > 0.5:
                print('    ⚠  Borderline: model slightly prefers image over text')
            else:
                print('    ✓  Healthy: model attends to both modalities')

        # ── Save heatmap ────────────────────────────────────────────────────
        title = (
            f'Cross-Attention [{layer_tag}]  |  Sample {sample_idx}\n'
            f'Prompt: "{text}"\n'
            + (f'IMG ratio: {ratio["img_ratio"]*100:.1f}%  |  ' if num_visual_tokens > 0 else '')
            + f'Text ratio: {ratio["text_ratio"]*100:.1f}%'
        )
        save_path = pjoin(
            save_dir,
            f'sample{sample_idx:03d}_{layer_tag}.png',
        )
        save_heatmap(
            attn, key_labels, save_path, title,
            num_visual_tokens=num_visual_tokens,
            colormap=colormap,
        )

    print(f'{"─"*60}')


# ─────────────────────── Main entry point ────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Visualize MaskTransformer cross-attention weights.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ── Required ──────────────────────────────────────────────────────────────
    p.add_argument('--dataset_name', type=str, required=True,
                   help='Dataset name (e.g. realestate10k_rotmat)')
    p.add_argument('--name', type=str, required=True,
                   help='MaskTransformer checkpoint name (e.g. mtrans_5k_newdata_1frame2)')

    # ── Text input ────────────────────────────────────────────────────────────
    p.add_argument('--text_prompt', type=str, default='',
                   help='Single text prompt to visualize')
    p.add_argument('--text_path', type=str, default='',
                   help='Path to text file (one prompt per line, or prompt#length format)')

    # ── Visual conditioning (optional) ───────────────────────────────────────
    p.add_argument('--first_frame_path', type=str, default='',
                   help='Path to a single first-frame image (enables IMG key token). '
                        'Only used if the model was trained with use_first_frame.')
    # Sparse keyframe conditioning — provide a directory, not individual paths
    p.add_argument('--sparse_frame_dir', type=str, default='',
                   help='Directory containing frame images (jpg/png). '
                        'The script will sample --sparse_frame_count frames from it. '
                        'Only used if the model was trained with use_sparse_frames.')
    p.add_argument('--sparse_frame_count', type=int, default=0,
                   help='How many frames to sample from --sparse_frame_dir. '
                        'Defaults to the model\'s max_sparse_frames (usually 4).')
    p.add_argument('--sparse_frame_sample', type=str, default='uniform',
                   choices=['uniform', 'first', 'endpoints'],
                   help='Frame sampling strategy inside --sparse_frame_dir:\n'
                        '  uniform   — evenly spaced across the full sorted file list (default)\n'
                        '  first     — the first N files (chronological start)\n'
                        '  endpoints — first frame + last frame + N-2 evenly spaced middle frames')

    # ── Model & device ────────────────────────────────────────────────────────
    p.add_argument('--conditioning_mode', type=str, default='t5',
                   choices=['clip', 't5', 'id_embedding'])
    p.add_argument('--t5_model_name', type=str, default='t5-base')
    p.add_argument('--gpu_id', type=int, default=0)
    p.add_argument('--motion_length', type=int, default=200,
                   help='Trajectory length in raw frames (will be divided by 4 for token len).')
    p.add_argument('--which_epoch', type=str, default='latest.tar',
                   help='Checkpoint file to load (e.g. latest.tar, net_best_acc.tar).')

    # ── Output ────────────────────────────────────────────────────────────────
    p.add_argument('--save_dir', type=str, default='debug/attention_plots',
                   help='Output directory for PNG heatmaps.')
    p.add_argument('--colormap', type=str, default='magma',
                   help='Matplotlib colormap (e.g. magma, viridis, plasma).')
    p.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                   help='Root checkpoints directory.')
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    # ── Device ────────────────────────────────────────────────────────────────
    if args.gpu_id >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpu_id}')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    # ── Collect prompts ───────────────────────────────────────────────────────
    prompts: list[str] = []
    if args.text_prompt:
        prompts.append(args.text_prompt)
    elif args.text_path:
        with open(args.text_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append(line.split('#')[0].strip())
    else:
        raise ValueError('Provide --text_prompt or --text_path.')

    print(f'Prompts to visualize: {len(prompts)}')

    # ── Load model options ────────────────────────────────────────────────────
    root_dir = pjoin(args.checkpoints_dir, args.dataset_name, args.name)
    model_opt_path = pjoin(root_dir, 'opt.txt')
    model_opt = get_opt(model_opt_path, device=device)
    model_opt.device = device
    # Resolve VQ num_tokens from VQ opt
    vq_opt_path = pjoin(args.checkpoints_dir, args.dataset_name,
                        model_opt.vq_name, 'opt.txt')
    vq_opt = get_opt(vq_opt_path, device=device)
    model_opt.num_tokens = vq_opt.nb_code
    model_opt.num_quantizers = vq_opt.num_quantizers
    model_opt.code_dim = vq_opt.code_dim

    # ── Load transformer ──────────────────────────────────────────────────────
    # Override conditioning mode if explicitly provided
    args_conditioning_mode = args.conditioning_mode
    model_opt_cond_mode = getattr(model_opt, 'conditioning_mode',
                                  args_conditioning_mode)

    # Build a thin 'opt' namespace that matches what load_trans_model expects
    import types
    thin_opt = types.SimpleNamespace(
        conditioning_mode=model_opt_cond_mode,
        num_id_samples=getattr(model_opt, 'num_id_samples', 50),
        t5_model_name=getattr(model_opt, 't5_model_name', args.t5_model_name),
    )
    t2m_transformer = load_trans_model(model_opt, thin_opt, args.which_epoch)
    t2m_transformer.eval().to(device)

    # ── Determine token sequence length ───────────────────────────────────────
    seq_len = args.motion_length // 4  # VQ downsampling factor = 4
    print(f'Token sequence length: {seq_len}  (motion_length={args.motion_length})')

    # ── Identify conditioning mode used by the loaded model ──────────────────
    conditioning_mode = getattr(model_opt, 'conditioning_mode',
                                args.conditioning_mode)
    print(f'Conditioning mode: {conditioning_mode}')

    # ── Obtain tokenizer for key labels (T5 mode only) ────────────────────────
    tokenizer = None
    if conditioning_mode == 't5':
        if (hasattr(t2m_transformer, 'cond_provider') and
                hasattr(t2m_transformer.cond_provider, 't5_tokenizer')):
            tokenizer = t2m_transformer.cond_provider.t5_tokenizer
            print('T5 tokenizer loaded from model.')
        else:
            try:
                from transformers import T5Tokenizer
                t5_name = getattr(model_opt, 't5_model_name', args.t5_model_name)
                tokenizer = T5Tokenizer.from_pretrained(t5_name)
                print(f'T5 tokenizer loaded standalone: {t5_name}')
            except Exception as e:
                print(f'Warning: could not load T5 tokenizer ({e}). '
                      'Key labels will be numeric indices.')

    # ── Prepare visual conditioning (if requested and model supports it) ──────
    first_frame_pixels = None
    sparse_frames_tensor = None
    visual_indices_tensor = None
    visual_valid_mask_tensor = None
    num_visual_tokens = 0
    visual_key_labels: list = []    # short per-slot labels for the heatmap x-axis

    use_first_frame = getattr(model_opt, 'use_first_frame', False)
    use_sparse_frames = getattr(model_opt, 'use_sparse_frames', False)
    max_sparse_frames = getattr(model_opt, 'max_sparse_frames', 4)

    if use_first_frame and args.first_frame_path:
        print(f'Loading first-frame image: {args.first_frame_path}')
        first_frame_pixels = load_first_frame(args.first_frame_path, device)
        num_visual_tokens = 1
        visual_key_labels = ['IMG']
        print(f'  Visual tokens: 1 (use_first_frame)')

    elif use_sparse_frames and args.sparse_frame_dir:
        sparse_frames_tensor, visual_indices_tensor, visual_valid_mask_tensor = \
            load_sparse_frames_from_dir(
                frame_dir=args.sparse_frame_dir,
                K=max_sparse_frames,
                seq_len=seq_len,
                count=args.sparse_frame_count or max_sparse_frames,
                strategy=args.sparse_frame_sample,
                device=device,
            )
        num_visual_tokens = max_sparse_frames   # K key positions in cond_seq
        n_valid = int(visual_valid_mask_tensor.sum().item())
        # Build short labels: "f001", "f025", ... from VQ temporal positions
        visual_key_labels = [
            f'f{int(visual_indices_tensor[0, s]):03d}'
            if visual_valid_mask_tensor[0, s] else 'pad'
            for s in range(max_sparse_frames)
        ]
        print(f'  Visual tokens: {max_sparse_frames} sparse slots '
              f'({n_valid} valid)')

    else:
        if (use_first_frame or use_sparse_frames):
            print('Note: model supports visual conditioning but no image paths '
                  'were provided — running text-only visualization.')

    # ── Process each prompt ────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    print(f'\nSaving heatmaps to: {args.save_dir}\n')

    for sample_idx, text in enumerate(prompts):
        print(f'\n[{sample_idx+1}/{len(prompts)}] "{text}"')

        # Build conditioning tensors for batch_size=1
        bs = 1
        padding_mask = torch.zeros(bs, seq_len, dtype=torch.bool, device=device)

        # Encode condition
        cond_vector, _, cond_mask = t2m_transformer.encode_condition(
            [text], bs, device)

        # Build motion token input: all-masked (diagnostic — no specific tokens)
        motion_ids = torch.full(
            (bs, seq_len), t2m_transformer.mask_id,
            dtype=torch.long, device=device,
        )

        # Tile visual tensors to batch size
        ff_pixels = (first_frame_pixels.expand(bs, -1, -1, -1)
                     if first_frame_pixels is not None else None)
        sp_frames = (sparse_frames_tensor.expand(bs, -1, -1, -1, -1)
                     if sparse_frames_tensor is not None else None)
        sp_indices = (visual_indices_tensor.expand(bs, -1)
                      if visual_indices_tensor is not None else None)
        sp_valid = (visual_valid_mask_tensor.expand(bs, -1)
                    if visual_valid_mask_tensor is not None else None)

        # ── Forward pass capturing attention weights ───────────────────────
        result = t2m_transformer.trans_forward(
            motion_ids,
            cond_vector,
            padding_mask,
            force_mask=False,
            cond_mask=cond_mask,
            first_frame_pixels=ff_pixels,
            sparse_frames=sp_frames,
            visual_indices=sp_indices,
            visual_valid_mask=sp_valid,
            return_attn_weights=True,
        )
        # result is (logits, attn_weights_list)
        _, attn_weights_list = result

        if not attn_weights_list:
            print('  Warning: no attention weights captured. '
                  'Check that num_layers >= 2.')
            continue

        # ── Build key labels ───────────────────────────────────────────────
        T_cond = attn_weights_list[0].shape[2]
        key_labels = build_key_labels(
            text, t2m_transformer,
            num_visual_tokens=num_visual_tokens,
            T_cond=T_cond,
            tokenizer=tokenizer,
            visual_labels=visual_key_labels,
        )

        # ── Visualize ─────────────────────────────────────────────────────
        visualize_sample(
            sample_idx=0,          # within-batch index is always 0 (bs=1)
            text=text,
            attn_weights_list=attn_weights_list,
            key_labels=key_labels,
            num_visual_tokens=num_visual_tokens,
            save_dir=pjoin(args.save_dir, f'sample_{sample_idx:03d}'),
            colormap=args.colormap,
        )

    print(f'\nAll done. Heatmaps saved in: {args.save_dir}')


if __name__ == '__main__':
    main()
