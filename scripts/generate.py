#!/usr/bin/env python3
"""
Generate camera trajectories from a text prompt, optionally conditioned on a first frame
(firstframe) or on up to 4 keyframes placed at user-given times (xframe).

Examples:
    python scripts/generate.py --mode noframe --duration 6 \\
        --text "The camera dollies forward slowly, revealing the living room."
    python scripts/generate.py --mode firstframe --duration 5 --first_frame room.jpg \\
        --text "The camera pans left smoothly across the kitchen."
    python scripts/generate.py --mode xframe --duration 8 \\
        --keyframe hall.jpg@0 --keyframe stairs.jpg@4.5 \\
        --text "The camera moves forward through the hallway, then tilts up toward the stairs."

Duration is in seconds at 30 fps (the RealEstate10K frame rate). It is clamped to
[3, 11] s and rounded to a multiple of 4 frames (one VQ token); the training clips are
~4.0-9.3 s, so durations outside that range are extrapolation.

Outputs, per sample i, in --out_dir:
    sample{i}_9d.npy    (T, 9)    [r11, r21, r31, r12, r22, r32, x, y, z]   (CLaTr / benchmark layout)
    sample{i}_c2w.npy   (T, 4, 4) camera-to-world matrices
    sample{i}.mp4       trajectory animation (skip with --no_video)
    meta.json           prompt, mode, checkpoints, duration, keyframe times / frames / tokens
Trajectories are camera-to-world relative to the first frame (frame 0 is the identity up to the
model's prediction), OpenGL camera axes (+X right, +Y up, -Z forward); translation is reconstructed
from the predicted velocities, as in the benchmark.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from run_tkcam_benchmark_inference import (
    _clip_preprocess,
    _load_track_models,
    _resolve_checkpoints_dir,
    _to_clatr_with_velocity_integration,
)
from tkcam.gen_camera import plot_camera_trajectory_animation_vel_integrated
from tkcam.utils.camera_geometry import sixd_to_matrix
from tkcam.utils.fixseed import fixseed
from tkcam.utils.unified_data_format import CameraDataFormat, detect_format_from_dataset_name

FPS = 30
FRAMES_PER_TOKEN = 4
DURATION_CLAMP = (3.0, 11.0)        # seconds; requests outside are clamped
DURATION_TRAINED = (4.0, 9.3)       # seconds covered by the training clips (119-279 frames)
DEFAULT_MODELS = {                  # mode -> (mask transformer, residual transformer)
    'noframe': ('mtrans_25kfixed_noframe_r4', 'rtrans_25kfixed_noframe_r4'),
    'firstframe': ('mtrans_25kfixed_1frame_r4', 'rtrans_25kfixed_1frame_r4'),
    'xframe': ('mtrans_25kfixed_xframe_r4', 'rtrans_25kfixed_xframe_r4'),
}


def resolve_duration(seconds):
    """Clamp to DURATION_CLAMP and round to whole tokens. Returns (n_frames, n_tokens, warnings)."""
    warnings = []
    lo, hi = DURATION_CLAMP
    if not lo <= seconds <= hi:
        clamped = min(max(seconds, lo), hi)
        warnings.append(f'duration {seconds:g}s is outside [{lo:g}, {hi:g}]s; clamped to {clamped:g}s')
        seconds = clamped
    if not DURATION_TRAINED[0] <= seconds <= DURATION_TRAINED[1]:
        warnings.append(f'duration {seconds:g}s is outside the trained range '
                        f'{DURATION_TRAINED[0]:g}-{DURATION_TRAINED[1]:g}s; results may degrade')
    min_tokens = int(np.ceil(lo * FPS / FRAMES_PER_TOKEN))      # whole tokens inside the clamp range
    max_tokens = int(np.floor(hi * FPS / FRAMES_PER_TOKEN))
    n_tokens = int(np.floor(seconds * FPS / FRAMES_PER_TOKEN + 0.5))
    n_tokens = min(max(n_tokens, min_tokens), max_tokens)
    return n_tokens * FRAMES_PER_TOKEN, n_tokens, warnings


def parse_keyframes(specs, n_frames, max_keyframes):
    """'path@seconds' -> sorted list of dicts with path / time / frame / token."""
    if len(specs) > max_keyframes:
        raise ValueError(f'at most {max_keyframes} keyframes are supported, got {len(specs)}')
    keyframes = []
    for spec in specs:
        path, sep, t = spec.rpartition('@')
        if not sep or not path:
            raise ValueError(f"keyframe must be 'image_path@seconds', got {spec!r}")
        t = float(t)
        frame = int(round(t * FPS))
        if not 0 <= frame < n_frames:
            raise ValueError(f'keyframe time {t:g}s is outside the trajectory (0 - {n_frames / FPS:g}s)')
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        keyframes.append({'path': path, 'time': t, 'frame': frame, 'token': frame // FRAMES_PER_TOKEN})
    keyframes.sort(key=lambda k: k['frame'])
    tokens = [k['token'] for k in keyframes]
    if len(set(tokens)) != len(tokens):
        print('[generate] Warning: several keyframes fall on the same 4-frame token '
              f'({FRAMES_PER_TOKEN / FPS:.3f}s); they may conflict.')
    return keyframes


def nine_d_to_c2w(traj9):
    c2w = np.tile(np.eye(4, dtype=np.float32), (len(traj9), 1, 1))
    c2w[:, :3, :3] = sixd_to_matrix(traj9[:, :6])
    c2w[:, :3, 3] = traj9[:, 6:9]
    return c2w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=list(DEFAULT_MODELS), required=True)
    ap.add_argument('--text', required=True, help='camera-motion description')
    ap.add_argument('--duration', type=float, default=6.0, help='seconds (clamped to [3, 11])')
    ap.add_argument('--first_frame', help='firstframe mode: path to the first-frame image')
    ap.add_argument('--keyframe', action='append', default=[],
                    help="xframe mode: 'image_path@seconds' (repeat, up to 4)")
    ap.add_argument('--num_samples', type=int, default=1)
    ap.add_argument('--seed', type=int, default=10107)
    ap.add_argument('--out_dir', default='./generation/samples')
    ap.add_argument('--no_video', action='store_true')
    ap.add_argument('--checkpoints_dir', default=None, help='default: <repo>/checkpoints')
    ap.add_argument('--dataset_name', default='realestate10k_rotmat')
    ap.add_argument('--mtrans_name', default=None, help='override the mode default')
    ap.add_argument('--rtrans_name', default=None, help='override the mode default')
    ap.add_argument('--which_epoch', default='net_best_fid', help='mask transformer checkpoint file')
    ap.add_argument('--res_which_epoch', default='net_best_fid', help='residual transformer checkpoint file')
    ap.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    fixseed(args.seed)
    device = torch.device(args.device)
    n_frames, n_tokens, warnings = resolve_duration(args.duration)
    for w in warnings:
        print(f'[generate] Warning: {w}')

    if args.mode != 'firstframe' and args.first_frame:
        raise ValueError('--first_frame is only used with --mode firstframe')
    if args.mode != 'xframe' and args.keyframe:
        raise ValueError('--keyframe is only used with --mode xframe')
    if args.mode == 'firstframe' and not args.first_frame:
        raise ValueError('--mode firstframe needs --first_frame')
    if args.mode == 'xframe' and not args.keyframe:
        raise ValueError('--mode xframe needs at least one --keyframe image@seconds')

    checkpoints_dir = _resolve_checkpoints_dir(args.checkpoints_dir)
    mtrans_name = args.mtrans_name or DEFAULT_MODELS[args.mode][0]
    rtrans_name = args.rtrans_name or DEFAULT_MODELS[args.mode][1]
    m_opt, t2m_transformer, res_model, vq_model, mean, std = _load_track_models(
        checkpoints_dir=checkpoints_dir, dataset_name=args.dataset_name,
        mtrans_name=mtrans_name, rtrans_name=rtrans_name,
        which_epoch=args.which_epoch, res_which_epoch=args.res_which_epoch, device=device)
    expected = {'noframe': (False, False), 'firstframe': (True, False), 'xframe': (False, True)}[args.mode]
    found = (bool(getattr(m_opt, 'use_first_frame', False)), bool(getattr(m_opt, 'use_sparse_frames', False)))
    if found != expected:
        raise ValueError(f'{mtrans_name} is not a {args.mode} model (use_first_frame, use_sparse_frames = {found})')

    B = args.num_samples
    preprocess = _clip_preprocess()
    first_frame_pixels = sparse_frames = visual_indices = visual_valid_mask = None
    keyframes = []
    if args.mode == 'firstframe':
        pixels = preprocess(Image.open(args.first_frame).convert('RGB'))
        first_frame_pixels = pixels.unsqueeze(0).expand(B, -1, -1, -1).to(device)
    elif args.mode == 'xframe':
        K = int(getattr(m_opt, 'max_sparse_frames', 4))
        keyframes = parse_keyframes(args.keyframe, n_frames, K)
        frames = torch.zeros(K, 3, 224, 224)
        indices = torch.zeros(K, dtype=torch.long)
        valid = torch.zeros(K, dtype=torch.bool)
        for slot, kf in enumerate(keyframes):
            frames[slot] = preprocess(Image.open(kf['path']).convert('RGB'))
            indices[slot] = kf['token']
            valid[slot] = True
        sparse_frames = frames.unsqueeze(0).expand(B, -1, -1, -1, -1).to(device)
        visual_indices = indices.unsqueeze(0).expand(B, -1).to(device)
        visual_valid_mask = valid.unsqueeze(0).expand(B, -1).to(device)

    texts = [args.text] * B
    token_lens = torch.full((B,), n_tokens, dtype=torch.long, device=device)
    vis = dict(first_frame_pixels=first_frame_pixels, sparse_frames=sparse_frames,
               visual_indices=visual_indices, visual_valid_mask=visual_valid_mask)
    with torch.no_grad():   # sampling settings: the training-validation defaults, as in the benchmark
        mids = t2m_transformer.generate(
            texts, token_lens,
            timesteps=int(getattr(m_opt, 'eval_time_steps', 18)),
            cond_scale=float(getattr(m_opt, 'eval_mask_cond_scale', 3.0)),
            temperature=float(getattr(m_opt, 'eval_temperature', 1.0)),
            topk_filter_thres=float(getattr(m_opt, 'eval_topkr', 0.9)),
            gsample=bool(getattr(m_opt, 'gumbel_sample', False)), **vis)
        ids = res_model.generate(mids, texts, token_lens, temperature=1.0,
                                 cond_scale=float(getattr(m_opt, 'eval_res_cond_scale', 5.0)),
                                 num_res_layers=-1, **vis)
        pred = vq_model.forward_decoder(ids).cpu().numpy().astype(np.float32)
    pred = pred[:, :n_frames] * std.astype(np.float32) + mean.astype(np.float32)   # de-normalized 12D
    fmt = detect_format_from_dataset_name(args.dataset_name) or CameraDataFormat.FULL_12_ROTMAT
    traj9 = _to_clatr_with_velocity_integration(pred, format_type=fmt, dataset_type=args.dataset_name)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(B):
        np.save(out / f'sample{i}_9d.npy', traj9[i])
        np.save(out / f'sample{i}_c2w.npy', nine_d_to_c2w(traj9[i]))
        if not args.no_video:
            plot_camera_trajectory_animation_vel_integrated(
                pred[i], str(out / f'sample{i}.mp4'), title=args.text, fps=FPS,
                show_trail=True, trail_length=20, format_type=fmt, smooth=True, smooth_sigma=1.5)
    meta = {
        'text': args.text, 'mode': args.mode, 'mtrans': mtrans_name, 'rtrans': rtrans_name,
        'which_epoch': args.which_epoch, 'res_which_epoch': args.res_which_epoch, 'seed': args.seed, 'num_samples': B, 'fps': FPS,
        'duration_requested_s': args.duration, 'duration_s': n_frames / FPS, 'num_frames': n_frames,
        'first_frame': args.first_frame, 'keyframes': keyframes, 'warnings': warnings,
        'outputs': {'9d': '[r11, r21, r31, r12, r22, r32, x, y, z], camera-to-world relative to frame 0, OpenGL axes',
                    'c2w': '(T, 4, 4) camera-to-world matrices from the 9D output'},
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'[generate] {B} sample(s), {n_frames} frames ({n_frames / FPS:.2f}s) -> {out}')


if __name__ == '__main__':
    main()
