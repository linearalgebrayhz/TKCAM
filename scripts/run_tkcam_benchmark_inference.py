#!/usr/bin/env python3
"""
Run TKCAM benchmark inference from Benchmark_hub/test.json.

This script runs one track per invocation:
1) text/no-frame conditioning
2) first-frame conditioning
3) sparse-keyframe conditioning

For each sample, it saves one .npy trajectory in CLaTr-compatible 9D layout:
    [r11, r21, r31, r12, r22, r32, x, y, z]

The RVQ decoder emits trajectories in the VQ pipeline's normalized 12D space.
This script first de-normalizes with the VQ meta mean/std, then converts to 9D.
By default, translation is reconstructed from raw velocity channels to match
the CLaTr validation wrapper used during transformer training.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import subprocess
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from tkcam.gen_camera import (
    _resolve_text_conditioning,
    load_res_model,
    load_trans_model,
    load_vq_model,
)
from tkcam.utils.camera_geometry import integrate_velocity_to_positions, matrix_to_sixd, sixd_to_matrix
from tkcam.utils.fixseed import fixseed
from tkcam.utils.get_opt import get_opt
from tkcam.utils.unified_data_format import (
    CameraDataFormat,
    UnifiedCameraData,
    detect_format_from_dataset_name,
    gram_schmidt_orthogonalize,
)


_ROOT = Path(__file__).resolve().parent


def _resolve_benchmark_root(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    return (_ROOT.parent / "Benchmark_hub").resolve()


def _resolve_checkpoints_dir(cli_path: Optional[str]) -> Path:
    if cli_path:
        return Path(cli_path).expanduser().resolve()
    # repo-root/checkpoints (scripts/ is one level under the repo root)
    return (_ROOT.parent / "checkpoints").resolve()


def _find_prompts_file(benchmark_root: Path, cli_prompts_path: Optional[str]) -> Path:
    if cli_prompts_path:
        p = Path(cli_prompts_path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Prompts JSON not found: {p}")
        return p
    p = benchmark_root / "test.json"
    if not p.is_file():
        raise FileNotFoundError(f"Prompts JSON not found: {p}")
    return p


def _load_prompts(prompts_path: Path) -> List[Dict[str, Any]]:
    with open(prompts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return data["items"]
    raise ValueError(
        f"Unsupported prompt json format at {prompts_path}; expected list or {{'items': [...]}}"
    )


def _clip_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ]
    )


def _make_first_frame_tensor(image_path: str, preprocess: transforms.Compose) -> torch.Tensor:
    img = Image.open(image_path).convert("RGB")
    return preprocess(img)


def _resolve_path(raw_path: str, benchmark_root: Path) -> Path:
    p = Path(raw_path).expanduser()
    if p.is_absolute():
        return p
    return (benchmark_root / p).resolve()


def _item_id(item: Dict[str, Any], fallback: str) -> str:
    sid = str(item.get("id", item.get("ID", fallback))).strip()
    return sid or fallback


def _candidate_frame_paths(item: Dict[str, Any], benchmark_root: Path) -> List[Path]:
    for key in ("sparse_image_paths", "keyframe_image_paths", "keyframe_paths", "frame_paths", "images"):
        value = item.get(key)
        if isinstance(value, list) and value:
            paths = []
            for raw in value:
                if isinstance(raw, dict):
                    raw = raw.get("image_path", raw.get("path", ""))
                if raw:
                    paths.append(_resolve_path(str(raw), benchmark_root))
            if paths:
                return paths

    image_path = str(item.get("image_path", "")).strip()
    if not image_path:
        return []
    first = _resolve_path(image_path, benchmark_root)
    if not first.is_file():
        return []

    scene_dir = first.parent
    globbed = sorted(scene_dir.glob("frame_*.jpg"))
    if not globbed:
        globbed = sorted(scene_dir.glob("*.jpg")) + sorted(scene_dir.glob("*.png"))
    return globbed if globbed else [first]


def _make_sparse_frame_tensors(
    item: Dict[str, Any],
    target_len: int,
    max_sparse_frames: int,
    preprocess: transforms.Compose,
    benchmark_root: Path,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    paths = [p for p in _candidate_frame_paths(item, benchmark_root) if p.is_file()]
    sparse = torch.zeros(max_sparse_frames, 3, 224, 224)
    valid = torch.zeros(max_sparse_frames, dtype=torch.bool)
    indices = torch.zeros(max_sparse_frames, dtype=torch.long)
    if not paths:
        raise FileNotFoundError("no sparse keyframe image paths found")

    keyframe_indices = item.get("keyframe_indices", None)
    visual_indices = item.get("visual_indices", None)
    if isinstance(keyframe_indices, list) and keyframe_indices:
        chosen = list(range(min(len(paths), max_sparse_frames, len(keyframe_indices))))
        raw_indices = [int(keyframe_indices[i]) for i in chosen]
        vq_indices = [max(0, raw_idx) // 4 for raw_idx in raw_indices]
    elif isinstance(visual_indices, list) and visual_indices:
        chosen = list(range(min(len(paths), max_sparse_frames, len(visual_indices))))
        vq_indices = [max(0, int(visual_indices[i])) for i in chosen]
    else:
        n_use = min(len(paths), max_sparse_frames)
        chosen = np.linspace(0, len(paths) - 1, n_use, dtype=int).tolist()
        raw_max = max(target_len - 1, 1)
        raw_indices = [
            int(round(path_i * raw_max / max(len(paths) - 1, 1)))
            for path_i in chosen
        ]
        vq_indices = [max(0, raw_idx) // 4 for raw_idx in raw_indices]

    for slot, (path_i, vq_idx) in enumerate(zip(chosen, vq_indices)):
        img = Image.open(paths[path_i]).convert("RGB")
        sparse[slot] = preprocess(img)
        indices[slot] = int(vq_idx)
        valid[slot] = True
    return sparse, valid, indices


def _linear_resample_np(arr: np.ndarray, out_len: int) -> np.ndarray:
    in_len = int(arr.shape[0])
    if out_len <= 0:
        raise ValueError(f"out_len must be positive, got {out_len}")
    if in_len == out_len:
        return arr
    if in_len <= 1:
        return np.repeat(arr[:1], out_len, axis=0)
    x_old = np.linspace(0.0, 1.0, num=in_len, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=out_len, dtype=np.float64)
    out = np.empty((out_len, arr.shape[1]), dtype=np.float32)
    for j in range(arr.shape[1]):
        out[:, j] = np.interp(x_new, x_old, arr[:, j]).astype(np.float32)
    return out


def _safe_traj_len(item: Dict[str, Any], length_key: str, default_len: int) -> int:
    try:
        v = int(item.get(length_key, default_len))
    except (TypeError, ValueError):
        return int(default_len)
    return max(1, v)


def _infer_track_defaults(track: str) -> Tuple[str, str, str]:
    if track in ("noframe", "textonly"):
        return "mtrans_25k_noframe_r4", "rtrans_25k_noframe_r4", "result_TKCAM_textOnly"
    if track == "firstframe":
        return "mtrans_25k_1frame_r4", "rtrans_25k_1frame_r4", "result_TKCAM_firstFrame"
    if track == "sparsekeyframe":
        return "mtrans_25k_sparse_r4", "rtrans_25k_sparse_r4", "result_TKCAM_sparseKeyframe"
    raise ValueError(f"Unsupported track: {track}")


def _residual_suffix(apply_residual: bool, num_res_layers: int) -> str:
    if not apply_residual:
        return "base"
    if num_res_layers == -1:
        return "full"
    total_levels = int(num_res_layers) + 1
    return f"levels{total_levels}"


def _normalize_format_type(
    format_type: Optional[CameraDataFormat | str],
) -> Optional[CameraDataFormat]:
    if format_type is None:
        return None
    if isinstance(format_type, CameraDataFormat):
        return format_type
    if isinstance(format_type, str):
        key = format_type.strip().upper()
        if key in CameraDataFormat.__members__:
            return CameraDataFormat[key]
    raise ValueError(f"Unsupported format_type: {format_type}")


def _tkcam_to_clatr_format_local(
    tkcam_trajectories: torch.Tensor,
    *,
    format_type: Optional[CameraDataFormat | str] = None,
    dataset_type: Optional[str] = None,
) -> torch.Tensor:
    """Convert [B, N, D] TKCAM trajectories to CLaTr [B, N, 9] without importing CLaTr evaluator."""
    if tkcam_trajectories.dim() != 3:
        raise ValueError(f"Expected [B, N, D], got {tuple(tkcam_trajectories.shape)}")

    bsz, _, feat_dim = tkcam_trajectories.shape
    resolved_format = _normalize_format_type(format_type)
    if resolved_format is None and dataset_type:
        resolved_format = detect_format_from_dataset_name(str(dataset_type))
    if resolved_format is None and feat_dim == 12:
        resolved_format = CameraDataFormat.FULL_12_ROTMAT
    elif resolved_format is None and feat_dim == 10:
        resolved_format = CameraDataFormat.QUATERNION_10
    elif resolved_format is None and feat_dim == 6:
        resolved_format = CameraDataFormat.POSITION_ORIENTATION_6

    if resolved_format == CameraDataFormat.FULL_12_ROTMAT:
        pos = tkcam_trajectories[..., :3]
        rot6d = gram_schmidt_orthogonalize(tkcam_trajectories[..., 6:12])
        return torch.cat([rot6d, pos], dim=-1)
    if resolved_format == CameraDataFormat.POSITION_ROTMAT_9:
        pos = tkcam_trajectories[..., :3]
        rot6d = gram_schmidt_orthogonalize(tkcam_trajectories[..., 3:9])
        return torch.cat([rot6d, pos], dim=-1)

    converted = []
    for i in range(bsz):
        unified = UnifiedCameraData(tkcam_trajectories[i], format_type=resolved_format)
        as_rotmat = unified.to_format(CameraDataFormat.FULL_12_ROTMAT).data
        pos = as_rotmat[:, :3]
        rot6d = gram_schmidt_orthogonalize(as_rotmat[:, 6:12])
        converted.append(torch.cat([rot6d, pos], dim=-1))
    return torch.stack(converted, dim=0)


def _to_clatr_with_velocity_integration(
    denorm_feats: np.ndarray,
    *,
    format_type: CameraDataFormat,
    dataset_type: str,
) -> np.ndarray:
    if denorm_feats.shape[-1] < 10:
        denorm_t = torch.from_numpy(denorm_feats.astype(np.float32))
        return _tkcam_to_clatr_format_local(
            denorm_t,
            format_type=format_type,
            dataset_type=dataset_type,
        ).detach().cpu().numpy().astype(np.float32)

    # Start from canonical CLaTr conversion to get robust rot6d handling.
    denorm_t = torch.from_numpy(denorm_feats.astype(np.float32))
    clatr_direct = _tkcam_to_clatr_format_local(
        denorm_t,
        format_type=format_type,
        dataset_type=dataset_type,
    ).detach().cpu().numpy().astype(np.float32)

    # Replace translation with velocity-integrated positions.
    # For 12D TKCAM features, velocity channels are [:, 3:6].
    bsz = denorm_feats.shape[0]
    for bi in range(bsz):
        pos_int = integrate_velocity_to_positions(
            denorm_feats[bi],
            smooth=False,
        ).astype(np.float32)  # (T,3)
        clatr_direct[bi, :, 6:9] = pos_int
    return clatr_direct


def _canonicalize_clatr_rotation(arr9: np.ndarray) -> np.ndarray:
    """
    Canonicalize CLaTr 9D rotation to frame-0-relative rotation.

    Input / output layout:
        [r11, r21, r31, r12, r22, r32, x, y, z]
    """
    if arr9.ndim != 2 or arr9.shape[1] != 9:
        raise ValueError(f"Expected (T,9) CLaTr trajectory, got {arr9.shape}")
    if arr9.shape[0] < 1:
        return arr9

    rot6d = arr9[:, :6].astype(np.float32)
    rotmats = sixd_to_matrix(rot6d)  # (T,3,3)
    r0_t = rotmats[0].T
    # R_rel[t] = R0^T * Rt, so frame-0 becomes identity.
    rel = np.einsum("ij,tjk->tik", r0_t, rotmats)
    rel6d = matrix_to_sixd(rel).astype(np.float32)

    out = arr9.copy().astype(np.float32)
    out[:, :6] = rel6d
    out[0, :6] = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    return out


def _resolve_track_model_opts(
    checkpoints_dir: Path,
    dataset_name: str,
    mtrans_name: str,
    rtrans_name: str,
    device: torch.device,
) -> Tuple[Any, Any, Any]:
    m_opt_path = checkpoints_dir / dataset_name / mtrans_name / "opt.txt"
    r_opt_path = checkpoints_dir / dataset_name / rtrans_name / "opt.txt"
    if not m_opt_path.is_file():
        raise FileNotFoundError(f"Mask transformer opt not found: {m_opt_path}")
    if not r_opt_path.is_file():
        raise FileNotFoundError(f"Residual transformer opt not found: {r_opt_path}")

    m_opt = get_opt(str(m_opt_path), device=device)
    r_opt = get_opt(str(r_opt_path), device=device)
    if str(m_opt.vq_name) != str(r_opt.vq_name):
        raise ValueError(
            f"vq mismatch between mtrans({m_opt.vq_name}) and rtrans({r_opt.vq_name})"
        )

    vq_opt_path = checkpoints_dir / dataset_name / str(m_opt.vq_name) / "opt.txt"
    if not vq_opt_path.is_file():
        raise FileNotFoundError(f"RVQ opt inferred from transformer not found: {vq_opt_path}")
    vq_opt = get_opt(str(vq_opt_path), device=device)
    return m_opt, r_opt, vq_opt


def _load_track_models(
    *,
    checkpoints_dir: Path,
    dataset_name: str,
    mtrans_name: str,
    rtrans_name: str,
    which_epoch: str,
    res_which_epoch: Optional[str],
    device: torch.device,
) -> Tuple[Any, Any, Any, np.ndarray, np.ndarray]:
    m_opt, r_opt, vq_opt = _resolve_track_model_opts(
        checkpoints_dir=checkpoints_dir,
        dataset_name=dataset_name,
        mtrans_name=mtrans_name,
        rtrans_name=rtrans_name,
        device=device,
    )

    m_opt.checkpoints_dir = str(checkpoints_dir)
    r_opt.checkpoints_dir = str(checkpoints_dir)
    vq_opt.checkpoints_dir = str(checkpoints_dir)

    vq_model, vq_opt = load_vq_model(vq_opt)

    m_opt.num_tokens = vq_opt.nb_code
    m_opt.num_quantizers = vq_opt.num_quantizers
    m_opt.code_dim = vq_opt.code_dim

    text_cond_m = _resolve_text_conditioning(m_opt, m_opt, role="mask_transformer")
    text_cond_r = _resolve_text_conditioning(r_opt, r_opt, role="res_transformer")

    ckpt_name = which_epoch if which_epoch.endswith(".tar") else f"{which_epoch}.tar"
    t2m_transformer = load_trans_model(m_opt, m_opt, ckpt_name, text_cond=text_cond_m)

    class _TmpEvalOpt:
        pass

    tmp_eval_opt = _TmpEvalOpt()
    tmp_eval_opt.device = device
    tmp_eval_opt.res_which_epoch = res_which_epoch
    res_model = load_res_model(r_opt, vq_opt, tmp_eval_opt, text_cond=text_cond_r)

    t2m_transformer.eval().to(device)
    res_model.eval().to(device)
    vq_model.eval().to(device)

    mean = np.load(checkpoints_dir / dataset_name / str(vq_opt.name) / "meta" / "mean.npy")
    std = np.load(checkpoints_dir / dataset_name / str(vq_opt.name) / "meta" / "std.npy")
    return m_opt, t2m_transformer, res_model, vq_model, mean, std


def _run_track(
    *,
    track_name: str,
    prompts: Sequence[Dict[str, Any]],
    out_dir: Path,
    m_opt: Any,
    t2m_transformer: Any,
    res_model: Any,
    vq_model: Any,
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    benchmark_root: Path,
    length_key: str,
    batch_size: int,
    translation_source: str,
    canonicalize_rotation: bool,
    apply_residual: bool,
    num_res_layers: int,
    skip_existing: bool,
    log: logging.Logger,
    timing_json: Optional[str] = None,
    warmup: int = 3,
) -> Tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = None
    if timing_json:
        from tkcam.utils.timing import Profiler
        prof = Profiler(f"TKCAM-{track_name}", warmup=warmup)
    err_path = out_dir / "inference_failures.log"
    skip_path = out_dir / "inference_skips.log"
    if err_path.exists():
        err_path.unlink()
    if skip_path.exists():
        skip_path.unlink()

    use_first_frame = track_name == "firstframe" and bool(getattr(m_opt, "use_first_frame", False))
    use_sparse_frames = track_name == "sparsekeyframe" and bool(getattr(m_opt, "use_sparse_frames", False))
    max_sparse_frames = int(getattr(m_opt, "max_sparse_frames", 4))
    preprocess = _clip_preprocess() if (use_first_frame or use_sparse_frames) else None

    # Match training validation config by default.
    time_steps = int(getattr(m_opt, "eval_time_steps", 18))
    cond_scale = float(getattr(m_opt, "eval_mask_cond_scale", 3.0))
    temperature = float(getattr(m_opt, "eval_temperature", 1.0))
    topkr = float(getattr(m_opt, "eval_topkr", 0.9))
    res_cond_scale = float(getattr(m_opt, "eval_res_cond_scale", 5.0))
    gumbel_sample = bool(getattr(m_opt, "gumbel_sample", False))

    log.info(
        "[%s] using eval cfg: time_steps=%s cond_scale=%s temperature=%s topkr=%s res_cond_scale=%s apply_residual=%s num_res_layers=%s canonicalize_rotation=%s",
        track_name,
        time_steps,
        cond_scale,
        temperature,
        topkr,
        res_cond_scale,
        apply_residual,
        num_res_layers,
        canonicalize_rotation,
    )

    n_ok, n_skip, n_fail = 0, 0, 0
    n_items = len(prompts)
    for start in tqdm(range(0, n_items, batch_size), desc=f"TKCAM {track_name}"):
        batch = prompts[start : start + batch_size]

        valid_rows = []
        for i, item in enumerate(batch):
            sid = _item_id(item, f"row_{start+i}")
            text = str(item.get("text", item.get("prompt", ""))).strip()
            if not sid:
                sid = f"row_{start+i}"
            out_path = out_dir / f"{sid}.npy"

            if skip_existing and out_path.is_file():
                n_ok += 1
                continue
            if not text:
                n_skip += 1
                with open(skip_path, "a", encoding="utf-8") as sf:
                    sf.write(f"id={sid!r} reason='empty text'\n")
                continue

            target_len = _safe_traj_len(item, length_key=length_key, default_len=200)
            token_len = max(1, int(np.ceil(target_len / 4.0)))

            frame_tensor = None
            if use_first_frame:
                image_path = str(item.get("image_path", "")).strip()
                image_file = _resolve_path(image_path, benchmark_root) if image_path else None
                if image_file is None or not image_file.is_file():
                    n_skip += 1
                    with open(skip_path, "a", encoding="utf-8") as sf:
                        sf.write(f"id={sid!r} reason='missing image_path for first-frame track'\n")
                    continue
                try:
                    frame_tensor = _make_first_frame_tensor(str(image_file), preprocess)
                except Exception as exc:
                    n_skip += 1
                    with open(skip_path, "a", encoding="utf-8") as sf:
                        sf.write(f"id={sid!r} reason='failed to read first frame: {exc!r}'\n")
                    continue

            sparse_tensor = None
            sparse_valid = None
            sparse_indices = None
            if use_sparse_frames:
                try:
                    sparse_tensor, sparse_valid, sparse_indices = _make_sparse_frame_tensors(
                        item=item,
                        target_len=target_len,
                        max_sparse_frames=max_sparse_frames,
                        preprocess=preprocess,
                        benchmark_root=benchmark_root,
                    )
                except Exception as exc:
                    n_skip += 1
                    with open(skip_path, "a", encoding="utf-8") as sf:
                        sf.write(f"id={sid!r} reason='failed to build sparse keyframes: {exc!r}'\n")
                    continue

            valid_rows.append((sid, text, target_len, token_len, frame_tensor, sparse_tensor, sparse_valid, sparse_indices))

        if not valid_rows:
            continue

        sids = [r[0] for r in valid_rows]
        texts = [r[1] for r in valid_rows]
        target_lens = [r[2] for r in valid_rows]
        token_lens = torch.LongTensor([r[3] for r in valid_rows]).to(device)

        first_frame_pixels = None
        if use_first_frame:
            first_frame_pixels = torch.stack([r[4] for r in valid_rows], dim=0).to(device)
        sparse_frames = visual_valid_mask = visual_indices = None
        if use_sparse_frames:
            sparse_frames = torch.stack([r[5] for r in valid_rows], dim=0).to(device)
            visual_valid_mask = torch.stack([r[6] for r in valid_rows], dim=0).to(device)
            visual_indices = torch.stack([r[7] for r in valid_rows], dim=0).to(device)

        try:
            _tctx = (prof.timed(n_traj=len(valid_rows), n_frames=int(sum(target_lens)))
                     if prof else nullcontext())
            with _tctx, torch.no_grad():
                mids = t2m_transformer.generate(
                    texts,
                    token_lens,
                    timesteps=time_steps,
                    cond_scale=cond_scale,
                    temperature=temperature,
                    topk_filter_thres=topkr,
                    gsample=gumbel_sample,
                    first_frame_pixels=first_frame_pixels,
                    sparse_frames=sparse_frames,
                    visual_indices=visual_indices,
                    visual_valid_mask=visual_valid_mask,
                )
                if apply_residual:
                    decode_ids = res_model.generate(
                        mids,
                        texts,
                        token_lens,
                        temperature=1.0,
                        cond_scale=res_cond_scale,
                        num_res_layers=num_res_layers,
                        first_frame_pixels=first_frame_pixels,
                        sparse_frames=sparse_frames,
                        visual_indices=visual_indices,
                        visual_valid_mask=visual_valid_mask,
                    )
                else:
                    # t2m_transformer.generate() returns coarse indices [B, N] (q=1),
                    # while RVQ decoder expects [B, N, Q]. Add singleton quantizer dim.
                    decode_ids = mids if mids.dim() == 3 else mids.unsqueeze(-1)
                pred = vq_model.forward_decoder(decode_ids)
        except Exception as exc:
            n_fail += len(valid_rows)
            for sid in sids:
                with open(err_path, "a", encoding="utf-8") as ef:
                    ef.write(f"id={sid!r} error={exc!r}\n")
            log.exception("Track %s failed batch with %d samples", track_name, len(valid_rows))
            continue

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        pred_denorm = pred_np * std.astype(np.float32)[None, None, :] + mean.astype(np.float32)[None, None, :]

        dataset_type = str(getattr(m_opt, "dataset_name", ""))
        resolved_format = detect_format_from_dataset_name(dataset_type)
        if resolved_format is None:
            resolved_format = CameraDataFormat.FULL_12_ROTMAT if pred_denorm.shape[-1] == 12 else CameraDataFormat.POSITION_ROTMAT_9

        if translation_source == "integrated_velocity":
            pred_clatr = _to_clatr_with_velocity_integration(
                pred_denorm,
                format_type=resolved_format,
                dataset_type=dataset_type,
            )
        else:
            pred_denorm_t = torch.from_numpy(pred_denorm)
            pred_clatr_t = _tkcam_to_clatr_format_local(
                pred_denorm_t,
                format_type=resolved_format,
                dataset_type=dataset_type,
            )
            pred_clatr = pred_clatr_t.detach().cpu().numpy().astype(np.float32)

        for bi, sid in enumerate(sids):
            out_path = out_dir / f"{sid}.npy"
            target_len = int(target_lens[bi])
            native_len = int(token_lens[bi].item() * 4)
            arr = pred_clatr[bi, :native_len]
            if arr.shape[0] != target_len:
                arr = _linear_resample_np(arr, target_len)
            if canonicalize_rotation:
                arr = _canonicalize_clatr_rotation(arr)
            np.save(out_path, arr.astype(np.float32))
            n_ok += 1

    if prof is not None:
        prof.save(timing_json, meta={"batch_size": batch_size, "device": str(device),
                                     "track": track_name})
    return n_ok, n_skip, n_fail


def _write_manifest(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    prompts_path: Path,
    checkpoints_dir: Path,
    mtrans_name: str,
    rtrans_name: str,
    m_opt: Any,
    mean: np.ndarray,
    std: np.ndarray,
    ok: int,
    skip: int,
    fail: int,
) -> None:
    residual_mode = _residual_suffix(bool(args.apply_residual), int(args.num_res_layers))
    manifest = {
        "track": args.track,
        "dataset_name": args.dataset_name,
        "prompts_json": str(prompts_path),
        "checkpoints_dir": str(checkpoints_dir),
        "mtrans_name": mtrans_name,
        "rtrans_name": rtrans_name,
        "vq_name": getattr(m_opt, "vq_name", None),
        "which_epoch": args.which_epoch,
        "res_which_epoch": args.res_which_epoch,
        "seed": int(args.seed),
        "residual_mode": residual_mode,
        "apply_residual": bool(args.apply_residual),
        "num_res_layers": int(args.num_res_layers),
        "translation_source": args.translation_source,
        "canonicalize_rotation": bool(args.canonicalize_rotation),
        "length_key": args.length_key,
        "generation": {
            "eval_time_steps": getattr(m_opt, "eval_time_steps", None),
            "eval_mask_cond_scale": getattr(m_opt, "eval_mask_cond_scale", None),
            "eval_temperature": getattr(m_opt, "eval_temperature", None),
            "eval_topkr": getattr(m_opt, "eval_topkr", None),
            "eval_res_cond_scale": getattr(m_opt, "eval_res_cond_scale", None),
            "gumbel_sample": getattr(m_opt, "gumbel_sample", None),
        },
        "model_conditioning": {
            "conditioning_mode": getattr(m_opt, "conditioning_mode", None),
            "use_first_frame": getattr(m_opt, "use_first_frame", False),
            "use_sparse_frames": getattr(m_opt, "use_sparse_frames", False),
            "max_sparse_frames": getattr(m_opt, "max_sparse_frames", None),
            "condition_fusion": getattr(m_opt, "condition_fusion", "cross_attn"),
        },
        "normalization": {
            "mean_shape": list(mean.shape),
            "std_shape": list(std.shape),
        },
        "summary": {"ok": ok, "skip": skip, "fail": fail},
    }
    with open(out_dir / "inference_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def _write_sanity_check(out_dir: Path, prompts: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    expected_ids = [_item_id(item, f"row_{i}") for i, item in enumerate(prompts)]
    files = {p.stem: p for p in out_dir.glob("*.npy")}
    missing = [sid for sid in expected_ids if sid not in files]
    bad = []
    shapes: Dict[str, int] = {}
    for sid, path in files.items():
        try:
            arr = np.load(path)
            shapes[str(tuple(arr.shape))] = shapes.get(str(tuple(arr.shape)), 0) + 1
            if arr.ndim != 2 or arr.shape[1] != 9 or arr.shape[0] < 1 or not np.isfinite(arr).all():
                bad.append({"id": sid, "shape": list(arr.shape), "finite": bool(np.isfinite(arr).all())})
        except Exception as exc:
            bad.append({"id": sid, "error": repr(exc)})
    report = {
        "expected": len(expected_ids),
        "found_npy": len(files),
        "missing": len(missing),
        "bad": len(bad),
        "missing_ids_sample": missing[:20],
        "bad_sample": bad[:20],
        "shape_histogram": shapes,
    }
    with open(out_dir / "sanity_check.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    return report


def _write_eval_recipe(out_dir: Path, benchmark_root: Path, prompts_path: Path) -> None:
    baseline_name = out_dir.name
    prefix = ""
    name = baseline_name
    if baseline_name.startswith("result_"):
        prefix = "result_"
        name = baseline_name[len(prefix):]
    cmd = [
        "python",
        str(_ROOT / "eval_benchmark_clatr.py"),
        "--benchmark-root",
        str(benchmark_root),
        "--prompts-json",
        str(prompts_path),
        "--result-prefix",
        prefix,
        "--baseline-names",
        name,
    ]
    with open(out_dir / "eval_command.sh", "w", encoding="utf-8") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -euo pipefail\n")
        f.write(subprocess.list2cmdline(cmd) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one TKCAM benchmark track and save CLaTr-format 9D npy files."
    )
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["textonly", "noframe", "firstframe", "sparsekeyframe"],
        help="Which track to run in this invocation.",
    )
    parser.add_argument("--benchmark-root", type=str, default=None, help="Path to Benchmark_hub.")
    parser.add_argument("--prompts-json", type=str, default=None, help="Path to test.json.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Torch device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument("--seed", type=int, default=10107)
    parser.add_argument("--max-samples", type=int, default=None, help="Only run first N samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Generation batch size.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip existing <id>.npy outputs.")
    parser.add_argument("--timing-json", type=str, default=None,
                        help="If set, profile per-batch generation time (cuda-synced) -> JSON summary.")
    parser.add_argument("--warmup", type=int, default=3, help="Timed batches to discard as warm-up.")
    parser.add_argument(
        "--apply-residual",
        action="store_true",
        help="Apply residual transformer refinement after mask-transformer generation.",
    )
    parser.add_argument(
        "--num-res-layers",
        type=int,
        default=-1,
        help=(
            "Residual depth passed to ResidualTransformer.generate. "
            "-1 uses all quantizer levels; N decodes N+1 total RVQ levels."
        ),
    )
    parser.add_argument(
        "--translation-source",
        type=str,
        default="integrated_velocity",
        choices=["position", "integrated_velocity"],
        help=(
            "Choose CLaTr translation source: "
            "'integrated_velocity' uses cumulative sum of dx/dy/dz (default); "
            "'position' uses xyz channels directly."
        ),
    )
    parser.add_argument(
        "--canonicalize-rotation",
        dest="canonicalize_rotation",
        action="store_true",
        help="Canonicalize rotation as frame-0-relative (R0^T Rt); frame 0 becomes identity.",
    )
    parser.add_argument(
        "--no-canonicalize-rotation",
        dest="canonicalize_rotation",
        action="store_false",
        help="Disable rotation canonicalization. Default: disabled to match CLaTr validation conversion.",
    )
    parser.add_argument(
        "--length-key",
        type=str,
        default="traj_len",
        help="JSON key that stores target trajectory length.",
    )

    parser.add_argument(
        "--checkpoints-dir",
        type=str,
        default=None,
        help="Path to momask checkpoints root. Default: <momask-codes>/checkpoints",
    )
    parser.add_argument("--dataset-name", type=str, default="realestate10k_rotmat")
    parser.add_argument(
        "--which-epoch",
        type=str,
        default="net_best_fid",
        help="Mask transformer checkpoint basename. Defaults to validation-selected net_best_fid.",
    )
    parser.add_argument(
        "--res-which-epoch",
        type=str,
        default=None,
        help=(
            "Residual checkpoint basename. If unset, load_res_model prefers validation-selected "
            "checkpoints and only uses latest when no best checkpoint exists."
        ),
    )

    parser.add_argument(
        "--mtrans-name",
        type=str,
        default=None,
        help="Mask transformer experiment name. If omitted, default is inferred from --track.",
    )
    parser.add_argument(
        "--rtrans-name",
        type=str,
        default=None,
        help="Residual transformer experiment name. If omitted, default is inferred from --track.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. If omitted, default is inferred from --track under benchmark root.",
    )
    parser.set_defaults(canonicalize_rotation=False)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log = logging.getLogger("run_tkcam_benchmark_inference")

    benchmark_root = _resolve_benchmark_root(args.benchmark_root)
    checkpoints_dir = _resolve_checkpoints_dir(args.checkpoints_dir)
    mtrans_default, rtrans_default, out_default_name = _infer_track_defaults(str(args.track))
    mtrans_name = str(args.mtrans_name) if args.mtrans_name else mtrans_default
    rtrans_name = str(args.rtrans_name) if args.rtrans_name else rtrans_default
    residual_mode = _residual_suffix(bool(args.apply_residual), int(args.num_res_layers))
    out_dir = (
        Path(args.out_dir).expanduser().resolve()
        if args.out_dir
        else (benchmark_root / f"{out_default_name}_{residual_mode}").resolve()
    )

    prompts_path = _find_prompts_file(benchmark_root, args.prompts_json)
    prompts = _load_prompts(prompts_path)
    if args.max_samples is not None:
        prompts = prompts[: int(args.max_samples)]
    log.info("Loaded %d prompts from %s", len(prompts), prompts_path)

    device = torch.device(args.device)
    fixseed(int(args.seed))

    log.info(
        "[%s] model=%s res=%s (RVQ inferred from opt.txt), translation_source=%s",
        args.track,
        mtrans_name,
        rtrans_name,
        args.translation_source,
    )
    m_opt, t2m, res, vq, mean, std = _load_track_models(
        checkpoints_dir=checkpoints_dir,
        dataset_name=str(args.dataset_name),
        mtrans_name=mtrans_name,
        rtrans_name=rtrans_name,
        which_epoch=str(args.which_epoch),
        res_which_epoch=args.res_which_epoch,
        device=device,
    )
    ok, skip, fail = _run_track(
        track_name=str(args.track),
        prompts=prompts,
        out_dir=out_dir,
        m_opt=m_opt,
        t2m_transformer=t2m,
        res_model=res,
        vq_model=vq,
        mean=mean,
        std=std,
        device=device,
        benchmark_root=benchmark_root,
        length_key=str(args.length_key),
        batch_size=int(args.batch_size),
        translation_source=str(args.translation_source),
        canonicalize_rotation=bool(args.canonicalize_rotation),
        apply_residual=bool(args.apply_residual),
        num_res_layers=int(args.num_res_layers),
        skip_existing=bool(args.skip_existing),
        log=log,
        timing_json=args.timing_json,
        warmup=int(args.warmup),
    )
    _write_manifest(
        out_dir,
        args=args,
        prompts_path=prompts_path,
        checkpoints_dir=checkpoints_dir,
        mtrans_name=mtrans_name,
        rtrans_name=rtrans_name,
        m_opt=m_opt,
        mean=mean,
        std=std,
        ok=ok,
        skip=skip,
        fail=fail,
    )
    sanity = _write_sanity_check(out_dir, prompts)
    _write_eval_recipe(out_dir, benchmark_root, prompts_path)
    del t2m, res, vq
    if device.type == "cuda":
        torch.cuda.empty_cache()
    log.info(
        "[%s] done ok=%d skip=%d fail=%d missing=%d bad=%d out=%s",
        args.track, ok, skip, fail, sanity["missing"], sanity["bad"], out_dir,
    )
    return 0 if ok > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
