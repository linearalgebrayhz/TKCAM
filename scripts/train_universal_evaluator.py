#!/usr/bin/env python3
"""
Train a universal CLaTr-style contrastive evaluator on mixed camera datasets.

Expected mixed dataset layout (already converted to CLaTr-compatible 9D):
  <data_root>/
    trajectories/*.npy       # (T, 9) trajectory (legacy name: new_joint_vecs/)
    texts/*.txt              # caption lines, supports "caption#tokens" format
    train.txt
    val.txt
    Mean.npy                 # optional
    Std.npy                  # optional

Default root is tuned for the planned universal benchmark:
  ./dataset/RE10K_ET_GenDoP
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tkcam.utils.dataset_layout import traj_dir

try:
    import clip as clip_module
except ImportError:
    print("ERROR: `clip` package not found. Install via: pip install git+https://github.com/openai/CLIP.git")
    sys.exit(1)

from tkcam.models.evaluator.clatr_models import (
    TrajectoryEncoder,
    CLIPTextEncoder,
    InfoNCELoss,
    compute_diversity,
    compute_matching_score,
    compute_r_precision,
)


class UniversalTrajectoryTextDataset(Dataset):
    """Loads mixed trajectories + captions and applies optional normalization."""

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        input_dim: int = 9,
        max_motion_length: int = 300,
        min_motion_length: int = 24,
        normalize: bool = True,
        augment: bool = False,
        translation_slice=(6, 9),
        translation_noise_std: float = 0.01,
        norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        fit_normalization: bool = False,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.motion_dir = Path(traj_dir(self.data_root))
        self.text_dir = self.data_root / "texts"
        self.max_motion_length = max_motion_length
        self.min_motion_length = min_motion_length
        self.input_dim = input_dim
        self.normalize = normalize
        self.augment = augment
        self.translation_slice = translation_slice
        self.translation_noise_std = translation_noise_std

        self.mean = np.zeros(self.input_dim, dtype=np.float32)
        self.std = np.ones(self.input_dim, dtype=np.float32)
        need_fit_stats = False
        if self.normalize:
            if norm_stats is not None:
                m, s = norm_stats
                self.mean = np.asarray(m, dtype=np.float32).copy()
                self.std = np.asarray(s, dtype=np.float32).copy()
            else:
                mean_path = self.data_root / "Mean.npy"
                std_path = self.data_root / "Std.npy"
                if mean_path.exists() and std_path.exists():
                    self.mean = np.load(str(mean_path)).astype(np.float32)
                    self.std = np.load(str(std_path)).astype(np.float32)
                elif fit_normalization:
                    need_fit_stats = True
                else:
                    print(
                        f"[UniversalTrajectoryTextDataset] Mean/Std not found in {self.data_root}; "
                        "using identity normalization."
                    )

            if self.mean.shape[-1] != self.input_dim or self.std.shape[-1] != self.input_dim:
                raise ValueError(
                    f"Mean/Std dim mismatch: mean={self.mean.shape}, std={self.std.shape}, "
                    f"expected last dim {self.input_dim}"
                )
            self._apply_translation_only_normalization_mask()

        split_file = self.data_root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        with open(split_file, "r") as f:
            id_list = [line.strip() for line in f.readlines() if line.strip()]

        self.samples = []
        skipped = 0
        for name in id_list:
            motion_path = self.motion_dir / f"{name}.npy"
            text_path = self.text_dir / f"{name}.txt"
            if not motion_path.exists() or not text_path.exists():
                skipped += 1
                continue

            motion = np.load(str(motion_path))
            if motion.ndim != 2 or motion.shape[1] != self.input_dim:
                skipped += 1
                continue
            if motion.shape[0] < self.min_motion_length or motion.shape[0] >= self.max_motion_length:
                skipped += 1
                continue

            captions = []
            with open(text_path, "r") as f:
                for line in f.readlines():
                    cap = line.strip().split("#")[0].strip()
                    if cap:
                        captions.append(cap)
            if not captions:
                skipped += 1
                continue

            self.samples.append({"name": name, "motion": motion.astype(np.float32), "captions": captions})

        if self.normalize and need_fit_stats:
            self._fit_translation_stats_from_samples()

        print(
            f"[UniversalTrajectoryTextDataset] split={split}: loaded {len(self.samples)} "
            f"samples, skipped {skipped}"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        motion = sample["motion"].copy()
        m_length = motion.shape[0]

        if self.augment and m_length > self.min_motion_length:
            crop_ratio = np.random.uniform(0.8, 1.0)
            crop_len = max(self.min_motion_length, int(m_length * crop_ratio))
            if crop_len < m_length:
                start = np.random.randint(0, m_length - crop_len)
                motion = motion[start:start + crop_len]
                m_length = crop_len

            lo, hi = self.translation_slice
            lo = max(0, lo)
            hi = min(self.input_dim, hi)
            if hi > lo and self.translation_noise_std > 0:
                motion[:, lo:hi] += (
                    np.random.randn(m_length, hi - lo).astype(np.float32) * self.translation_noise_std
                )

        if self.normalize:
            motion = (motion - self.mean) / self.std

        if motion.shape[0] < self.max_motion_length:
            pad = np.zeros((self.max_motion_length - motion.shape[0], motion.shape[1]), dtype=np.float32)
            motion = np.concatenate([motion, pad], axis=0)

        caption = np.random.choice(sample["captions"])
        return {"motion": motion, "m_length": m_length, "caption": caption}

    def _apply_translation_only_normalization_mask(self):
        """Keep only translation channels normalized; leave rotation channels untouched."""
        lo, hi = self.translation_slice
        lo = max(0, lo)
        hi = min(self.input_dim, hi)

        masked_mean = np.zeros(self.input_dim, dtype=np.float32)
        masked_std = np.ones(self.input_dim, dtype=np.float32)
        if hi > lo:
            masked_mean[lo:hi] = self.mean[lo:hi]
            masked_std[lo:hi] = self.std[lo:hi]
        self.mean = masked_mean
        self.std = np.where(masked_std < 1e-8, 1.0, masked_std)

    def _fit_translation_stats_from_samples(self):
        """Fit translation-only mean/std from loaded samples."""
        lo, hi = self.translation_slice
        lo = max(0, lo)
        hi = min(self.input_dim, hi)
        if hi <= lo or not self.samples:
            self.mean = np.zeros(self.input_dim, dtype=np.float32)
            self.std = np.ones(self.input_dim, dtype=np.float32)
            return

        trans_chunks = [s["motion"][:, lo:hi] for s in self.samples if s["motion"].shape[0] > 0]
        if not trans_chunks:
            self.mean = np.zeros(self.input_dim, dtype=np.float32)
            self.std = np.ones(self.input_dim, dtype=np.float32)
            return

        trans = np.concatenate(trans_chunks, axis=0).astype(np.float32)
        self.mean = np.zeros(self.input_dim, dtype=np.float32)
        self.std = np.ones(self.input_dim, dtype=np.float32)
        self.mean[lo:hi] = trans.mean(axis=0)
        self.std[lo:hi] = np.where(trans.std(axis=0) < 1e-8, 1.0, trans.std(axis=0))
        print(
            f"[UniversalTrajectoryTextDataset] Fitted translation stats from split={self.data_root.name}: "
            f"indices [{lo}:{hi}]"
        )


def collate_evaluator(batch):
    motions = torch.from_numpy(np.stack([b["motion"] for b in batch]))
    lengths = torch.tensor([b["m_length"] for b in batch], dtype=torch.long)
    captions = [b["caption"] for b in batch]
    return motions, lengths, captions


def train_one_epoch(
    traj_encoder,
    text_encoder,
    criterion,
    optimizer,
    dataloader,
    device,
    epoch,
    clip_tokenize,
    emb_dropout: float = 0.1,
):
    traj_encoder.train()
    text_encoder.train()

    total_loss = 0.0
    total_samples = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for motions, lengths, captions in pbar:
        motions = motions.to(device)
        lengths = lengths.to(device)
        text_tokens = clip_tokenize(captions, truncate=True).to(device)

        traj_emb = traj_encoder(motions, lengths)
        text_emb = text_encoder(text_tokens)

        if emb_dropout > 0:
            traj_emb = F.dropout(traj_emb, p=emb_dropout, training=True)
            text_emb = F.dropout(text_emb, p=emb_dropout, training=True)

        traj_emb = F.normalize(traj_emb, dim=-1)
        text_emb = F.normalize(text_emb, dim=-1)

        loss = criterion(text_emb, traj_emb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(traj_encoder.parameters()) + list(text_encoder.parameters()),
            max_norm=1.0,
        )
        optimizer.step()

        bs = motions.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        pbar.set_postfix(loss=loss.item())

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def validate(
    traj_encoder,
    text_encoder,
    dataloader,
    device,
    clip_tokenize,
):
    traj_encoder.eval()
    text_encoder.eval()

    all_traj_emb = []
    all_text_emb = []

    for motions, lengths, captions in dataloader:
        motions = motions.to(device)
        lengths = lengths.to(device)
        text_tokens = clip_tokenize(captions, truncate=True).to(device)

        traj_emb = F.normalize(traj_encoder(motions, lengths), dim=-1)
        text_emb = F.normalize(text_encoder(text_tokens), dim=-1)

        all_traj_emb.append(traj_emb.cpu().numpy())
        all_text_emb.append(text_emb.cpu().numpy())

    traj_emb_all = np.concatenate(all_traj_emb, axis=0)
    text_emb_all = np.concatenate(all_text_emb, axis=0)

    r_prec = compute_r_precision(text_emb_all, traj_emb_all, top_k=3)
    match_score = compute_matching_score(text_emb_all, traj_emb_all)
    diversity = compute_diversity(traj_emb_all)

    sim = text_emb_all @ traj_emb_all.T
    ranks_t2m = []
    for i in range(sim.shape[0]):
        sorted_idx = np.argsort(-sim[i])
        rank = np.where(sorted_idx == i)[0][0]
        ranks_t2m.append(rank)
    med_rank = float(np.median(ranks_t2m))
    recall_at_1 = float(np.mean([r < 1 for r in ranks_t2m]))
    recall_at_5 = float(np.mean([r < 5 for r in ranks_t2m]))

    return {
        "r_prec_top1": r_prec[0],
        "r_prec_top2": r_prec[1],
        "r_prec_top3": r_prec[2],
        "matching_score": match_score,
        "diversity": diversity,
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
        "median_rank": med_rank,
    }


def main():
    parser = argparse.ArgumentParser(description="Train universal CLaTr-style evaluator")
    parser.add_argument("--data_root", type=str, default="./dataset/RE10K_ET_GenDoP")
    parser.add_argument("--dataset_name", type=str, default="re10k_et_gendop")
    parser.add_argument("--input_dim", type=int, default=9,
                        help="Trajectory feature dim (default: 9 for CLaTr-compatible data)")
    parser.add_argument("--output_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--ff_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--min_lr_ratio", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=10)
    parser.add_argument("--save_dir", type=str, default="./checkpoints/evaluator")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_motion_length", type=int, default=300)
    parser.add_argument("--min_motion_length", type=int, default=24)
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Use Mean.npy/Std.npy normalization from data_root")
    parser.add_argument("--disable_normalize", action="store_true",
                        help="Disable Mean/Std normalization")
    parser.add_argument("--translation_start_idx", type=int, default=6,
                        help="Start index (inclusive) of translation channels in 9D input")
    parser.add_argument("--translation_end_idx", type=int, default=9,
                        help="End index (exclusive) of translation channels in 9D input")
    parser.add_argument("--translation_noise_std", type=float, default=0.01)
    parser.add_argument("--ortho_normalize", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.disable_normalize:
        args.normalize = False

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.run_name is None:
        ds = Path(args.data_root).name
        args.run_name = f"clatr_universal_eval_{ds}_{args.input_dim}d_ld{args.latent_dim}_od{args.output_dim}"
    save_dir = Path(args.save_dir) / args.run_name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving to: {save_dir}")

    with open(save_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_dataset = UniversalTrajectoryTextDataset(
        data_root=args.data_root,
        split="train",
        input_dim=args.input_dim,
        max_motion_length=args.max_motion_length,
        min_motion_length=args.min_motion_length,
        normalize=args.normalize,
        augment=True,
        translation_slice=(args.translation_start_idx, args.translation_end_idx),
        translation_noise_std=args.translation_noise_std,
        fit_normalization=True,
    )
    val_dataset = UniversalTrajectoryTextDataset(
        data_root=args.data_root,
        split="val",
        input_dim=args.input_dim,
        max_motion_length=args.max_motion_length,
        min_motion_length=args.min_motion_length,
        normalize=args.normalize,
        augment=False,
        translation_slice=(args.translation_start_idx, args.translation_end_idx),
        translation_noise_std=args.translation_noise_std,
        norm_stats=(train_dataset.mean, train_dataset.std) if args.normalize else None,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_evaluator,
        drop_last=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_evaluator,
        drop_last=False,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches")

    traj_encoder = TrajectoryEncoder(
        input_dim=args.input_dim,
        latent_dim=args.latent_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_size=args.ff_size,
        dropout=args.dropout,
        ortho_normalize=args.ortho_normalize,
    ).to(device)

    text_encoder = CLIPTextEncoder(output_dim=args.output_dim).to(device)
    criterion = InfoNCELoss(temperature=args.temperature)

    trainable_params = (
        list(traj_encoder.parameters())
        + list(text_encoder.projection.parameters())
        + list(criterion.parameters())
    )
    optimizer = optim.AdamW(
        trainable_params, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.99)
    )

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return max(cosine, args.min_lr_ratio)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    start_epoch = 0
    best_r1 = 0.0
    patience_counter = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        traj_encoder.load_state_dict(ckpt["traj_encoder"])
        text_encoder.load_state_dict(ckpt["text_encoder"], strict=False)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_r1 = ckpt.get("best_r1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best R@1={best_r1:.4f}")

    if train_dataset.mean is None:
        mean_tensor = torch.zeros(args.input_dim)
        std_tensor = torch.ones(args.input_dim)
    else:
        mean_tensor = torch.from_numpy(train_dataset.mean)
        std_tensor = torch.from_numpy(train_dataset.std)

    clip_tokenize = clip_module.tokenize

    traj_params = sum(p.numel() for p in traj_encoder.parameters() if p.requires_grad)
    text_proj_params = sum(p.numel() for p in text_encoder.projection.parameters())
    print(f"Trajectory encoder: {traj_params:,} trainable params")
    print(f"Text projection:    {text_proj_params:,} trainable params")
    print(f"Total trainable:    {traj_params + text_proj_params:,} params")

    print(f"\n{'='*60}")
    print(f"Starting universal evaluator training for {args.epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            traj_encoder,
            text_encoder,
            criterion,
            optimizer,
            train_loader,
            device,
            epoch + 1,
            clip_tokenize,
            emb_dropout=args.emb_dropout,
        )
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0
        print(f"Epoch {epoch + 1}/{args.epochs} | loss={train_loss:.4f} | lr={lr:.2e} | time={elapsed:.1f}s")

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            metrics = validate(traj_encoder, text_encoder, val_loader, device, clip_tokenize)
            print(
                f"  Val | R@1={metrics['recall_at_1']:.4f} R@5={metrics['recall_at_5']:.4f} "
                f"MedR={metrics['median_rank']:.1f} | "
                f"R-Prec [1/2/3]={metrics['r_prec_top1']:.4f}/{metrics['r_prec_top2']:.4f}/{metrics['r_prec_top3']:.4f} | "
                f"Match={metrics['matching_score']:.4f} Div={metrics['diversity']:.4f}"
            )

            if metrics["recall_at_1"] > best_r1:
                best_r1 = metrics["recall_at_1"]
                patience_counter = 0
                ckpt = {
                    "traj_encoder": traj_encoder.state_dict(),
                    "text_encoder": text_encoder.state_dict(),
                    "epoch": epoch,
                    "best_r1": best_r1,
                    "metrics": metrics,
                    "mean": mean_tensor,
                    "std": std_tensor,
                    "config": vars(args),
                }
                torch.save(ckpt, save_dir / "best.pt")
                print(f"  ** New best R@1={best_r1:.4f} -> saved best.pt")
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"  Early stopping: no improvement for {args.patience} eval intervals")
                    break

        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "traj_encoder": traj_encoder.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch,
                "best_r1": best_r1,
                "mean": mean_tensor,
                "std": std_tensor,
                "config": vars(args),
            }
            torch.save(ckpt, save_dir / "latest.pt")
            if (epoch + 1) % (args.save_every * 5) == 0:
                torch.save(ckpt, save_dir / f"epoch_{epoch + 1:04d}.pt")
            print(f"  Saved checkpoint -> {save_dir / 'latest.pt'}")

    print(f"\n{'='*60}")
    print(f"Training complete. Best R@1 = {best_r1:.4f}")
    print(f"Best checkpoint: {save_dir / 'best.pt'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

