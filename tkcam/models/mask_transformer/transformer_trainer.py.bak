import torch
from collections import defaultdict
import torch.optim as optim
from utils.logger import create_experiment_logger
from collections import OrderedDict
from utils.utils import *
from os.path import join as pjoin
from utils.eval_t2m import evaluation_mask_transformer, evaluation_res_transformer
from models.mask_transformer.tools import *

from einops import rearrange, repeat

from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

def def_value():
    return 0.0

class MaskTransformerTrainer:
    def __init__(self, args, t2m_transformer, vq_model):
        self.opt = args
        self.t2m_transformer = t2m_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        
        # Mixed precision training
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("✓ Mixed precision training (AMP) enabled for MaskTransformer")

        if args.is_train:
            # Use enhanced logger with hyperparameter tracking
            self.logger = create_experiment_logger(args)

    def encode_frames(self, frames_batch, m_lens, target_seq_len=None):
        """
        Encode frames using SparseKeyframeEncoder (ResNet + Temporal Conv).
        
        Args:
            frames_batch: List of List[Path], length B
                Each element is a list of Path objects to frame files (NOT loaded images!)
                Images will be loaded AFTER sampling indices (optimized!)
            m_lens: Tensor of motion lengths (B,) ORIGINAL lengths (before VQ downsampling)
            target_seq_len: Target sequence length to pad to (after //4). If None, uses max(m_lens)//4
        
        Returns:
            frame_embeddings: Tensor (target_seq_len, B, latent_dim)
                Padded to match VQ-encoded motion token sequence length
            has_frames: bool - Whether any frames were actually processed
        """
        # Lazy load sparse keyframe encoder
        if not hasattr(self, 'sparse_keyframe_encoder'):
            from models.sparse_keyframe_encoder import SparseKeyframeEncoder
            
            resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')  # Default to resnet18
            latent_dim = self.t2m_transformer.latent_dim
            
            print(f"Loading SparseKeyframeEncoder ({resnet_arch})...")
            self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                resnet_arch=resnet_arch,
                latent_dim=latent_dim,
                pretrained=True
            ).to(self.device)
            
            # Important: Enable training for ResNet fine-tuning
            self.sparse_keyframe_encoder.train()
            
            print(f"✓ SparseKeyframeEncoder loaded (arch={resnet_arch}, latent_dim={latent_dim})")
            print(f"  ResNet parameters will be fine-tuned during training")
        
        # Forward pass through sparse keyframe encoder
        # frames_batch contains lists of Path objects (not loaded tensors!)
        # Images will be loaded AFTER sampling indices inside encoder (optimized!)
        # Use deterministic sampling during validation (when encoder is in eval mode)
        frame_embeddings, has_frames = self.sparse_keyframe_encoder(frames_batch, m_lens, 
                                                         deterministic=not self.sparse_keyframe_encoder.training)
        
        # Pad to target sequence length if needed (to match VQ-padded motion tokens)
        if target_seq_len is not None and frame_embeddings.shape[0] < target_seq_len:
            pad_len = target_seq_len - frame_embeddings.shape[0]
            padding = torch.zeros(pad_len, frame_embeddings.shape[1], frame_embeddings.shape[2], 
                                 device=frame_embeddings.device)
            frame_embeddings = torch.cat([frame_embeddings, padding], dim=0)
            # print(f"[TRAINER DEBUG] Padded frame_emb from {frame_embeddings.shape[0] - pad_len} to {target_seq_len}")
        
        return frame_embeddings, has_frames

    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_t2m_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data, step=None):

        # Unpack batch - handle both with/without frames
        if len(batch_data) == 4:
            conds, motion, m_lens, frames_batch = batch_data  # frames_batch = List[List[Path]]
            has_frames = True
        else:
            conds, motion, m_lens = batch_data
            frames_batch = None
            has_frames = False
        
        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q)
        code_idx, _ = self.vq_model.encode(motion)
        
        # Encode frames BEFORE downsampling m_lens (need original lengths)
        frame_emb = None
        has_frames_flag = False
        if has_frames and frames_batch is not None:
            # frames_batch contains lists of Path objects (not loaded tensors!)
            # Images will be loaded AFTER sampling inside SparseKeyframeEncoder
            # Use ORIGINAL m_lens (motion lengths), not frame tensor sizes
            # m_lens represents the actual trajectory length
            # Pass target_seq_len to match code_idx sequence length (after VQ padding)
            target_seq_len = code_idx.shape[1]
            frame_emb, has_frames_flag = self.encode_frames(frames_batch, m_lens, target_seq_len=target_seq_len)
            
            # Debug: Log frame usage (only once at startup)
            if step is not None and step == 0:
                print(f"\n{'='*60}")
                print(f"OPTIMIZED FRAME LOADING IS ACTIVE")
                print(f"  Only loading {2}-{4} sampled frames per scene (not all frames!)")
                print(f"  Batch size: {len(frames_batch)}")
                print(f"  Sample 0: {len(frames_batch[0])} total frames available")
                print(f"  Frame embedding shape: {frame_emb.shape}")
                print(f"  ResNet parameters: {sum(p.numel() for p in self.sparse_keyframe_encoder.parameters())/1e6:.1f}M")
                print(f"{'='*60}\n")
        
        # Downsample m_lens for VQ tokens
        m_lens = torch.div(m_lens, 4, rounding_mode='floor')
            # frame_emb shape: (T//4, B, latent_dim) - matches motion token sequence

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        # Pass frame_emb and has_frames flag to transformer
        # Note: We need logits for Gumbel-softmax smoothness loss
        _loss, _pred_ids, _acc, logits = self.t2m_transformer.forward_with_logits(code_idx[..., 0], conds, m_lens, 
                                                       frame_emb=frame_emb, has_frames=has_frames_flag)

        # Store base reconstruction loss for monitoring
        base_recon_loss = _loss.item()
        
        # Add CLIP direction loss if fine-tuning (computed every 4 steps to reduce overhead)
        direction_loss_value = 0.0
        if self.t2m_transformer.finetune_clip and hasattr(self.opt, 'direction_loss_weight') and self.opt.direction_loss_weight > 0:
            # Compute direction loss every 4 steps (25% overhead instead of 100%)
            # Still provides strong training signal while reducing computation
            compute_dir_loss = (step is None) or (step % 4 == 0)
            
            if compute_dir_loss:
                from utils.clip_direction_loss import contrastive_direction_loss
                with torch.set_grad_enabled(True):
                    clip_embeddings = self.t2m_transformer.encode_text(conds)
                
                # Check for NaN in CLIP embeddings before direction loss
                if torch.isnan(clip_embeddings).any():
                    dir_loss = torch.tensor(0.0, device=_loss.device)
                    dir_stats = {'direction_loss/total': 0.0, 'direction_loss/same': 0.0, 'direction_loss/opposite': 0.0}
                else:
                    dir_loss, dir_stats = contrastive_direction_loss(clip_embeddings, conds, temperature=0.1, margin=0.2)
                
                # Cache the loss value for non-compute steps
                if not hasattr(self, '_cached_dir_loss'):
                    self._cached_dir_loss = 0.0
                self._cached_dir_loss = dir_loss.item()
                
                # Use conservative weight 0.05 (half of default 0.1) for stability
                actual_weight = min(self.opt.direction_loss_weight, 0.05)
                direction_loss_value = actual_weight * dir_loss
                
                # Check direction loss for NaN before adding
                if torch.isnan(direction_loss_value):
                    direction_loss_value = torch.tensor(0.0, device=_loss.device)
                else:
                    _loss = _loss + direction_loss_value
                
                if not hasattr(self, 'last_direction_stats'):
                    self.last_direction_stats = {}
                self.last_direction_stats = dir_stats
                self.last_direction_stats['direction_loss/weighted'] = direction_loss_value.item()
                self.last_direction_stats['direction_loss/weight'] = actual_weight
            else:
                # Use cached value for monitoring (no gradient computation)
                if hasattr(self, '_cached_dir_loss'):
                    actual_weight = min(self.opt.direction_loss_weight, 0.05)
                    direction_loss_value = actual_weight * self._cached_dir_loss
        
        # Add trajectory smoothness loss using Gumbel-softmax (DIFFERENTIABLE!)
        smooth_loss_value = 0.0
        if hasattr(self.opt, 'smooth_loss_weight') and self.opt.smooth_loss_weight > 0:
            # Use Gumbel-softmax to get soft token predictions (differentiable)
            # logits: (batch, num_tokens, seqlen) -> (batch, seqlen, num_tokens)
            logits_permuted = logits.permute(0, 2, 1)  # (batch, seqlen, num_tokens)
            
            # Apply Gumbel-softmax with temperature (lower = closer to one-hot)
            tau = 0.5  # Temperature for Gumbel-softmax
            soft_tokens = torch.nn.functional.gumbel_softmax(logits_permuted, tau=tau, hard=False, dim=-1)
            # soft_tokens: (batch, seqlen, num_tokens) - soft distribution over tokens
            
            # Get VQ codebook embeddings from quantizer (ResidualVQ structure)
            with torch.no_grad():
                # Access first quantizer's codebook (we only predict first codebook in masked modeling)
                codebook = self.vq_model.quantizer.layers[0].codebook  # (num_tokens, code_dim)
            
            # Soft lookup: weighted sum of codebook vectors (DIFFERENTIABLE!)
            # soft_codes: (batch, seqlen, code_dim)
            soft_codes = torch.matmul(soft_tokens, codebook)
            
            # Prepare for decoder: (batch, seqlen, code_dim) -> (batch, code_dim, seqlen)
            soft_codes_permuted = soft_codes.permute(0, 2, 1)
            
            # Decode soft codes to motion using VQ decoder (DIFFERENTIABLE!)
            with torch.set_grad_enabled(True):
                predicted_motion = self.vq_model.decoder(soft_codes_permuted)
                # predicted_motion: (batch, motion_dim, seqlen*4) - upsampled
                # Permute back: (batch, seqlen*4, motion_dim)
                predicted_motion = predicted_motion.permute(0, 2, 1)
            
            # Compute acceleration (second-order differences) on PREDICTED motion
            # Gradients flow: acceleration <- predicted_motion <- decoder <- soft_codes <- soft_tokens <- logits!
            accel = predicted_motion[:, 2:, :] - 2 * predicted_motion[:, 1:-1, :] + predicted_motion[:, :-2, :]
            smooth_loss = torch.mean(accel ** 2)
            
            # Check for NaN
            if torch.isnan(smooth_loss):
                smooth_loss = torch.tensor(0.0, device=_loss.device)
            else:
                smooth_loss_value = self.opt.smooth_loss_weight * smooth_loss
                _loss = _loss + smooth_loss_value
            
            if not hasattr(self, 'last_smooth_stats'):
                self.last_smooth_stats = {}
            self.last_smooth_stats['smooth_loss/value'] = smooth_loss.item()
            self.last_smooth_stats['smooth_loss/weighted'] = smooth_loss_value.item() if isinstance(smooth_loss_value, torch.Tensor) else smooth_loss_value
        
        # Store combined loss components for logging
        if not hasattr(self, 'last_loss_components'):
            self.last_loss_components = {}
        self.last_loss_components['loss/reconstruction'] = base_recon_loss
        self.last_loss_components['loss/direction'] = direction_loss_value.item() if isinstance(direction_loss_value, torch.Tensor) else direction_loss_value
        self.last_loss_components['loss/smooth'] = smooth_loss_value.item() if isinstance(smooth_loss_value, torch.Tensor) else smooth_loss_value
        self.last_loss_components['loss/total'] = _loss.item()

        return _loss, _acc

    def update(self, batch_data, step=None):
        self.opt_t2m_transformer.zero_grad()
        
        if self.use_amp:
            # Mixed precision forward pass
            with autocast():
                loss, acc = self.forward(batch_data, step=step)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            # Clip gradients to prevent explosion
            self.scaler.unscale_(self.opt_t2m_transformer)
            torch.nn.utils.clip_grad_norm_(self.t2m_transformer.parameters(), max_norm=1.0)
            self.scaler.step(self.opt_t2m_transformer)
            self.scaler.update()
        else:
            # Standard precision
            loss, acc = self.forward(batch_data, step=step)
            
            # Check for NaN loss BEFORE backward
            if torch.isnan(loss):
                print(f"\n{'='*60}")
                print(f"NaN DETECTED at iteration {step}")
                print(f"{'='*60}")
                # Print batch info
                if len(batch_data) == 4:
                    conds, motion, m_lens, _ = batch_data
                else:
                    conds, motion, m_lens = batch_data
                print(f"Batch info:")
                print(f"  Conditions (text): {conds if isinstance(conds, list) else 'tensor'}")
                print(f"  Motion shape: {motion.shape}")
                print(f"  Lengths: {m_lens}")
                print(f"  Motion stats: min={motion.min().item():.4f}, max={motion.max().item():.4f}, mean={motion.mean().item():.4f}")
                print(f"  Motion has NaN: {torch.isnan(motion).any().item()}")
                print(f"  Motion has Inf: {torch.isinf(motion).any().item()}")
                
                # Check CLIP output
                if isinstance(conds, list):
                    clip_out = self.t2m_transformer.encode_text(conds)
                    print(f"\nCLIP output:")
                    print(f"  Shape: {clip_out.shape}")
                    print(f"  Has NaN: {torch.isnan(clip_out).any().item()}")
                    print(f"  Has Inf: {torch.isinf(clip_out).any().item()}")
                    print(f"  Stats: min={clip_out.min().item():.4f}, max={clip_out.max().item():.4f}")
                
                print(f"{'='*60}\n")
                raise ValueError("NaN loss detected - stopping training for debugging")
            
            loss.backward()
            
            # Monitor gradients every 100 iterations (simple logging)
            if step is not None and step % 100 == 0:
                grad_stats = self.compute_gradient_stats()
                for key, value in grad_stats.items():
                    self.logger.log_scalar(f'Gradients/{key}', value, step)
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.t2m_transformer.parameters(), max_norm=1.0)
            self.opt_t2m_transformer.step()
        
        self.scheduler.step()
        return loss.item(), acc
    
    def compute_gradient_stats(self):
        """Compute simple gradient statistics for monitoring"""
        stats = {}
        
        # Separate transformer and CLIP gradients
        transformer_grads = []
        clip_grads = []
        cond_emb_grads = []
        
        for name, param in self.t2m_transformer.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if 'clip_model' in name:
                    clip_grads.append(grad_norm)
                elif 'cond_emb' in name:
                    cond_emb_grads.append(grad_norm)
                else:
                    transformer_grads.append(grad_norm)
        
        # Compute max gradient norms for each component
        if transformer_grads:
            stats['transformer_max'] = max(transformer_grads)
            stats['transformer_mean'] = sum(transformer_grads) / len(transformer_grads)
        if clip_grads:
            stats['clip_max'] = max(clip_grads)
            stats['clip_mean'] = sum(clip_grads) / len(clip_grads)
        if cond_emb_grads:
            stats['cond_emb_max'] = max(cond_emb_grads)
        
        return stats

    def save(self, file_name, ep, total_it):
        t2m_trans_state_dict = self.t2m_transformer.state_dict()
        # Only exclude CLIP weights if NOT fine-tuning
        if not self.t2m_transformer.finetune_clip:
            clip_weights = [e for e in t2m_trans_state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del t2m_trans_state_dict[e]
        
        state = {
            't2m_transformer': t2m_trans_state_dict,
            'opt_t2m_transformer': self.opt_t2m_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        
        # Save SparseKeyframeEncoder if using frames
        if hasattr(self, 'sparse_keyframe_encoder'):
            state['sparse_keyframe_encoder'] = self.sparse_keyframe_encoder.state_dict()
            print(f"DEBUG: Saving checkpoint with SparseKeyframeEncoder (epoch {ep})")
        
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        
        # Load transformer model
        missing_keys, unexpected_keys = self.t2m_transformer.load_state_dict(checkpoint['t2m_transformer'], strict=False)
        
        # Filter out expected missing keys
        # When NOT fine-tuning CLIP: expect clip_model keys to be missing (not in checkpoint)
        # When fine-tuning CLIP: expect clip_model keys to be present and loaded
        expected_missing = []
        if not self.t2m_transformer.finetune_clip:
            expected_missing.extend([k for k in missing_keys if k.startswith('clip_model.')])
        expected_missing.extend([k for k in missing_keys if k.startswith('frame_')])
        
        unexpected_missing = [k for k in missing_keys if k not in expected_missing]
        
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        if len(unexpected_missing) > 0:
            print(f"Warning: Unexpected missing keys: {unexpected_missing}")
        
        print(f"DEBUG: Loaded transformer model")

        # Resume SparseKeyframeEncoder if available
        if 'sparse_keyframe_encoder' in checkpoint:
            if not hasattr(self, 'sparse_keyframe_encoder'):
                # Initialize encoder if not already created
                from models.sparse_keyframe_encoder import SparseKeyframeEncoder
                resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
                latent_dim = self.t2m_transformer.latent_dim
                self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                    resnet_arch=resnet_arch,
                    latent_dim=latent_dim,
                    pretrained=False  # Will load from checkpoint
                ).to(self.device)
            
            self.sparse_keyframe_encoder.load_state_dict(checkpoint['sparse_keyframe_encoder'])
            print(f"DEGUG: Loaded SparseKeyframeEncoder")

        # Note: Optimizer and scheduler will be loaded in train() after creation
        return checkpoint['ep'], checkpoint['total_it']


    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        # Move models to GPU (if not already there)
        self.t2m_transformer.to(self.device)
        self.vq_model.to(self.device)
        
        # Ensure VQ model stays in eval mode (frozen during training)
        self.vq_model.eval()
        for param in self.vq_model.parameters():
            param.requires_grad = False

        epoch = 0
        it = 0

        # Resume from checkpoint if continuing training
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            if os.path.exists(model_dir):
                print(f"Resuming training from {model_dir}")
                epoch, it = self.resume(model_dir)
                print(f"Resumed from epoch {epoch}, iteration {it}")
            else:
                print(f"Warning: --is_continue set but checkpoint not found at {model_dir}")
                print(f"Starting training from scratch...")
                self.opt.is_continue = False

        # Collect parameters: transformer + sparse keyframe encoder (if using frames)
        params_to_optimize = list(self.t2m_transformer.parameters())
        
        if hasattr(self, 'sparse_keyframe_encoder'):
            # Add ResNet + Temporal Conv parameters for fine-tuning
            params_to_optimize += list(self.sparse_keyframe_encoder.parameters())
            print(f"Optimizer will update: transformer + SparseKeyframeEncoder ({self.opt.keyframe_arch})")
        else:
            print("Optimizer will update: transformer only")
        
        self.opt_t2m_transformer = optim.AdamW(params_to_optimize, betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_t2m_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)
        
        # Load optimizer and scheduler states if resuming
        if self.opt.is_continue and os.path.exists(pjoin(self.opt.model_dir, 'latest.tar')):
            try:
                checkpoint = torch.load(pjoin(self.opt.model_dir, 'latest.tar'), map_location=self.device)
                if 'opt_t2m_transformer' in checkpoint:
                    self.opt_t2m_transformer.load_state_dict(checkpoint['opt_t2m_transformer'])
                    print("Loaded optimizer state")
                if 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    print("Loaded scheduler state")
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")
                print("    Continuing with fresh optimizer...")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
            self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_acc = 0.

        while epoch < self.opt.max_epoch:
            self.t2m_transformer.train()
            self.vq_model.eval()
            
            # Set frame encoder to train mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.train()

            for i, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{self.opt.max_epoch}")):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)
                # where does clip encoding happen? in transformer.py. Batch contains text.
                # import pdb; pdb.set_trace()

                loss, acc = self.update(batch_data=batch, step=it)
                logs['loss'] += loss
                logs['acc'] += acc
                logs['lr'] += self.opt_t2m_transformer.param_groups[0]['lr']
                
                # Log direction loss stats if available (RE-ENABLED)
                if hasattr(self, 'last_direction_stats') and self.last_direction_stats:
                    for key, value in self.last_direction_stats.items():
                        if key not in logs:
                            logs[key] = 0.0
                        logs[key] += value
                
                # Log smoothness loss stats if available
                if hasattr(self, 'last_smooth_stats') and self.last_smooth_stats:
                    for key, value in self.last_smooth_stats.items():
                        if key not in logs:
                            logs[key] = 0.0
                        logs[key] += value
                
                # Log loss components for detailed monitoring
                if hasattr(self, 'last_loss_components') and self.last_loss_components:
                    for key, value in self.last_loss_components.items():
                        if key not in logs:
                            logs[key] = 0.0
                        logs[key] += value

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.log_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)
            epoch += 1

            print('Validation time:')
            self.vq_model.eval()
            self.t2m_transformer.eval()
            
            # Set frame encoder to eval mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, accuracy:{np.mean(val_acc):.3f}")

            self.logger.log_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.log_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_acc) > best_acc:
                print(f"Improved accuracy from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            # Skip expensive evaluation for camera datasets or only run every N epochs
            is_camera_dataset = any(name in self.opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
            should_evaluate = not is_camera_dataset or (epoch % self.opt.eval_every_e == 0)
            
            if should_evaluate:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_mask_transformer(
                    self.opt.save_root, eval_val_loader, self.t2m_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
                )
            else:
                print(f"Skipping evaluation for epoch {epoch} (camera dataset, eval_every_e={self.opt.eval_every_e})")


class ResidualTransformerTrainer:
    def __init__(self, args, res_transformer, vq_model):
        self.opt = args
        self.res_transformer = res_transformer
        self.vq_model = vq_model
        self.device = args.device
        self.vq_model.eval()
        
        # Mixed precision training
        self.use_amp = getattr(args, 'use_amp', False)
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            print("✓ Mixed precision training (AMP) enabled for ResidualTransformer")

        if args.is_train:
            # Use enhanced logger with hyperparameter tracking
            self.logger = create_experiment_logger(args)
            # self.l1_criterion = torch.nn.SmoothL1Loss()

    def encode_frames(self, frames_batch, m_lens, target_seq_len=None):
        """
        Encode frames using SparseKeyframeEncoder (ResNet + Temporal Conv).
        Reuses the same encoder instance as MaskTransformerTrainer if available.
        
        Args:
            frames_batch: List of List[Path], length B
                Each element is a list of Path objects to frame files (NOT loaded images!)
                Images will be loaded AFTER sampling indices (optimized!)
            m_lens: Tensor of motion lengths (B,) ORIGINAL lengths (before VQ downsampling)
            target_seq_len: Target sequence length to pad to (after //4). If None, uses max(m_lens)//4
        
        Returns:
            frame_embeddings: Tensor (target_seq_len, B, latent_dim)
                Padded to match VQ-encoded motion token sequence length
            has_frames: bool - Whether any frames were actually processed
        """
        # Lazy load sparse keyframe encoder
        if not hasattr(self, 'sparse_keyframe_encoder'):
            from models.sparse_keyframe_encoder import SparseKeyframeEncoder
            
            resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
            latent_dim = self.res_transformer.latent_dim
            
            print(f"Loading SparseKeyframeEncoder for ResidualTransformer ({resnet_arch})...")
            self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                resnet_arch=resnet_arch,
                latent_dim=latent_dim,
                pretrained=True
            ).to(self.device)
            
            # Enable training for ResNet fine-tuning
            self.sparse_keyframe_encoder.train()
            
            print(f"✓ SparseKeyframeEncoder loaded (arch={resnet_arch}, latent_dim={latent_dim})")
            print(f"  ResNet parameters will be fine-tuned during training")
        
        # Forward pass through sparse keyframe encoder
        # frames_batch contains lists of Path objects (not loaded tensors!)
        # Images will be loaded AFTER sampling indices inside encoder (optimized!)
        # Use deterministic sampling during validation (when encoder is in eval mode)
        frame_embeddings, has_frames = self.sparse_keyframe_encoder(frames_batch, m_lens,
                                                         deterministic=not self.sparse_keyframe_encoder.training)
        
        # Pad to target sequence length if needed (to match VQ-padded motion tokens)
        if target_seq_len is not None and frame_embeddings.shape[0] < target_seq_len:
            pad_len = target_seq_len - frame_embeddings.shape[0]
            padding = torch.zeros(pad_len, frame_embeddings.shape[1], frame_embeddings.shape[2], 
                                 device=frame_embeddings.device)
            frame_embeddings = torch.cat([frame_embeddings, padding], dim=0)
        
        return frame_embeddings, has_frames


    def update_lr_warm_up(self, nb_iter, warm_up_iter, lr):

        current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
        for param_group in self.opt_res_transformer.param_groups:
            param_group["lr"] = current_lr

        return current_lr


    def forward(self, batch_data):
        # Unpack batch - handle both with/without frames
        if len(batch_data) == 4:
            conds, motion, m_lens, frames_batch = batch_data  # frames_batch = List[Tensor(T_i, 3, 224, 224)]
            has_frames = True
        else:
            conds, motion, m_lens = batch_data
            frames_batch = None
            has_frames = False

        motion = motion.detach().float().to(self.device)
        m_lens = m_lens.detach().long().to(self.device)

        # (b, n, q), (q, b, n ,d)
        code_idx, all_codes = self.vq_model.encode(motion)
        
        # Encode frames BEFORE downsampling m_lens (need original lengths)
        frame_emb = None
        has_frames_flag = False
        if has_frames and frames_batch is not None:
            # frames_batch contains PRE-LOADED tensors from DataLoader workers
            # Use ORIGINAL m_lens (motion lengths), not frame tensor sizes
            # Pass target_seq_len to match code_idx sequence length (after VQ padding)
            target_seq_len = code_idx.shape[1]
            frame_emb, has_frames_flag = self.encode_frames(frames_batch, m_lens, target_seq_len=target_seq_len)
            
            # Debug: Log frame usage (only once at startup)
            # if not hasattr(self, '_frames_logged'):
            #     self._frames_logged = True
            #     print(f"\n{'='*60}")
            #     print(f"✓ FRAMES ARE BEING USED IN RESIDUAL TRANSFORMER TRAINING")
            #     print(f"  Batch size: {len(frames_batch)}")
            #     print(f"  Sample frame tensor shapes: {[f.shape for f in frames_batch[:3]]}")
            #     print(f"  Frame embedding shape: {frame_emb.shape}")
            #     print(f"  ResNet parameters: {sum(p.numel() for p in self.sparse_keyframe_encoder.parameters())/1e6:.1f}M")
            #     print(f"{'='*60}\n")
        
        # Now downsample m_lens to match VQ token length
        m_lens = torch.div(m_lens, 4, rounding_mode='floor')

        conds = conds.to(self.device).float() if torch.is_tensor(conds) else conds

        ce_loss, pred_ids, acc = self.res_transformer(code_idx, conds, m_lens, frame_emb=frame_emb, has_frames=has_frames_flag)

        # Store base reconstruction loss for monitoring
        base_recon_loss = ce_loss.item()
        
        # Add CLIP direction loss if fine-tuning (computed every 4 steps to reduce overhead)
        direction_loss_value = 0.0
        if self.res_transformer.finetune_clip and hasattr(self.opt, 'direction_loss_weight') and self.opt.direction_loss_weight > 0:
            # Compute direction loss every 4 iterations (25% overhead instead of 100%)
            if not hasattr(self, '_res_step_counter'):
                self._res_step_counter = 0
            self._res_step_counter += 1
            compute_dir_loss = (self._res_step_counter % 4 == 0)
            
            if compute_dir_loss:
                from utils.clip_direction_loss import contrastive_direction_loss
                with torch.set_grad_enabled(True):
                    clip_embeddings = self.res_transformer.encode_text(conds)
                
                # Check for NaN in CLIP embeddings before direction loss
                if torch.isnan(clip_embeddings).any():
                    dir_loss = torch.tensor(0.0, device=ce_loss.device)
                    dir_stats = {'direction_loss/total': 0.0, 'direction_loss/same': 0.0, 'direction_loss/opposite': 0.0}
                else:
                    dir_loss, dir_stats = contrastive_direction_loss(clip_embeddings, conds, temperature=0.1, margin=0.2)
                
                # Cache the loss value for non-compute steps
                if not hasattr(self, '_cached_res_dir_loss'):
                    self._cached_res_dir_loss = 0.0
                self._cached_res_dir_loss = dir_loss.item()
                
                # Use conservative weight 0.05 (half of default 0.1) for stability
                actual_weight = min(self.opt.direction_loss_weight, 0.05)
                direction_loss_value = actual_weight * dir_loss
                
                # Check direction loss for NaN before adding
                if torch.isnan(direction_loss_value):
                    direction_loss_value = torch.tensor(0.0, device=ce_loss.device)
                else:
                    ce_loss = ce_loss + direction_loss_value
                
                if not hasattr(self, 'last_direction_stats'):
                    self.last_direction_stats = {}
                self.last_direction_stats = dir_stats
                self.last_direction_stats['direction_loss/weighted'] = direction_loss_value.item()
                self.last_direction_stats['direction_loss/weight'] = actual_weight
            else:
                # Use cached value for monitoring (no gradient computation)
                if hasattr(self, '_cached_res_dir_loss'):
                    actual_weight = min(self.opt.direction_loss_weight, 0.05)
                    direction_loss_value = actual_weight * self._cached_res_dir_loss
        
        # Add trajectory smoothness loss (penalize acceleration - second-order differences)
        smooth_loss_value = 0.0
        if hasattr(self.opt, 'smooth_loss_weight') and self.opt.smooth_loss_weight > 0:
            # Compute acceleration (second-order differences) - penalizes jerky motion, not speed
            # Optimized: compute in one pass without intermediate storage
            acceleration = motion[:, 2:, :] - 2 * motion[:, 1:-1, :] + motion[:, :-2, :]  # (batch, seq_len-2, motion_dim)
            smooth_loss = torch.mean(acceleration ** 2)
            
            # Check for NaN (rare with second-order, but still check)
            if torch.isnan(smooth_loss):
                smooth_loss = torch.tensor(0.0, device=ce_loss.device)
            else:
                smooth_loss_value = self.opt.smooth_loss_weight * smooth_loss
                ce_loss = ce_loss + smooth_loss_value
            
            if not hasattr(self, 'last_smooth_stats'):
                self.last_smooth_stats = {}
            self.last_smooth_stats['smooth_loss/value'] = smooth_loss.item() if not torch.isnan(smooth_loss) else 0.0
            self.last_smooth_stats['smooth_loss/weighted'] = smooth_loss_value.item() if isinstance(smooth_loss_value, torch.Tensor) else smooth_loss_value
            self.last_smooth_stats['smooth_loss/weight'] = self.opt.smooth_loss_weight
        
        # Store combined loss components for logging
        if not hasattr(self, 'last_loss_components'):
            self.last_loss_components = {}
        self.last_loss_components['loss/reconstruction'] = base_recon_loss
        self.last_loss_components['loss/direction'] = direction_loss_value.item() if isinstance(direction_loss_value, torch.Tensor) else direction_loss_value
        self.last_loss_components['loss/smooth'] = smooth_loss_value.item() if isinstance(smooth_loss_value, torch.Tensor) else smooth_loss_value
        self.last_loss_components['loss/total'] = ce_loss.item()

        return ce_loss, acc

    def update(self, batch_data):
        self.opt_res_transformer.zero_grad()
        
        if self.use_amp:
            # Mixed precision forward pass
            with autocast():
                loss, acc = self.forward(batch_data)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            # Clip gradients to prevent explosion
            self.scaler.unscale_(self.opt_res_transformer)
            torch.nn.utils.clip_grad_norm_(self.res_transformer.parameters(), max_norm=1.0)
            self.scaler.step(self.opt_res_transformer)
            self.scaler.update()
        else:
            # Standard precision
            loss, acc = self.forward(batch_data)
            loss.backward()
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.res_transformer.parameters(), max_norm=1.0)
            self.opt_res_transformer.step()
        
        self.scheduler.step()
        return loss.item(), acc

    def save(self, file_name, ep, total_it):
        res_trans_state_dict = self.res_transformer.state_dict()
        # Only exclude CLIP weights if NOT fine-tuning
        if not self.res_transformer.finetune_clip:
            clip_weights = [e for e in res_trans_state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del res_trans_state_dict[e]
        
        state = {
            'res_transformer': res_trans_state_dict,
            'opt_res_transformer': self.opt_res_transformer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'ep': ep,
            'total_it': total_it,
        }
        
        # Save SparseKeyframeEncoder if using frames
        if hasattr(self, 'sparse_keyframe_encoder'):
            state['sparse_keyframe_encoder'] = self.sparse_keyframe_encoder.state_dict()
            print(f"Saving ResidualTransformer checkpoint with SparseKeyframeEncoder (epoch {ep})")
        
        torch.save(state, file_name)

    def resume(self, model_dir):
        checkpoint = torch.load(model_dir, map_location=self.device)
        
        # Load residual transformer model
        missing_keys, unexpected_keys = self.res_transformer.load_state_dict(checkpoint['res_transformer'], strict=False)
        
        # Filter out expected missing keys
        # When NOT fine-tuning CLIP: expect clip_model keys to be missing (not in checkpoint)
        # When fine-tuning CLIP: expect clip_model keys to be present and loaded
        expected_missing = []
        if not self.res_transformer.finetune_clip:
            expected_missing.extend([k for k in missing_keys if k.startswith('clip_model.')])
        expected_missing.extend([k for k in missing_keys if k.startswith('frame_')])
        
        unexpected_missing = [k for k in missing_keys if k not in expected_missing]
        
        if len(unexpected_keys) > 0:
            print(f"Warning: Unexpected keys in checkpoint: {unexpected_keys}")
        if len(unexpected_missing) > 0:
            print(f"Warning: Unexpected missing keys: {unexpected_missing}")
        
        print(f"Loaded residual transformer model")

        # Resume SparseKeyframeEncoder if available
        if 'sparse_keyframe_encoder' in checkpoint:
            if not hasattr(self, 'sparse_keyframe_encoder'):
                # Initialize encoder if not already created
                from models.sparse_keyframe_encoder import SparseKeyframeEncoder
                resnet_arch = getattr(self.opt, 'keyframe_arch', 'resnet18')
                latent_dim = self.res_transformer.latent_dim
                self.sparse_keyframe_encoder = SparseKeyframeEncoder(
                    resnet_arch=resnet_arch,
                    latent_dim=latent_dim,
                    pretrained=False  # Will load from checkpoint
                ).to(self.device)
            
            self.sparse_keyframe_encoder.load_state_dict(checkpoint['sparse_keyframe_encoder'])
            print(f"Loaded SparseKeyframeEncoder for ResidualTransformer")

        # Note: Optimizer and scheduler will be loaded in train() after creation
        return checkpoint['ep'], checkpoint['total_it']

    def train(self, train_loader, val_loader, eval_val_loader, eval_wrapper, plot_eval):
        # Move models to GPU (if not already there)
        self.res_transformer.to(self.device)
        self.vq_model.to(self.device)
        
        # Ensure VQ model stays in eval mode (frozen during training)
        self.vq_model.eval()
        for param in self.vq_model.parameters():
            param.requires_grad = False

        epoch = 0
        it = 0

        # Resume from checkpoint if continuing training
        if self.opt.is_continue:
            model_dir = pjoin(self.opt.model_dir, 'latest.tar')
            if os.path.exists(model_dir):
                print(f"Resuming training from {model_dir}")
                epoch, it = self.resume(model_dir)
                print(f"Resumed from epoch {epoch}, iteration {it}")
            else:
                print(f"Warning: --is_continue set but checkpoint not found at {model_dir}")
                print(f"    Starting training from scratch...")
                self.opt.is_continue = False

        # Collect parameters: residual transformer + sparse keyframe encoder (if using frames)
        params_to_optimize = list(self.res_transformer.parameters())
        
        if hasattr(self, 'sparse_keyframe_encoder'):
            params_to_optimize += list(self.sparse_keyframe_encoder.parameters())
            print(f"ResidualTransformer optimizer: transformer + SparseKeyframeEncoder ({self.opt.keyframe_arch})")
        else:
            print("ResidualTransformer optimizer: transformer only")

        self.opt_res_transformer = optim.AdamW(params_to_optimize, betas=(0.9, 0.99), lr=self.opt.lr, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.opt_res_transformer,
                                                        milestones=self.opt.milestones,
                                                        gamma=self.opt.gamma)
        
        # Load optimizer and scheduler states if resuming
        if self.opt.is_continue and os.path.exists(pjoin(self.opt.model_dir, 'latest.tar')):
            try:
                checkpoint = torch.load(pjoin(self.opt.model_dir, 'latest.tar'), map_location=self.device)
                if 'opt_res_transformer' in checkpoint:
                    self.opt_res_transformer.load_state_dict(checkpoint['opt_res_transformer'])
                    print("Loaded optimizer state")
                if 'scheduler' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler'])
                    print("Loaded scheduler state")
            except Exception as e:
                print(f"Warning: Could not load optimizer/scheduler state: {e}")
                print("    Continuing with fresh optimizer...")

        start_time = time.time()
        total_iters = self.opt.max_epoch * len(train_loader)
        print(f'Total Epochs: {self.opt.max_epoch}, Total Iters: {total_iters}')
        print('Iters Per Epoch, Training: %04d, Validation: %03d' % (len(train_loader), len(val_loader)))
        logs = defaultdict(def_value, OrderedDict())

        best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
            self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch,
            best_fid=100, best_div=100,
            best_top1=0, best_top2=0, best_top3=0,
            best_matching=100, eval_wrapper=eval_wrapper,
            plot_func=plot_eval, save_ckpt=False, save_anim=False
        )
        best_loss = 100
        best_acc = 0

        while epoch < self.opt.max_epoch:
            self.res_transformer.train()
            self.vq_model.eval()
            
            # Set frame encoder to train mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.train()

            for i, batch in enumerate(tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{self.opt.max_epoch}")):
                it += 1
                if it < self.opt.warm_up_iter:
                    self.update_lr_warm_up(it, self.opt.warm_up_iter, self.opt.lr)

                loss, acc = self.update(batch_data=batch)
                logs['loss'] += loss
                logs["acc"] += acc
                logs['lr'] += self.opt_res_transformer.param_groups[0]['lr']
                
                # Log direction loss stats if available (RE-ENABLED)
                if hasattr(self, 'last_direction_stats') and self.last_direction_stats:
                    for key, value in self.last_direction_stats.items():
                        if key not in logs:
                            logs[key] = 0.0
                        logs[key] += value
                
                # Log smoothness loss stats if available
                if hasattr(self, 'last_smooth_stats') and self.last_smooth_stats:
                    for key, value in self.last_smooth_stats.items():
                        if key not in logs:
                            logs[key] = 0.0
                        logs[key] += value
                
                # Log loss components for detailed monitoring
                if hasattr(self, 'last_loss_components') and self.last_loss_components:
                    for key, value in self.last_loss_components.items():
                        if key not in logs:
                            logs[key] = 0.0
                        logs[key] += value

                if it % self.opt.log_every == 0:
                    mean_loss = OrderedDict()
                    for tag, value in logs.items():
                        self.logger.log_scalar('Train/%s'%tag, value / self.opt.log_every, it)
                        mean_loss[tag] = value / self.opt.log_every
                    logs = defaultdict(def_value, OrderedDict())
                    print_current_loss(start_time, it, total_iters, mean_loss, epoch=epoch, inner_iter=i)

                if it % self.opt.save_latest == 0:
                    self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            epoch += 1
            self.save(pjoin(self.opt.model_dir, 'latest.tar'), epoch, it)

            print('Validation time:')
            self.vq_model.eval()
            self.res_transformer.eval()
            
            # Set frame encoder to eval mode if it exists
            if hasattr(self, 'sparse_keyframe_encoder'):
                self.sparse_keyframe_encoder.eval()

            val_loss = []
            val_acc = []
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    loss, acc = self.forward(batch_data)
                    val_loss.append(loss.item())
                    val_acc.append(acc)

            print(f"Validation loss:{np.mean(val_loss):.3f}, Accuracy:{np.mean(val_acc):.3f}")

            self.logger.log_scalar('Val/loss', np.mean(val_loss), epoch)
            self.logger.log_scalar('Val/acc', np.mean(val_acc), epoch)

            if np.mean(val_loss) < best_loss:
                print(f"Improved loss from {best_loss:.02f} to {np.mean(val_loss)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_loss.tar'), epoch, it)
                best_loss = np.mean(val_loss)

            if np.mean(val_acc) > best_acc:
                print(f"Improved acc from {best_acc:.02f} to {np.mean(val_acc)}!!!")
                self.save(pjoin(self.opt.model_dir, 'net_best_acc.tar'), epoch, it)
                best_acc = np.mean(val_acc)

            # Skip expensive evaluation for camera datasets or only run every N epochs
            is_camera_dataset = any(name in self.opt.dataset_name.lower() for name in ["cam", "estate", "realestate"])
            should_evaluate = not is_camera_dataset or (epoch % self.opt.eval_every_e == 0)
            
            if should_evaluate:
                best_fid, best_div, best_top1, best_top2, best_top3, best_matching, writer = evaluation_res_transformer(
                    self.opt.save_root, eval_val_loader, self.res_transformer, self.vq_model, self.logger, epoch, best_fid=best_fid,
                    best_div=best_div, best_top1=best_top1, best_top2=best_top2, best_top3=best_top3,
                    best_matching=best_matching, eval_wrapper=eval_wrapper,
                    plot_func=plot_eval, save_ckpt=True, save_anim=(epoch%self.opt.eval_every_e==0)
                )
            else:
                print(f"Skipping evaluation for epoch {epoch} (camera dataset, eval_every_e={self.opt.eval_every_e})")