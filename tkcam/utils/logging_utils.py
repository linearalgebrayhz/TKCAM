import os
import torch
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, Any, Optional, List

try:
    import wandb  # type: ignore
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("wandb not installed. Install with: pip install wandb")

try:
    from torch.utils.tensorboard.writer import SummaryWriter  # type: ignore
    TENSORBOARD_AVAILABLE = True
except ImportError:
    try:
        from torch.utils.tensorboard import SummaryWriter  # type: ignore
        TENSORBOARD_AVAILABLE = True
    except ImportError:
        TENSORBOARD_AVAILABLE = False
        print("tensorboard not installed. Install with: pip install tensorboard")


class VQLogger:
    """
    Comprehensive logging utility for VQ training that supports both TensorBoard and wandb
    """
    
    def __init__(self, opt, resume_run_id=None):
        self.opt = opt
        self.use_wandb = opt.use_wandb and WANDB_AVAILABLE
        self.use_tensorboard = not opt.disable_tensorboard and TENSORBOARD_AVAILABLE
        self.should_log_gradients = opt.log_gradients
        self.should_log_codebook_usage = opt.log_codebook_usage
        self.should_log_model_weights = opt.log_model_weights
        
        # Initialize loggers
        self.wandb_run = None
        self.tensorboard_writer = None
        
        if self.use_wandb:
            self._init_wandb(resume_run_id)
        
        if self.use_tensorboard:
            self._init_tensorboard()
    
    def _init_wandb(self, resume_run_id=None):
        """Initialize Weights & Biases logging"""
        if not WANDB_AVAILABLE:
            print("wandb not available, skipping wandb initialization")
            return
            
        # Prepare config
        config = {
            # Training config
            'dataset_name': self.opt.dataset_name,
            'batch_size': self.opt.batch_size,
            'max_epoch': self.opt.max_epoch,
            'lr': self.opt.lr,
            'warm_up_iter': self.opt.warm_up_iter,
            'weight_decay': self.opt.weight_decay,
            'seed': self.opt.seed,
            
            # Loss config
            'commit_weight': self.opt.commit,
            'loss_vel_weight': self.opt.loss_vel,
            'recons_loss': self.opt.recons_loss,
            
            # VQ config
            'code_dim': self.opt.code_dim,
            'nb_code': self.opt.nb_code,
            'num_quantizers': self.opt.num_quantizers,
            'shared_codebook': self.opt.shared_codebook,
            'quantize_dropout_prob': self.opt.quantize_dropout_prob,
            'mu': self.opt.mu,
            
            # Model architecture
            'down_t': self.opt.down_t,
            'stride_t': self.opt.stride_t,
            'width': self.opt.width,
            'depth': self.opt.depth,
            'dilation_growth_rate': self.opt.dilation_growth_rate,
            'vq_act': self.opt.vq_act,
            'vq_norm': self.opt.vq_norm,
        }
        
        # Initialize wandb
        run_name = self.opt.wandb_run_name if self.opt.wandb_run_name else self.opt.name
        
        self.wandb_run = wandb.init(
            project=self.opt.wandb_project,
            entity=self.opt.wandb_entity,
            name=run_name,
            config=config,
            tags=self.opt.wandb_tags,
            notes=self.opt.wandb_notes,
            resume="allow" if resume_run_id else None,
            id=resume_run_id
        )
        
        # Watch model if specified
        if self.should_log_model_weights:
            print("Model weight logging enabled - will watch model during training")
    
    def _init_tensorboard(self):
        """Initialize TensorBoard logging"""
        if not TENSORBOARD_AVAILABLE:
            print("tensorboard not available, skipping tensorboard initialization")
            return
            
        self.tensorboard_writer = SummaryWriter(self.opt.log_dir)
    
    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = ""):
        """
        Log metrics to both wandb and tensorboard
        
        Args:
            metrics: Dictionary of metric names and values
            step: Current step/iteration
            prefix: Prefix for metric names (e.g., "train/", "val/")
        """
        if not metrics:
            return
            
        # Log to wandb
        if self.use_wandb and self.wandb_run:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    wandb_metrics[f"{prefix}{key}"] = value
            if wandb_metrics:
                self.wandb_run.log(wandb_metrics, step=step)
        
        # Log to tensorboard
        if self.use_tensorboard and self.tensorboard_writer:
            for key, value in metrics.items():
                if isinstance(value, (int, float, np.number)):
                    self.tensorboard_writer.add_scalar(f"{prefix}{key}", value, step)
    
    def log_training_metrics(self, losses: Dict[str, float], lr: float, step: int):
        """Log training metrics"""
        metrics = losses.copy()
        metrics['lr'] = lr
        self.log_metrics(metrics, step, "Train/")
    
    def log_validation_metrics(self, losses: Dict[str, float], step: int):
        """Log validation metrics"""
        self.log_metrics(losses, step, "Val/")
    
    def log_evaluation_metrics(self, eval_metrics: Dict[str, float], step: int):
        """Log evaluation metrics"""
        self.log_metrics(eval_metrics, step, "Eval/")
    
    def log_gradients(self, model, step: int):
        """Log gradient norms and histograms"""
        if not self.should_log_gradients:
            return
            
        gradient_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms[f"grad_norm/{name}"] = grad_norm
                
                # Log gradient histograms to tensorboard
                if self.use_tensorboard and self.tensorboard_writer:
                    self.tensorboard_writer.add_histogram(f"grad_hist/{name}", param.grad, step)
        
        if gradient_norms:
            self.log_metrics(gradient_norms, step, "")
    
    def log_codebook_usage(self, quantizer, step: int):
        """Log codebook usage statistics"""
        if not self.should_log_codebook_usage:
            return
            
        usage_stats = {}
        
        # Get codebook usage from each quantizer layer
        for i, layer in enumerate(quantizer.layers):
            if hasattr(layer, 'code_count'):
                code_count = layer.code_count.cpu().numpy()
                
                # Calculate usage statistics
                total_codes = len(code_count)
                used_codes = np.sum(code_count > 0)
                usage_percentage = (used_codes / total_codes) * 100
                
                usage_stats[f"codebook_usage/layer_{i}_used_codes"] = used_codes
                usage_stats[f"codebook_usage/layer_{i}_usage_percentage"] = usage_percentage
                usage_stats[f"codebook_usage/layer_{i}_mean_count"] = np.mean(code_count)
                usage_stats[f"codebook_usage/layer_{i}_std_count"] = np.std(code_count)
                
                # Log codebook histogram to tensorboard
                if self.use_tensorboard and self.tensorboard_writer:
                    self.tensorboard_writer.add_histogram(f"codebook_hist/layer_{i}", code_count, step)
        
        if usage_stats:
            self.log_metrics(usage_stats, step, "")
    
    def log_model_weights(self, model, step: int):
        """Log model weight histograms"""
        if not self.should_log_model_weights:
            return
            
        if self.use_tensorboard and self.tensorboard_writer:
            for name, param in model.named_parameters():
                if param.requires_grad:
                    self.tensorboard_writer.add_histogram(f"weights/{name}", param, step)
        
        # Log weight norms to both loggers
        weight_norms = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                weight_norms[f"weight_norm/{name}"] = param.norm().item()
        
        if weight_norms:
            self.log_metrics(weight_norms, step, "")
    
    def log_learning_rate_schedule(self, lr_schedule: List[float], step: int):
        """Log learning rate schedule"""
        # Convert list to individual metrics
        lr_metrics = {f"lr_schedule_{i}": lr for i, lr in enumerate(lr_schedule)}
        self.log_metrics(lr_metrics, step, "")
    
    def log_epoch_summary(self, epoch: int, train_losses: Dict[str, float], 
                         val_losses: Dict[str, float], best_metrics: Dict[str, float]):
        """Log epoch summary"""
        summary = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_losses.items()},
            **{f"val_{k}": v for k, v in val_losses.items()},
            **{f"best_{k}": v for k, v in best_metrics.items()}
        }
        
        self.log_metrics(summary, epoch, "Epoch/")
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters with final metrics"""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparams, metrics)
    
    def save_checkpoint_info(self, epoch: int, step: int, model_path: str, 
                           metrics: Dict[str, float]):
        """Save checkpoint information"""
        checkpoint_info = {
            "epoch": epoch,
            "step": step,
            "model_path": model_path,
            **metrics
        }
        
        if self.use_wandb and self.wandb_run:
            # Save model as artifact
            artifact = wandb.Artifact(
                name=f"model-{self.opt.name}-epoch-{epoch}",
                type="model",
                description=f"Model checkpoint at epoch {epoch}",
                metadata=checkpoint_info
            )
            artifact.add_file(model_path)
            self.wandb_run.log_artifact(artifact)
    
    def log_dataset_info(self, train_size: int, val_size: int, 
                        motion_length_stats: Dict[str, float]):
        """Log dataset information"""
        dataset_info = {
            "dataset/train_size": train_size,
            "dataset/val_size": val_size,
            **{f"dataset/motion_length_{k}": v for k, v in motion_length_stats.items()}
        }
        
        self.log_metrics(dataset_info, 0, "")
    
    def watch_model(self, model):
        """Watch model for automatic gradient and parameter logging"""
        if self.use_wandb and self.wandb_run and self.should_log_model_weights:
            self.wandb_run.watch(model, log="all", log_freq=self.opt.log_every)
    
    def add_scalar(self, tag: str, scalar_value: float, global_step: Optional[int] = None):
        """
        Backward compatibility method for TensorBoard SummaryWriter interface
        This allows existing evaluation code to work without modifications
        """
        # Remove any leading './' from tag for cleaner logging
        clean_tag = tag.lstrip('./')
        
        # Log to both backends
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, scalar_value, global_step)
        
        if self.use_wandb and self.wandb_run:
            self.wandb_run.log({clean_tag: scalar_value}, step=global_step)
    
    def add_histogram(self, tag: str, values, global_step: Optional[int] = None, bins: str = 'tensorflow'):
        """
        Backward compatibility method for TensorBoard histogram logging
        """
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_histogram(tag, values, global_step, bins)
        
        # wandb histograms are handled differently, skip for now to avoid conflicts
    
    def add_hparams(self, hparam_dict: Dict[str, Any], metric_dict: Dict[str, float]):
        """
        Backward compatibility method for hyperparameter logging
        """
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_hparams(hparam_dict, metric_dict)
        
        # wandb automatically tracks config, so no additional action needed
    
    def close(self):
        """Close all loggers"""
        if self.use_wandb and self.wandb_run:
            self.wandb_run.finish()
        
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def create_loss_dict(loss, loss_rec, loss_explicit, loss_commit, perplexity):
    """Create standardized loss dictionary"""
    return {
        'loss': loss,
        'loss_rec': loss_rec,
        'loss_explicit': loss_explicit,
        'loss_commit': loss_commit,
        'perplexity': perplexity
    }


def aggregate_losses(loss_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate losses from multiple batches"""
    if not loss_list:
        return {}
    
    aggregated = defaultdict(float)
    for losses in loss_list:
        for key, value in losses.items():
            aggregated[key] += value
    
    # Calculate averages
    num_batches = len(loss_list)
    return {key: value / num_batches for key, value in aggregated.items()} 