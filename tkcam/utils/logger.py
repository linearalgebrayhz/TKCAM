"""
Enhanced logging utilities for experiment tracking.
Supports both TensorBoard and Weights & Biases (wandb).
"""
import os
import json
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    """
    Unified logger supporting TensorBoard and optionally W&B.
    Automatically logs hyperparameters and creates organized experiment structure.
    """
    
    def __init__(self, log_dir, config, use_wandb=False, wandb_project=None, wandb_entity=None):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            config: Dictionary or argparse.Namespace with hyperparameters
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name (required if use_wandb=True)
            wandb_entity: W&B entity/username (optional)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert argparse.Namespace to dict if needed
        if hasattr(config, '__dict__'):
            self.config = vars(config)
        else:
            self.config = config
        
        # Initialize TensorBoard
        self.tb_writer = SummaryWriter(str(self.log_dir))
        
        # Log hyperparameters to TensorBoard
        self._log_hparams_to_tensorboard()
        
        # Save config as JSON for easy reference
        self._save_config_json()
        
        # Initialize W&B if requested
        self.use_wandb = use_wandb
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                
                # Initialize W&B run
                run_name = self.config.get('name', 'unnamed_experiment')
                self.wandb_run = wandb.init(
                    project=wandb_project or "momask-camera-trajectory",
                    entity=wandb_entity,
                    name=run_name,
                    config=self.config,
                    dir=str(self.log_dir.parent),
                    resume="allow",
                    id=self._get_wandb_run_id()
                )
                print(f"✓ W&B logging enabled: {wandb.run.get_url()}")
            except ImportError:
                print("⚠ wandb not installed. Run: pip install wandb")
                print("  Falling back to TensorBoard only")
                self.use_wandb = False
            except Exception as e:
                print(f"⚠ Failed to initialize W&B: {e}")
                print("  Falling back to TensorBoard only")
                self.use_wandb = False
    
    def _log_hparams_to_tensorboard(self):
        """Log hyperparameters to TensorBoard with metrics."""
        # Filter config to only include serializable types
        hparam_dict = {}
        for key, value in self.config.items():
            if isinstance(value, (int, float, str, bool)):
                hparam_dict[key] = value
            elif value is None:
                hparam_dict[key] = 'None'
            else:
                hparam_dict[key] = str(value)
        
        # Define metrics that will be tracked
        metric_dict = {
            'hparam/train_loss': 0.0,
            'hparam/train_acc': 0.0,
            'hparam/val_loss': 0.0,
            'hparam/val_acc': 0.0,
        }
        
        # Log to TensorBoard
        self.tb_writer.add_hparams(hparam_dict, metric_dict, run_name='.')
    
    def _save_config_json(self):
        """Save configuration as JSON for easy reference."""
        config_path = self.log_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)
        print(f"✓ Config saved to: {config_path}")
    
    def _get_wandb_run_id(self):
        """Generate or load persistent W&B run ID."""
        run_id_file = self.log_dir / 'wandb_run_id.txt'
        
        if run_id_file.exists():
            with open(run_id_file, 'r') as f:
                return f.read().strip()
        else:
            # Generate new ID based on experiment name
            import hashlib
            run_name = self.config.get('name', 'unnamed')
            run_id = hashlib.md5(run_name.encode()).hexdigest()[:8]
            with open(run_id_file, 'w') as f:
                f.write(run_id)
            return run_id
    
    def log_scalar(self, tag, value, step):
        """Log scalar to both TensorBoard and W&B."""
        # TensorBoard
        self.tb_writer.add_scalar(tag, value, step)
        
        # W&B
        if self.use_wandb and self.wandb_run is not None:
            # Convert nested tags like 'Train/loss' to wandb format
            wandb_dict = {tag: value}
            self.wandb.log(wandb_dict, step=step)
    
    def log_scalars(self, tag_dict, step):
        """Log multiple scalars at once."""
        for tag, value in tag_dict.items():
            self.log_scalar(tag, value, step)
    
    def log_image(self, tag, image, step):
        """Log image to both TensorBoard and W&B."""
        self.tb_writer.add_image(tag, image, step)
        
        if self.use_wandb and self.wandb_run is not None:
            self.wandb.log({tag: self.wandb.Image(image)}, step=step)
    
    def log_histogram(self, tag, values, step):
        """Log histogram to TensorBoard."""
        self.tb_writer.add_histogram(tag, values, step)
    
    def log_text(self, tag, text, step):
        """Log text to both TensorBoard and W&B."""
        self.tb_writer.add_text(tag, text, step)
        
        if self.use_wandb and self.wandb_run is not None:
            self.wandb.log({tag: text}, step=step)
    
    def watch_model(self, model, log_freq=100):
        """Watch model gradients in W&B."""
        if self.use_wandb and self.wandb_run is not None:
            self.wandb.watch(model, log='all', log_freq=log_freq)
    
    # Backward compatibility aliases for TensorBoard API
    def add_scalar(self, tag, value, step):
        """Alias for log_scalar() - TensorBoard API compatibility."""
        self.log_scalar(tag, value, step)
    
    def add_image(self, tag, image, step):
        """Alias for log_image() - TensorBoard API compatibility."""
        self.log_image(tag, image, step)
    
    def add_histogram(self, tag, values, step):
        """Alias for log_histogram() - TensorBoard API compatibility."""
        self.log_histogram(tag, values, step)
    
    def add_text(self, tag, text, step):
        """Alias for log_text() - TensorBoard API compatibility."""
        self.log_text(tag, text, step)
    
    def finish(self):
        """Close loggers."""
        self.tb_writer.close()
        if self.use_wandb and self.wandb_run is not None:
            self.wandb.finish()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()


def create_experiment_logger(opt, use_wandb=None, wandb_project=None):
    """
    Convenient factory function to create logger from training options.
    
    Args:
        opt: Training options (argparse.Namespace or dict)
        use_wandb: Override use_wandb setting (default: check opt.use_wandb)
        wandb_project: W&B project name (default: 'momask-{dataset_name}')
    
    Returns:
        ExperimentLogger instance
    """
    # Check if wandb should be used
    if use_wandb is None:
        use_wandb = getattr(opt, 'use_wandb', False)
    
    # Determine project name
    if wandb_project is None:
        dataset_name = getattr(opt, 'dataset_name', 'unknown')
        wandb_project = f"momask-{dataset_name}"
    
    # Get wandb entity from opt
    wandb_entity = getattr(opt, 'wandb_entity', None)
    
    # Create logger
    log_dir = getattr(opt, 'log_dir', './log')
    logger = ExperimentLogger(
        log_dir=log_dir,
        config=opt,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity
    )
    
    # Print experiment info
    exp_name = getattr(opt, 'name', 'unnamed')
    print(f"\n{'='*70}")
    print(f"Experiment: {exp_name}")
    print(f"Log directory: {log_dir}")
    print(f"TensorBoard: tensorboard --logdir={log_dir}")
    if use_wandb:
        print(f"W&B: {logger.wandb_run.get_url() if logger.wandb_run else 'Failed to initialize'}")
    print(f"{'='*70}\n")
    
    return logger
