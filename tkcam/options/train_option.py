from tkcam.options.base_option import BaseOptions
import argparse

class TrainT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
        self.parser.add_argument('--max_epoch', type=int, default=500, help='Maximum number of epoch for training')
        # self.parser.add_argument('--max_iters', type=int, default=150_000, help='Training iterations')

        '''LR scheduler'''
        self.parser.add_argument('--lr', type=float, default=2e-4, help='Learning rate')
        self.parser.add_argument('--gamma', type=float, default=0.1, help='Learning rate schedule factor')
        self.parser.add_argument('--milestones', default=[50_000], nargs="+", type=int,
                            help="learning rate schedule (iterations)")
        self.parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')

        '''Condition'''
        self.parser.add_argument('--cond_drop_prob', type=float, default=0.1, help='Drop ratio of condition, for classifier-free guidance')
        self.parser.add_argument('--mask_replace_prob', type=float, default=0.1,
                                help='Fraction of masked tokens to randomly replace with a random token (BERT-style noise). '
                                     'Set to 0.0 to disable replacement and keep-unchanged tricks (pure masking, recommended for overfitting experiments). '
                                     'Default 0.1 = 10%% replace + ~79%% mask + ~11%% keep-unchanged.')
        self.parser.add_argument("--seed", default=3407, type=int, help="Seed")
        self.parser.add_argument('--conditioning_mode', type=str, default='clip',
                                choices=['clip', 't5', 'id_embedding'],
                                help='Conditioning encoder: clip (default), t5 (token-level), id_embedding (per-sample learnable)')
        self.parser.add_argument('--condition_fusion', type=str, default='cross_attn',
                                choices=['cross_attn', 'prefix_self_attn'],
                                help='How condition tokens are fused: current self+cross attention, or MoMask-style condition prefix in self-attention.')
        self.parser.add_argument('--num_id_samples', type=int, default=50,
                                help='Number of learnable sample embeddings for id_embedding mode')
        self.parser.add_argument('--t5_model_name', type=str, default='t5-base',
                                help='HuggingFace T5 model name for t5 conditioning mode')

        self.parser.add_argument('--is_continue', action="store_true", help='Is this trial continuing previous state?')
        self.parser.add_argument('--gumbel_sample', action="store_true", help='Strategy for token sampling, True: Gumbel sampling, False: Categorical sampling')
        self.parser.add_argument('--share_weight', action="store_true", help='Whether to share weight for projection/embedding, for residual transformer.')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress, (iteration)')
        # self.parser.add_argument('--save_every_e', type=int, default=100, help='Frequency of printing training progress')
        self.parser.add_argument('--eval_every_e', type=int, default=5, help='Frequency of animating eval results, (epoch)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving checkpoint, (iteration)')
        self.parser.add_argument('--evaluator_ckpt', type=str, default=None,
                                help='Path to pre-trained CLaTr evaluator checkpoint. '
                                     'Replaces legacy GloVe+BiGRU evaluator with CLaTr metrics.')
        self.parser.add_argument('--eval_time_steps', type=int, default=18,
                                help='Demasking steps for mask transformer during CLaTr eval (align with gen_camera --time_steps).')
        self.parser.add_argument('--eval_mask_cond_scale', type=float, default=3.0,
                                help='CFG scale for mask stage during CLaTr eval.')
        self.parser.add_argument('--eval_res_cond_scale', type=float, default=5.0,
                                help='CFG scale for residual stage during CLaTr eval (matches gen_camera default).')
        self.parser.add_argument('--eval_temperature', type=float, default=1.0,
                                help='Sampling temperature during CLaTr eval.')
        self.parser.add_argument('--eval_topkr', type=float, default=0.9,
                                help='Top-k filtering threshold during CLaTr mask eval.')

        '''Frame Conditioning'''
        self.parser.add_argument('--use_first_frame', action="store_true",
                                help='Enable First-Frame (frame-0) visual conditioning via frozen CLIP image encoder. '
                                     'Concatenates a visual token to T5/CLIP text tokens in cross-attention.')
        self.parser.add_argument('--use_sparse_frames', action="store_true",
                                help='Enable Dynamic Sparse Keyframe Conditioning (0-4 frames per sample). '
                                     'Randomly samples K frames from trajectory, encodes via frozen CLIP + learnable temporal positional embedding.')
        self.parser.add_argument('--max_sparse_frames', type=int, default=4,
                                help='Maximum number of sparse keyframes to sample per sample (default: 4)')
        self.parser.add_argument('--visual_drop_prob', type=float, default=0.0,
                                help='Independent per-sample dropout probability for visual tokens')
        self.parser.add_argument('--frame_dir', type=str,
                                default='/data4/haozhe/CamTraj/data/processed_estate/train_frames',
                                help='Directory containing per-scene frame subdirectories')
        
        '''CLIP Fine-tuning'''
        self.parser.add_argument('--finetune_clip', action="store_true", help='Fine-tune CLIP last layers for camera direction understanding')
        self.parser.add_argument('--finetune_clip_layers', type=int, default=2, help='Number of CLIP transformer layers to unfreeze (default: 2)')
        self.parser.add_argument('--direction_loss_weight', type=float, default=0.1, help='Weight for direction contrastive loss (default: 0.1)')
        self.parser.add_argument('--smooth_loss_weight', type=float, default=0.0, help='Weight for trajectory smoothness regularization (default: 0.0 = disabled)')
        
        '''Visualization'''
        self.parser.add_argument('--vis_vel_integration', action="store_true",
                                help='Also generate visualizations reconstructed via velocity-channel '
                                     'time-integration (cumulative sum of dx/dy/dz) with Gaussian '
                                     'smoothing, alongside the standard position-based visualization.')

        '''Mixed Precision Training'''
        self.parser.add_argument('--use_amp', action="store_true", help='Enable automatic mixed precision (FP16) training for 2x speedup')
        
        '''Logging & Experiment Tracking'''
        self.parser.add_argument('--use_wandb', action="store_true", help='Enable Weights & Biases logging')
        self.parser.add_argument('--wandb_project', type=str, default=None, help='W&B project name (default: momask-{dataset_name})')
        self.parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/username (optional)')

        """Dataset"""
        # self.parser.add_argument('--data_suffix', type=str, default='', help='Suffix for dataset folder, e.g. _cam for camera datasets')

        self.is_train = True


class TrainLenEstOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--name', type=str, default="test", help='Name of this trial')
        self.parser.add_argument("--gpu_id", type=int, default=-1, help='GPU id')

        self.parser.add_argument('--dataset_name', type=str, default='t2m', help='Dataset Name')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--batch_size', type=int, default=64, help='Batch size')

        self.parser.add_argument("--unit_length", type=int, default=4, help="Length of motion")
        self.parser.add_argument("--max_text_len", type=int, default=20, help="Length of motion")

        self.parser.add_argument('--max_epoch', type=int, default=300, help='Maximum number of training epochs')

        self.parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')

        self.parser.add_argument('--is_continue', action="store_true", help='Resume training from latest checkpoint')

        self.parser.add_argument('--log_every', type=int, default=50, help='Frequency of printing training progress (iterations)')
        self.parser.add_argument('--save_every_e', type=int, default=5, help='Frequency of saving model checkpoint (epochs)')
        self.parser.add_argument('--eval_every_e', type=int, default=3, help='Frequency of running evaluation (epochs)')
        self.parser.add_argument('--save_latest', type=int, default=500, help='Frequency of saving latest checkpoint (iterations)')

    def parse(self):
        self.opt = self.parser.parse_args()
        self.opt.is_train = True
        # args = vars(self.opt)
        return self.opt
