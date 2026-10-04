import argparse
import os
import torch

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## dataloader
    parser.add_argument('--dataset_name', type=str, default='humanml3d', help='dataset name (e.g. t2m, realestate10k_rotmat)')
    parser.add_argument('--data_root', type=str, default=None, help='override default dataset directory (e.g. ./dataset/RealEstate10K_rotmat1_overfit50)')
    parser.add_argument('--batch_size', default=256, type=int, help='batch size')
    parser.add_argument('--window_size', type=int, default=64, help='training motion length')
    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')

    ## optimization
    parser.add_argument('--max_epoch', default=50, type=int, help='number of total epochs to run')
    # parser.add_argument('--total_iter', default=None, type=int, help='number of total iterations to run')
    parser.add_argument('--warm_up_iter', default=2000, type=int, help='number of total iterations for warmup')
    parser.add_argument('--lr', default=2e-4, type=float, help='max learning rate')
    parser.add_argument('--milestones', default=[150000, 250000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument('--gamma', default=0.1, type=float, help="learning rate decay")

    parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')
    parser.add_argument("--commit", type=float, default=0.02, help="hyper-parameter for the commitment loss")
    parser.add_argument('--loss_vel', type=float, default=0.5, help='hyper-parameter for the velocity loss')
    parser.add_argument('--loss_smoothness', type=float, default=0.0, help='hyper-parameter for temporal smoothness (acceleration/jerk) loss. Set to 0 to disable.')
    parser.add_argument('--recons_loss', type=str, default='l1_smooth', help='reconstruction loss')

    ## vqvae arch
    parser.add_argument("--code_dim", type=int, default=512, help="embedding dimension")
    parser.add_argument("--nb_code", type=int, default=512, help="nb of embedding")
    parser.add_argument("--mu", type=float, default=0.99, help="exponential moving average to update the codebook")
    parser.add_argument("--down_t", type=int, default=2, help="downsampling rate")
    parser.add_argument("--stride_t", type=int, default=2, help="stride size")
    parser.add_argument("--width", type=int, default=512, help="width of the network")
    parser.add_argument("--depth", type=int, default=3, help="num of resblocks for each res")
    parser.add_argument("--dilation_growth_rate", type=int, default=3, help="dilation growth rate")
    parser.add_argument("--output_emb_width", type=int, default=512, help="output embedding width")
    parser.add_argument('--vq_act', type=str, default='relu', choices=['relu', 'silu', 'gelu'],
                        help='activation function for VQ encoder/decoder')
    parser.add_argument('--vq_norm', type=str, default=None, help='normalization type for VQ encoder/decoder')

    parser.add_argument('--num_quantizers', type=int, default=3, help='num_quantizers')
    parser.add_argument('--shared_codebook', action="store_true")
    parser.add_argument('--quantize_dropout_prob', type=float, default=0.2, help='quantize_dropout_prob')
    # parser.add_argument('--use_vq_prob', type=float, default=0.8, help='quantize_dropout_prob')

    parser.add_argument('--ext', type=str, default='default', help='extension/suffix for eval output filenames')


    ## other
    parser.add_argument('--name', type=str, default="test", help='Name of this trial')
    parser.add_argument('--is_continue', action="store_true", help='resume training from latest checkpoint')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--log_every', default=10, type=int, help='iter log frequency')
    parser.add_argument('--save_latest', default=500, type=int, help='iter save latest model frequency')
    parser.add_argument('--eval_every_e', default=1, type=int, help='save eval results every n epoch')
    parser.add_argument('--eval_on', action="store_true", help='turn off evaluation when there is no pretrained evaluator')
    parser.add_argument('--evaluator_ckpt', type=str, default=None,
                        help='Path to pre-trained CLaTr evaluator checkpoint (from train_evaluator.py). '
                             'When set, replaces the legacy GloVe+BiGRU evaluator with CLaTr metrics '
                             '(FID, R-Precision, Matching Score in a CLIP-trajectory contrastive space).')
    # parser.add_argument('--early_stop_e', default=5, type=int, help='early stopping epoch')
    parser.add_argument('--feat_bias', type=float, default=5, help='feature bias scaling factor for normalization')
    
    ## logging
    parser.add_argument('--use_wandb', action="store_true", help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='momask-vq', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity/team name')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='W&B run name (defaults to experiment name)')
    parser.add_argument('--wandb_tags', type=str, nargs='*', default=[], help='W&B tags for the run')
    parser.add_argument('--wandb_notes', type=str, default='', help='W&B run notes')
    parser.add_argument('--disable_tensorboard', action="store_true", help='Disable TensorBoard logging')
    parser.add_argument('--log_gradients', action="store_true", help='Log gradient norms and histograms')
    parser.add_argument('--log_codebook_usage', action="store_true", help='Log codebook usage statistics')
    parser.add_argument('--log_model_weights', action="store_true", help='Log model weight histograms periodically')
    parser.add_argument('--num_vis_samples', type=int, default=4, help='number of GT/Pred sample pairs to visualize per epoch')
    parser.add_argument('--vis_vel_integration', action="store_true",
                        help='Also generate visualizations using velocity-channel time-integration '
                             '(cumulative sum of dx/dy/dz) with Gaussian smoothing, in addition to '
                             'the standard position-based visualization.')

    parser.add_argument('--which_epoch', type=str, default="all", help='which epoch checkpoint to evaluate (e.g. all, best, latest)')

    ## For Res Predictor only
    parser.add_argument('--vq_name', type=str, default="rvq_nq6_dc512_nc512_noshare_qdp0.2", help='VQ model checkpoint name for residual predictor')
    # parser.add_argument('--n_res', type=int, default=2, help='Name of this trial')
    # parser.add_argument('--do_vq_res', action="store_true")
    parser.add_argument("--seed", default=3407, type=int)

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)

    args = vars(opt)

    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    return opt