from tkcam.options.base_option import BaseOptions

class EvalT2MOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default="net_best_fid",
            help='Mask transformer checkpoint to use. Defaults to validation-selected net_best_fid; pass latest explicitly only for diagnostics.',
        )
        self.parser.add_argument(
            '--res_which_epoch',
            type=str,
            default=None,
            help='Residual transformer checkpoint (e.g. latest, net_best_loss). '
                 'If unset, camera datasets try net_best_acc → net_best_loss → latest '
                 '(mask transformer still follows --which_epoch only — mismatched epochs can hurt quality).',
        )
        self.parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

        self.parser.add_argument('--ext', type=str, default='text2motion', help='Extension of the result file or folder')
        self.parser.add_argument("--num_batch", default=2, type=int,
                                 help="Number of batch for generation")
        self.parser.add_argument("--repeat_times", default=1, type=int,
                                 help="Number of repetitions, per sample text prompt")
        self.parser.add_argument("--cond_scale", default=4, type=float,
                                 help="CFG scale for the mask transformer stage.")
        self.parser.add_argument("--res_cond_scale", default=5, type=float,
                                 help="CFG scale for the residual transformer stage (matches training CLaTr eval / common gen_camera usage).")
        self.parser.add_argument("--temperature", default=1., type=float,
                                 help="Sampling Temperature.")
        self.parser.add_argument("--topkr", default=0.9, type=float,
                                 help="Filter out percentil low prop entries.")
        self.parser.add_argument("--time_steps", default=18, type=int,
                                 help="Mask Generate steps.")
        self.parser.add_argument("--seed", default=10107, type=int)

        self.parser.add_argument('--gumbel_sample', action="store_true", help='True: gumbel sampling, False: categorical sampling.')
        self.parser.add_argument('--use_res_model', action="store_true", help='Whether to use residual transformer.')
        # self.parser.add_argument('--est_length', action="store_true", help='Training iterations')

        self.parser.add_argument('--res_name', type=str, default='tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw', help='Model name of residual transformer')
        self.parser.add_argument('--text_path', type=str, default="", help='Text prompt file')


        self.parser.add_argument('-msec', '--mask_edit_section', nargs='*', type=str, help='Indicate sections for editing, use comma to separate the start and end of a section'
                                 'type int will specify the token frame, type float will specify the ratio of seq_len')
        self.parser.add_argument('--text_prompt', default='', type=str, help="A text prompt to be generated. If empty, will take text prompts from dataset.")
        self.parser.add_argument('--source_motion', default='example_data/000612.npy', type=str, help="Source motion path for editing. (new_joint_vecs format .npy file)")
        self.parser.add_argument("--motion_length", default=0, type=int,
                                 help="Motion length for generation, only applicable with single text prompt.")
        self.parser.add_argument("--default_camera_length", default=200, type=int,
                                 help="Default motion length for camera datasets when --motion_length is not set (frames). "
                                      "Use --motion_length N to override and generate any length. Default: 200 frames (~6.7s at 30fps).")
        
        '''Conditioning Mode (Inference)'''
        self.parser.add_argument('--conditioning_mode', type=str, default='clip', choices=['clip', 't5', 'id_embedding'],
                                 help='Conditioning mode: clip (text), t5 (token-level text), id_embedding (per-sample learnable)')
        self.parser.add_argument('--num_id_samples', type=int, default=50,
                                 help='Number of learnable sample embeddings for id_embedding mode')
        self.parser.add_argument('--t5_model_name', type=str, default='t5-base',
                                 help='HuggingFace T5 model name for t5 mode')
        self.parser.add_argument('--sample_ids', type=str, default='',
                                 help='Comma-separated sample IDs for id_embedding generation (e.g., "0,1,2,3"). '
                                      'If empty, generates for all IDs [0, num_id_samples).')

        self.parser.add_argument('--vis_vel_integration', action="store_true",
                                 help='Also generate velocity-integrated trajectory animations '
                                      '(cumulative sum of dx/dy/dz with Gaussian smoothing) '
                                      'alongside the standard position-based visualizations.')

        '''Keyframe Conditioning (Inference)'''
        self.parser.add_argument('--keyframe_dir', type=str, default=None, help='Directory containing keyframe images (jpg/png)')
        self.parser.add_argument('--keyframe_indices', type=str, default=None, 
                                help='Comma-separated frame indices for keyframes (e.g., "0,30,60,90"). Must match number of images in keyframe_dir')
        self.parser.add_argument('--use_keyframes', action="store_true", help='Enable keyframe conditioning during inference')
        
        self.is_train = False
