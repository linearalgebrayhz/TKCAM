from tkcam.models.t2m_eval_modules import *
from tkcam.utils.word_vectorizer import POS_enumerator
from os.path import join as pjoin

def build_models(opt):
    # For camera datasets, don't subtract 4 (no foot contact features)
    is_camera_dataset = opt.dataset_name in ['realestate10k_6', 'realestate10k_12', 'cam']
    movement_input_dim = opt.dim_pose if is_camera_dataset else (opt.dim_pose - 4)
    
    movement_enc = MovementConvEncoder(movement_input_dim, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)
    if opt.eval_on:
        checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
        movement_enc.load_state_dict(checkpoint['movement_encoder'])
        # Load with strict=False to allow for architecture changes (e.g., added Dropout layers)
        text_enc.load_state_dict(checkpoint['text_encoder'], strict=False)
        motion_enc.load_state_dict(checkpoint['motion_encoder'], strict=False)
        print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
        print('Note: Loaded with strict=False due to architecture updates (Dropout layers added)')
    else:
        print('evaluation off, no evaluation model loaded. All parameters are randomly initialized.')
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        if opt.dataset_name == 't2m':
            opt.dim_pose = 263
        elif opt.dataset_name == 'kit':
            opt.dim_pose = 251
        elif opt.dataset_name == 'cam':
            opt.dim_pose = 5
        elif opt.dataset_name == 'realestate10k_6':
            opt.dim_pose = 6
        elif opt.dataset_name == 'realestate10k_12':
            opt.dim_pose = 12
        elif opt.dataset_name == 'realestate10k_quat':
            opt.dim_pose = 10
        elif opt.dataset_name == 'realestate10k_rotmat':
            opt.dim_pose = 12
        else:
            raise KeyError('Dataset not Recognized!!!')

        opt.dim_word = 300
        opt.max_motion_length = 196 if opt.dataset_name == 'kit' else 240
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20 if opt.dataset_name == 'kit' else 60
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512

        # print(opt)

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not follow the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            
            # Also reorder text inputs to match the sorted order
            word_embs = word_embs[align_idx]
            pos_ohot = pos_ohot[align_idx]
            cap_lens = cap_lens[align_idx]

            '''Movement Encoding'''
            # For camera datasets, don't remove last 4 features (no foot contact)
            is_camera_dataset = self.opt.dataset_name in ['realestate10k_6', 'realestate10k_12', 'cam']
            motion_input = motions if is_camera_dataset else motions[..., :-4]
            movements = self.movement_encoder(motion_input).detach()
            m_lens = torch.div(m_lens, self.opt.unit_length, rounding_mode='floor')
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            # No need to reorder text_embedding since inputs are already sorted
        return text_embedding, motion_embedding

    # Please note that the results does not follow the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            # For camera datasets, don't remove last 4 features (no foot contact)
            is_camera_dataset = self.opt.dataset_name in ['realestate10k_6', 'realestate10k_12', 'cam']
            motion_input = motions if is_camera_dataset else motions[..., :-4]
            movements = self.movement_encoder(motion_input).detach()
            m_lens = torch.div(m_lens, self.opt.unit_length, rounding_mode='floor')
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding

## Borrowed form MDM
# our version
def build_evaluators(opt):
    # For camera datasets, don't subtract 4 (no foot contact features)
    is_camera_dataset = opt['dataset_name'] in ['realestate10k_6', 'realestate10k_12', 'cam']
    movement_input_dim = opt['dim_pose'] if is_camera_dataset else (opt['dim_pose'] - 4)
    
    movement_enc = MovementConvEncoder(movement_input_dim, opt['dim_movement_enc_hidden'], opt['dim_movement_latent'])
    text_enc = TextEncoderBiGRUCo(word_size=opt['dim_word'],
                                  pos_size=opt['dim_pos_ohot'],
                                  hidden_size=opt['dim_text_hidden'],
                                  output_size=opt['dim_coemb_hidden'],
                                  device=opt['device'])

    motion_enc = MotionEncoderBiGRUCo(input_size=opt['dim_movement_latent'],
                                      hidden_size=opt['dim_motion_hidden'],
                                      output_size=opt['dim_coemb_hidden'],
                                      device=opt['device'])

    ckpt_dir = opt['dataset_name']
    if opt['dataset_name'] == 'humanml':
        ckpt_dir = 't2m'

    checkpoint = torch.load(pjoin(opt['checkpoints_dir'], ckpt_dir, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt['device'])
    movement_enc.load_state_dict(checkpoint['movement_encoder'], strict=False)
    text_enc.load_state_dict(checkpoint['text_encoder'], strict=False)
    motion_enc.load_state_dict(checkpoint['motion_encoder'], strict=False)
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc

# our wrapper
class EvaluatorWrapper(object):

    def __init__(self, dataset_name, device):
        # Determine dim_pose based on dataset
        if dataset_name == 'humanml':
            dim_pose = 263
        elif dataset_name == 'kit':
            dim_pose = 251
        elif dataset_name == 'realestate10k_6':
            dim_pose = 6
        elif dataset_name == 'realestate10k_12':
            dim_pose = 12
        elif dataset_name == 'cam':
            dim_pose = 5
        else:
            dim_pose = 251  # default fallback
        
        opt = {
            'dataset_name': dataset_name,
            'device': device,
            'dim_word': 300,
            'max_motion_length': 196,
            'dim_pos_ohot': len(POS_enumerator),
            'dim_motion_hidden': 1024,
            'max_text_len': 20,
            'dim_text_hidden': 512,
            'dim_coemb_hidden': 512,
            'dim_pose': dim_pose,
            'dim_movement_enc_hidden': 512,
            'dim_movement_latent': 512,
            'checkpoints_dir': './checkpoints',
            'unit_length': 4,
        }

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_evaluators(opt)
        self.opt = opt
        self.device = opt['device']

        self.text_encoder.to(opt['device'])
        self.motion_encoder.to(opt['device'])
        self.movement_encoder.to(opt['device'])

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, word_embs, pos_ohot, cap_lens, motions, m_lens):
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]
            
            # Also reorder text inputs to match the sorted order
            word_embs = word_embs[align_idx]
            pos_ohot = pos_ohot[align_idx]
            cap_lens = cap_lens[align_idx]

            '''Movement Encoding'''
            # For camera datasets, don't remove last 4 features (no foot contact)
            is_camera_dataset = self.opt['dataset_name'] in ['realestate10k_6', 'realestate10k_12', 'cam']
            motion_input = motions if is_camera_dataset else motions[..., :-4]
            movements = self.movement_encoder(motion_input).detach()
            m_lens = torch.div(m_lens, self.opt['unit_length'], rounding_mode='floor')
            motion_embedding = self.motion_encoder(movements, m_lens)
            # print(motions.shape, movements.shape, motion_embedding.shape, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            # No need to reorder text_embedding since inputs are already sorted
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            movements = self.movement_encoder(motions[..., :-4]).detach()
            m_lens = torch.div(m_lens, self.opt['unit_length'], rounding_mode='floor')
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding