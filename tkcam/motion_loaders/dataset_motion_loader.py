from tkcam.data.t2m_dataset import Text2MotionDatasetEval, collate_fn, collate_fn_text2motion_camera, collate_fn_text2motion_camera_train # TODO
from tkcam.utils.word_vectorizer import WordVectorizer
import numpy as np
from os.path import join as pjoin
from torch.utils.data import DataLoader
from tkcam.utils.get_opt import get_opt
from tkcam.utils.dataset_layout import traj_dir

def get_dataset_motion_loader(opt_path, batch_size, fname, device, load_frames=False, data_root_override=None):
    """Create evaluation dataloader for motion datasets.

    Mean/std are loaded from opt.meta_dir (e.g. checkpoints/dataset_name/Comp_v6_KLD005/meta).
    For camera datasets, ensure that meta has format-specific mean.npy, std.npy matching the
    dataset the evaluator was trained on (e.g. RealEstate10K_rotmat_5k). You can copy meta
    from the dataset or from a VQ run's meta folder once; no need to override per VQ retrain.

    Args:
        opt_path: Path to opt.txt (defines dataset_name, checkpoints_dir, name, etc.)
        batch_size: Batch size for the loader
        fname: Split name ('train', 'val', 'test')
        device: Torch device
        load_frames: Whether to load frame images
        data_root_override: Override data_root (e.g. for overfit experiments)
    """
    opt = get_opt(opt_path, device)

    # Override data_root if provided (e.g., for overfit experiments)
    if data_root_override is not None:
        opt.data_root = data_root_override
        opt.motion_dir = traj_dir(opt.data_root)
        opt.text_dir = pjoin(opt.data_root, 'texts')

    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 't2m' or opt.dataset_name == 'kit' or opt.dataset_name == 'cam':
        print('Loading dataset %s ...' % opt.dataset_name)

        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'), allow_pickle=True)
        std = np.load(pjoin(opt.meta_dir, 'std.npy'), allow_pickle=True)

        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer, load_frames=load_frames)
        # Use camera-specific collate function for camera datasets
        if opt.dataset_name == 'cam':
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=False,
                                    collate_fn=collate_fn_text2motion_camera, shuffle=True)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=False,
                                    collate_fn=collate_fn, shuffle=True)
    elif opt.dataset_name in ['realestate10k_6', 'realestate10k_12', 'realestate10k_quat', 'realestate10k_rotmat', 'realestate10k_rotmat9']:
        print('Loading dataset %s ...' % opt.dataset_name)
        mean = np.load(pjoin(opt.meta_dir, 'mean.npy'), allow_pickle=True)
        std = np.load(pjoin(opt.meta_dir, 'std.npy'), allow_pickle=True)
        w_vectorizer = WordVectorizer('./glove', 'our_vab')
        split_file = pjoin(opt.data_root, '%s.txt'%fname)
        # print(f"opt.data_root: {opt.data_root}")
        # exit()
        dataset = Text2MotionDatasetEval(opt, mean, std, split_file, w_vectorizer, load_frames=load_frames)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4, drop_last=False, collate_fn=collate_fn_text2motion_camera, shuffle=True) # collate fn for RealEstate10K datasets
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset