import os
from argparse import Namespace
import re
from os.path import join as pjoin
from tkcam.utils.word_vectorizer import POS_enumerator


def is_float(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    try:
        reg = re.compile(r'^[-+]?[0-9]+\.[0-9]+$')
        res = reg.match(str(numStr))
        if res:
            flag = True
    except Exception as ex:
        print("is_float() - error: " + str(ex))
    return flag


def is_number(numStr):
    flag = False
    numStr = str(numStr).strip().lstrip('-').lstrip('+')    # 去除正数(+)、负数(-)符号
    if str(numStr).isdigit():
        flag = True
    return flag


def get_opt(opt_path, device, **kwargs):
    opt = Namespace()
    opt_dict = vars(opt)
    skip = ('-------------- End ----------------',
            '------------ Options -------------',
            '\n')
    print('Reading', opt_path)
    with open(opt_path, 'r') as f:
        for line in f:
            if line.strip() not in skip:
                # print(line.strip())
                key, value = line.strip('\n').split(': ')
                if value in ('True', 'False'):
                    opt_dict[key] = (value == 'True')
                #     print(key, value)
                elif is_float(value):
                    opt_dict[key] = float(value)
                elif is_number(value):
                    opt_dict[key] = int(value)
                else:
                    opt_dict[key] = str(value)

    # Only print key model info, not the full Namespace
    _name = getattr(opt, 'name', '?')
    _dataset = getattr(opt, 'dataset_name', '?')
    print(f'  -> {_name} (dataset={_dataset})')
    opt_dict['which_epoch'] = 'finest'
    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')

    # Dataset-specific defaults — only set data_root if not already loaded from opt.txt
    _dataset_defaults = {
        't2m':               {'data_root': './dataset/HumanML3D/',           'joints_num': 22, 'dim_pose': 263, 'max_motion_length': 196, 'max_motion_frame': 196, 'max_motion_token': 55},
        'kit':               {'data_root': './dataset/KIT-ML/',              'joints_num': 21, 'dim_pose': 251, 'max_motion_length': 196, 'max_motion_frame': 196, 'max_motion_token': 55},
        'cam':               {'data_root': './dataset/CameraTraj/',          'joints_num': 1,  'dim_pose': 5,   'max_motion_length': 500, 'max_motion_frame': 500, 'max_motion_token': 125},
        'realestate10k_6':   {'data_root': './dataset/RealEstate10K_6feat/', 'joints_num': 1,  'dim_pose': 6,   'max_motion_length': 500, 'max_motion_frame': 500, 'max_motion_token': 125},
        'realestate10k_12':  {'data_root': './dataset/RealEstate10K_12feat/','joints_num': 1,  'dim_pose': 12,  'max_motion_length': 500, 'max_motion_frame': 500, 'max_motion_token': 125},
        'realestate10k_quat':{'data_root': './dataset/RealEstate10K_quat/', 'joints_num': 1,  'dim_pose': 10,  'max_motion_length': 500, 'max_motion_frame': 500, 'max_motion_token': 125},
        'realestate10k_rotmat':{'data_root': './dataset/RealEstate10K_rotmat/','joints_num': 1,'dim_pose': 12,  'max_motion_length': 500, 'max_motion_frame': 500, 'max_motion_token': 125},
        'realestate10k_rotmat9':{'data_root': './dataset/RealEstate10K_rotmat9/','joints_num': 1,'dim_pose': 9,  'max_motion_length': 500, 'max_motion_frame': 500, 'max_motion_token': 125},
    }

    if opt.dataset_name not in _dataset_defaults:
        raise KeyError(f'Dataset not recognized: {opt.dataset_name}')

    defaults = _dataset_defaults[opt.dataset_name]

    # Respect data_root from opt.txt if it was saved there; otherwise use default
    if not hasattr(opt, 'data_root') or not opt.data_root:
        opt.data_root = defaults['data_root']

    opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
    opt.text_dir = pjoin(opt.data_root, 'texts')
    opt.joints_num = defaults['joints_num']
    opt.dim_pose = defaults['dim_pose']
    opt.max_motion_length = defaults['max_motion_length']
    opt.max_motion_frame = defaults['max_motion_frame']
    opt.max_motion_token = defaults['max_motion_token']

    if not hasattr(opt, 'unit_length'):
        opt.unit_length = 4
    if not hasattr(opt, 'max_text_len'):
        opt.max_text_len = 20  # Default max text length
    opt.dim_word = 300
    opt.num_classes = 200 // opt.unit_length
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.is_train = False
    opt.is_continue = False
    opt.device = device

    opt_dict.update(kwargs) # Overwrite with kwargs params
    return opt