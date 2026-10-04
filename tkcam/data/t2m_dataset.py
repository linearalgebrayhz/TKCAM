from os.path import join as pjoin
import torch
from torch.utils import data
import numpy as np
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import random
import codecs as cs
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms


CAMERA_DATASET_NAMES = (
    "cam",
    "realestate10k_6",
    "realestate10k_12",
    "realestate10k_quat",
    "realestate10k_rotmat",
    "realestate10k_rotmat9",
)


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)

def collate_fn_camera(batch):
    """
    Custom collate function for MotionDataset with camera trajectories.
    Pads all trajectories to the maximum length in the batch.
    """
    # For MotionDataset, each item is just a motion tensor
    # Sort by length (descending) to get the maximum length first
    batch.sort(key=lambda x: x.shape[0], reverse=True)
    
    # Get the maximum length in this batch
    max_len = batch[0].shape[0]
    
    # Pad all motions to the same length
    padded_batch = []
    for motion in batch:
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        padded_batch.append(motion)
    
    # Use default collate on the padded batch
    return default_collate(padded_batch)

def collate_fn_text2motion_camera(batch):
    """
    Custom collate function for Text2MotionDatasetEval with camera trajectories.
    Pads all motions to the maximum length in the batch.
    Expected format: (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token)
    or with frames: (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, keyframes)
    """
    # Check if batch includes frames
    has_frames = len(batch[0]) == 8
    
    # Sort by length (descending) to get the maximum length first
    if has_frames:
        batch.sort(key=lambda x: x[5], reverse=True)  # m_length is at index 5
    else:
        batch.sort(key=lambda x: x[5], reverse=True)  # m_length is at index 5
    
    # Get the maximum length in this batch
    max_len = batch[0][5]  # motion length is at index 5
    
    # Pad all motions to the same length
    padded_batch = []
    for item in batch:
        if has_frames:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, keyframes = item
        else:
            word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token = item
            keyframes = None
        
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # Update the item with padded motion
        if has_frames:
            padded_item = (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, keyframes)
        else:
            padded_item = (word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token)
        padded_batch.append(padded_item)
    
    # Use default collate on the padded batch
    return default_collate(padded_batch)


def collate_fn_text2motion_camera_train(batch):
    """
    Custom collate function for Text2MotionDataset with camera trajectories (training).
    Pads all motions to the maximum length in the batch.
    Expected format: (caption, motion, m_length)
    """
    # Sort by length (descending) to get the maximum length first
    batch.sort(key=lambda x: x[2], reverse=True)  # m_length is at index 2 for Text2MotionDataset
    
    # Get the maximum length in this batch
    max_len = batch[0][2]  # motion length is at index 2
    
    # Pad all motions to the same length
    padded_batch = []
    for item in batch:
        caption, motion, m_length = item
        
        # Pad motion to max_len
        if motion.shape[0] < max_len:
            # Create padding with zeros
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        
        # Update the item with padded motion
        padded_item = (caption, motion, m_length)
        padded_batch.append(padded_item)
    
    # Use default collate on the padded batch
    return default_collate(padded_batch)


def collate_fn_text2motion_camera_train_first_frame(batch):
    """Collate for first-frame conditioning.

    Expected format: (caption, motion, m_length, first_frame_tensor)
    where first_frame_tensor is (3, 224, 224) or None.
    Returns: (captions, motions, m_lengths, first_frame_batch)
    where first_frame_batch is (B, 3, 224, 224).
    """
    batch.sort(key=lambda x: x[2], reverse=True)
    max_len = batch[0][2]

    captions = []
    motions = []
    m_lengths = []
    first_frames = []

    for item in batch:
        caption, motion, m_length, first_frame = item

        if motion.shape[0] < max_len:
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)

        captions.append(caption)
        motions.append(motion)
        m_lengths.append(m_length)
        first_frames.append(first_frame)

    motions = torch.from_numpy(np.stack(motions, axis=0))
    m_lengths = torch.tensor(m_lengths)
    first_frame_batch = torch.stack(first_frames, dim=0)  # (B, 3, 224, 224)

    return captions, motions, m_lengths, first_frame_batch


def collate_fn_text2motion_camera_train_sparse_frames(batch):
    """Collate for sparse keyframe conditioning (0-4 frames).

    Expected format: (caption, motion, m_length, sparse_frames, visual_valid_mask, visual_indices)
    where:
        sparse_frames: (4, 3, 224, 224) tensor with K valid frames + zeros
        visual_valid_mask: (4,) bool tensor — True for valid frames
        visual_indices: (4,) int tensor — frame indices in trajectory

    Returns:
        captions: list of strings
        motions: (B, max_len, D) tensor
        m_lengths: (B,) tensor
        sparse_frames_batch: (B, 4, 3, 224, 224) tensor
        visual_valid_mask_batch: (B, 4) bool tensor
        visual_indices_batch: (B, 4) long tensor
    """
    batch.sort(key=lambda x: x[2], reverse=True)
    max_len = batch[0][2]

    captions = []
    motions = []
    m_lengths = []
    sparse_frames_list = []
    visual_valid_masks = []
    visual_indices_list = []

    for item in batch:
        caption, motion, m_length, sparse_frames, visual_valid_mask, visual_indices = item

        if motion.shape[0] < max_len:
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)

        captions.append(caption)
        motions.append(motion)
        m_lengths.append(m_length)
        sparse_frames_list.append(sparse_frames)
        visual_valid_masks.append(visual_valid_mask)
        visual_indices_list.append(visual_indices)

    motions = torch.from_numpy(np.stack(motions, axis=0))
    m_lengths = torch.tensor(m_lengths)
    sparse_frames_batch = torch.stack(sparse_frames_list, dim=0)      # (B, 4, 3, 224, 224)
    visual_valid_mask_batch = torch.stack(visual_valid_masks, dim=0)  # (B, 4)
    visual_indices_batch = torch.stack(visual_indices_list, dim=0)    # (B, 4)

    return captions, motions, m_lengths, sparse_frames_batch, visual_valid_mask_batch, visual_indices_batch


class MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file):
        self.opt = opt
        joints_num = opt.joints_num

        self.data = []
        self.lengths = []
        id_list = []
        with open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                if motion.shape[0] < opt.window_size:
                    continue
                self.lengths.append(motion.shape[0] - opt.window_size)
                self.data.append(motion)
            except Exception as e:
                # Some motion may not exist in KIT dataset
                print(e)
                pass

        self.cumsum = np.cumsum([0] + self.lengths)
        if opt.dataset_name == "cam":
            if opt.is_train:
                std = std / opt.feat_bias  # Scale all features by feat_bias
                # Save updated mean and std to meta_dir
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            assert mean.shape[-1] == 5, f"Expected 5 features for cam dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_6":
            if opt.is_train:
                # Save updated mean and std to meta_dir for realestate10k_6
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            assert mean.shape[-1] == 6, f"Expected 6 features for realestate10k_6 dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_quat":
            # 10D quaternion format: [x, y, z, dx, dy, dz, qw, qx, qy, qz]
            if opt.is_train:
                # Only normalize position & velocity (first 6 dimensions)
                # Quaternion (last 4 dimensions) has unit constraint, no normalization needed
                # Set mean=0, std=1 for quaternion to skip normalization
                mean_quat = np.concatenate([mean[:6], np.zeros(4)])
                std_quat = np.concatenate([std[:6], np.ones(4)])
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean_quat)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std_quat)
                mean = mean_quat
                std = std_quat
            assert mean.shape[-1] == 10, f"Expected 10 features for realestate10k_quat dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_rotmat":
            # 12D rotation matrix format: [x, y, z, dx, dy, dz, r1x, r1y, r1z, r2x, r2y, r2z]
            if opt.is_train:
                # Only normalize position & velocity (first 6 dimensions)
                # Rotation matrix columns (last 6 dimensions) have orthonormal constraint, no normalization needed
                # Set mean=0, std=1 for rotation to skip normalization
                mean_rot = np.concatenate([mean[:6], np.zeros(6)])
                std_rot = np.concatenate([std[:6], np.ones(6)])
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean_rot)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std_rot)
                mean = mean_rot
                std = std_rot
            assert mean.shape[-1] == 12, f"Expected 12 features for realestate10k_rotmat dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_rotmat9":
            # 9D rotation matrix format: [x, y, z, r1x, r1y, r1z, r2x, r2y, r2z]
            if opt.is_train:
                # Only normalize translation; rot6d columns keep their unit-scale geometry.
                mean_rot9 = np.concatenate([mean[:3], np.zeros(6)])
                std_rot9 = np.concatenate([std[:3], np.ones(6)])
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean_rot9)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std_rot9)
                mean = mean_rot9
                std = std_rot9
            assert mean.shape[-1] == 9, f"Expected 9 features for realestate10k_rotmat9 dataset, got {mean.shape[-1]}"
        elif opt.dataset_name == "realestate10k_12":
            # 12D Euler format (existing): [x, y, z, dx, dy, dz, pitch, yaw, roll, dpitch, dyaw, droll]. The Euler Angle is really not a good representation
            if opt.is_train:
                # Save updated mean and std to meta_dir for realestate10k_12
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)
            assert mean.shape[-1] == 12, f"Expected 12 features for realestate10k_12 dataset, got {mean.shape[-1]}"
        else: 
            if opt.is_train:
                # root_rot_velocity (B, seq_len, 1)
                std[0:1] = std[0:1] / opt.feat_bias
                # root_linear_velocity (B, seq_len, 2)
                std[1:3] = std[1:3] / opt.feat_bias
                # root_y (B, seq_len, 1)
                std[3:4] = std[3:4] / opt.feat_bias
                # ric_data (B, seq_len, (joint_num - 1)*3)
                std[4: 4 + (joints_num - 1) * 3] = std[4: 4 + (joints_num - 1) * 3] / 1.0
                # rot_data (B, seq_len, (joint_num - 1)*6)
                std[4 + (joints_num - 1) * 3: 4 + (joints_num - 1) * 9] = std[4 + (joints_num - 1) * 3: 4 + (
                        joints_num - 1) * 9] / 1.0
                # local_velocity (B, seq_len, joint_num*3)
                std[4 + (joints_num - 1) * 9: 4 + (joints_num - 1) * 9 + joints_num * 3] = std[
                                                                                        4 + (joints_num - 1) * 9: 4 + (
                                                                                                joints_num - 1) * 9 + joints_num * 3] / 1.0
                # foot contact (B, seq_len, 4)
                std[4 + (joints_num - 1) * 9 + joints_num * 3:] = std[
                                                                4 + (
                                                                            joints_num - 1) * 9 + joints_num * 3:] / opt.feat_bias

                assert 4 + (joints_num - 1) * 9 + joints_num * 3 + 4 == mean.shape[-1]
                np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
                np.save(pjoin(opt.meta_dir, 'std.npy'), std)

        self.mean = mean
        self.std = std
        print("Total number of motions {}, snippets {}".format(len(self.data), self.cumsum[-1]))

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return self.cumsum[-1]

    def __getitem__(self, item):
        if item != 0:
            motion_id = np.searchsorted(self.cumsum, item) - 1
            idx = item - self.cumsum[motion_id] - 1
        else:
            motion_id = 0
            idx = 0
        motion = self.data[motion_id][idx:idx + self.opt.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion


class Text2MotionDatasetEval(data.Dataset):
    def __init__(self, opt, mean, std, split_file, w_vectorizer, load_frames=False):
        self.opt = opt
        self.w_vectorizer = w_vectorizer
        self.max_length = 20 #?
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        
        # Frame loading configuration for evaluation
        self.load_frames = load_frames
        if load_frames:
            from pathlib import Path
            import json
            from PIL import Image
            import torchvision.transforms as transforms
            
            # Frame preprocessing transforms (matching CLaTr)
            self.frame_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ])
            
            # Load scene_id_mapping
            mapping_path = Path(opt.data_root) / 'scene_id_mapping.json'
            if mapping_path.exists():
                with open(mapping_path) as f:
                    self.scene_id_mapping = json.load(f)
                print(f"✅ Loaded scene ID mapping: {len(self.scene_id_mapping)} entries")
            else:
                print(f"⚠️  Warning: scene_id_mapping.json not found at {mapping_path}")
                self.scene_id_mapping = {}
            
            # Frame directory
            frame_dir_str = getattr(opt, 'frame_dir', 
                                    '/data4/haozhe/CamTraj/data/processed_estate/train_frames')
            self.frame_dir = Path(frame_dir_str)
            self.max_frames = getattr(opt, 'max_frames', 8)
            
            print(f"✅ Frame loading enabled:")
            print(f"   - Frame dir: {self.frame_dir}")
            print(f"   - Max frames: {self.max_frames}")
            print(f"   - Frame size: 224x224")

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]
        # print("[Testing] id list:")
        # print("Length of id_list:", len(id_list))
        # print(id_list[:10])
        
        new_name_list = []
        length_list = []
        if opt.dataset_name in CAMERA_DATASET_NAMES:
            for name in tqdm(id_list):
                try:
                    # import pdb; pdb.set_trace()
                    motion_path = pjoin(opt.motion_dir, name + '.npy')
                    text_path = pjoin(opt.text_dir, name + '.txt')
                    motion = np.load(motion_path)
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = 0.0
                            to_tag = 0.0
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens 
                            # print(f"loading text file:\ncaption - {caption},\ntokens - {tokens},\n{f_tag}, {to_tag}")
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    print("error loading text")
        else:
            for name in tqdm(id_list):
                try:
                    motion_path = pjoin(opt.motion_dir, name + '.npy')
                    text_path = pjoin(opt.text_dir, name + '.txt')
                    print(f"Attempting to load motion: {motion_path}")
                    print(f"Attempting to load text: {text_path}")
                    motion = np.load(motion_path)
                    print(f"Motion loaded, shape: {motion.shape}")
                    # motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            print(f"loading text file: \n caption {caption},\n tokens {tokens},\n {f_tag}, {to_tag}")
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 200):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                        'length': len(n_motion),
                                                        'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                        'length': len(motion),
                                        'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except:
                    print("error loading text")
                
                

        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list
        self.reset_max_len(self.max_length)

    def reset_max_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d"%self.pointer)
        self.max_length = length

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def load_and_process_frames(self, scene_id):
        """
        Load and preprocess frames for a scene (for evaluation).
        
        Args:
            scene_id: Sequential scene ID (e.g., '000000')
        
        Returns:
            frames: Tensor of shape (max_frames, 3, 224, 224)
        """
        from pathlib import Path
        from PIL import Image
        import torch
        
        # Map sequential ID to hash ID
        hash_id = self.scene_id_mapping.get(scene_id, scene_id)
        scene_frame_dir = self.frame_dir / hash_id
        
        if not scene_frame_dir.exists():
            # Return dummy black frames
            return torch.zeros((self.max_frames, 3, 224, 224))
        
        # Get all frame files (sorted by name)
        frame_files = sorted(scene_frame_dir.glob('frame_*.jpg'))
        
        if len(frame_files) == 0:
            return torch.zeros((self.max_frames, 3, 224, 224))
        
        # Uniform sampling to get max_frames
        if len(frame_files) > self.max_frames:
            indices = np.linspace(0, len(frame_files) - 1, self.max_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        # Load and transform frames
        frames = []
        for frame_file in frame_files:
            try:
                img = Image.open(frame_file).convert("RGB")
                frame_tensor = self.frame_transform(img)
                frames.append(frame_tensor)
            except Exception as e:
                print(f"Warning: failed to load {frame_file}: {e}")
                continue
        
        if len(frames) == 0:
            return torch.zeros((self.max_frames, 3, 224, 224))
        
        # Stack frames
        frames_tensor = torch.stack(frames)  # (num_frames, 3, 224, 224)
        
        # Pad to max_frames if needed
        if frames_tensor.shape[0] < self.max_frames:
            padding = torch.zeros((self.max_frames - frames_tensor.shape[0], 3, 224, 224))
            frames_tensor = torch.cat([frames_tensor, padding], dim=0)
        
        return frames_tensor  # (max_frames, 3, 224, 224)

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if len(tokens) < self.opt.max_text_len:
            # pad with "unk"
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
            tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        else:
            # crop
            tokens = tokens[:self.opt.max_text_len]
            tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
            sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            word_emb, pos_oh = self.w_vectorizer[token]
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        idx = random.randint(0, len(motion) - m_length)
        motion = motion[idx:idx+m_length]

        "Z Normalization"
        # import pdb; pdb.set_trace()
        # print(f"Motion shape: {motion.shape}")
        # print(f"Mean shape: {self.mean.shape}")
        # print(f"Std shape: {self.std.shape}")
        motion = (motion - self.mean) / self.std


        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        # Load frames if enabled (for evaluation)
        if self.load_frames:
            scene_id = self.name_list[idx]
            keyframes = self.load_and_process_frames(scene_id)
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens), keyframes
        
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class Text2MotionDataset(data.Dataset):
    def __init__(self, opt, mean, std, split_file, load_frames=False, load_first_frame=False,
                 load_sparse_frames=False):
        self.opt = opt
        self.max_length = 20
        self.pointer = 0
        self.max_motion_length = opt.max_motion_length
        min_motion_len = 40 if self.opt.dataset_name =='t2m' else 24
        
        # Frame loading configuration
        self.load_frames = load_frames
        self.load_first_frame = load_first_frame
        self.load_sparse_frames = load_sparse_frames
        self.max_sparse_frames = 4  # Maximum number of sparse keyframes
        
        if load_frames or load_first_frame or load_sparse_frames:
            import json
            
            # Use configurable frame directory
            frame_dir_str = getattr(opt, 'frame_dir', 
                                    '/data4/haozhe/CamTraj/data/processed_estate/train_frames')
            self.frame_dir = Path(frame_dir_str)
            
            scene_id_mapping_path = Path(opt.data_root) / 'scene_id_mapping.json'
            with open(scene_id_mapping_path) as f:
                self.scene_id_mapping = json.load(f)
            
            print(f"Frame loading enabled. Frame dir: {self.frame_dir}")
            print(f"Scene ID mapping: {len(self.scene_id_mapping)} scenes")
            
            if load_first_frame or load_sparse_frames:
                # CLIP-standard preprocessing for visual conditioning
                self.first_frame_transform = transforms.Compose([
                    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    ),
                ])
                if load_first_frame:
                    print(f"First-frame (frame-0) conditioning: ENABLED")
                if load_sparse_frames:
                    print(f"Sparse keyframe conditioning (0-{self.max_sparse_frames} frames): ENABLED")

        data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        # id_list = id_list[:250]
        
        print(f"\n{'='*60}")
        print(f"Loading dataset: {opt.dataset_name}")
        print(f"Split file: {split_file}")
        print(f"Total scenes to load: {len(id_list)}")
        print(f"{'='*60}\n")

        new_name_list = []
        length_list = []
        if opt.dataset_name in CAMERA_DATASET_NAMES:
            # Camera dataset: text files have format "caption#tokens" (2 fields)
            for name in tqdm(id_list, desc="Loading motion & text data"):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            # print(line)
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = 0.0
                            to_tag = 0.0
                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(f"Error loading {name}: {e}")
                    pass
        else:
            # Human motion dataset: text files have format "caption#tokens#f_tag#to_tag" (4 fields)
            for name in tqdm(id_list):
                try:
                    motion = np.load(pjoin(opt.motion_dir, name + '.npy'))
                    if (len(motion)) < min_motion_len or (len(motion) >= 300):
                        continue
                    text_data = []
                    flag = False
                    with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:
                        for line in f.readlines():
                            text_dict = {}
                            line_split = line.strip().split('#')
                            # print(line)
                            caption = line_split[0]
                            tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                try:
                                    n_motion = motion[int(f_tag*20) : int(to_tag*20)]
                                    if (len(n_motion)) < min_motion_len or (len(n_motion) >= 300):
                                        continue
                                    new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    while new_name in data_dict:
                                        new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name
                                    data_dict[new_name] = {'motion': n_motion,
                                                           'length': len(n_motion),
                                                           'text':[text_dict]}
                                    new_name_list.append(new_name)
                                    length_list.append(len(n_motion))
                                except:
                                    print(line_split)
                                    print(line_split[2], line_split[3], f_tag, to_tag, name)
                                    # break

                    if flag:
                        data_dict[name] = {'motion': motion,
                                           'length': len(motion),
                                           'text': text_data}
                        new_name_list.append(name)
                        length_list.append(len(motion))
                except Exception as e:
                    print(f"Error loading {name}: {e}")
                    pass

        # name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))
        name_list, length_list = new_name_list, length_list

        self.mean = mean
        self.std = std
        self.length_arr = np.array(length_list)
        self.data_dict = data_dict
        self.name_list = name_list

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def load_scene_frames(self, scene_id, num_frames=None):
        """
        Load frame paths for a scene.
        
        Args:
            scene_id: Sequential ID (e.g., '000000')
            num_frames: Number of frames to return
                - If None: return ALL frame paths (for sparse keyframe sampling)
                - If int: uniformly sample that many frame paths
        
        Returns:
            frame_paths: List of Path objects to frame files (NOT loaded images!)
        """
        # Map scene_id to hash_id
        if scene_id not in self.scene_id_mapping:
            return []
        
        hash_id = self.scene_id_mapping[scene_id]
        scene_frame_dir = self.frame_dir / hash_id
        
        if not scene_frame_dir.exists():
            return []
        
        # Get all frame files, sorted by name
        frame_files = sorted(scene_frame_dir.glob('frame_*.jpg'))
        
        if num_frames is not None and len(frame_files) > num_frames:
            # Uniformly sample num_frames indices
            indices = np.linspace(0, len(frame_files) - 1, num_frames, dtype=int)
            frame_files = [frame_files[i] for i in indices]
        
        # Return ALL frame paths for sparse keyframe sampling
        # SparseKeyframeEncoder will randomly sample N∈[0,4] indices and load those images
        return frame_files

    def load_first_frame_image(self, scene_id):
        """Load and preprocess the first (index-0) frame for a scene.

        Args:
            scene_id: Sequential scene ID (e.g., '000000').

        Returns:
            Tensor (3, 224, 224) — CLIP-preprocessed RGB image,
            or zeros if the frame is unavailable.
        """
        if scene_id not in self.scene_id_mapping:
            return torch.zeros(3, 224, 224)

        hash_id = self.scene_id_mapping[scene_id]
        scene_frame_dir = self.frame_dir / hash_id

        if not scene_frame_dir.exists():
            return torch.zeros(3, 224, 224)

        frame_files = sorted(scene_frame_dir.glob('frame_*.jpg'))
        if len(frame_files) == 0:
            return torch.zeros(3, 224, 224)

        # Always take the first frame (index 0)
        try:
            img = Image.open(frame_files[0]).convert('RGB')
            return self.first_frame_transform(img)
        except Exception as e:
            print(f"Warning: failed to load first frame for {scene_id}: {e}")
            return torch.zeros(3, 224, 224)

    def load_sparse_frames_data(self, scene_id, motion_length, motion_start_idx=0):
        """Load 0-4 sparse keyframes for a scene.

        Randomly samples K ∈ [0, max_sparse_frames] unique frame indices from the
        trajectory, loads and preprocesses them. Returns fixed-shape tensors for
        PyTorch batching.

        Args:
            scene_id: Sequential scene ID (e.g., '000000').
            motion_length: Actual trajectory length (number of frames in motion).
            motion_start_idx: Starting index of the motion crop (for alignment).

        Returns:
            sparse_frames: (4, 3, 224, 224) tensor — K valid frames + (4-K) zeros.
            visual_valid_mask: (4,) bool tensor — True for valid frame slots.
            visual_indices: (4,) long tensor — frame indices in [0, motion_length),
                            0 for invalid slots.
        """
        max_k = self.max_sparse_frames  # 4
        
        # Initialize outputs with zeros/False
        sparse_frames = torch.zeros(max_k, 3, 224, 224)
        visual_valid_mask = torch.zeros(max_k, dtype=torch.bool)
        visual_indices = torch.zeros(max_k, dtype=torch.long)
        
        # Check if scene has frames
        if scene_id not in self.scene_id_mapping:
            return sparse_frames, visual_valid_mask, visual_indices
        
        hash_id = self.scene_id_mapping[scene_id]
        scene_frame_dir = self.frame_dir / hash_id
        
        if not scene_frame_dir.exists():
            return sparse_frames, visual_valid_mask, visual_indices
        
        frame_files = sorted(scene_frame_dir.glob('frame_*.jpg'))
        num_available_frames = len(frame_files)
        if num_available_frames == 0:
            return sparse_frames, visual_valid_mask, visual_indices
        
        # Randomly determine K ∈ [0, min(max_k, available_frames, motion_length)]
        max_possible_k = min(max_k, num_available_frames, motion_length)
        K = random.randint(0, max_possible_k)
        
        if K == 0:
            return sparse_frames, visual_valid_mask, visual_indices
        
        # Sample K unique frame indices from [0, motion_length)
        # These indices are relative to the cropped motion segment
        sampled_indices = sorted(random.sample(range(motion_length), K))
        
        # Load the sampled frames
        for slot_idx, frame_idx in enumerate(sampled_indices):
            # Map motion index to actual frame file index
            # (accounting for the motion crop start)
            actual_frame_idx = motion_start_idx + frame_idx
            
            # Clamp to available frames
            actual_frame_idx = min(actual_frame_idx, num_available_frames - 1)
            
            try:
                img = Image.open(frame_files[actual_frame_idx]).convert('RGB')
                sparse_frames[slot_idx] = self.first_frame_transform(img)
                visual_valid_mask[slot_idx] = True
                visual_indices[slot_idx] = frame_idx  # Index relative to motion
            except Exception as e:
                print(f"Warning: failed to load frame {actual_frame_idx} for {scene_id}: {e}")
                # Leave as zeros/False
        
        return sparse_frames, visual_valid_mask, visual_indices

    def __len__(self):
        return len(self.name_list) - self.pointer

    def __getitem__(self, item):
        idx = self.pointer + item
        data = self.data_dict[self.name_list[idx]]
        motion, m_length, text_list = data['motion'], data['length'], data['text']
        # Randomly select a caption
        text_data = random.choice(text_list)
        caption, tokens = text_data['caption'], text_data['tokens']

        if self.opt.unit_length < 10:
            coin2 = np.random.choice(['single', 'single', 'double'])
        else:
            coin2 = 'single'

        if coin2 == 'double':
            m_length = (m_length // self.opt.unit_length - 1) * self.opt.unit_length
        elif coin2 == 'single':
            m_length = (m_length // self.opt.unit_length) * self.opt.unit_length
        
        # Random temporal crop
        motion_start_idx = random.randint(0, len(motion) - m_length)
        motion = motion[motion_start_idx:motion_start_idx+m_length]

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1]))
                                     ], axis=0)
        
        # Load frames if enabled - OPTIMIZED: Only return paths, not loaded images
        if self.load_frames:
            scene_id = self.name_list[idx]
            frame_paths = self.load_scene_frames(scene_id, num_frames=None)
            
            # Return frame paths as list (will be loaded in SparseKeyframeEncoder after sampling)
            return caption, motion, m_length, frame_paths
        
        # Load sparse keyframes (0-4 frames) for sparse keyframe conditioning
        if self.load_sparse_frames:
            scene_id = self.name_list[idx]
            sparse_frames, visual_valid_mask, visual_indices = self.load_sparse_frames_data(
                scene_id, m_length, motion_start_idx)
            return caption, motion, m_length, sparse_frames, visual_valid_mask, visual_indices
        
        # Load first frame for first-frame conditioning
        if self.load_first_frame:
            scene_id = self.name_list[idx]
            first_frame = self.load_first_frame_image(scene_id)  # (3, 224, 224)
            return caption, motion, m_length, first_frame
        
        # print(word_embeddings.shape, motion.shape)
        # print(tokens)
        return caption, motion, m_length

    def reset_min_len(self, length):
        assert length <= self.max_motion_length
        self.pointer = np.searchsorted(self.length_arr, length)
        print("Pointer Pointing at %d" % self.pointer)


# ──────────────────── ID-Embedding Dataset Wrapper ────────────────────

class Text2MotionDatasetIDWrapped(data.Dataset):
    """Wraps a Text2MotionDataset to additionally return the sample index.

    For id_embedding conditioning, the model needs a per-sample integer index
    instead of (or alongside) the text caption.  This wrapper replaces the
    caption string with the integer index in the returned batch.

    The sample index runs from 0 to len(base_dataset)-1.  When the dataset
    has more samples than ``num_id_samples`` in the embedding table, indices
    are taken modulo ``num_id_samples``.

    Returned tuple:  (sample_idx: int, motion, m_length [, frame_paths])
    """

    def __init__(self, base_dataset: Text2MotionDataset, num_id_samples: int = None):
        self.base = base_dataset
        self.num_id_samples = num_id_samples

    # Forward dataset attributes expected by the training pipeline
    @property
    def opt(self):
        return self.base.opt

    @property
    def mean(self):
        return self.base.mean

    @property
    def std(self):
        return self.base.std

    def inv_transform(self, data):
        return self.base.inv_transform(data)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, item):
        out = self.base[item]
        # out is (caption, motion, m_length) or (caption, motion, m_length, frame_paths)
        sample_idx = item
        if self.num_id_samples is not None:
            sample_idx = item % self.num_id_samples
        # Replace caption with sample_idx
        return (sample_idx, *out[1:])


def collate_fn_text2motion_id_train(batch):
    """Collate function for id_embedding mode.

    Expected format: (sample_idx, motion, m_length)
    Returns: (sample_indices_tensor, motions_tensor, m_lengths_tensor)
    """
    batch.sort(key=lambda x: x[2], reverse=True)
    max_len = batch[0][2]

    sample_indices = []
    motions = []
    m_lengths = []

    for item in batch:
        sample_idx, motion, m_length = item[:3]
        if motion.shape[0] < max_len:
            padding = np.zeros((max_len - motion.shape[0], motion.shape[1]))
            motion = np.concatenate([motion, padding], axis=0)
        sample_indices.append(sample_idx)
        motions.append(motion)
        m_lengths.append(m_length)

    sample_indices = torch.tensor(sample_indices, dtype=torch.long)
    motions = torch.from_numpy(np.stack(motions, axis=0))
    m_lengths = torch.tensor(m_lengths)

    return sample_indices, motions, m_lengths