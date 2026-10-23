"""
On-disk layout of TKCAM trajectory datasets.

    <data_root>/
        trajectories/*.npy    # (T, D) camera trajectory per sample
        texts/*.txt           # "caption#tokens" lines
        untagged_text/*.txt   # raw captions
        metadata/*.json
        train.txt / val.txt / test.txt
        Mean.npy / Std.npy

Datasets built before the rename store trajectories in ``new_joint_vecs/``
(the name inherited from MoMask / HumanML3D). Readers go through
``traj_dir`` so both layouts keep working; writers always use TRAJ_DIR.
"""

import os

TRAJ_DIR = 'trajectories'
LEGACY_TRAJ_DIR = 'new_joint_vecs'


def traj_dir(data_root):
    """Trajectory directory under data_root, falling back to the legacy name."""
    path = os.path.join(data_root, TRAJ_DIR)
    legacy = os.path.join(data_root, LEGACY_TRAJ_DIR)
    if not os.path.isdir(path) and os.path.isdir(legacy):
        return legacy
    return path
