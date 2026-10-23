#!/bin/bash
# rtrans_25kfixed_noframe_r4: text-only (noFrame) residual transformer on the pose-fixed 25k data, on top of the
# fixed-data RVQ (rvq_w128_nq4_cb256_25kfixed_60ep, loads model/net_best_recon.tar = epoch 25).
# Hyperparameters = the paper run rtrans_25k_noframe_r4. LR schedule = MoMask default
# (milestones [50000], gamma 0.1, warm_up_iter 2000); lr 5e-5 as in the paper runs.
# Frozen T5 is now loaded pretrained and kept in eval mode (encoder-init fix).
#
# Frame tracks: EXTRA_ARGS='--use_first_frame --visual_drop_prob 0.2' (firstFrame) or
#               EXTRA_ARGS='--use_sparse_frames --visual_drop_prob 0.2' (sparseKeyframe), with a matching EXPERIMENT_NAME.
# Usage: GPU=4 bash configs/train_r_trans_fixed.sh
GPU=${GPU:-4}
DATASET_NAME="realestate10k_rotmat"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"rtrans_25kfixed_noframe_r4"}
echo "Experiment: ${EXPERIMENT_NAME}  GPU: ${GPU}"
echo "TensorBoard: ./log/res/${DATASET_NAME}/${EXPERIMENT_NAME}   Checkpoints: ./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}"

CUDA_VISIBLE_DEVICES=${GPU} python scripts/train_res_transformer.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${DATASET_NAME} \
    --data_root /data5/haozhe/CamTraj/data/posefix/RealEstate10K_rotmat_25k_fixed \
    --frame_dir /data5/haozhe/CamTraj/data/posefix/frames_25k \
    --batch_size 64 \
    --vq_name rvq_w128_nq4_cb256_25kfixed_60ep \
    --conditioning_mode t5 \
    --cond_drop_prob 0.2 \
    --latent_dim 384 \
    --ff_size 1024 \
    --n_layers 4 \
    --n_heads 6 \
    --dropout 0.2 \
    --max_epoch ${MAX_EPOCH:-600} \
    --lr 5e-5 \
    --evaluator_ckpt ./checkpoints/evaluator/clatr_universal_eval_RE10K_ET_GenDoP_9d_ld256_od512/best.pt \
    --vis_vel_integration \
    ${EXTRA_ARGS}

echo "Training completed!"
