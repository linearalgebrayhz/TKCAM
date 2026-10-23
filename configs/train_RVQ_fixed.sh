#!/bin/bash
# RVQ on the pose-fixed RealEstate10K-Cap 25k data (convention fix + duplicated first frame removed).
# Same hyperparameters as train_RVQ.sh; checkpoint selection by masked reconstruction
# (net_best_recon.tar, see utils/clatr_camera_eval.py: evaluation_camera_vqvae_clatr), CLaTr
# metrics logged. With milestones 150K/250K the LR is 2.5e-7 after 250K iterations; the previous
# 25k run (rvq_window128_25k_4quantizers) plateaued at ~300K iterations (~epoch 50), so 60 epochs
# (~365K iterations at ~6.1K iterations/epoch) covers convergence with margin.
#
# Usage: GPU=7 MAX_EPOCH=60 bash configs/train_RVQ_fixed.sh
LOG_DIR="./log"

GPU=${GPU:-7}
MAX_EPOCH=${MAX_EPOCH:-60}
DATASET_NAME="realestate10k_rotmat"
DATA_ROOT="/data5/haozhe/CamTraj/data/posefix/RealEstate10K_rotmat_25k_fixed"
EVALUATOR_CKPT="./checkpoints/evaluator/clatr_universal_eval_RE10K_ET_GenDoP_9d_ld256_od512/best.pt"
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"rvq_w128_nq4_cb256_25kfixed_${MAX_EPOCH}ep"}
TENSORBOARD_DIR="./log/vq/${DATASET_NAME}/${EXPERIMENT_NAME}"
CHECKPOINT_DIR="./checkpoints/${DATASET_NAME}/${EXPERIMENT_NAME}"

echo "Starting Camera RVQ Training (fixed data)..."
echo "- Experiment name: ${EXPERIMENT_NAME}"
echo "- GPU: ${GPU}, max_epoch: ${MAX_EPOCH}"
echo "- TensorBoard logs: ${TENSORBOARD_DIR}"
echo "- Model checkpoints: ${CHECKPOINT_DIR}"

CUDA_VISIBLE_DEVICES=${GPU} python scripts/train_vq.py \
    --name ${EXPERIMENT_NAME} \
    --gpu_id 0 \
    --dataset_name ${DATASET_NAME} \
    --data_root ${DATA_ROOT} \
    --evaluator_ckpt ${EVALUATOR_CKPT} \
    --batch_size 256 \
    --num_quantizers 4 \
    --max_epoch ${MAX_EPOCH} \
    --quantize_dropout_prob 0.2 \
    --gamma 0.05 \
    --lr 1e-4 \
    --warm_up_iter 500 \
    --commit 0.02 \
    --loss_vel 0.5 \
    --recons_loss l1_smooth \
    --log_every 100 \
    --log_codebook_usage \
    --log_gradients \
    --window_size 128 \
    --loss_smoothness 0.0 \
    --num_vis_samples 2 \
    --code_dim 512  \
    --output_emb_width 512  \
    --nb_code 256 \
    --vis_vel_integration

echo "RVQ Training completed!"
