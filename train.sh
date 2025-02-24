#!/bin/bash
#$ -q gpu
#$ -l gpu_card=4
#$ -N NbIrisSeg


# Activate your conda custom environment
source activate infants

# pip install --pre torchvision
# pip install madgrad
# pip install opencv-python
# pip install matplotlib

# Source directory
cd ./NbIrisSeg

# Train your custom dataset by running train.py
python train.py \
    --multi_gpu \
    --cuda \
    --cudnn \
    --train_image_dir_mateusz='./data/SegNetWarm-Mateusz-coarse/all-images/' \
    --train_mask_dir_mateusz='./data/SegNetWarm-Mateusz-coarse/all-masks/' \
    --train_image_dir_openeds='./data/Piotr-NB-Dataset/train-data/all-images/' \
    --train_mask_dir_openeds='./data/Piotr-NB-Dataset/train-data/all-masks/' \
    --test_image_dir_newborn='./data/Piotr-NB-Dataset/test-data/all-images/' \
    --test_mask_dir_newborn='./data/Piotr-NB-Dataset/test-data/all-masks/' \
    --circle_model_path='./models/convnext_tiny-1076-0.030622-maskIoU-0.938355.pth' \
    --seg_model_name='nestedsharedatrousresunet' \
    --num_classes=1 \
    --num_channels=1 \
    --width=64 \
    --circle_model_name='convnext' \
    --mode='train' \
    --aug_num_repetitions=2 \
    --batch_size=32 \
    --log_batch=10 \
    --loss_type='cross_entropy+dice' \
    --lr=0.001 \
    --solver_name='madgrad' \
    --num_epochs=300 \
    --num_workers=0 \
    --log_txt \
    --tag='coarsemasks' \
    --label_smoothing
    # --state='model_state.pth' \
    # --pupil_pixel_range 109 200 \
    # --gpu=$CUDA_VISIBLE_DEVICES \
