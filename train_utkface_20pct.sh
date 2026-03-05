#!/bin/bash
# Training script for UTKFace dataset with ImageNet pretrained weights
# 20% labeled ratio (0.2) with ImageNet normalization

echo "Starting training for UTKFace dataset (20% labeled ratio)..."

docker run -it --rm \
    --gpus '"device=2,3"' \
    --shm-size=16g \
    -v $(pwd):/workspace \
    -v /home/ammar/tree/UTKFace:/workspace/data \
    -v /home/ammar/tree/SimRegMatch/results:/workspace/results \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset utkface \
        --data_dir /workspace/data \
        --labeled-ratio 0.2 \
        --batch_size 256 \
        --epochs 100 \
        --lr 0.002 \
        --optimizer adam \
        --loss mse \
        --lambda-u 0.01 \
        --beta 0.6 \
        --t 1.0 \
        --percentile 0.95 \
        --threshold 10 \
        --iter-u 5 \
        --img_size 224 \
        --model resnet50 \
        --dropout 0.1 \
        --num_workers 4 \
        --seed 0 \
        --gpu 0
