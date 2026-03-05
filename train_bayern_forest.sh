#!/bin/bash
# Training script for Bayern Forest Height dataset

echo "Starting training for Bayern Forest Height dataset..."

docker run -it --rm \
    --gpus '"device=0"' \
    --shm-size=16g \
    -v $(pwd):/workspace \
    -v /home/ammar/data:/workspace/data \
    -v /home/ammar/tree/SimRegMatch/results:/workspace/results \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset bayern_forest \
        --data_dir /workspace/data \
        --labeled-ratio 0.1 \
        --batch_size 16 \
        --epochs 100 \
        --lr 0.001 \
        --optimizer adam \
        --loss mse \
        --lambda-u 0.01 \
        --beta 0.6 \
        --t 1.0 \
        --percentile 0.95 \
        --threshold 10 \
        --iter-u 5 \
        --img_size 256 \
        --model unet \
        --dropout 0.1 \
        --normalize-labels \
        --num_workers 4 \
        --seed 0 \
        --gpu 0
