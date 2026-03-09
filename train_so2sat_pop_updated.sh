#!/bin/bash
# Training script for So2Sat_POP dataset with UPDATED changes:
# 1. Simple random 50/50 labeled/unlabeled split (no stratification)
# 2. Fixed normalization: mean=1085, std=2800
# 3. Resize BEFORE RandAugment for better augmentation quality

echo "Starting training for So2Sat_POP dataset (UPDATED)..."

docker run -it --rm \
    --gpus '"device=8"' \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -v /home/ammar/data:/home/ammar/data \
    -v /work/ammar/sslrp/SimRegMatch/results:/workspace/results \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset so2sat_pop \
        --data_dir /home/ammar/data \
        --data-source sen2 \
        --labeled-ratio 0.4 \
        --normalize-labels \
        --batch_size 192 \
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
        --img_size 224 \
        --model efficientnet_b0 \
        --dropout 0.1 \
        --num-workers 8 \
        --seed 123 \
        --gpu 0

echo ""
echo "Training complete! Check results directory for outputs."
