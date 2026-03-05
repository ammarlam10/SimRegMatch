#!/bin/bash
# Bayern Forest Height: 5% labeled, GPU 2
# Parameters from experiment_13
# Usage: ./train_bayern_forest.sh [seed]

set -e

GPU_ID=2
SEED=${1:-0}

echo "=========================================="
echo "SimRegMatch: Bayern Forest 5% labeled"
echo "=========================================="
echo "GPU: $GPU_ID | Seed: $SEED | Labeled: 5%"
echo "Params from experiment_13"
echo ""

docker run -it --rm \
    --gpus "device=${GPU_ID}" \
    --shm-size=16g \
    -v $(pwd):/workspace \
    -v ~/data:/workspace/data \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset simreg_bayern_forest \
        --data-source sen2 \
        --data_dir /workspace/data \
        --labeled-ratio 0.05 \
        --batch_size 64 \
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
        --model unet-small \
        --dropout 0.1 \
        --gpu 0 \
        --normalize-labels \
        --num-workers 4 \
        --seed "$SEED"
