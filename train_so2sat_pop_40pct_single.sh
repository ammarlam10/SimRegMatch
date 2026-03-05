#!/bin/bash
# Single interactive training run for So2Sat_POP 40% labeled data.
# Run this in separate tmux panes/windows for parallel runs.
#
# Usage: ./train_so2sat_pop_40pct_single.sh <gpu_id> <seed>
# Example:
#   tmux pane 1: ./train_so2sat_pop_40pct_single.sh 0 0
#   tmux pane 2: ./train_so2sat_pop_40pct_single.sh 1 42
#   tmux pane 3: ./train_so2sat_pop_40pct_single.sh 2 123

set -e

GPU_ID=${1:-0}
SEED=${2:-0}

echo "=========================================="
echo "SimRegMatch: So2Sat_POP 40% (interactive)"
echo "=========================================="
echo "GPU: $GPU_ID | Seed: $SEED"
echo ""

docker run -it --rm \
    --gpus "device=${GPU_ID}" \
    --shm-size=8g \
    -v $(pwd):/workspace \
    -v /work/ammar/sslrp/data:/workspace/data \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset so2sat_pop \
        --data-source sen2 \
        --data_dir /workspace/data \
        --labeled-ratio 0.4 \
        --batch_size 64 \
        --epochs 200 \
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
        --gpu 0 \
        --normalize-labels \
        --use-cache \
        --seed "$SEED"
