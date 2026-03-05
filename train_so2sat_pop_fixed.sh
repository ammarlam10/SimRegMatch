#!/bin/bash
# Training script for So2Sat_POP dataset with 40% labeled data (FIXED - no softplus bug)

echo "Starting training for So2Sat_POP dataset with 40% labeled data..."

docker run -it --rm \
    --gpus '"device=1,4"' \
    -v $(pwd):/workspace \
    -v /work/ammar/sslrp/data:/workspace/data \
    -v /work/ammar/sslrp/SimRegMatch/results:/workspace/results \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset so2sat_pop \
        --data-source sen2 \
        --data_dir /workspace/data \
        --labeled-ratio 0.4 \
        --batch_size 128 \
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
        --seed 0 \
        --gpu 0 \
        --normalize-labels
