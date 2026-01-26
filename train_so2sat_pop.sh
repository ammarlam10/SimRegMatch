#!/bin/bash
# Training script for So2Sat_POP dataset

echo "Starting training for So2Sat_POP dataset..."

docker run -it --rm \
    --gpus '"device=5"' \
    -v $(pwd):/workspace \
    -v /work/ammar/sslrp/data:/workspace/data \
    -v /work/ammar/sslrp/SimRegMatch/results:/workspace/results \
    -v /work/ammar/sslrp/UCVME:/workspace/ucvme \
    simregmatch:latest \
    python /workspace/main.py \
        --dataset so2sat_pop \
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
        --img_size 224 \
        --model resnet50 \
        --dropout 0.1 \
        --seed 0 \
        --gpu 0
