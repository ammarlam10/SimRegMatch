#!/bin/bash
# Build and run the Docker container

echo "Building Docker image..."
docker build -t simregmatch:latest .

echo "Running Docker container..."
docker run -it --rm \
    --gpus '"device=5"' \
    -v $(pwd):/workspace \
    -v /work/ammar/sslrp/data:/workspace/data \
    -v /work/ammar/sslrp/SimRegMatch/results:/workspace/results \
    -v /work/ammar/sslrp/UCVME:/workspace/ucvme \
    --name simregmatch \
    simregmatch:latest \
    /bin/bash
