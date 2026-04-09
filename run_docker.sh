#!/bin/bash
# Docker run script with memory limits to prevent segmentation faults

echo "Building and running LexGuard API with memory limits..."

# Build the image
docker build -t lexguard-api .

# Run with memory limits and single-threaded execution
docker run \
  --memory=4g \
  --cpus=1.0 \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env NUMEXPR_NUM_THREADS=1 \
  --env OPENBLAS_NUM_THREADS=1 \
  --volume "$(pwd)/data:/home/app/data:ro" \
  --publish 8000:8000 \
  --name lexguard-container \
  lexguard-api

echo "API should be running at http://localhost:8000"
echo "Test with: curl http://localhost:8000/health"