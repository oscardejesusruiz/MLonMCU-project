#!/bin/bash

# Simple script to run post-training quantization

echo "Starting post-training quantization..."

# Create output directories if they don't exist
mkdir -p outputs_2
mkdir -p trained_models_2

# Run the quantization script
python quantize_models.py \
    --source-dir trained_models_2 \
    --output-dir outputs_2 \
    --batch-size 64 \
    --n-iter 10 \
    --skip-export

echo "Quantization completed. Check outputs_2 directory for results."