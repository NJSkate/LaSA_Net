#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOW_CONV_ANYSIZE_DEPTHWISE=1

python tools/test.py configs/lasanet_multi3drefer.yaml \
    --gpu_id 0 --num_gpus 1 \
    --checkpoint checkpoints/lasanet_multi3drefer.pth
