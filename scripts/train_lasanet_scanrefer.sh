#!/bin/bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_ENDPOINT=https://hf-mirror.com
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOW_CONV_ANYSIZE_DEPTHWISE=1

python tools/train.py configs/lasanet_scanrefer.yaml \
    --gpu_ids 0 --num_gpus 1
