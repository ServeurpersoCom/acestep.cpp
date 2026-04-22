#!/bin/bash

set -eu

# Multi-GPU: set GGML_BACKEND to pick a device (CUDA0, CUDA1, Vulkan0...)
#export GGML_BACKEND=CUDA0
#export GGML_BACKEND=Vulkan0

./build/ace-server \
    --host 0.0.0.0 \
    --port 8085 \
    --models ./models \
    --loras ./loras \
    --max-batch 1
    #--mp3-bitrate 128    # MP3 encoding bitrate in kbps (valid: 32,40,48,56,64,80,96,112,128,160,192,224,256,320)
