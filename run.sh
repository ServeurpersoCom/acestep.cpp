#!/bin/bash

set -eu

# LM
./ace-qwen3 checkpoints/acestep-5Hz-lm-4B \
    --prompt-json prompt.json \
    --output-codes /tmp/codes.txt \
    --cfg-scale 2.2 \
    --temperature 0.80 \
    --top-p 0.9

# TextEnc -> CondEnc -> DiT -> VAE -> WAV
./pipeline \
    --prompt prompt.json \
    --input-codes /tmp/codes.txt \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo \
    --vae checkpoints/vae \
    --output output.wav
