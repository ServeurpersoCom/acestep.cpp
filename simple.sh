#!/bin/bash
# Inspiration: query -> LLM metadata+lyrics+codes -> DiT -> WAV

set -eu
SEED="${SEED:--1}"

mkdir -p /tmp/ace

./build/ace-qwen3 --model checkpoints/acestep-5Hz-lm-4B \
    --system $'# Instruction\nExpand the user\'s input into a more detailed and specific musical description:\n' \
    --user $'Une chanson française sur la ville de Paris\n\ninstrumental: false' \
    --fsm --cfg-scale 2.2 \
    --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed "$SEED"

./build/dit-vae \
    --input-dir /tmp/ace \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo --vae checkpoints/vae \
    --seed "$SEED" --output simple.wav
