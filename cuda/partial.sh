#!/bin/bash
# Partial-metas: caption+lyrics only -> LLM fills bpm/key/etc via CoT -> codes -> DiT -> WAV

set -eu
SEED="${SEED:--1}"

CAPTION="TODO"
LYRICS="[Instrumental]"

mkdir -p /tmp/ace

./ace-qwen3 --model ../checkpoints/acestep-5Hz-lm-4B \
    --caption "$CAPTION" --lyrics "$LYRICS" --duration 180 \
    --fsm --cfg-scale 2.2 \
    --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed "$SEED"

./dit-vae \
    --input-dir /tmp/ace \
    --text-encoder ../checkpoints/Qwen3-Embedding-0.6B \
    --dit ../checkpoints/acestep-v15-turbo --vae ../checkpoints/vae \
    --seed "$SEED" --output partial.wav
