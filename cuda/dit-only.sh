#!/bin/bash
# DiT-only: all metas -> no LLM codes -> DiT generates from noise

set -eu
SEED="${SEED:--1}"

CAPTION="Ambient electronic soundscape with warm analog pads"
LYRICS="[Instrumental]"

mkdir -p /tmp/ace

./ace-qwen3 --model ../checkpoints/acestep-5Hz-lm-4B \
    --caption "$CAPTION" --lyrics "$LYRICS" \
    --bpm 90 --duration 180 --keyscale "C minor" --timesignature 4 --language en \
    --fsm --cfg-scale 2.2 --no-codes \
    --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed "$SEED"

./dit-vae \
    --input-dir /tmp/ace \
    --text-encoder ../checkpoints/Qwen3-Embedding-0.6B \
    --dit ../checkpoints/acestep-v15-turbo --vae ../checkpoints/vae \
    --seed "$SEED" --output dit-only.wav
