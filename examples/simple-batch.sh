#!/bin/bash
# Generate 2 LM variations, then 2 DiT variations per LM output = 4 total WAVs
#
# LM phase:
# simple.json -> simple0.json, simple1.json
#
# DiT phase:
# simple0.json -> simple00.wav, simple01.wav
#            simple1.json -> simple10.wav, simple11.wav

set -eu

# Phase 1: LM generates 2 code variations
../build/ace-qwen3 \
    --request simple.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
    --batch 2

# Phase 2: DiT+VAE generates 2 noise variations per LM output (single call)
../build/dit-vae \
    --request simple0.json simple1.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --batch 2
