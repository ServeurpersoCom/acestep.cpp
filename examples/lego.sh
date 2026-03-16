#!/bin/bash
# Generate a source track, then lego a guitar stem over it
#
# Note: lego requires acestep-v15-base; turbo/sft do not support it
#
# LM + DiT phase (source track):
# simple.json -> simple0.json -> simple00.wav
#
# Lego phase (guitar stem over source):
# lego.json + simple00.wav -> lego0.wav

set -eu

# Phase 1: generate a source track with the simple prompt
../build/ace-lm \
    --request simple.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf

../build/ace-synth \
    --request simple0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --wav

# Phase 2: lego guitar on the generated track (base model required)
../build/ace-synth \
    --src-audio simple00.wav \
    --request lego.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-base-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --wav
