#!/bin/bash
# Lego test: three-step self-contained pipeline.
#
# step zero: download the base DiT model if not already present
#            (lego requires acestep-v15-base; turbo/sft do not support it)
# step one:  generate a track from the simple prompt
# step two:  apply lego guitar to that generated track

set -eu

# Step 1: generate a source track with the simple prompt
../build/ace-qwen3 \
    --request simple.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf

../build/dit-vae \
    --request simple0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --wav

# Step 2: lego guitar on the generated track (base model required)
../build/dit-vae \
    --src-audio simple00.wav \
    --request lego.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-base-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --wav
