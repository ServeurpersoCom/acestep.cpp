#!/bin/bash

set -eu

../build/ace-qwen3 \
    --request simple.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
    --batch 4

../build/dit-vae \
    --request simple0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --output simple-batch0.wav

../build/dit-vae \
    --request simple1.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --output simple-batch1.wav

../build/dit-vae \
    --request simple2.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --output simple-batch2.wav

../build/dit-vae \
    --request simple3.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --output simple-batch3.wav
