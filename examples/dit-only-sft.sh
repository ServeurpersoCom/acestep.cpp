#!/bin/bash

set -eu

cp dit-only-sft.json request.json

../build/dit-vae \
    --request request.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q6_K.gguf \
    --vae ../models/vae-BF16.gguf \
    --output dit-only-sft.wav
