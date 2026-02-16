#!/bin/bash

set -eu

cp dit-only-sft.json /tmp/request.json

../build/dit-vae \
    --request /tmp/request.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-bf16.gguf \
    --dit ../models/acestep-v15-sft-bf16.gguf \
    --vae ../models/vae-bf16.gguf \
    --output dit-only-sft.wav
