#!/bin/bash

set -eu

cp dit-only.json /tmp/request.json

../build/dit-vae \
    --request /tmp/request.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-bf16.gguf \
    --dit ../models/acestep-v15-turbo-bf16.gguf \
    --vae ../models/vae-bf16.gguf \
    --output dit-only.wav
