#!/bin/bash

set -eu

cp full-sft.json /tmp/request.json

../build/ace-qwen3 \
    --request /tmp/request.json \
    --model ../models/acestep-5Hz-lm-4B-bf16.gguf

../build/dit-vae \
    --request /tmp/request.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-bf16.gguf \
    --dit ../models/acestep-v15-sft-bf16.gguf \
    --vae ../models/vae-bf16.gguf \
    --output full-sft.wav
