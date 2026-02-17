#!/bin/bash

set -eu

cp partial.json /tmp/request.json

../build/ace-qwen3 \
    --request /tmp/request.json \
    --model ../models/acestep-5Hz-lm-4B-Q6_K.gguf

../build/dit-vae \
    --request /tmp/request.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q6_K.gguf \
    --vae ../models/vae-BF16.gguf \
    --output partial.wav
