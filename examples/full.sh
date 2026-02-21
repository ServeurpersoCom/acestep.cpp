#!/bin/bash

set -eu

../build/ace-qwen3 \
    --request full.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf

../build/dit-vae \
    --request full0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
