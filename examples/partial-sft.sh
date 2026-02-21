#!/bin/bash

set -eu

../build/ace-qwen3 \
    --request partial-sft.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf

../build/dit-vae \
    --request partial-sft0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
