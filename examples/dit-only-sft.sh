#!/bin/bash

set -eu

../build/dit-vae \
    --request dit-only-sft.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
