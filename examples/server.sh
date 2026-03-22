#!/bin/bash
# full server: all pipelines loaded (~19 GB VRAM).
# /lm, /synth, /understand all available.
# /synth runs concurrently with /lm (auto, disjoint GPU mem).

set -eu

../build/ace-server \
    --host 0.0.0.0 \
    --port 8085 \
    --lm ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
    --embedding ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-sft-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --max-batch 2 \
    --sleep 60
