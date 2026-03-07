#!/bin/bash
# LoRA example: generate with a PEFT LoRA adapter (e.g. duckdbot/acestep-lora-cryda).
# Requires adapter_model.safetensors in lora/ (download once; see below).
set -eu
cd "$(dirname "$0")"

ADAPTER="lora/adapter_model.safetensors"
if [ ! -f "$ADAPTER" ]; then
    echo "LoRA adapter not found at $ADAPTER"
    exit 1
fi

# LLM: fill lyrics + codes
../build/ace-qwen3 \
    --request lora.json \
    --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf

# DiT+VAE with LoRA (scale = alpha/rank; 1.0 is typical)
../build/dit-vae \
    --request lora0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --lora "$ADAPTER" \
    --lora-scale 1.0

echo "Done. Check lora00.wav"
