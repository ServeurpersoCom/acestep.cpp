#!/bin/bash

set -eu

cp dit-only-sft.json /tmp/request.json

../build/dit-vae   --request /tmp/request.json --output dit-only-sft.wav \
    --dit ../checkpoints/acestep-v15-sft \
    --text-encoder ../checkpoints/Qwen3-Embedding-0.6B \
    --vae ../checkpoints/vae
