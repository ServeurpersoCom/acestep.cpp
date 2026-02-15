#!/bin/bash

set -eu

cp dit-only.json /tmp/request.json

../build/dit-vae   --request /tmp/request.json --output dit-only.wav \
    --dit ../checkpoints/acestep-v15-turbo \
    --text-encoder ../checkpoints/Qwen3-Embedding-0.6B \
    --vae ../checkpoints/vae
