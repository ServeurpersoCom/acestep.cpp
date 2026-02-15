#!/bin/bash

set -eu

cp simple.json /tmp/request.json

../build/ace-qwen3 --request /tmp/request.json --model ../checkpoints/acestep-5Hz-lm-4B

../build/dit-vae   --request /tmp/request.json --output simple.wav \
    --dit ../checkpoints/acestep-v15-turbo \
    --text-encoder ../checkpoints/Qwen3-Embedding-0.6B \
    --vae ../checkpoints/vae
