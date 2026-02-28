#!/bin/bash
# Test reference_audio (WAV) and audio_cover_strength.
# Put a WAV file at reference.wav (or set reference_audio path in request-reference.json).
# Requires: built dit-vae, --vae with encoder weights, and models in ../models/.

set -eu
cd "$(dirname "$0")"

if [ ! -f "reference.wav" ]; then
    echo "No reference.wav found. Copy a WAV file to reference.wav (stereo 48kHz or any rate; will be resampled)."
    echo "Then run: $0"
    exit 1
fi

../build/dit-vae \
    --request request-reference.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf

echo "Done. Check request-reference0.wav (and request-reference1.wav if --batch 2)."
