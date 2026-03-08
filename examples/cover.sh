#!/bin/bash
# cover.sh – ACEStep cover-mode generation example
#
# Cover mode takes an existing song as a reference ("source audio") and
# generates a new song that shares its musical structure / timbre while
# following the caption you provide.
#
# Supported source-audio formats
# ─────────────────────────────────────────────────────────────────────
#  • WAV  – any bit depth / sample rate (16-bit PCM, 24-bit, float32 …)
#  • MP3  – any standard bit-rate
#
# Both formats are decoded and automatically resampled to 48 kHz stereo,
# which is the native rate of the ACEStep VAE encoder.  No pre-conversion
# with external tools (ffmpeg, sox, …) is needed.
#
# Usage
# ─────────────────────────────────────────────────────────────────────
#  1. Set SRC_AUDIO to the path of your reference track (WAV or MP3).
#  2. Edit cover.json to describe the style you want.
#  3. Run:  bash cover.sh
#
# The output file will be named cover0.wav (same stem as cover.json).
#
# Optional: adjust cover strength (0.0 = ignore reference, 1.0 = max)
#   add  --cover-strength 0.7  to the dit-vae invocation below.

set -eu

# ── user settings ────────────────────────────────────────────────────
SRC_AUDIO="my_reference.mp3"   # path to your source WAV or MP3 file
# ─────────────────────────────────────────────────────────────────────

# Step 1: run the DiT+VAE pipeline with the reference audio (no LLM needed)
#
#   --src-audio accepts WAV or MP3; any sample rate is accepted and
#   automatically converted to 48 kHz stereo before encoding.
../build/dit-vae \
    --request cover0.json \
    --text-encoder ../models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-turbo-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf \
    --src-audio "${SRC_AUDIO}"
