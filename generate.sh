#!/bin/bash
# Unified ACE-Step C++ pipeline: LLM -> DiT+VAE -> WAV
# All args are passed to ace-qwen3 (it handles mode detection internally)
#
# Usage:
#   ./generate.sh --query "Une chanson française sur Paris"
#   ./generate.sh --query "ambient piano" --instrumental
#   ./generate.sh --caption "French house..." --lyrics "..." --bpm 124 --duration 220 --keyscale "F# minor" --timesignature 4 --language fr
#   ./generate.sh --caption "Ambient piano" --duration 180

set -eu

# Model paths (override via env)
LM="${ACE_LM:-checkpoints/acestep-5Hz-lm-4B}"
DIT="${ACE_DIT:-checkpoints/acestep-v15-turbo}"
TE="${ACE_TE:-checkpoints/Qwen3-Embedding-0.6B}"
VAE="${ACE_VAE:-checkpoints/vae}"
SEED="${SEED:--1}"
OUT="${OUT:-output.wav}"
TMP="${TMP:-/tmp/ace}"
BIN="${BIN:-./build}"

mkdir -p "$TMP"

# LLM: metadata + lyrics + audio codes
"$BIN/ace-qwen3" --model "$LM" --output-dir "$TMP" --seed "$SEED" "$@"

# DiT + VAE: flow matching -> WAV
"$BIN/dit-vae" --input-dir "$TMP" \
    --text-encoder "$TE" --dit "$DIT" --vae "$VAE" \
    --seed "$SEED" --output "$OUT"

echo "Output: $OUT"
