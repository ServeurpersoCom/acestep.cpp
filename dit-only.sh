#!/bin/bash
# DiT-only: all metas -> no LLM codes -> DiT generates from noise

set -eu

export SEED="${SEED:--1}" OUT="${OUT:-dit-only.wav}"
exec ./generate.sh \
    --caption "Ambient electronic soundscape with warm analog pads" \
    --lyrics "[Instrumental]" \
    --bpm 90 --duration 180 --keyscale "C minor" --timesignature 4 --language en \
    --no-codes
