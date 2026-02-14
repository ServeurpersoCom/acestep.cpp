#!/bin/bash
# DiT-only SFT: all metas -> no LLM codes -> DiT (CFG 7.0) generates from noise

set -eu

export SEED="${SEED:--1}" OUT="${OUT:-dit-only-sft.wav}"
export ACE_DIT="${ACE_DIT:-checkpoints/acestep-v15-sft}"
export SHIFT=1.0 STEPS=50 GUIDANCE_SCALE=7.0

exec ./generate.sh \
    --caption "Ambient electronic soundscape with warm analog pads" \
    --lyrics "[Instrumental]" \
    --bpm 90 --duration 180 --keyscale "C minor" --timesignature 4 --language en \
    --no-codes
