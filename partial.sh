#!/bin/bash
# Custom mode (partial metas): caption+lyrics -> LLM fills bpm/key/etc via CoT -> codes -> DiT -> WAV

set -eu

export SEED="${SEED:--1}" OUT="${OUT:-partial.wav}"
exec ./generate.sh --caption "Ambient electronic soundscape with warm analog pads" --lyrics "[Instrumental]" --duration 180
