#!/bin/bash
# Simple mode: query -> LLM inspiration + codes -> DiT -> WAV

set -eu

export SEED="${SEED:--1}" OUT="${OUT:-simple.wav}"
exec ./generate.sh --query "Une chanson française sur la ville de Paris"
