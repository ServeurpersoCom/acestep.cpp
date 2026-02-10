#!/bin/bash
# Standard: all metas on CLI -> LLM codes -> DiT -> WAV

set -eu
SEED="${SEED:-42}"

CAPTION="Vibrant French house meets tech-house fusion track featuring filtered disco samples, \
driving funky basslines, and classic four-on-the-floor beats with signature Bob Sinclar vocal chops"

LYRICS='[Intro]

[Verse 1]
Optimisation des poids synaptiques en temps reel
Reseaux neuronaux convolutifs, profondeur ideale
Backpropagation stochastique, gradient ajuste
Modeles GAN generatifs, data-set finalise

[Chorus]
Deep learning en action, reseau fully connected
Processing en temps reel, le futur est lance'

./ace-qwen3 checkpoints/acestep-5Hz-lm-4B \
    --caption "$CAPTION" --lyrics "$LYRICS" \
    --bpm 124 --duration 220 --keyscale "F# minor" --timesignature 4 --language fr \
    --fsm --cfg-scale 2.2 \
    --output-codes /tmp/codes.txt --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed "$SEED"

./dit-vae \
    --caption "$(cat /tmp/ace/caption)" \
    --lyrics "$(cat /tmp/ace/lyrics)" \
    --bpm "$(cat /tmp/ace/bpm)" \
    --duration "$(cat /tmp/ace/duration)" \
    --keyscale "$(cat /tmp/ace/keyscale)" \
    --timesignature "$(cat /tmp/ace/timesignature)" \
    --language "$(cat /tmp/ace/language)" \
    --input-codes /tmp/codes.txt \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo --vae checkpoints/vae \
    --seed "$SEED" --output full.wav
