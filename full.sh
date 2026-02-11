#!/bin/bash
# Standard: all metas on CLI -> LLM codes -> DiT -> WAV

set -eu
SEED="${SEED:--1}"

CAPTION="Vibrant French house meets tech-house fusion track featuring filtered disco samples, driving funky basslines, and classic four-on-the-floor beats with signature Bob Sinclar vocal chops. Analytical yet euphoric mood blending advanced AI technical vocabulary with dancefloor energy. Instruments include talkbox lead vocals, analog Moog bass synths, glitchy arpeggiated sequencers, punchy TR-808 drum machine, and shimmering high-hat rolls. Production style showcases crisp retro-modern mix with dynamic sidechain compression, warm vinyl crackle, and modular synth modulations. Vocal delivery combines rhythmic French rap verses with melodic, pitch-shifted choruses celebrating machine learning breakthroughs"

LYRICS='[Intro - Filtered Disco Sample & Synth Arp]

[Verse 1]
Optimisation des poids synaptiques en temps réel
Réseaux neuronaux convolutifs, profondeur idéale
Backpropagation stochastique, gradient ajusté
Modèles GAN génératifs, data-set finalisé

[Pre-Chorus]
Latence zéro, flux continu
L'\''IA évolue, circuit virtuel

[Chorus - Talkbox Vocals]
C'\''est l'\''ère de l'\''intelligence artificielle
Algorithmes de backpropagation dansent sur le beat
Deep learning en action, réseau fully connected
Processing en temps réel, le futur est lancé !

[Verse 2]
Transformers multi-tâches, attention mécanique
Zones de pooling maximisées, features extraites
Loss function minimisée, convergence assurée
Overfitting évité, dataset équilibré

[Chorus - Talkbox Vocals]
C'\''est l'\''ère de l'\''intelligence artificielle (AI!)
Algorithmes de backpropagation dansent sur le beat
Deep learning en action, réseau fully connected
Processing en temps réel, le futur est lancé !

[Bridge - Synth Pad Build-Up]
Stochastic gradient descent, optimise le modèle
Early stopping activé, le réseau se stabilise
Cloud computing massif, calcul distribué
L'\''IA révolutionne, monde transformé

[Outro - Vinyl Crackles & Fade Out]
IA... Intelligence Artificielle...
Réseaux neuronaux... dansent... forever...'

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
