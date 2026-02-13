#!/bin/bash
# Full pipeline: LLM (ace-qwen3 GGML) -> DiT+VAE (GGML) -> WAV

set -eu

SEED="${SEED:--1}"

CAPTION="Vibrant French house meets tech-house fusion track featuring filtered disco samples, driving funky basslines, and classic four-on-the-floor beats with signature Bob Sinclar vocal chops. Analytical yet euphoric mood blending advanced AI technical vocabulary with dancefloor energy. Instruments include talkbox lead vocals, analog Moog bass synths, glitchy arpeggiated sequencers, punchy TR-808 drum machine, and shimmering high-hat rolls. Production style showcases crisp retro-modern mix with dynamic sidechain compression, warm vinyl crackle, and modular synth modulations. Vocal delivery combines rhythmic French rap verses with melodic, pitch-shifted choruses celebrating machine learning breakthroughs"

LYRICS="[Intro - Filtered Disco Sample & Synth Arp]

[Verse 1]
Optimisation des poids synaptiques en temps reel
Reseaux neuronaux convolutifs, profondeur ideale
Backpropagation stochastique, gradient ajuste
Modeles GAN generatifs, data-set finalise

[Pre-Chorus]
Latence zero, flux continu
L'IA evolue, circuit virtuel

[Chorus - Talkbox Vocals]
C'est l'ere de l'intelligence artificielle
Algorithmes de backpropagation dansent sur le beat
Deep learning en action, reseau fully connected
Processing en temps reel, le futur est lance !

[Verse 2]
Transformers multi-taches, attention mecanique
Zones de pooling maximisees, features extraites
Loss function minimisee, convergence assuree
Overfitting evite, dataset equilibre

[Chorus - Talkbox Vocals]
C'est l'ere de l'intelligence artificielle (AI!)
Algorithmes de backpropagation dansent sur le beat
Deep learning en action, reseau fully connected
Processing en temps reel, le futur est lance !

[Bridge - Synth Pad Build-Up]
Stochastic gradient descent, optimise le modele
Early stopping active, le reseau se stabilise
Cloud computing massif, calcul distribue
L'IA revolutionne, monde transforme

[Outro - Vinyl Crackles & Fade Out]
IA... Intelligence Artificielle...
Reseaux neuronaux... dansent... forever..."

mkdir -p /tmp/ace

./build/ace-qwen3 --model checkpoints/acestep-5Hz-lm-4B \
    --caption "$CAPTION" --lyrics "$LYRICS" \
    --bpm 124 --duration 220 --keyscale "F# minor" --timesignature 4 --language fr \
    --fsm --cfg-scale 2.2 \
    --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed "$SEED"

./build/dit-vae \
    --input-dir /tmp/ace \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo --vae checkpoints/vae \
    --seed "$SEED" --output full.wav
