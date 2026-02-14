#!/bin/bash
# Custom mode (all metas) with SFT model: caption + lyrics + metadata -> codes -> DiT (CFG 7.0) -> WAV

set -eu
export SEED="${SEED:--1}" OUT="${OUT:-full-sft.wav}"
export ACE_DIT="${ACE_DIT:-checkpoints/acestep-v15-sft}"
export SHIFT=1.0 STEPS=50 GUIDANCE_SCALE=7.0

CAPTION="Vibrant French house meets tech-house fusion track featuring filtered disco samples, driving funky basslines, and classic four-on-the-floor beats with signature Bob Sinclar vocal chops. Analytical yet euphoric mood blending advanced AI technical vocabulary with dancefloor energy. Instruments include talkbox lead vocals, analog Moog bass synths, glitchy arpeggiated sequencers, punchy TR-808 drum machine, and shimmering high-hat rolls. Production style showcases crisp retro-modern mix with dynamic sidechain compression, warm vinyl crackle, and modular synth modulations. Vocal delivery combines rhythmic French rap verses with melodic, pitch-shifted choruses celebrating machine learning breakthroughs"

LYRICS="[Intro - Filtered Disco Sample & Synth Arp]

[Verse 1]
Optimisation des poids synaptiques en temps réel
Réseaux neuronaux convolutifs, profondeur idéale
Backpropagation stochastique, gradient ajusté
Modèles GAN génératifs, data-set finalisé

[Pre-Chorus]
Latence zéro, flux continu
L'IA évolue, circuit virtuel

[Chorus - Talkbox Vocals]
C'est l'ère de l'intelligence artificielle
Algorithmes de backpropagation dansent sur le beat
Deep learning en action, réseau fully connected
Processing en temps réel, le futur est lancé !

[Verse 2]
Transformers multi-tâches, attention mécanique
Zones de pooling maximisées, features extraites
Loss function minimisée, convergence assurée
Overfitting évité, dataset équilibré

[Chorus - Talkbox Vocals]
C'est l'ère de l'intelligence artificielle
Algorithmes de backpropagation dansent sur le beat
Deep learning en action, réseau fully connected
Processing en temps réel, le futur est lancé !

[Bridge - Synth Pad Build-Up]
Stochastic gradient descent, optimise le modèle
Early stopping activé, le réseau se stabilise
Cloud computing massif, calcul distribué
L'IA révolutionne, monde transformé

[Outro - Vinyl Crackles & Fade Out]
IA... Intelligence Artificielle...
Réseaux neuronaux... dansent... forever..."

exec ./generate.sh \
    --caption "$CAPTION" --lyrics "$LYRICS" \
    --bpm 124 --duration 220 --keyscale "F# minor" --timesignature 4 --language fr
