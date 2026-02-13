#!/bin/bash
# Format: caption+lyrics -> LLM enriched metadata (no codes, no dit-vae)
# Output: /tmp/ace/ -> feed to dit-vae via --input-dir

set -eu
SEED="${SEED:--1}"

CAPTION="Indie rock, jangly guitars, upbeat energy with nostalgic vocals"
LYRICS="[Verse 1]
Walking down the street with the radio on
Every song reminds me of when you were gone
[Chorus]
Turn it up, turn it up, let the music play
We got nothing left but a beautiful day"

mkdir -p /tmp/ace

./ace-qwen3 --model ../checkpoints/acestep-5Hz-lm-4B \
    --system "Format the user's input into a more detailed and specific musical description:" \
    --user "$(printf '# Caption\n%s\n\n# Lyric\n%s' "$CAPTION" "$LYRICS")" \
    --fsm --no-codes \
    --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed "$SEED"
