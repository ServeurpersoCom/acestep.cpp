#!/bin/bash
# Download ACE-Step checkpoints from HuggingFace
# Usage: ./checkpoints.sh [--all]
#   default: Qwen3-Embedding-0.6B + acestep-5Hz-lm-4B + acestep-v15-turbo + vae
#   --all:   + all LM variants + all DiT variants from ACE-Step registry

set -eu

DIR="checkpoints"
mkdir -p "$DIR"

HF="hf download --quiet"
MAIN="ACE-Step/Ace-Step1.5"

dl_main() {
    local name="$1"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls "$target"/*.safetensors "$target"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $MAIN"
    $HF "$MAIN" --include "$name/*" --local-dir "$DIR"
}

dl_repo() {
    local name="$1" repo="$2"
    local target="$DIR/$name"
    if [ -d "$target" ] && [ "$(ls "$target"/*.safetensors "$target"/*.bin 2>/dev/null | wc -l)" -gt 0 ]; then
        echo "[OK] $name"
        return
    fi
    echo "[Download] $name <- $repo"
    $HF "$repo" --local-dir "$target"
}

# === Core (required) ===
dl_main "Qwen3-Embedding-0.6B"
dl_repo "acestep-5Hz-lm-4B" "ACE-Step/acestep-5Hz-lm-4B"
dl_main "acestep-v15-turbo"
dl_main "vae"

# === --all: every model from ACE-Step registry ===
if [ "${1:-}" = "--all" ]; then
    # LM variants (from main repo)
    dl_main "acestep-5Hz-lm-1.7B"
    # LM variants (separate repos)
    dl_repo "acestep-5Hz-lm-0.6B" "ACE-Step/acestep-5Hz-lm-0.6B"
    # DiT variants (separate repos)
    dl_repo "acestep-v15-turbo-shift3" "ACE-Step/acestep-v15-turbo-shift3"
    dl_repo "acestep-v15-turbo-shift1" "ACE-Step/acestep-v15-turbo-shift1"
    dl_repo "acestep-v15-turbo-continuous" "ACE-Step/acestep-v15-turbo-continuous"
    dl_repo "acestep-v15-sft" "ACE-Step/acestep-v15-sft"
    dl_repo "acestep-v15-base" "ACE-Step/acestep-v15-base"
fi

# Convert silence_latent.pt -> .bin (raw float32, [T, 64])
SL="$DIR/acestep-v15-turbo/silence_latent"
if [ -f "$SL.pt" ] && [ ! -f "$SL.bin" ]; then
    echo "[Convert] silence_latent.pt -> .bin"
    SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
    "$SCRIPT_DIR/pt2bin" "$SL.pt" "$SL.bin"
fi

find "$DIR" -name '.cache' -type d -exec rm -rf {} + 2>/dev/null
echo "[Done] Checkpoints ready in $DIR"
