#!/bin/bash
# Download pre-quantized ACE-Step GGUF models from HuggingFace
#
# Usage: ./models.sh [options]
#   default:    Q8_0 turbo essentials (text-encoder + LM-4B + DiT-turbo + VAE)
#   --all:      all models, all quants
#   --quant X:  use quant X (Q4_K_M, Q5_K_M, Q6_K, Q8_0, BF16)
#   --lm SIZE:  LM size (0.6B, 1.7B, 4B, default: 4B)
#   --sft:      include SFT DiT variant
#   --base:     include base DiT variant
#   --shifts:   include shift1/shift3/continuous DiT variants

set -eu

REPO="Serveurperso/ACE-Step-1.5-GGUF"
DIR="models"
QUANT="Q8_0"
LM_SIZE="4B"
ALL=0
SFT=0
BASE=0
SHIFTS=0

while [ $# -gt 0 ]; do
    case "$1" in
        --all)    ALL=1 ;;
        --quant)  QUANT="$2"; shift ;;
        --lm)     LM_SIZE="$2"; shift ;;
        --sft)    SFT=1 ;;
        --base)   BASE=1 ;;
        --shifts) SHIFTS=1 ;;
        *)        echo "Unknown option: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p "$DIR"

dl() {
    local file="$1"
    if [ -f "$DIR/$file" ]; then
        echo "[OK] $file"
        return
    fi
    echo "[Download] $file"
    hf download --quiet "$REPO" "$file" --local-dir "$DIR"
}

# Resolve quant to best available for each model type.
# Matches quantize.sh matrix exactly:
#   Embedding/LM-small: BF16, Q8_0
#   LM-4B:              BF16, Q5_K_M, Q6_K, Q8_0  (Q4_K_M breaks audio codes)
#   DiT:                BF16, Q4_K_M, Q5_K_M, Q6_K, Q8_0
# Order: Q4_K_M < Q5_K_M < Q6_K < Q8_0 < BF16
# If requested quant unavailable, picks next larger available.
resolve_quant() {
    local requested="$1" model_type="$2"
    case "$model_type" in
        emb|lm_small)
            case "$requested" in
                BF16) echo "BF16" ;;
                *)    echo "Q8_0" ;;
            esac ;;
        lm_4B)
            case "$requested" in
                BF16)              echo "BF16" ;;
                Q8_0)              echo "Q8_0" ;;
                Q6_K)              echo "Q6_K" ;;
                Q5_K_M|Q4_K_M)    echo "Q5_K_M" ;;
                *)                 echo "Q8_0" ;;
            esac ;;
        dit)
            echo "$requested" ;;
    esac
}

# VAE is always BF16 (small, quality-critical)
dl "vae-BF16.gguf"

# Text encoder
dl "Qwen3-Embedding-0.6B-$(resolve_quant "$QUANT" emb).gguf"

# LM
if [ "$LM_SIZE" = "4B" ]; then
    dl "acestep-5Hz-lm-4B-$(resolve_quant "$QUANT" lm_4B).gguf"
else
    dl "acestep-5Hz-lm-${LM_SIZE}-$(resolve_quant "$QUANT" lm_small).gguf"
fi

# DiT turbo (always included)
dl "acestep-v15-turbo-${QUANT}.gguf"

# Optional DiT variants
if [ "$SFT" = 1 ] || [ "$ALL" = 1 ]; then
    dl "acestep-v15-sft-${QUANT}.gguf"
fi
if [ "$BASE" = 1 ] || [ "$ALL" = 1 ]; then
    dl "acestep-v15-base-${QUANT}.gguf"
fi
if [ "$SHIFTS" = 1 ] || [ "$ALL" = 1 ]; then
    dl "acestep-v15-turbo-shift1-${QUANT}.gguf"
    dl "acestep-v15-turbo-shift3-${QUANT}.gguf"
    dl "acestep-v15-turbo-continuous-${QUANT}.gguf"
fi

# --all: every model with its valid quants (matches quantize.sh)
if [ "$ALL" = 1 ]; then
    # Embedding: BF16 + Q8_0 only
    dl "Qwen3-Embedding-0.6B-BF16.gguf"

    # Small/medium LM: BF16 + Q8_0 only (too small for aggressive quant)
    for lm in 0.6B 1.7B; do
        dl "acestep-5Hz-lm-${lm}-BF16.gguf"
        dl "acestep-5Hz-lm-${lm}-Q8_0.gguf"
    done

    # Large LM: BF16 + Q5_K_M/Q6_K/Q8_0 (Q4_K_M breaks audio codes)
    for q in BF16 Q5_K_M Q6_K Q8_0; do
        dl "acestep-5Hz-lm-4B-${q}.gguf"
    done

    # DiT variants: BF16 + Q4_K_M/Q5_K_M/Q6_K/Q8_0
    for dit in turbo sft base turbo-shift1 turbo-shift3 turbo-continuous; do
        for q in BF16 Q4_K_M Q5_K_M Q6_K Q8_0; do
            dl "acestep-v15-${dit}-${q}.gguf"
        done
    done
fi

echo "[Done] Models ready in $DIR/"
