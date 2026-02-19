#!/bin/bash
set -e

Q="./build/quantize"

quantize() {
    local bf16="$1" type="$2"
    local out="${bf16/-BF16.gguf/-${type}.gguf}"
    if [ -f "$out" ]; then
        echo "[Skip] $out"
    else
        $Q "$bf16" "$out" "$type"
    fi
}

# Embedding: Q8_0 only (single-shot, precision matters)
quantize models/Qwen3-Embedding-0.6B-BF16.gguf Q8_0

# Small/medium LM: Q8_0 only (too small to survive aggressive quant)
quantize models/acestep-5Hz-lm-0.6B-BF16.gguf Q8_0
quantize models/acestep-5Hz-lm-1.7B-BF16.gguf Q8_0

# Large LM: full range (Q4_K_M confirmed broken for audio codes)
for type in Q5_K_M Q6_K Q8_0; do
    quantize models/acestep-5Hz-lm-4B-BF16.gguf "$type"
done

# DiT models: full range
for bf16 in models/acestep-v15-*-BF16.gguf; do
    for type in Q4_K_M Q5_K_M Q6_K Q8_0; do
        quantize "$bf16" "$type"
    done
done

# VAE: never quantized (stays BF16)
