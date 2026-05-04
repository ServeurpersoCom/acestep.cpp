#!/bin/bash

declare -A DISABLE_VAR=(
    [CUDA0]=GGML_CUDA_DISABLE_FUSION
    [Vulkan0]=GGML_VK_DISABLE_FUSION
    [CPU]=GGML_CPU_DISABLE_FUSION
)

for backend in "${!DISABLE_VAR[@]}"; do
    var=${DISABLE_VAR[$backend]}
    for fusion in "" NOFUSION; do
        for quant in BF16 Q8_0 Q6_K Q5_K_M; do
            env ${fusion:+$var=1} GGML_BACKEND=$backend ./debug-dit-cossim.py \
                --mode all --quant $quant 2>&1 \
                | tee ${backend}-${fusion:+$fusion-}${quant}.log
        done
    done
done
