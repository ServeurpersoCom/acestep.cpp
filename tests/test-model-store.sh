#!/bin/bash

../build/test-model-store \
    --lm ../models/acestep-5Hz-lm-0.6B-Q8_0.gguf \
    --dit ../models/acestep-v15-base-Q8_0.gguf \
    --vae ../models/vae-BF16.gguf
