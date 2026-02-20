#!/bin/bash

cp ../examples/simple.json .
cp ../examples/partial.json .
cp ../examples/full.json .

../build/ace-qwen3 --request simple.json \
  --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
  --dump-logits logits.bin --dump-tokens tokens.csv

python3 debug-lm-logits.py ../checkpoints/acestep-5Hz-lm-4B logits.bin tokens.csv

../build/ace-qwen3 --request partial.json \
  --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
  --dump-logits logits.bin --dump-tokens tokens.csv

python3 debug-lm-logits.py ../checkpoints/acestep-5Hz-lm-4B logits.bin tokens.csv

../build/ace-qwen3 --request full.json \
  --model ../models/acestep-5Hz-lm-4B-Q8_0.gguf \
  --dump-logits logits.bin --dump-tokens tokens.csv

python3 debug-lm-logits.py ../checkpoints/acestep-5Hz-lm-4B logits.bin tokens.csv
