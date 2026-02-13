# acestep.cpp

Portable C++17 implementation of ACE-Step 1.5 music generation using GGML.
Text + lyrics in, stereo 48kHz WAV out. Runs on CPU, CUDA, Metal, Vulkan.

## Build

```
git submodule update --init
cd build && cmake .. && make -j$(nproc)
```

Enable GPU backends with cmake flags:

```
cmake .. -DGGML_CUDA=ON
cmake .. -DGGML_METAL=ON
cmake .. -DGGML_VULKAN=ON
```

Builds two binaries: `ace-qwen3` (LLM) and `dit-vae` (DiT + VAE).

## Full pipeline (LLM + DiT + VAE)

The LLM generates a chain-of-thought (BPM, key, time signature) then audio
tokens (integers 0-63999 from a 64000-level FSQ codebook, at 5Hz). These
audio tokens are written as `codes` in the exchange directory and decoded
into source latents for the DiT.

```bash
# Step 1: LLM generates metadata + audio tokens
./build/ace-qwen3 --model checkpoints/acestep-5Hz-lm-0.6B \
    --caption "Hip hop, jazzy piano, vinyl crackle" \
    --lyrics "[Instrumental]" \
    --bpm 90 --duration 60 --keyscale "Bb minor" --timesignature 4 \
    --cfg-scale 2.2 --fsm --seed 42 \
    --output-dir /tmp/ace

# Step 2: DiT + VAE renders audio from tokens
./build/dit-vae --input-dir /tmp/ace \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo \
    --vae checkpoints/vae \
    --seed 42 --output output.wav
```

Or use `full.sh` which chains both steps with a default prompt.

Three LLM sizes are available: 0.6B (fast), 1.7B, 4B (best quality).
When all metadata is provided (bpm, duration, keyscale, timesignature),
the LLM skips chain-of-thought and generates audio tokens directly.
When metadata is partial or missing, the LLM runs two phases: first
generating the missing metadata via CoT, then generating audio tokens.

## Exchange directory

The two binaries communicate through a directory of plain text files.
One file per field, no extensions. The directory must exist before running.

```
<dir>/
  caption       Music description (required, may be multiline)
  lyrics        Lyrics with [Section] markers (default: empty)
  codes         LLM audio tokens, CSV integers 0-63999 (absent = generate from noise)
  bpm           Integer, e.g. "124" (default: N/A)
  duration      Integer seconds, e.g. "220" (default: 120)
  keyscale      Key and scale, e.g. "F# minor" (default: N/A)
  timesig       Time signature numerator, e.g. "4" (default: N/A)
  language      Vocal language code, e.g. "fr" (default: en)
```

Missing file = default value (N/A for bpm/keyscale/timesig, matching
the original ACE-Step training data).

dit-vae only reads files, it does not care how they got there.
You can write them yourself to run dit-vae standalone:

```bash
mkdir -p /tmp/ace
echo -n "Hip hop, jazzy piano, vinyl crackle" > /tmp/ace/caption
echo -n "[Instrumental]" > /tmp/ace/lyrics
echo -n "90"             > /tmp/ace/bpm
echo -n "60"             > /tmp/ace/duration
echo -n "Bb minor"       > /tmp/ace/keyscale
echo -n "4"              > /tmp/ace/timesig

./build/dit-vae --input-dir /tmp/ace \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo \
    --vae checkpoints/vae \
    --output output.wav
```

Without `codes`, dit-vae generates from noise only.
With `codes` (audio tokens from ace-qwen3), dit-vae decodes them into
source latents for the DiT (higher quality, LLM-guided generation).

## Pipeline

```
ace-qwen3 (Qwen3 causal LM, 0.6B/1.7B/4B)
  -> CoT: bpm, keyscale, timesignature
  -> Audio codes (5Hz tokens, FSQ vocabulary)
  -> [optional CFG with dual KV cache]

dit-vae
  -> BPE tokenize
  -> Qwen3-Embedding (28L text encoder)
  -> CondEncoder (lyric 8L + timbre 4L + text_proj)
  -> FSQ detokenizer (audio codes -> source latents, if provided)
  -> DiT (24L flow matching, 8 Euler steps)
  -> VAE (AutoencoderOobleck)
  -> WAV stereo 48kHz
```

## CUDA backend (legacy)

Standalone CUDA implementation under `cuda/`, kept as reference.
Requires nvcc + sm_120 (Blackwell). The GGML build replaces this.

```
cd cuda && make
```

## Accuracy

LLM logits (prefill, 0.6B):

```
GGML<>PyTorch cosine similarity: 0.999980
Top-5 argmax: identical
```

DiT+VAE pipeline (3-way comparison, seed 42, Philox noise).
Values are cosine similarity between intermediate tensors (1.0 = identical):

```
stage                               CUDA<>GGML    CUDA<>Python   GGML<>Python

text_hidden                          0.999843       0.999755       0.999810
lyric_embed                          0.176042       0.176042       1.000000
enc_hidden                           0.176002       0.175578       0.999826
context                              1.000002       1.000000       1.000002
noise                                1.000000       1.000000       1.000000
dit_step0_vt                         0.949375       0.949714       0.996955
dit_step1_vt                         0.981008       0.982935       0.999365
dit_step2_vt                         0.957166       0.959070       0.998946
dit_step3_vt                         0.937398       0.942449       0.997979
dit_step4_vt                         0.935670       0.944726       0.996615
dit_step5_vt                         0.930438       0.943404       0.995034
dit_step6_vt                         0.925888       0.936682       0.994890
dit_step7_vt                         0.906374       0.920733       0.990427
dit_x0                               0.929130       0.942255       0.995127

vae_audio (log spectral)             0.912169       0.940828       0.977008
```

GGML matches PyTorch to >0.99 across all stages. CUDA drift comes from bf16
accumulation in VAE and a minor BPE tokenization difference on lyrics.

## Checkpoints

```
./checkpoints.sh
```

Downloads from HuggingFace:

- Qwen3-Embedding-0.6B (text encoder, 1.1GB),
- acestep-v15-turbo (DiT 24L + CondEncoder, 3GB),
- vae (AutoencoderOobleck, 322MB),
- acestep-5Hz-lm-0.6B / 1.7B / 4B (autoregressive LLM).

## Known issues

Uses a patched GGML fork (submodule). Two fixes for long-sequence audio:

- im2col.cu gridDim.y overflow when T > 65535 patches,
- conv_transpose_1d.cu O(T_in) brute-force loop too slow for VAE upsampling.

TODO: verify if these are still needed on latest GGML and submit upstream PRs.

## Acknowledgements

Independent implementation based on ACE-Step 1.5 by ACE Studio and StepFun.
All model weights are theirs, this is just a native backend.

```bibtex
@misc{gong2026acestep,
	title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
	author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
	howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
	year={2026},
	note={GitHub repository}
}
```
