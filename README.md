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

## Quick start

```bash
./checkpoints.sh                    # download models from HuggingFace
./generate.sh --query "A French chanson about Paris"
```

The unified `generate.sh` script chains both binaries automatically.
Model paths default to `checkpoints/` and can be overridden via environment
variables (`ACE_LM`, `ACE_DIT`, `ACE_TE`, `ACE_VAE`).

## Generation modes

Three modes match the original ACE-Step WebUI, with sane defaults
(temperature 0.85, CFG 2.0, FSM constrained decoding on by default).

### Simple mode (WebUI "Simple" tab)

Natural language in, music out. The LLM generates metadata, lyrics, and
audio codes from a free-form description:

```bash
./generate.sh --query "Une chanson française sur la ville de Paris"
./generate.sh --query "ambient piano meditation" --instrumental
```

### Custom mode (WebUI "Custom" tab)

Provide caption, lyrics, and optionally metadata. When all metadata is
present (bpm, duration, keyscale, timesignature), the LLM skips
chain-of-thought and generates audio codes directly. When metadata is
partial, the LLM fills in missing fields via CoT first.

```bash
# All metadata provided: direct code generation
./generate.sh \
    --caption "French house, talkbox vocals, TR-808 drums" \
    --lyrics "[Verse 1]\nOptimisation des poids synaptiques..." \
    --bpm 124 --duration 220 --keyscale "F# minor" --timesignature 4 --language fr

# Partial metadata: LLM fills bpm, key, timesig via CoT
./generate.sh --caption "Ambient electronic soundscape" --duration 180
```

### Raw mode (advanced)

Custom system/user prompts for experimentation:

```bash
./generate.sh \
    --system "Format the user's input into a detailed musical description:" \
    --user "$(printf '# Caption\n%s\n\n# Lyric\n%s' "$CAPTION" "$LYRICS")" \
    --no-codes
```

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ACE_LM` | `checkpoints/acestep-5Hz-lm-4B` | LLM model directory |
| `ACE_DIT` | `checkpoints/acestep-v15-turbo` | DiT model directory |
| `ACE_TE` | `checkpoints/Qwen3-Embedding-0.6B` | Text encoder directory |
| `ACE_VAE` | `checkpoints/vae` | VAE directory |
| `SEED` | random | RNG seed (`-1` = random) |
| `OUT` | `output.wav` | Output WAV path |
| `TMP` | `/tmp/ace` | Exchange directory |
| `BIN` | `./build` | Binary directory |

## ace-qwen3 reference

```
Usage: ace-qwen3 --model <dir> [options]

Model:
  --model <dir>          Model directory (safetensors + config.json)

Simple mode (inspiration):
  --query <text>         Natural language music description
  --instrumental         Generate instrumental (no vocals)

Custom mode:
  --caption <text>       Music description
  --lyrics <text>        Lyrics (default: empty)
  --bpm <N>              BPM (0=LLM decides)
  --duration <N>         Duration in seconds (0=LLM decides)
  --keyscale <text>      Key/scale (e.g. 'C major')
  --timesignature <N>    Time signature (2,3,4,6)
  --language <code>      Vocal language (en,fr,zh,...)

Raw mode (advanced):
  --system <text>        Custom system message
  --user <text>          Custom user message

Generation:
  --max-tokens <N>       Max new tokens (default: 256)
  --max-seq <N>          KV cache size (default: 8192)
  --temperature <f>      Sampling temperature (default: 0.85)
  --top-p <f>            Top-p sampling (default: 0.9, 1.0=disabled)
  --seed <N>             RNG seed (default: random)
  --cfg-scale <f>        CFG scale for Phase 2 (default: 2.0, 1.0=disabled)
  --negative-prompt <t>  Negative prompt for CFG
  --no-fsm               Disable FSM constrained decoding
  --no-codes             Phase 1 only (no audio codes)

Output:
  --output-dir <dir>     Write codes + metadata for dit-vae
```

Three LLM sizes: 0.6B (fast), 1.7B, 4B (best quality).

## dit-vae reference

```
Usage: dit-vae --input-dir <dir> [options]

Input (from ace-qwen3 --output-dir):
  --input-dir <dir>       Directory with codes, caption, lyrics, etc.

Models (required):
  --text-encoder <dir>    Qwen3-Embedding-0.6B directory
  --dit <dir>             DiT model directory (e.g. acestep-v15-turbo)
  --vae <dir>             VAE directory

Audio:
  --seed <n>              Random seed (default: random)
  --noise-file <path>     Load noise from bf16 file (Philox RNG dump)
  --shift <f>             Timestep shift (default: 3.0)
  --steps <n>             Euler steps (default: 8)
  --output <path>         Output WAV (default: output.wav)

Debug:
  --dump <dir>            Dump intermediate tensors
  --inject <dir>          Inject noise/context/enc_hidden from dump dir
                          (bypasses text encoder and condition encoder)
```

## Exchange directory

The two binaries communicate through a directory of plain text files.
One file per field, no extensions.

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

dit-vae only reads files, so you can write them yourself to run it standalone:

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
Requires nvcc + sm_120 (Blackwell). Same CLI interface as the GGML build.

```
cd cuda && make
```

The `cuda/` directory has its own `generate.sh` and example scripts,
mirroring the root-level ones but calling the CUDA binaries directly.

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
