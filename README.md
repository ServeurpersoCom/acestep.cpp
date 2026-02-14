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
./generate.sh --query "A smooth RnB track with soulful vocals and warm keys"
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
./generate.sh --query "Indie rock with jangly guitars and nostalgic vocals"
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
    --caption "90s RnB slow jam, silky vocals, Rhodes piano, 808 bass" \
    --lyrics "[Verse 1]\nI've been thinking about you all night long..." \
    --bpm 85 --duration 200 --keyscale "Eb major" --timesignature 4 --language en

# Partial metadata: LLM fills bpm, key, timesig via CoT
./generate.sh --caption "Garage rock, distorted guitars, raw energy" --duration 180
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

## Accuracy

LLM logits (prefill, 0.6B, `tests/debug-lm-logits.py`):

```
GGML<>PyTorch cosine similarity: 0.999980
Top-5 argmax: identical
```

DiT+VAE pipeline (`tests/debug-cos-sim.py`, seed 42, Philox noise).
Cosine similarity between GGML and Python intermediate tensors (1.0 = identical):

| Stage | GGML <-> Python |
|---|---:|
| text_hidden | 0.9998 |
| lyric_embed | 1.0000 |
| enc_hidden | 0.9998 |
| context | 1.0000 |
| noise | 1.0000 |
| dit_step0_vt | 0.9970 |
| dit_step1_vt | 0.9994 |
| dit_step2_vt | 0.9989 |
| dit_step3_vt | 0.9980 |
| dit_step4_vt | 0.9966 |
| dit_step5_vt | 0.9950 |
| dit_step6_vt | 0.9949 |
| dit_step7_vt | 0.9904 |
| dit_x0 | 0.9951 |
| vae_audio | 0.9897 |
| vae_audio (log spectral) | 0.9770 |

Residual drift is FP32 (GGML) vs BF16 (PyTorch) accumulation over 8 Euler steps.

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
