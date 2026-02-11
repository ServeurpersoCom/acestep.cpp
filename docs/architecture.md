# Architecture Reference (acestep.cpp)

Authority: Python upstream ACE-Step-1.5 (github.com/ace-step/ACE-Step-1.5).
C++ must produce identical token sequences and equivalent audio.


## System Overview

Two standalone CUDA binaries, connected by bash scripts via plain text files.

```
                        ace-qwen3                                    dit-vae
                   (LLM inference)                          (DiT + VAE inference)
                                                           
 CLI args ──────►┌──────────────────┐                     ┌──────────────────────┐
 --caption       │ Qwen3-4B LM      │   /tmp/ace/         │ Qwen3-Embedding 0.6B │
 --lyrics        │                  │   (7 text files)    │ (text encoder)       │
 --bpm ...       │ Phase 1: CoT     │──────────────────►  │                      │
   OR            │  YAML metadata   │   caption,lyrics,   │ ConditionEncoder     │
 --system        │                  │   bpm,duration,     │  LyricEncoder (8L)   │
 --user          │ Phase 2: audio   │   keyscale,timesig, │  TimbreEncoder (4L)  │
                 │  code tokens     │   language          │                      │
                 │  0-63999         │                     │ AudioDetokenizer (2L)│
                 └───────┬──────────┘                     │                      │
                         │                                │ DiT (24 layers)      │
                         │ --output-codes                 │  flow matching       │
                         │ /tmp/codes.txt                 │  8 steps (turbo)     │
                         │ (CSV integers)                 │                      │
                         └──────────────────────────────► │ VAE decode           │
                                                          │  48kHz stereo        │
                                                          └──────────┬───────────┘
                                                                     │
                                                                     ▼
                                                                  song.wav
```


## Full Pipeline Diagram

```
User Input (caption, lyrics, metadata)
    │
    ▼
┌────────────────────────────────────────┐
│ LM 4B (Qwen3-based)   ace-qwen3 binary |
│ ChatML template                        │
│ Phase 1: CoT ─► YAML metadata          │
│ Phase 2: audio codes as text tokens    │
│   <|audio_code_XXXXX|> (0-63999)       │
│   FSQ codebook = 64000                 │
│     levels = [8,8,8,5,5,5]             │
│     8*8*8*5*5*5 = 64000                │
│   Token rate: 5 Hz (1 code per 200ms)  │
└──────────┬──────────────────────────────┘
           │  integers 0-63999 (CSV file)
           │  + text files (caption, lyrics, bpm, duration, keyscale, timesig, language)
           ▼
┌─────────────────────────────────────────────────────┐
│                                      dit-vae binary |
│                                                     │
│ ┌─ Qwen3-Embedding 0.6B (text encoder) ───────────┐ │
│ │ 28 layers, hidden=1024, 16 heads, 8 kv_heads    │ │
│ │ caption ─► text_hidden_states [T_text, 1024]    │ │
│ └─────────────────────────────────────────────────┘ │
│                                                     │
│ ┌─ AceStepConditionGenerationModel (4.79 GB bf16) ─┐│
│ │                                                  ││
│ │ ┌─ ConditionEncoder ────────────────────────────┐││
│ │ │ text_projector: Linear(1024 ─► 2048)          │││
│ │ │ LyricEncoder: 8 layers (GQA + RoPE)           │││
│ │ │ TimbreEncoder: 4 layers                       │││
│ │ └───────────────────────────────────────────────┘││
│ │                                                  ││
│ │ ┌─ AudioTokenizer ─────────────────────────────┐ ││
│ │ │ ResidualFSQ(levels=[8,8,8,5,5,5])            │ ││
│ │ │ codebook = 64000, vocab_size = 64003 (+pad)  │ ││
│ │ │ AttentionPooler (2L, pool_window=5: 25─►5Hz) │ ││
│ │ └──────────────────────────────────────────────┘ ││
│ │                                                  ││
│ │ ┌─ AudioDetokenizer ────────────────────────────┐││
│ │ │ 2 layers (num_attention_pooler_hidden_layers) │││
│ │ │ Expands 5Hz ─► 25Hz via special_tokens        │││
│ │ │ proj_out: Linear(2048 ─► 64)                  │││
│ │ └───────────────────────────────────────────────┘││
│ │                                                  ││
│ │ ┌─ DiT (24 layers) ─────────────────────────────┐││
│ │ │ hidden_size=2048, heads=16, kv_heads=8 (GQA)  │││
│ │ │ head_dim=128, intermediate=6144 (SwiGLU)      │││
│ │ │ RoPE theta=1M, sliding_window=128             │││
│ │ │ layer_types: alternating sliding/full attn    │││
│ │ │ + cross-attention to encoder outputs          │││
│ │ │ AdaLN (scale-shift from timestep embedding)   │││
│ │ │ Input: cat(noise[64], src[64], mask[64]) = 192│││
│ │ │ proj_in: Conv1d(192─►2048, patch=2, stride=2) │││
│ │ │ proj_out: ConvT1d(2048─►64, patch=2, stride=2)│││
│ │ └───────────────────────────────────────────────┘││
│ │                                                  ││
│ │ ┌─ Diffusion Scheduler ────────────────────────┐ ││
│ │ │ Flow matching, 8 steps (turbo), shift=3.0    │ ││
│ │ │ Predefined timestep schedules (not learned)  │ ││
│ │ └──────────────────────────────────────────────┘ ││
│ └──────────────────────────────────────────────────┘│
│                                                     │
│ ┌─ AutoencoderOobleck (VAE, diffusers) ────────────┐│
│ │ 48kHz stereo, decoder_input=64 channels          ││
│ │ downsampling: [2,4,4,6,10] = 1920x total         ││
│ │ 48000 / 1920 = 25 Hz latent rate                 ││
│ │ latents [B, T_25Hz, 64] ─► audio [B, 2, T_48k]   ││
│ └──────────────────────────────────────────────────┘│
│                                                     │
└──────────────────────────┬──────────────────────────┘
                           │
                           ▼
                        song.wav (48kHz stereo)
```


## Model Specifications

### LLM: acestep-5Hz-lm-4B (Qwen3-based)
```
vocab_size:     217204 (151669 base Qwen3 + 65535 audio code slots, 64000 valid)
hidden_size:    2560
num_layers:     36
num_heads:      32
num_kv_heads:   8
intermediate:   9728
max_seq:        40960
rope_theta:     1000000
```

### Text Encoder: Qwen3-Embedding-0.6B
```
vocab_size:     151669
hidden_size:    1024 (= text_hidden_dim in DiT config)
num_layers:     28
num_heads:      16
num_kv_heads:   8
intermediate:   3072
max_seq:        32768
```

### DiT: acestep-v15-turbo (AceStepConditionGenerationModel)
```
hidden_size:      2048
num_layers:       24 (alternating sliding/full attention)
num_heads:        16
num_kv_heads:     8
head_dim:         128
intermediate:     6144 (SwiGLU)
rope_theta:       1000000
sliding_window:   128
in_channels:      192 (64 noise + 64 src_latents + 64 chunk_masks)
patch_size:       2
audio_acoustic_hidden_dim: 64

ConditionEncoder:
  text_projector:  Linear(1024, 2048)
  lyric_encoder:   8 layers
  timbre_encoder:  4 layers

AudioTokenizer:
  FSQ levels:      [8,8,8,5,5,5] = 64000 codebook
  pool_window:     5 (25Hz latent -> 5Hz codes)
  pooler_layers:   2

AudioDetokenizer:
  layers:          2 (num_attention_pooler_hidden_layers)
  special_tokens:  pool_window_size=5 (5Hz -> 25Hz expansion)
  proj_out:        Linear(2048, 64)
```

### VAE: AutoencoderOobleck (diffusers)
```
sampling_rate:          48000
audio_channels:         2 (stereo)
decoder_input_channels: 64
encoder_hidden_size:    128
decoder_channels:       128
channel_multiples:      [1, 2, 4, 8, 16]
downsampling_ratios:    [2, 4, 4, 6, 10] = 1920x total
latent_rate:            48000 / 1920 = 25 Hz
```


## Sampling Parameters

### LLM (ace-qwen3)

| Parameter      | acestep.cpp default | Python upstream default |
|----------------|---------------------|-------------------------|
| cfg_scale      | 2.2                 | 2.0                     |
| temperature    | 0.80                | 0.85                    |
| top_p          | 0.9                 | 0.9                     |
| top_k          | disabled            | 0 (disabled)            |
| seed           | 42 (scripts)        | random                  |
| rep_penalty    | 1.0 (none)          | 1.0 (none)              |

### DiT (dit-vae)

| Parameter      | acestep.cpp default | Python UI default |
|----------------|---------------------|-------------------|
| steps          | 8                   | 8                 |
| shift          | 3.0                 | 3.0               |
| guidance_scale | NOT IMPLEMENTED     | 7.0 (dead code?)  |
| infer_method   | ode (euler)         | ode               |
| seed           | 42 (scripts)        | random            |

Note: Python handler.py has shift=1.0 as code default but the Gradio UI sets it
to 3.0. The turbo model (acestep-v15-turbo) is designed for shift=3.0.
C++ snaps non-standard shift values to nearest valid (1.0, 2.0, 3.0) with a warning.


## Token IDs (Qwen3 vocabulary)

```
151643  <|endoftext|>           EOS for DiT text encoder
151644  <|im_start|>            TOKEN_IM_START
151645  <|im_end|>              TOKEN_IM_END (LLM EOS)
151667  <think>                 TOKEN_THINK
151668  </think>                TOKEN_THINK_END
151669  <|audio_code_0|>        AUDIO_CODE_BASE (range 0-63999)
```

LLM token ID for audio code N = 151669 + N.
Valid range: N in [0, 63999]. Codebook size = 64000 = product([8,8,8,5,5,5]).


## Data Flow Between Binaries

### ace-qwen3 outputs

| Output              | Flag                 | Content                              |
|---------------------|----------------------|--------------------------------------|
| Audio codes         | --output-codes FILE  | CSV integers (0-63999), one per line |
| Enriched metadata   | --output-dir DIR     | 7 text files (see below)             |
| Raw LLM text        | --output-text FILE   | Full LLM output including CoT        |

Output directory (--output-dir /tmp/ace):
```
/tmp/ace/
  caption          Music description (text, may be multiline)
  lyrics           Lyrics with [Section] markers (multiline)
  bpm              Integer (e.g. "124")
  duration         Integer seconds (e.g. "220")
  keyscale         Key and scale (e.g. "F# minor")
  timesignature    Time signature numerator (e.g. "4")
  language         Vocal language code (e.g. "fr")
```

All fields populated after LLM generation (CoT fills missing metas).
Each file contains one plain text value, no trailing newline.

### dit-vae inputs

| Input               | Flag                | Source                               |
|---------------------|---------------------|--------------------------------------|
| Caption             | --caption TEXT      | cat /tmp/ace/caption                 |
| Lyrics              | --lyrics TEXT       | cat /tmp/ace/lyrics                  |
| BPM                 | --bpm N             | cat /tmp/ace/bpm                     |
| Duration            | --duration SEC      | cat /tmp/ace/duration                |
| Key/scale           | --keyscale TEXT     | cat /tmp/ace/keyscale                |
| Time signature      | --timesignature TEXT| cat /tmp/ace/timesignature           |
| Language            | --language TEXT     | cat /tmp/ace/language                |
| Audio codes         | --input-codes FILE  | /tmp/codes.txt (optional)            |
| Text encoder        | --text-encoder DIR  | checkpoints/Qwen3-Embedding-0.6B    |
| DiT model           | --dit DIR           | checkpoints/acestep-v15-turbo       |
| VAE model           | --vae DIR           | checkpoints/vae                      |
| Seed                | --seed N            | CLI (default: random)                |

### dit-vae outputs

| Output              | Flag                | Content                              |
|---------------------|---------------------|--------------------------------------|
| Audio               | --output FILE       | WAV 48kHz stereo (default: output.wav)|


## Generation Paths

### 1. Custom mode (simple.sh, format.sh)

Triggered by --system + --user. Two phases:

Phase 1: build_custom_prompt(system, user) -> generate_text() (no CFG) -> parse_cot_and_lyrics()
Phase 2 (unless --no-codes): run_phase2() -> reset KV -> CFG + CoT injected -> codes

format.sh uses --no-codes (text enrichment only, no audio codes).

### 2. All-metas (full.sh)

When --bpm, --keyscale, --timesignature, and --duration are all present.
Single phase: CoT injected from CLI args, LLM generates codes only.

### 3. Partial-metas (partial.sh)

When one or more of bpm/keyscale/timesignature/duration are missing. Two phases:

Phase 1: build_lm_prompt() -> generate() with CFG + stop_at_reasoning -> parse CoT
Phase 2 (unless --no-codes): run_phase2() -> reset KV -> CFG + CoT injected -> codes

dit-only.sh uses --no-codes to skip codes (DiT generates from noise).


## Chat Template Format

Qwen3 chat template:
```
<|im_start|>system
# Instruction
{instruction}

<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
<think>
```

For all-metas/partial-metas, the standard instruction is:
"Generate audio semantic tokens based on the given conditions:"

For simple mode:
"Expand the user's input into a more detailed and specific musical description:"

For format mode:
"Format the user's input into a more detailed and specific musical description:"


## CoT (Chain of Thought) YAML Format

The LLM generates metadata inside `<think>...</think>`:

```yaml
bpm: 124
caption: Funky French house with deep bass...
duration: 120
keyscale: F# minor
language: fr
timesignature: 4
```

Fields are in alphabetical order (Python uses yaml.dump sort_keys=True).
After `</think>`, the LLM emits lyrics (custom instruction modes) or audio codes
(all-metas/partial-metas Phase 2).


## User Content Format

### Standard (all-metas, partial-metas)
```
# Caption
{caption}

# Lyric
{lyrics}
```

### Simple/Inspiration
```
{query}

instrumental: false
```

### Format/Rewrite
```
# Caption
{caption}

# Lyric
{lyrics}
```


## CFG (Classifier-Free Guidance)

CFG in the LLM uses dual forward passes:
  Conditional: full prompt (system + user with caption/lyrics)
  Unconditional: same structure but user content replaced:
    CoT phase (partial-metas Phase 1):
      With negative_prompt: "# Caption\n{negative_prompt}\n\n# Lyric\n{lyrics}\n"
      Without negative_prompt: "# Lyric\n{lyrics}\n"
    Codes phase (Phase 2, with injected CoT):
      With negative_prompt: caption replaced by negative_prompt, lyrics kept
      Without negative_prompt: caption AND lyrics kept as-is (only CoT emptied)

The final logits: logits = uncond + cfg_scale * (cond - uncond)

Note: CFG is NOT used in simple or format mode Phase 1 (generate_text, no CFG).
CFG IS used in Phase 2 (codes generation) and in partial-metas Phase 1.


## Prompt Differences: LLM vs DiT

The LLM and DiT text encoder receive DIFFERENT prompts built from the same metadata:

| Aspect         | LLM prompt (ace-qwen3.cu)             | DiT prompt (dit-vae.cu)           |
|----------------|----------------------------------|------------------------------------|
| Template       | Qwen3 chat template              | SFT_GEN_PROMPT + _dict_to_meta_string |
| Metas order    | Alphabetical (yaml.dump)         | bpm, timesig, keyscale, duration   |
| Missing metas  | Field omitted entirely           | "N/A" placeholder                  |
| Caption        | In user turn                     | In SFT_GEN_PROMPT (# Instruction + # Caption + # Metas) |
| Lyrics         | In user turn                     | Separate lyric prompt (# Languages + # Lyric) |
| EOS token      | 151645 <|im_end|>               | 151643 <|endoftext|>              |

### DiT text encoder prompt format (SFT_GEN_PROMPT + _dict_to_meta_string):
```
# Instruction
Fill the audio semantic mask based on the given conditions:

# Caption
{caption}

# Metas
- bpm: {bpm}
- timesignature: {timesig}
- keyscale: {keyscale}
- duration: {duration} seconds
<|endoftext|>
```

### DiT lyric prompt format (_format_lyrics):
```
# Languages
{language}

# Lyric
{lyrics}<|endoftext|>
```

Missing metadata fields use "N/A" as placeholder.
DiT field names match CLI names: "timesignature", "keyscale", "bpm", "duration".


## CoT Injection

Python never exposes the LLM's raw CoT to the user. For all-metas mode, it fakes
the CoT by injecting a pre-built `<think>...</think>` as if the LLM generated it,
then decodes only audio codes. This is a critical performance trick: the LLM sees
its own CoT as if it was pre-built.

Our C++ matches this behavior:
  All-metas:     CoT injected from CLI args, LLM generates codes only
  Partial-metas: Phase 1 generates CoT (stop at `</think>`), KV reset,
                 Phase 2 generates codes with CoT injected

Both paths produce identical token sequences for the codes generation phase.


## Constrained Decoding

C++ MetadataFSM (ace-qwen3.cu) matches Python constrained_logits_processor.py.
PrefixTree based FSM enforces valid values during CoT generation in Phase 1:

  Alphabetical field order (bpm -> caption -> duration -> [genres] -> keyscale -> language -> timesig)
  genres is optional (skip_genres=True by default)
  BPM: integer 30-300
  Duration: integer 10-600
  Keyscale: valid note + optional accidental + mode from closed list
  Time signatures: from closed list (4/4, 3/4, 6/8, etc.)
  Language: from closed list (en, zh, ja, ko, etc.)
  Caption: free text (no constraint)

After `</think>`: only audio codes 0-63999 + EOS.
FSM is enabled in both generate() and generate_text() via optional parameter.

Reference: constrained_logits_processor.py:53-100 (FSMState), :500-600 (MetadataConstrainedLogitsProcessor)
Reference: constants.py:1-50 (VALID_LANGUAGES, KEYSCALE_*, BPM_MIN/MAX, DURATION_MIN/MAX)


## CLI Reference (ace-qwen3)

Generic inference binary. Two mutually exclusive modes:

```
Custom mode (--system + --user):
  --system <text>          System instruction (INSPIRED/REWRITE/custom)
  --user <text>            User content (query, caption+lyrics, etc.)

Standard mode (--caption):
  --caption <text>         Music caption/description
  --lyrics <text>          Lyrics text (use [Instrumental] for instrumental)
  --bpm <n>                BPM (0 = generate via CoT)
  --duration <f>           Duration in seconds (default: 120)
  --keyscale <text>        Key scale (e.g. "F# minor")
  --timesignature <text>   Time signature (e.g. "4")
  --language <text>        Vocal language code (e.g. "en")

Generation control:
  --cfg-scale <f>          CFG scale (default: 1.0 = off)
  --negative-prompt <s>    Negative prompt for CFG
  --no-codes               Skip audio codes generation
  --fsm                    Enable FSM constrained decoding for metadata

Output:
  --output-codes <file>    Write audio codes CSV
  --output-dir <dir>       Write enriched prompt fields (7 text files)
  --output-text <file>     Write raw LLM output text

Sampling:
  --temperature <f>        Sampling temperature (default: 0.8)
  --top-p <f>              Top-p sampling (default: 0.9)
  --max-tokens <n>         Max new tokens (default: auto)
  --max-seq <n>            Max KV cache length (default: 8192)
  --seed <n>               Random seed (default: random)

Flow logic:
  --system present:   Custom prompt (generate_text) + optional codes via run_phase2
  --caption present:  all-metas = single pass codes, partial-metas = two-phase
  --no-codes:         Skip codes generation (output metadata only)
```

## CLI Reference (dit-vae)

```
Prompt (required):
  --caption <text>         Music caption/description
  --lyrics <text>          Lyrics text
  --bpm <n>                BPM
  --duration <sec>         Duration in seconds (default: 120)
  --keyscale <text>        Key scale (e.g. "F# minor")
  --timesignature <text>   Time signature (e.g. "4")
  --language <text>        Vocal language code (e.g. "en")

Models (required):
  --text-encoder <dir>     Qwen3-Embedding-0.6B directory
  --dit <dir>              DiT model directory (e.g. acestep-v15-turbo)
  --vae <dir>              VAE directory

Audio:
  --input-codes <file>     LM audio codes (from ace-qwen3 --output-codes)
  --seed <n>               Random seed (default: random)
  --shift <f>              Timestep shift (default: 3.0)
  --steps <n>              Euler steps (default: 8)
  --output <path>          Output WAV (default: output.wav)
```

## Bash Scripts

Self-contained examples, one per mode. All hardcode example values.
Bash is the glue: `cat` reads ace-qwen3 output files to feed dit-vae CLI args.

| Script       | Python equivalent                    | Description                          |
|--------------|--------------------------------------|--------------------------------------|
| simple.sh    | create_sample (inference.py)         | Query -> metadata + lyrics + codes   |
| full.sh      | generate_music (all-metas)           | All CLI metas -> codes -> WAV        |
| partial.sh   | generate_music (partial-metas)       | Caption+lyrics -> CoT fills metas    |
| format.sh    | format_sample (inference.py)         | Caption+lyrics rewrite (no codes)    |
| dit-only.sh  | generate_music (no LM codes)         | LLM enriches caption, DiT from noise |

Example pipeline (simple.sh):
```bash
./ace-qwen3 checkpoints/acestep-5Hz-lm-4B \
    --system "Expand the user's input into a more detailed and specific musical description:" \
    --user "funky French house track" \
    --fsm --cfg-scale 2.2 \
    --output-codes /tmp/codes.txt --output-dir /tmp/ace \
    --temperature 0.80 --top-p 0.9 --seed 42

./dit-vae \
    --caption "$(cat /tmp/ace/caption)" \
    --lyrics "$(cat /tmp/ace/lyrics)" \
    --bpm "$(cat /tmp/ace/bpm)" \
    --duration "$(cat /tmp/ace/duration)" \
    --keyscale "$(cat /tmp/ace/keyscale)" \
    --timesignature "$(cat /tmp/ace/timesignature)" \
    --language "$(cat /tmp/ace/language)" \
    --input-codes /tmp/codes.txt \
    --text-encoder checkpoints/Qwen3-Embedding-0.6B \
    --dit checkpoints/acestep-v15-turbo --vae checkpoints/vae \
    --seed 42 --output simple.wav
```


## Known Differences (C++ vs Python)

1. VAE decode: C++ full-frame, Python tiled_vae_decode -> edge differences
2. Float precision: bf16 op order, fused multiply-add -> accumulation errors
3. Constrained decoding: Implemented. C++ MetadataFSM validates metadata fields
   during CoT generation (BPM range, valid keyscales, etc.) matching Python FSM.
   Post-`</think>`: only audio codes 0-63999 + EOS.
