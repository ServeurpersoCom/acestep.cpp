# Prompt & Pipeline Reference (acestep.cpp)

Authority: Python upstream ACE-Step-1.5 (github.com/ace-step/ACE-Step-1.5).
C++ must produce identical token sequences and equivalent audio.


## Architecture Overview

```
prompt.json
|
+--> ace-qwen3 (main.cu) -- LLM Qwen3 4B
|    Reads: caption, lyrics, bpm, duration, keyscale, timesignature
|    Writes: /tmp/codes.txt (comma-separated audio code ints)
|
+--> pipeline (pipeline.cu) -- TextEncoder + DiT + VAE
     Reads: prompt.json + /tmp/codes.txt + noise + checkpoints
     Writes: output.wav (48kHz stereo)
```


## Sampling Parameters

### LLM (ace-qwen3)

| Parameter      | acestep.cpp default | Python upstream default | Notes                          |
|----------------|---------------------|------------------------|--------------------------------|
| cfg_scale      | 2.2                 | 2.0                    | Ours gives better coherence    |
| temperature    | 0.80                | 0.85                   | Ours reduces hallucinated codes|
| top_p          | 0.9                 | 0.9                    | Same                           |
| top_k          | disabled            | 0 (disabled)           | Same                           |
| seed           | 42 (run.sh)         | random                 | We fix it for reproducibility  |
| rep_penalty    | 1.0 (none)          | 1.0 (none)             | Same                           |

### DiT (pipeline)

| Parameter      | acestep.cpp default | Python upstream default | Python UI default | Notes                    |
|----------------|---------------------|------------------------|-------------------|--------------------------|
| steps          | 8                   | 8                      | 8                 | Same                     |
| shift          | 3.0                 | 1.0                    | 3.0               | UI overrides code default|
| guidance_scale | NOT IMPLEMENTED     | 7.0                    | 7.0 (hidden)      | C++ has no DiT CFG yet   |
| infer_method   | ode (euler)         | ode                    | ode               | Same                     |
| seed           | 42 (run.sh)         | random                 | random            | We fix it for repro      |

Note: Python handler.py has shift=1.0 as code default but the Gradio UI sets it
to 3.0. The turbo model (acestep-v15-turbo) is designed for shift=3.0.


## Token IDs (Qwen3 vocabulary)

```
151643  <|endoftext|>           EOS for DiT text encoder
151644  <|im_start|>            TOKEN_IM_START
151645  <|im_end|>              TOKEN_IM_END (also LLM EOS)
151667  <think>                 TOKEN_THINK
151668  </think>                TOKEN_THINK_END
151669  <|audio_code_0|>        AUDIO_CODE_BASE
217203  <|audio_code_65534|>    Last token in vocab
```

CRITICAL: The vocab has tokens for audio codes 0-65534, but Python constrained
decoding only allows codes 0-63999 (codebook size = 64000). Codes 64000-65534
exist in the vocab but are never emitted during generation.

Our C++ currently allows 0-65534 (AUDIO_CODE_COUNT=65535). This should be
tightened to 0-63999 to match Python.


## LLM Prompt (main.cu)

### Two Modes

Mode 1 -- All metas present (bpm > 0 AND keyscale AND timesignature AND duration > 0):
  CoT is injected from prompt.json. LLM only generates audio codes.
  max_tokens = duration * 5 + 100

Mode 2 -- Any meta missing:
  No CoT injected. LLM generates its own <think>...</think> then audio codes.
  max_tokens = duration * 5 + 800 (extra budget for CoT auto-generation)

Detection (same in C++ and Python):
```
has_all_metas = bpm > 0 && !keyscale.empty() && !timesignature.empty() && duration > 0
```
vocal_language is NOT a meta field for this check.

### System Instruction

Both modes use the same system instruction:
```
Generate audio semantic tokens based on the given conditions:
```
This is DEFAULT_LM_INSTRUCTION in Python (constants.py).
Note: the DiT uses a DIFFERENT instruction (see DiT section below).

### Mode 1: Conditional Prompt (CoT injected)

```
<|im_start|>system
# Instruction
Generate audio semantic tokens based on the given conditions:

<|im_end|>
<|im_start|>user
# Caption
{caption}

# Lyric
{lyrics}
<|im_end|>
<|im_start|>assistant
<think>
{cot_yaml}</think>

<|im_end|>
```

Token sequence for assistant turn (validated exact match C++ vs Python):
```
TOKEN_IM_START              151644
BPE("assistant\n")          77091 198
TOKEN_THINK                 151667      <- raw token ID, NOT BPE
BPE("\n")                   198
BPE("{cot_yaml}")           ...         <- BPE-encoded YAML lines
TOKEN_THINK_END             151668      <- raw token ID, NOT BPE
BPE("\n\n")                 271
TOKEN_IM_END                151645
BPE("\n")                   198
```

CRITICAL: <think> and </think> must be pushed as raw token IDs (151667/151668).
If passed through BPE they split into garbage subtokens (< + think + > = 3 tokens).

### Mode 1: Unconditional Prompt (CFG)

Empty CoT. Caption replaced by negative_prompt or removed.

```
<|im_start|>system
# Instruction
Generate audio semantic tokens based on the given conditions:

<|im_end|>
<|im_start|>user
# Caption                       <- OMITTED if no negative_prompt
{negative_prompt}                <- or caption removed entirely

# Lyric
{lyrics}
<|im_end|>
<|im_start|>assistant
<think>

</think>

<|im_end|>
```

The empty CoT contains: TOKEN_THINK + BPE("\n\n") + TOKEN_THINK_END.

Negative prompt logic (same in C++ and Python):
  - "NO USER INPUT" or empty string -> no meaningful negative prompt
  - Any other string -> used as caption replacement
  - When no meaningful negative prompt: caption section is removed entirely,
    user turn becomes just "# Lyric\n{lyrics}\n"

### Mode 2: Conditional Prompt (LLM generates CoT)

Same as Mode 1 conditional but WITHOUT the assistant turn pre-filled:
```
<|im_start|>system
# Instruction
Generate audio semantic tokens based on the given conditions:

<|im_end|>
<|im_start|>user
# Caption
{caption}

# Lyric
{lyrics}
<|im_end|>
<|im_start|>assistant
```
(generation prompt appended by apply_chat_template, LLM generates from here)

### Mode 2: Unconditional Prompt (CFG)

Same as Mode 1 uncond but for CoT phase. Python has subtlety here:
  - CoT phase: caption removed or replaced, no CoT at all (just generation prompt)
  - Codes phase: empty CoT injected (like Mode 1 uncond)

Our C++ does NOT implement two-phase generation. In Mode 2, C++ does single-pass
generation with the uncond prompt having no CoT. This is a known simplification.


## CoT Format

### Injected CoT (Mode 1, from user metadata)

4 fields only, alphabetical order (matching Python yaml.dump(sort_keys=True)):

```
bpm: 124
duration: 220
keyscale: F# minor
timesignature: 4
```

Fields NEVER present in Mode 1 injected CoT:
  - caption: already in user turn
  - language: not a CoT field, goes to DiT lyric prompt

### LLM-generated CoT (Mode 2)

When the LLM generates its own CoT in Mode 2, Python's constrained decoding FSM
allows up to 6 fields in alphabetical order:
  bpm, caption, duration, keyscale, language, timesignature

After Phase 1 (CoT generation), Python parses the CoT, then re-injects ALL parsed
fields (including caption and language) into the Phase 2 prompt via
_format_metadata_as_cot(). Caption and language from CoT can also override
the user-provided values for the DiT prompt.

Our C++ does not implement Phase 1/Phase 2 separation. In Mode 2, the LLM
generates CoT + codes in a single pass.

### YAML Formatting Rules

Python converts numeric strings to int before yaml.dump:
```python
if isinstance(value, str) and value.isdigit():
    value = int(value)
```
This makes `bpm: 124` (unquoted int) vs `bpm: '124'` (quoted string).

Python strips "/4" suffix from timesignature:
```python
if key == "timesignature" and value.endswith("/4"):
    value = value.split("/")[0]    # "4/4" -> "4", then int("4") -> 4
```
Non-"/4" values like "6/8" stay as strings. yaml.dump leaves "6/8" unquoted
(not a YAML reserved word).

C++ does raw string concatenation which matches Python output by accident:
  - Integers print without quotes (same as yaml.dump for int)
  - keyscale is never quoted by PyYAML (F#, C#, Db etc. are safe)
  - timesignature "6/8" is unquoted in both

C++ does NOT strip "/4" from timesignature. If someone passes "4/4", C++ emits
"timesignature: 4/4" while Python emits "timesignature: 4". This is a known
divergence but unlikely in practice (UI sends "4" not "4/4" by default).

YAML bool traps (yes/no/true/false) are quoted by PyYAML but never appear in
our music metadata values.

### Constrained Decoding

Python implements a full FSM (constrained_logits_processor.py) that enforces:
  - Alphabetical field order
  - BPM range: 30-300
  - Duration range: 10-600 seconds
  - Valid keyscales only (note + optional accidental + mode)
  - Valid time signatures only
  - Audio codes range: 0-63999

After </think>, only audio code tokens + EOS are allowed.

Our C++ only implements the post-</think> constraint (audio codes + EOS).
No metadata validation during CoT generation. This is acceptable in Mode 1
(CoT is injected, not generated) but matters in Mode 2.


## DiT Prompt (pipeline.cu)

### Text Prompt (caption branch)

Template from Python constants.py SFT_GEN_PROMPT:
```
# Instruction
{instruction}

# Caption
{caption}

# Metas
- bpm: {bpm}
- timesignature: {timesignature}
- keyscale: {keyscale}
- duration: {duration} seconds
<|endoftext|>
```

Instruction depends on mode:
  - text2music: "Fill the audio semantic mask based on the given conditions:"
  - cover (--input-codes): "Generate audio semantic tokens based on the given conditions:"

Note: the text2music instruction is DIFFERENT from the LLM instruction.
  LLM:  "Generate audio semantic tokens based on the given conditions:"
  DiT:  "Fill the audio semantic mask based on the given conditions:"

Missing metadata uses "N/A" as placeholder (matching Python _create_default_meta).

Duration always has " seconds" suffix in DiT metas (unlike LLM CoT where it is
just an integer).

### Lyric Prompt (lyric branch)

```
# Languages
{vocal_language}

# Lyric
{lyrics}<|endoftext|>
```

No space before <|endoftext|>, directly concatenated after lyrics.

### Encoding Pipeline

```
text_prompt --> BPE tokenize --> Qwen3-Embedding-0.6B --> text_hidden_states  --+
                                                                                +--> ConditionEncoder --> DiT
lyric_prompt -> BPE tokenize --> Qwen3-Embedding-0.6B --> lyric_hidden_states --+
```

Text encoder max_length = 256 tokens (truncation enabled).
The text encoder uses <|endoftext|> (151643) as EOS, not <|im_end|> (151645).


## LLM vs DiT Prompt Comparison

| Aspect           | LLM (main.cu)                    | DiT (pipeline.cu)                |
|------------------|----------------------------------|----------------------------------|
| Template         | Qwen3 chat (im_start/im_end)    | Flat text with <endoftext>       |
| Metas format     | YAML in <think> block            | Bulleted list (- key: value)     |
| Metas order      | Alphabetical (yaml.dump)         | bpm, timesig, keyscale, duration |
| Metas fields     | 4 fields (no suffix)             | 4 fields + "seconds" on duration |
| Missing metas    | Field omitted entirely           | "N/A" placeholder                |
| Caption          | In user turn                     | In # Caption section             |
| Lyrics           | In user turn                     | Separate lyric prompt            |
| Language         | NOT in LLM prompt                | In lyric prompt # Languages      |
| Instruction      | "Generate audio semantic tokens" | "Fill the audio semantic mask"   |
| EOS token        | 151645 (<im_end>)                | 151643 (<endoftext>)             |


## Web UI Flags (not implemented in C++)

Python has 3 boolean flags (all default True) controlling CoT generation:

  use_cot_metas:    Let LLM generate bpm/duration/keyscale/timesignature in CoT
  use_cot_caption:  Let LLM rewrite caption in CoT (feeds back to DiT)
  use_cot_language: Let LLM detect vocal language in CoT (feeds back to DiT)

These only matter in Mode 2 (LLM generates CoT). When all are True and the
LLM generates a CoT with caption/language, those values override the user input
for the DiT text/lyric prompts.

Our C++ ignores these flags. In Mode 1 they are irrelevant (CoT is injected).
In Mode 2 the LLM can generate whatever it wants in the CoT.


## Python Two-Phase vs C++ Single-Pass

Python generate_with_stop_condition() runs two phases:
  Phase 1: Generate CoT (stop_at_reasoning=True, stops at </think>)
  Phase 2: Parse CoT metadata, rebuild prompt with CoT injected, generate codes

This means Python ALWAYS injects the CoT for codes generation, even in Mode 2.
The LLM sees its own CoT as if it was pre-built.

Our C++ does single-pass in both modes:
  Mode 1: CoT injected, LLM generates codes only (same as Python Phase 2)
  Mode 2: LLM generates CoT + codes in one pass (differs from Python)

In Mode 1, C++ and Python produce identical token sequences.
In Mode 2, the token sequences differ because of the two-phase vs single-pass
approach, but the semantic result should be equivalent (same CoT, same codes).


## Audio Code Output

LLM emits tokens <|audio_code_N|> where N is the code value.
C++ extracts: code = token_id - AUDIO_CODE_BASE (151669).
Output format: comma-separated integers in /tmp/codes.txt.

Expected code count: approximately duration * 5 (5Hz codec).
For duration=220: ~1100 codes.

Valid code range:
  Python: 0-63999 (MAX_AUDIO_CODE=63999, codebook_size=64000)
  C++:    0-65534 (AUDIO_CODE_COUNT=65535) <-- BUG: should be 64000


## Known Divergences

1. Audio code range: C++ allows 0-65534, Python constrains to 0-63999
2. LLM RNG: C++ mt19937, Python Philox -> different codes (expected)
3. VAE decode: C++ full-frame, Python tiled_vae_decode -> edge differences
4. Float precision: bf16 op order, fused multiply-add -> accumulation errors
5. timesignature "/4" stripping: Python does it, C++ does not
6. DiT CFG: Python has guidance_scale=7.0, C++ not implemented
7. Mode 2 two-phase: Python splits CoT/codes, C++ single-pass
8. Constrained decoding: Python FSM validates metadata, C++ only constrains
   post-</think> to audio codes + EOS


## prompt.json Schema

```json
{
  "caption":        "string, music description",
  "lyrics":         "string, with [Section] markers and newlines",
  "duration":       220,
  "bpm":            124,
  "vocal_language": "fr",
  "keyscale":       "F# minor",
  "timesignature":  "4"
}
```

All fields optional. Missing fields trigger Mode 2 (LLM generates CoT).
seed is NOT in prompt.json (it is a CLI parameter: --seed).
negative_prompt is NOT in prompt.json (CLI parameter: --negative-prompt).
