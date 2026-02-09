# prompt.json Pipeline - Developer Reference

How `prompt.json` flows through the C++ LLM (`main.cu`) and DiT (`pipeline.cu`).

## prompt.json Schema

```json
{
  "caption":        "Vibrant French house...",     // string, required
  "lyrics":         "[Verse 1]\nOptimisation...",  // string, required
  "duration":       220,                           // number (seconds), optional (fallback 120)
  "bpm":            124,                           // number, optional (0 = not set)
  "vocal_language": "fr",                          // string, optional
  "keyscale":       "F# minor",                    // string, optional
  "timesignature":  "4",                           // string (quoted), optional
  "instrumental":   false                          // bool, optional (DiT only)
}
```

`duration` and `bpm` are JSON numbers (no quotes). `timesignature` is a quoted string.

`seed` is NOT part of the music description. It is a CLI runtime parameter (`--seed`).

## Field Routing

Each JSON field goes to specific consumers. No field should leak into the wrong template.

```
prompt.json
|
+--- main.cu (LLM, generates audio codes)
|    |
|    +-- caption ----------> User turn: "# Caption\n{caption}"
|    +-- lyrics ------------> User turn: "# Lyric\n{lyrics}"
|    +-- bpm ---------------> CoT only: "bpm: 124"
|    +-- duration ----------> CoT only: "duration: 220"
|    +-- keyscale ----------> CoT only: "keyscale: F# minor"
|    +-- timesignature -----> CoT only: "timesignature: 4"
|    +-- vocal_language ----> NOT IN LLM PROMPT. DiT only.
|    +-- instrumental ------> NOT USED BY LLM.
|
+--- pipeline.cu (DiT, generates audio from codes)
     |
     +-- caption ----------> Text prompt: "# Caption\n{caption}"
     +-- lyrics ------------> Lyric prompt: "# Lyric\n{lyrics}"
     +-- vocal_language ----> Lyric prompt: "# Languages\n{lang}"
     +-- bpm ---------------> Text prompt Metas: "- bpm: 124"
     +-- keyscale ----------> Text prompt Metas: "- keyscale: F# minor"
     +-- timesignature -----> Text prompt Metas: "- timesignature: 4"
     +-- duration ----------> Text prompt Metas: "- duration: 220 seconds"
     |                        + T_25Hz computation: duration * 25
     +-- instrumental ------> Lyrics override: "[Instrumental]"
```

## LLM Modes (main.cu)

### Mode 1: All metas provided (fast path)

When ALL 4 metadata fields are present in prompt.json (bpm + keyscale + timesignature + duration),
the CoT is pre-built and injected. The LLM only generates audio codes.

This matches the upstream behavior: their web interface skips Phase 1 when the user fills all fields.

```
has_all_metas = bpm > 0 AND keyscale not empty AND timesignature not empty AND duration > 0
```

Prompt sent to LLM:
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
bpm: 124
duration: 220
keyscale: F# minor
timesignature: 4
</think>
```
LLM generates audio codes after this point. max_tokens = duration * 5 + 100.

### Mode 2: Partial/no metas (LLM thinks for itself)

When any metadata field is missing, no CoT is injected. The LLM generates its own
`<think>...</think>` block first (guessing missing metas), then continues with audio codes.

This matches the upstream web interface default: the LLM decides BPM, key, etc.

Prompt sent to LLM:
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
LLM generates `<think>bpm: ...\n...</think>` then codes. max_tokens = duration * 5 + 800
(extra budget for CoT generation).

### CoT Format (critical, must match training data)

The `<think>` block contains up to 4 fields, alphabetical order (yaml.dump sort_keys=True):

| Field | Type | Notes |
|-------|------|-------|
| bpm | int | Beats per minute |
| duration | int | Seconds, cast to int |
| keyscale | string | e.g. "F# minor" |
| timesignature | int/string | e.g. "4" or 4 |

**Fields that MUST NOT appear in injected CoT:**
- `caption`: already in the user turn
- `language`: not a CoT field, goes to DiT lyric prompt only

The LLM was fine-tuned on this exact format. Extra fields = out-of-distribution = corrupted codes.

### CFG (Classifier-Free Guidance)

Unconditional prompt for CFG: empty CoT, caption removed or replaced by negative_prompt.

```
<|im_start|>assistant
<think>
</think>
```

## DiT Prompt (pipeline.cu)

### Text Prompt (TextEncoder input)

```
# Instruction
{instruction}

# Caption
{caption}

# Metas
- bpm: 124
- timesignature: 4
- keyscale: F# minor
- duration: 220 seconds
<|endoftext|>
```

Instruction depends on mode:
- text2music: "Fill the audio semantic mask based on the given conditions:"
- cover (when --input-codes provided): "Generate audio semantic tokens based on the given conditions:"

### Lyric Prompt (TextEncoder input)

```
# Languages
fr

# Lyric
{lyrics}<|endoftext|>
```

### Encoding Pipeline

```
text_prompt --> BPE tokenize --> TextEncoder --> text_hidden_states --+
                                                                      +--> ConditionEncoder --> DiT
lyric_prompt -> BPE tokenize --> TextEncoder --> lyric_hidden_states -+
```

## Key Differences: LLM vs DiT Prompt Formats

| Aspect | LLM (main.cu) | DiT (pipeline.cu) |
|--------|----------------|-------------------|
| Template | Qwen3 chat (im_start/im_end) | Flat text with endoftext |
| Metas format | YAML in `<think>` block | Bulleted list (- bpm: ...) |
| Metas content | 4 fields only | 4 fields + "seconds" suffix on duration |
| Caption | In user turn | In # Caption section |
| Lyrics | In user turn | In separate lyric prompt |
| Language | NOT in prompt | In lyric prompt # Languages header |
| Instruction | In system turn | In # Instruction header |

## YAML Quoting (TODO)

Python uses `yaml.dump(sort_keys=True)` which may quote values with special characters.
Current test: `yaml.dump({"keyscale": "F# minor"})` produces `keyscale: F# minor\n` (unquoted).
C++ does raw concatenation which matches for simple cases but may diverge on edge cases.

Needs systematic verification with the full set of possible keyscale values.

## Bugs Fixed

1. **duration/seed parsing**: `json_string()` returned "" for unquoted JSON numbers.
   New `json_get()` handles both quoted strings and unquoted numbers.

2. **has_meta OR -> AND**: Was triggering CoT injection with a single field present
   (including vocal_language which is not a CoT field). Now requires all 4 metadata fields.

3. **build_cot_text "all" removed**: Was emitting 6 fields including caption and language
   in the CoT block. Now emits only the 4 training-format fields.
   Dead code removed: `cot_has()` function, `"all"` mode.
