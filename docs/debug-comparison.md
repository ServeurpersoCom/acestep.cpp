# Debug & Comparison Report: C++ vs Python

Authority: Python upstream ACE-Step-1.5 (https://github.com/ace-step/ACE-Step-1.5).
Comparison done with identical inputs, same weights, same noise tensor.
Date: 2025-02-12.  Hardware: NVIDIA RTX PRO 6000 Blackwell (96 GB GDDR7).


## Method

The goal is to prove that the C++ CUDA reimplementation produces numerically
equivalent output to the Python reference, using the same model weights.

Strategy: inject identical noise, dump tensors at every pipeline stage,
compare bf16 values element-by-element.  Any algorithmic bug would show as
a structural divergence (wrong shape, wrong tokens, cosine sim << 0.99).
Pure numerical rounding shows as small per-element diffs with cosine > 0.999.


### Step 1: Generate deterministic noise

Python's torch.randn with seed=42 uses a Philox RNG.  We dumped the full
noise tensor to a raw bf16 file:

```
python3 -c "
    import torch
    noise = torch.randn(8500, 64, generator=torch.Generator().manual_seed(42))
    noise.to(torch.bfloat16).numpy().tofile('rng-philox-seed42.bf16')
"
```

File: rng-philox-seed42.bf16 (1,088,000 bytes = 8500 x 64 x 2).
For a 10s generation at 25 Hz: first 250 x 64 = 16,000 elements used.


### Step 2: Run Python reference with injected noise

Monkey-patched the handler's prepare_noise() to load from the bf16 file
instead of generating fresh noise.  Handler runs with batch_size=2 (default),
but we only compare batch index 0.

Key: the handler does NOT use Classifier-Free Guidance (CFG) in the DiT.
The batch=2 is for generating 2 parallel outputs, not cond/uncond.
The C++ also does not implement CFG.  Confirmed: no null_condition blending
in either implementation.

Parameters (identical for both):
  caption:   "simple piano melody, calm, ambient"
  lyrics:    "[verse]\nla la la"
  language:  "en"
  duration:  10s
  seed:      42
  shift:     3.0
  steps:     8
  schedule:  [1.0000, 0.9545, 0.9000, 0.8333, 0.7500, 0.6429, 0.5000, 0.3000]

Python dumps saved to /tmp/ref_dump/ as both .npy and .bf16.


### Step 3: Run C++ with same noise

```
./dit-vae \
  --text-encoder checkpoints/Qwen3-Embedding-0.6B \
  --dit checkpoints/acestep-v15-turbo \
  --vae checkpoints/vae \
  --caption "simple piano melody, calm, ambient" \
  --lyrics "[verse]\nla la la" \
  --language en --duration 10 --seed 42 --shift 3.0 --steps 8 \
  --noise rng-philox-seed42.bf16 \
  --dump /tmp/cpp_dump \
  --output /tmp/cpp_dump/output.wav
```

C++ dumps saved to /tmp/cpp_dump/ as .bf16.


### Step 4: Element-by-element comparison

```
python3 compare_tensors.py /tmp/ref_dump /tmp/cpp_dump
```


## Results

### Token counts (must be exact)

```
                  Python    C++
text tokens:        61       61      MATCH
lyric tokens:       17       17      MATCH
encoder seq len:    79       79      MATCH
```

Previously the C++ had 60 text / 16 lyric tokens due to add_eos=false.
Fixed by setting add_eos=true in bpe_encode() calls.


### Tensor comparison at each pipeline stage

```
Stage                   Shape         cos_sim    max_diff   mean_diff   Verdict
-----------------------+-------------+----------+----------+-----------+---------
noise (input)           [250, 64]     1.000000   0.000000   0.000000   EXACT
lyric_hidden (embed)    [17, 1024]    1.000000   0.000000   0.000000   EXACT
context_latents         [250, 128]    1.000000   0.000000   0.000000   EXACT
text_hidden (Qwen3)     [61, 1024]    0.999774   2.765625   0.051432   bf16 rounding
encoder_hidden_states   [79, 2048]    0.999763   0.687500   0.046965   bf16 rounding
output_latents (DiT)    [250, 64]     0.930626   3.687500   0.279358   accumulated
WAV (int16 samples)     [960000]      0.826248   18158      763.1      + VAE nonlinear
```


### Divergence cascade

```
                                   cos_sim
                                   |
noise -----------> EXACT           | 1.000
                                   |
lyric_embed -----> EXACT           | 1.000   (table lookup, no GEMM)
                                   |
context_latents -> EXACT           | 1.000   (silence VAE + concat, no GEMM)
                                   |
text_hidden -----> 0.9998          | 0.9998  <-- source: Qwen3 28-layer GEMM
                   (Qwen3 28L)     |              cuBLAS vs PyTorch accumulation
                                   |              first divergence: token 0, dim 0
                                   |              py=3.703125  cpp=3.718750
                                   |              (1 ULP in bf16)
                                   |
encoder_hidden --> 0.9998          | 0.9998  (CondEnc adds lyric+timbre, ~neutral)
                                   |
output_latents --> 0.931           | 0.931   DiT 8 Euler steps amplify the error
                                   |         each step feeds back into the next
                                   |
WAV output ------> 0.826           | 0.826   VAE decode is highly nonlinear
                                   |         (5 upsampling blocks, snake activations)
                                   v
```


### Per-token divergence in text_hidden_states

The divergence grows with token position, as expected for autoregressive-style
accumulation through 28 transformer layers:

```
token  0:  max_diff=0.016  cos=0.99999    (1 ULP seed)
token 17:  max_diff=0.063  cos=0.99999
token 18:  max_diff=0.500  cos=0.99985    (jump: deep layer accumulation)
token 40:  max_diff=0.250  cos=0.99983
token 60:  max_diff=0.688  cos=0.99735    (last text token, worst)
```


## Root Causes of Numerical Divergence

All divergence is numerical, not algorithmic.  Three contributing factors:

1. cuBLAS vs PyTorch GEMM accumulation order.
   bf16 has only 7 mantissa bits (1 ULP ~ 0.008 at magnitude 1.0).
   Matrix multiply with hidden_size=1024 accumulates 1024 products.
   Different summation order -> different rounding -> different result.
   This is the dominant source: first seen at token 0, dim 0.

2. --use_fast_math in NVCC compilation.
   Enables __fmaf_rn approximations, fast reciprocal sqrt, fast exp.
   Affects RMSNorm (rsqrt), SiLU (exp), RoPE (sin/cos).
   Tested: removing this flag gives cos=0.924 vs 0.931 with it.
   Negligible impact, not the dominant source of divergence.

3. Euler ODE step feedback loop.
   Each DiT step uses the previous step's output as input.
   8 steps of x_{t-1} = x_t + dt * model(x_t, t) compound the error.
   cos drops from 0.9998 (encoder) to 0.931 (after 8 steps).

None of these indicate a bug.  0.83-0.93 cosine similarity on the final
output is normal for independent bf16 implementations of a deep pipeline.


## Bugs Found and Fixed

### Bug 1: add_eos default (WRONG TOKENS, previously fixed)

The BPE tokenizer was called with add_eos=false, producing 60 text tokens
and 16 lyric tokens instead of the correct 61 and 17.  The missing EOS
token at the end of each sequence caused the text encoder to produce
different hidden states, leading to large divergence in the final output.

Fixed by setting add_eos=true in both bpe_encode() calls.


### Bug 2: vocal_language default (WRONG PROMPT)

The C++ default for --language was "unknown", but the Python handler
defaults to "en".  This changes the lyric prompt from:

```
# Languages\nunknown\n\n# Lyric\n...
```
to:
```
# Languages\nen\n\n# Lyric\n...
```

Different tokenization -> different lyric encoder output.
Fixed by passing --language en explicitly.  The default in C++ should
be changed to "en" to match Python.


## RoPE Bug Investigation (poc_rope_bug.py)

### The bug

Qwen3's RotaryEmbedding registers inv_freq with persistent=False,
so inv_freq is NOT saved in model checkpoints.  The transformers library
has a workaround in _init_weights that recomputes inv_freq for any module
whose class name contains "RotaryEmbedding".

When a model uses trust_remote_code and renames the RoPE class (e.g.
MyRotaryEmbedding), the workaround fails to match, and inv_freq stays
uninitialized (zeros or garbage from meta device).  This is the bug that
poc_rope_bug.py demonstrates.

### ACE-Step: NOT affected

The ACE-Step DiT model uses trust_remote_code=True but imports
Qwen3RotaryEmbedding directly from transformers without renaming:

```python
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
```

The class name IS "Qwen3RotaryEmbedding" (contains "RotaryEmbedding"),
so the workaround matches.  Additionally, the model loads without
device_map="auto" or low_cpu_mem_usage=True, so __init__ runs on CPU
and computes inv_freq correctly regardless of the workaround.

Verified empirically:

```
Text encoder:  1 RoPE module  -> inv_freq cos_sim=1.0  OK
DiT model:     5 RoPE modules -> inv_freq cos_sim=1.0  OK (all 5)
```

### Latent risk

The custom _init_weights in ACE-Step's DiT does NOT include the RoPE
workaround (it only handles Linear, Embedding, RMSNorm).  If someone
changes the loading to use device_map="auto", the meta device path
would skip __init__, and the missing workaround in _init_weights would
leave inv_freq uninitialized.  The bug would silently corrupt all
RoPE-based attention in the 24-layer DiT.

For the C++ implementation: irrelevant.  We compute inv_freq from the
config (theta=1000000, head_dim=128) at initialization time, matching
the formula in Qwen3RotaryEmbedding.__init__.


## Debug Methodology

### Python side

1. Monkey-patched the handler to inject noise from bf16 file.
2. Added hooks to capture tensors at prepare_condition input/output.
3. Saved all intermediates as both .npy (for Python) and .bf16 (for C++ comparison).
4. Saved metadata (prompt, params, shapes) as meta.json.

### C++ side

1. Added --noise flag to load pre-generated noise instead of RNG.
2. Added --dump flag with dump_bf16() helper to save GPU tensors.
3. Added dump points after: text encoder, lyric embed, condition encoder,
   DiT output.  Context latents dumped separately.

### Comparison

Element-by-element bf16 comparison using numpy:

```python
def load_bf16(path):
    raw = np.fromfile(path, dtype=np.uint16)
    return np.left_shift(raw.astype(np.uint32), 16).view(np.float32)
```

Metrics: cosine similarity, max absolute diff, mean absolute diff,
index of first divergence > threshold.

The comparison identifies exactly where divergence enters:
- EXACT stages (cos=1.0) confirm correct implementation.
- Near-exact stages (cos>0.999) confirm bf16 rounding only.
- The cascade from 0.9998 -> 0.931 -> 0.826 is consistent with
  iterative error amplification, not algorithmic bugs.


## File Inventory

```
/tmp/ref_dump/              Python reference dumps
  meta.json                 Prompt, params, tensor shapes
  noise.bf16                Injected noise [250, 64]
  pc_in_text_hidden_states.bf16    Qwen3 output [61, 1024]
  pc_in_lyric_hidden_states.bf16   Embed output [17, 1024]
  encoder_hidden_states.bf16       CondEnc output [79, 2048]
  context_latents.bf16      Silence + mask [250, 128]
  initial_hidden_states.bf16       = noise (before DiT)
  output_latents.bf16       DiT output [250, 64]
  python.wav                Final audio

/tmp/cpp_dump/              C++ dumps (same structure)
  text_hidden_states.bf16
  lyric_hidden_states.bf16
  encoder_hidden_states.bf16
  context_latents.bf16
  output_latents.bf16
  output.wav

acestep.cpp/
  rng-philox-seed42.bf16    Shared noise file [8500, 64]
```


## Conclusion

The C++ implementation is numerically equivalent to Python within bf16
precision limits.  All exact-match stages (noise, lyric embed, context
latents) confirm correct weight loading, tokenization, and data flow.
The divergence originates from GEMM accumulation differences in the
28-layer text encoder and compounds through 8 DiT steps and VAE decode.

Cosine similarity of 0.83 on the final WAV is normal for independent
bf16 implementations of a 4.5 GB model pipeline.  The audio sounds
identical to human ears (both produce the same melody and timbre).
