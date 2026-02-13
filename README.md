# acestep.cpp

Portable C++17 implementation of ACE-Step 1.5 music generation using GGML.
Text + lyrics in, stereo 48kHz WAV out. Runs on CPU, CUDA, Metal, Vulkan.

## Build

```
git submodule update --init
cd build && cmake .. && make dit-vae
```

Enable GPU backends with cmake flags:

```
cmake .. -DGGML_CUDA=ON
cmake .. -DGGML_METAL=ON
cmake .. -DGGML_VULKAN=ON
```

## Usage

```
./build/dit-vae --text-encoder checkpoints/Qwen3-Embedding-0.6B \
                --dit checkpoints/acestep-v15-turbo \
                --vae checkpoints/vae \
                --caption "Hip hop, jazzy piano, vinyl crackle" \
                --lyrics "[Instrumental]" \
                --duration 60 --steps 8 \
                --output output.wav
```

## Pipeline

```
BPE tokenize
  -> Qwen3-Embedding (28L causal)
  -> CondEncoder (lyric 8L + timbre 4L + text_proj)
  -> DiT (24L flow matching, 8 Euler steps)
  -> VAE (AutoencoderOobleck)
  -> WAV stereo 48kHz
```

The autoregressive LLM stage (audio code generation) is not yet ported to
GGML. In the meantime, the full pipeline including LLM is available in the
standalone CUDA backend under `cuda/`.

## CUDA backend

Complete pipeline in native CUDA under `cuda/`, including the autoregressive
LLM (ace-qwen3). Requires nvcc + sm_120 (Blackwell).

```
cd cuda && make
```

## Accuracy (3-way comparison)

Full pipeline cosine similarity, seed 42 with Philox noise:

```
stage                     CUDA<>GGML   CUDA<>Python   GGML<>Python
text_hidden                 0.9998      0.9998      0.9998
lyric_embed                 0.9292      0.9292      1.0000
enc_hidden                  0.9998      0.9997      0.9998
noise                       1.0000      1.0000      1.0000
dit_step0_vt                0.9783      0.9777      0.9970
dit_step7_vt                0.9788      0.9831      0.9904
dit_x0                      0.9856      0.9890      0.9951
vae_audio (log spectral)    0.9125      0.9374      0.9770
```

GGML matches PyTorch to >0.99 across all stages. CUDA drift comes from bf16
accumulation in VAE and a minor BPE tokenization difference on lyrics.

## Checkpoints

```
bash checkpoints.sh
```

Downloads from HuggingFace:
  Qwen3-Embedding-0.6B (text encoder, 1.1GB),
  acestep-v15-turbo (DiT 24L + CondEncoder, 3GB),
  vae (AutoencoderOobleck, 322MB),
  acestep-5Hz-lm-{0.6B,1.7B,4B} (autoregressive LLM, CUDA only for now).

## Known issues

Uses a patched ggml fork (submodule). Two fixes for long-sequence audio:

- `im2col.cu`: gridDim.y overflow when T > 65535 patches
- `conv_transpose_1d.cu`: O(T_in) brute-force loop, too slow for VAE upsampling

TODO: verify if these are still needed on latest ggml and submit upstream PRs.

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
