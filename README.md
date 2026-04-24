# acestep.cpp

Local AI music generation server with browser UI, powered by GGML.
Describe a song, get stereo 48kHz audio. Runs on CPU, CUDA, Metal, Vulkan.

<img width="1704" height="773" alt="Light" src="https://github.com/user-attachments/assets/aeda150a-46a2-4542-a2d6-57d238a7bbb4" />
<img width="1705" height="771" alt="Dark" src="https://github.com/user-attachments/assets/4941cec9-b6ff-4e09-8905-bdc3ee06d222" />

## Download models

Grab one GGUF of each type from Hugging Face and drop them in the `models/` folder:

https://huggingface.co/Serveurperso/ACE-Step-1.5-GGUF/tree/main

| Type | Pick one | Size |
|------|----------|------|
| LM | acestep-5Hz-lm-4B-Q8_0.gguf | 4.2 GB |
| Text encoder | Qwen3-Embedding-0.6B-Q8_0.gguf | 748 MB |
| DiT | acestep-v15-turbo-Q8_0.gguf | 2.4 GB |
| VAE | vae-BF16.gguf | 322 MB |

Three LM sizes available: 0.6B (fast), 1.7B, 4B (best quality).
Multiple DiT variants: turbo (8 steps), sft (50 steps, higher quality), base, shift1, shift3, continuous.

Alternative: `./models.sh` downloads the default set automatically (needs `pip install hf`).

## Build

```
git clone --recurse-submodules https://github.com/ServeurpersoCom/acestep.cpp.git
cd acestep.cpp
```

### Windows

Pre-built binaries (until CI is set up): https://www.serveurperso.com/temp/acestep.cpp-win64/

To build from source, install
[Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(select "Desktop development with C++" workload) and optionally the
[CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) and/or the
[Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

```cmd
buildcuda.cmd     # NVIDIA GPU
buildvulkan.cmd   # AMD/Intel GPU (Vulkan)
buildall.cmd      # all backends (CUDA + Vulkan + CPU, runtime loading)
```

### Linux / macOS

```bash
./buildcuda.sh    # NVIDIA GPU
./buildvulkan.sh  # AMD/Intel GPU (Vulkan)
./buildcpu.sh     # CPU only (with BLAS)
./buildall.sh     # all backends (CUDA + Vulkan + CPU, runtime loading)
```

macOS auto-enables Metal and Accelerate BLAS with any of the above.

## Run

```bash
./server.sh       # Linux / macOS
server.cmd        # Windows
```

Open http://localhost:8085 in your browser. The WebUI handles everything:
write a caption, set lyrics and metadata, generate, play, and download tracks.

Models are loaded on first request (zero GPU at startup) and swapped
automatically when you pick a different one in the UI.

## Adapters

Drop adapters in the `adapters/` folder and restart the server.
Supports LoRA today in two flavours: PEFT directories (with
`adapter_model.safetensors` + `adapter_config.json`) and ComfyUI single
`.safetensors` files. Select the active adapter from the WebUI.

## Server options

```
Usage: ./ace-server --models <dir> [options]

Required:
  --models <dir>          Directory of GGUF model files

Adapter:
  --adapters <dir>        Directory of adapters

Memory control:
  --keep-loaded           Keep models in VRAM between requests
  --vae-chunk <N>         Latent frames per tile (default: 1024)
  --vae-overlap <N>       Overlap frames per side (default: 64)

Output:
  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)

Server:
  --host <addr>           Listen address (default: 127.0.0.1)
  --port <N>              Listen port (default: 8080)
  --max-batch <N>         LM batch limit (default: 1)
  --max-seq <N>           KV cache size (default: 8192)

Debug:
  --no-fsm                Disable FSM constrained decoding
  --no-fa                 Disable flash attention
  --no-batch-cfg          Split CFG into two separate forwards (LM + DiT)
  --clamp-fp16            Clamp hidden states to FP16 range
```

<details>
<summary>API endpoints</summary>

The server exposes four POST endpoints and two GET endpoints:

**POST /lm** - Generate lyrics and audio codes from a caption. Returns JSON.

**POST /synth** - Render audio codes into MP3 or WAV (selected by the
`output_format` field in the request JSON). Accepts JSON or multipart
(with source audio or pre-encoded latents for cover/repaint modes; latents
win over audio when both are sent on the same side).

**POST /understand** - Reverse pipeline: audio in, metadata + lyrics + codes out.
Multipart only (source audio or pre-encoded latents required, optional request JSON for params).

**POST /vae** - Standalone VAE entrypoint: send `audio` to encode (latents
out), send `src_latents` to decode (audio out). Multipart only, the two
inputs are mutually exclusive. Lets the webui cache a cover latent on an
existing card, or play back a .vae file, without paying the LM cost of a
full /synth or /understand pass.

Synth and understand responses are multipart/mixed: the primary payload
(audio tracks or JSON) plus the cover latent the pipeline produced. The
client can replay that latent back into a later /synth or /understand call
to skip the VAE encode entirely on subsequent jobs.

**GET /health** - Returns `{"status":"ok"}`.

**GET /props** - Available models, server config, default parameters.

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full API reference
and AceRequest JSON specification.

</details>

<details>
<summary>CLI tools (advanced)</summary>

For scripting without the server, `ace-lm` and `ace-synth` work as a pipe:

```bash
# LM generates lyrics + codes
./build/ace-lm \
    --models models \
    --request /tmp/request.json

# DiT + VAE render to audio
./build/ace-synth \
    --models models \
    --request /tmp/request0.json
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full JSON reference,
task types, batching, and understand pipeline.

</details>

## Technical documentation

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the complete AceRequest JSON
reference, all task types (text2music, cover, repaint, lego, extract, complete),
FSM constrained decoding, custom GGML operators, quantization, and architecture
internals.

## Community

### ACE-Step official documentation

- [A Musician's Guide](https://github.com/ace-step/ACE-Step-1.5/discussions/235) - non-technical guide for music makers
- [Tutorial](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md) - design philosophy, model architecture, input control, inference hyperparameters

### Third-party UIs for acestep.cpp

- [acestep-cpp-ui](https://github.com/audiohacking/acestep-cpp-ui)
- [acestep.cpp-simple-GUI](https://github.com/Nurb4000/acestep.cpp-simple-GUI)
- [aceradio](https://github.com/IMbackK/aceradio)

## Samples

https://github.com/user-attachments/assets/9a50c1f4-9ec0-474a-bd14-e8c6b00622a1

https://github.com/user-attachments/assets/fb606249-0269-4153-b651-bf78e05baf22

https://github.com/user-attachments/assets/e0580468-5e33-4a1f-a0f4-b914e4b9a8c2

https://github.com/user-attachments/assets/292a31f1-f97e-4060-9207-ed8364d9a794

https://github.com/user-attachments/assets/34b1b781-a5bc-46c4-90a6-615a10bc2c6a

## Acknowledgements

Independent C++ implementation based on
[ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) by ACE Studio and StepFun.
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
