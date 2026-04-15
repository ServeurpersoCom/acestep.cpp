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
| VAE | vae-BF16.gguf (always this one) | 322 MB |

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

## LoRA

Drop LoRA adapters in the `loras/` folder and restart the server.
Supports PEFT directories and ComfyUI single `.safetensors` files.
Select the active LoRA from the WebUI.

## Server options

```
--models <dir>       Model directory (required)
--loras <dir>        LoRA adapters directory
--host <addr>        Listen address (default: 127.0.0.1)
--port <N>           Listen port (default: 8080)
--max-batch <N>      LM batch limit 1-9 (default: 1)
--vae-chunk <N>      VAE tile size (default: 256, lower = less VRAM)
--mp3-bitrate <N>    MP3 kbps (default: 128)
--wav-format <fmt>   WAV audio format (default: wav16)
                       Supported values: wav, wav16, wav24, wav32
                         wav/wav16: 16-bit signed-integer PCM audio
                         wav24: 24-bit signed-integer PCM audio
                         wav32: 32-bit IEEE floating-point PCM audio
                           (wav32 disables normalization & peak clip)
```

<details>
<summary>API endpoints</summary>

The server exposes three POST endpoints and two GET endpoints:

**POST /lm** - Generate lyrics and audio codes from a caption. Returns JSON.

**POST /synth** - Render audio codes into MP3 or WAV (`?wav=1`).
Accepts JSON or multipart (with source audio for cover/repaint modes).

**POST /understand** - Reverse pipeline: audio in, metadata + lyrics + codes out.
Accepts multipart (audio file) or JSON (codes-only).

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
    --request /tmp/request.json \
    --lm models/acestep-5Hz-lm-4B-Q8_0.gguf

# DiT + VAE render to audio
./build/ace-synth \
    --request /tmp/request0.json \
    --embedding models/Qwen3-Embedding-0.6B-Q8_0.gguf \
    --dit models/acestep-v15-turbo-Q8_0.gguf \
    --vae models/vae-BF16.gguf
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
