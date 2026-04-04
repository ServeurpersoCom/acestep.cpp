@echo off

set PATH=%~dp0build\Release;%PATH%

rem If you have both, uncomment one to use it
rem set GGML_BACKEND=CUDA0
rem set GGML_BACKEND=Vulkan0

ace-server.exe ^
    --host 0.0.0.0 ^
    --port 8085 ^
    --models .\models ^
    --loras .\loras ^
    --max-batch 1 ^
    --vae-chunk 1024 ^
    --unload-synth ^
    --unload-lm

pause
