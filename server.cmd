@echo off

set PATH=%~dp0build\Release;%PATH%

rem Multi-GPU: set GGML_BACKEND to pick a device (CUDA0, CUDA1, Vulkan0...)
rem set GGML_BACKEND=CUDA0
rem set GGML_BACKEND=Vulkan0

ace-server.exe ^
    --host 0.0.0.0 ^
    --port 8085 ^
    --models .\models ^
    --loras .\loras ^
    --max-batch 1

pause
