@echo off

chcp 65001
set PATH=%~dp0..\build-msvc\bin\Release;%PATH%

..\build-msvc\Release\ace-qwen3.exe ^
    --request simple.json ^
    --model ..\models\acestep-5Hz-lm-4B-Q6_K.gguf

..\build-msvc\Release\dit-vae.exe ^
    --request simple0.json ^
    --text-encoder ..\models\Qwen3-Embedding-0.6B-Q8_0.gguf ^
    --dit ..\models\acestep-v15-turbo-Q6_K.gguf ^
    --vae ..\models\vae-BF16.gguf ^
    --output simple.wav

pause
