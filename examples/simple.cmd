@echo off

set PATH=%~dp0..\build-msvc\bin\Release;%PATH%
copy /y simple.json "%TEMP%\request.json"

..\build-msvc\Release\ace-qwen3.exe ^
    --request "%TEMP%\request.json" ^
    --model ..\models\acestep-5Hz-lm-4B-bf16.gguf

..\build-msvc\Release\dit-vae.exe ^
    --request "%TEMP%\request.json" ^
    --text-encoder ..\models\Qwen3-Embedding-0.6B-bf16.gguf ^
    --dit ..\models\acestep-v15-turbo-bf16.gguf ^
    --vae ..\models\vae-bf16.gguf ^
    --output simple.wav

pause
