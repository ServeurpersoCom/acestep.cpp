@echo off

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

rem rd /s /q build 2>nul
mkdir build 2>nul
cd build

cmake .. -DGGML_CUDA=ON -DGGML_VULKAN=OFF
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..
