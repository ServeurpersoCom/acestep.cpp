# ace-cuda: CUDA native music generation (ACE-Step) + autoregressive LM
# Hardware: NVIDIA RTX PRO 6000 Blackwell (sm_120)

NVCC     ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CFLAGS   = -O3 -std=c++17 --use_fast_math
LDFLAGS  = -lcublas

# GPU arch (override: make SM=89)
SM ?= 120
ifeq ($(SM),native)
  ARCHFLAG = -arch=native
else
  ARCHFLAG = -arch=sm_$(SM)
endif

# Targets
all: pipeline ace-qwen3 test_bpe pt2bin

# Text+lyrics -> WAV music generation pipeline
pipeline: pipeline.cu kernels.cuh dit.cuh transformer.cuh text_encoder.cuh condition.cuh tokenizer.cuh vae.cuh safetensors.h bpe.h
	$(NVCC) $(CFLAGS) $(ARCHFLAG) -o $@ pipeline.cu $(LDFLAGS)

# Standalone Qwen3 autoregressive LM
ace-qwen3: main.cu kernels.cuh safetensors.h bpe.h
	$(NVCC) $(CFLAGS) $(ARCHFLAG) -o $@ main.cu $(LDFLAGS)

# BPE tokenizer regression test (CPU only)
test_bpe: test_bpe.cpp bpe.h
	g++ -O2 -std=c++17 -o $@ test_bpe.cpp

# silence_latent.pt -> .bin converter (CPU only, replaces Python)
pt2bin: pt2bin.cpp
	g++ -O2 -std=c++17 -o $@ pt2bin.cpp

test: test_bpe
	./test_bpe

clean:
	rm -f pipeline ace-qwen3 test_bpe pt2bin

.PHONY: all clean test
