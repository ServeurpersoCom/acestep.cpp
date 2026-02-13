# ace-cuda: CUDA native music generation (ACE-Step) + autoregressive LM
# Hardware: NVIDIA RTX PRO 6000 Blackwell (sm_120)
# Build: cd cuda && make

NVCC     ?= $(shell which nvcc 2>/dev/null || echo /usr/local/cuda/bin/nvcc)
CFLAGS   = -O3 -std=c++17 --use_fast_math -I..
LDFLAGS  = -lcublas

# GPU arch (override: make SM=89)
SM ?= 120
ifeq ($(SM),native)
  ARCHFLAG = -arch=native
else
  ARCHFLAG = -arch=sm_$(SM)
endif

# Shared headers
SHARED_HDRS = safetensors.h ../bpe.h

# Targets
all: dit-vae ace-qwen3 test-bpe pt2bin

# Text+lyrics -> WAV music generation (DiT + VAE)
dit-vae: dit-vae.cu kernels.cuh dit.cuh transformer.cuh text-encoder.cuh condition.cuh tokenizer.cuh vae.cuh $(SHARED_HDRS)
	$(NVCC) $(CFLAGS) $(ARCHFLAG) -o $@ dit-vae.cu $(LDFLAGS)

# Standalone Qwen3 autoregressive LM
ace-qwen3: ace-qwen3.cu kernels.cuh $(SHARED_HDRS)
	$(NVCC) $(CFLAGS) $(ARCHFLAG) -o $@ ace-qwen3.cu $(LDFLAGS)

# BPE tokenizer regression test (CPU only)
test-bpe: test-bpe.cpp ../bpe.h
	g++ -O2 -std=c++17 -I.. -o $@ test-bpe.cpp

# silence_latent.pt -> .bin converter (CPU only, replaces Python)
pt2bin: pt2bin.cpp
	g++ -O2 -std=c++17 -o $@ pt2bin.cpp

test: test-bpe
	./test-bpe

clean:
	rm -f dit-vae ace-qwen3 test-bpe pt2bin

.PHONY: all clean test
