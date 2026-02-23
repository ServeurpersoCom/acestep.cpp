CXX      ?= g++
CXXFLAGS ?= -std=c++17 -O2
INCLUDES  = -I..

test-philox: test-philox.cpp ../philox.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< -lm

clean:
	rm -f test-philox philox-noise.f32

.PHONY: clean
