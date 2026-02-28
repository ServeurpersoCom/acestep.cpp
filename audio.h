// audio.h: unified reference-audio loader (WAV + MP3 → stereo 48kHz float)
// Header-only for WAV; MP3 implementation in audio_loader.cpp (minimp3, no temp files).

#pragma once

#include <cstddef>
#include <string>
#include <vector>

// Load WAV or MP3 file into stereo float32 at 48kHz.
// Out: interleaved L,R,L,R,...; length = num_samples (per channel).
// Returns num_samples (per channel), or -1 on error.
// No temp files; MP3 decoded in memory via minimp3 (header-only dep).
int load_audio_48k_stereo(const char * path, std::vector<float> * out);

// MP3 implementation (in audio_loader.cpp; do not call from other TUs without linking it)
int mp3_load_48k_stereo(const char * path, std::vector<float> * out);
