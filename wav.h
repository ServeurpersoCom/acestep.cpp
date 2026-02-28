// wav.h: minimal WAV loader for reference audio (stereo 48kHz float out)
// No Python or external deps. Handles 16-bit PCM, mono/stereo, resamples to 48kHz if needed.

#pragma once

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// Load WAV file into stereo float32 at 48kHz.
// Out: interleaved L,R,L,R,... length = num_samples (both channels).
// Returns num_samples (per channel), or -1 on error.
static int wav_load_48k_stereo(const char * path, std::vector<float> * out) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;

    char riff[4], fmt[4];
    if (fread(riff, 1, 4, f) != 4 || memcmp(riff, "RIFF", 4) != 0) {
        fclose(f);
        return -1;
    }
    uint32_t file_len;
    if (fread(&file_len, 4, 1, f) != 1) { fclose(f); return -1; }
    if (fread(fmt, 1, 4, f) != 4 || memcmp(fmt, "WAVE", 4) != 0) {
        fclose(f);
        return -1;
    }

    uint16_t channels = 2, bits = 16;
    uint32_t sample_rate = 48000;
    bool found_fmt = false;

    while (1) {
        char chunk_id[4];
        if (fread(chunk_id, 1, 4, f) != 4) break;
        uint32_t chunk_size;
        if (fread(&chunk_size, 4, 1, f) != 1) break;
        long chunk_start = ftell(f);

        if (memcmp(chunk_id, "fmt ", 4) == 0 && chunk_size >= 16) {
            uint16_t fmt_tag, block_align;
            uint32_t byte_rate;
            if (fread(&fmt_tag, 2, 1, f) != 1) break;
            if (fread(&channels, 2, 1, f) != 1) break;
            if (fread(&sample_rate, 4, 1, f) != 1) break;
            if (fread(&byte_rate, 4, 1, f) != 1) break;
            if (fread(&block_align, 2, 1, f) != 1) break;
            if (fread(&bits, 2, 1, f) != 1) break;
            found_fmt = true;
        } else if (memcmp(chunk_id, "data", 4) == 0 && found_fmt) {
            size_t num_bytes = chunk_size;
            size_t num_samples = num_bytes / (channels * (bits / 8));
            if (num_samples == 0) { fclose(f); return -1; }

            std::vector<int16_t> raw(num_samples * channels);
            if (fread(raw.data(), 2, raw.size(), f) != raw.size()) {
                fclose(f);
                return -1;
            }

            out->resize(num_samples * 2);
            float scale = 1.0f / 32768.0f;
            if (channels == 1) {
                for (size_t i = 0; i < num_samples; i++) {
                    float s = (float)raw[i] * scale;
                    (*out)[i * 2] = s;
                    (*out)[i * 2 + 1] = s;
                }
            } else {
                for (size_t i = 0; i < num_samples * 2; i++)
                    (*out)[i] = (float)raw[i] * scale;
            }

            fclose(f);

            // Resample to 48kHz if needed (linear interpolation)
            if (sample_rate != 48000) {
                size_t in_len = num_samples;
                size_t out_len = (size_t)((double)in_len * 48000.0 / (double)sample_rate);
                std::vector<float> resampled(out_len * 2);
                for (size_t i = 0; i < out_len; i++) {
                    double t = (double)i * (double)in_len / (double)out_len;
                    size_t i0 = (size_t)t;
                    size_t i1 = std::min(i0 + 1, in_len - 1);
                    float w = (float)(t - (double)i0);
                    for (int c = 0; c < 2; c++)
                        resampled[i * 2 + c] = (*out)[i0 * 2 + c] * (1.0f - w) + (*out)[i1 * 2 + c] * w;
                }
                *out = std::move(resampled);
                return (int)out_len;
            }
            return (int)num_samples;
        }

        fseek(f, chunk_start + (long)chunk_size, SEEK_SET);
    }
    fclose(f);
    return -1;
}
