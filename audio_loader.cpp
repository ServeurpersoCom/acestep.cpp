// audio_loader.cpp: MP3 decode for reference audio (minimp3, no deps, no temp files)

#define MINIMP3_IMPLEMENTATION
#include "third_party/minimp3.h"

#include "wav.h"
#include "audio.h"
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

static bool path_ends_with_ci(const char * path, const char * suffix) {
    size_t pl = strlen(path), sl = strlen(suffix);
    if (pl < sl) return false;
    const char * p = path + pl - sl;
    for (size_t i = 0; i < sl; i++) {
        char a = (char)(p[i] >= 'A' && p[i] <= 'Z' ? p[i] + 32 : p[i]);
        char b = (char)(suffix[i] >= 'A' && suffix[i] <= 'Z' ? suffix[i] + 32 : suffix[i]);
        if (a != b) return false;
    }
    return true;
}

static void pcm_to_float_stereo_48k(
    const int16_t * pcm, size_t num_samples, int channels, unsigned int sample_rate,
    std::vector<float> * out)
{
    const float scale = 1.0f / 32768.0f;
    out->resize(num_samples * 2);
    if (channels == 1) {
        for (size_t i = 0; i < num_samples; i++) {
            float s = (float)pcm[i] * scale;
            (*out)[i * 2] = s;
            (*out)[i * 2 + 1] = s;
        }
    } else {
        for (size_t i = 0; i < num_samples * 2; i++)
            (*out)[i] = (float)pcm[i] * scale;
    }

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
    }
}

int mp3_load_48k_stereo(const char * path, std::vector<float> * out) {
    FILE * f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz > 200 * 1024 * 1024) {
        fclose(f);
        return -1;
    }
    std::vector<uint8_t> buf((size_t)sz);
    if (fread(buf.data(), 1, (size_t)sz, f) != (size_t)sz) {
        fclose(f);
        return -1;
    }
    fclose(f);

    mp3dec_t dec;
    mp3dec_init(&dec);
    mp3dec_frame_info_t info;
    std::vector<int16_t> pcm;
    const uint8_t * read_pos = buf.data();
    int remaining = (int)buf.size();
    int first_hz = 0, first_ch = 0;
    const size_t max_samples = (size_t)(60 * 48000 * 2);

    while (remaining > 0) {
        size_t old_size = pcm.size();
        if (old_size + (size_t)MINIMP3_MAX_SAMPLES_PER_FRAME > max_samples) break;
        pcm.resize(old_size + (size_t)MINIMP3_MAX_SAMPLES_PER_FRAME);
        int frame_samples = mp3dec_decode_frame(&dec, read_pos, remaining, pcm.data() + old_size, &info);
        if (frame_samples <= 0) {
            pcm.resize(old_size);
            read_pos++;
            remaining--;
            continue;
        }
        if (first_hz == 0) {
            first_hz = info.hz;
            first_ch = info.channels;
        }
        pcm.resize(old_size + (size_t)(frame_samples * info.channels));
        read_pos += info.frame_bytes;
        remaining -= info.frame_bytes;
    }

    if (pcm.empty() || first_hz == 0) return -1;
    size_t num_samples = pcm.size() / (size_t)first_ch;
    pcm_to_float_stereo_48k(pcm.data(), num_samples, first_ch, (unsigned)first_hz, out);
    return (int)(out->size() / 2);
}

int load_audio_48k_stereo(const char * path, std::vector<float> * out) {
    if (!path || !out) return -1;
    if (path_ends_with_ci(path, ".mp3"))
        return mp3_load_48k_stereo(path, out);
    if (path_ends_with_ci(path, ".wav"))
        return wav_load_48k_stereo(path, out);
    return -1;
}
