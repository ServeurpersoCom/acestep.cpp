// wav.h: minimal WAV reader/writer (16-bit PCM stereo)
//
// read_wav_buf: PCM16 or float32, mono/stereo, any rate -> interleaved [T, 2] float
// write_wav:    planar [ch0: T, ch1: T] float -> PCM16 stereo WAV

#pragma once
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Read WAV from memory buffer.
// Returns interleaved float [T, 2]. Sets *T_audio, *sr. Caller frees.
static float * read_wav_buf(const uint8_t * data, size_t size, int * T_audio, int * sr) {
    *T_audio = 0;
    *sr      = 0;

    if (size < 12 || memcmp(data, "RIFF", 4) != 0 || memcmp(data + 8, "WAVE", 4) != 0) {
        fprintf(stderr, "[WAV] Not a valid WAV buffer\n");
        return NULL;
    }

    int     n_channels = 0, sample_rate = 0, bits_per_sample = 0;
    short   audio_format = 0;
    float * audio        = NULL;
    int     n_samples    = 0;
    size_t  pos          = 12;

    while (pos + 8 <= size) {
        const uint8_t * chunk_id   = data + pos;
        int             chunk_size = 0;
        memcpy(&chunk_size, data + pos + 4, 4);
        pos += 8;

        if (memcmp(chunk_id, "fmt ", 4) == 0 && pos + 16 <= size) {
            memcpy(&audio_format, data + pos, 2);
            short nc;
            memcpy(&nc, data + pos + 2, 2);
            n_channels = nc;
            memcpy(&sample_rate, data + pos + 4, 4);
            // skip byte_rate(4) + block_align(2)
            short bps;
            memcpy(&bps, data + pos + 14, 2);
            bits_per_sample = bps;
            pos += (size_t) chunk_size;

        } else if (memcmp(chunk_id, "data", 4) == 0 && n_channels > 0) {
            size_t data_bytes = (size_t) chunk_size;
            if (pos + data_bytes > size) {
                data_bytes = size - pos;
            }

            if (audio_format == 1 && bits_per_sample == 16) {
                n_samples         = (int) (data_bytes / ((size_t) n_channels * 2));
                audio             = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                const short * pcm = (const short *) (data + pos);
                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        float s          = (float) pcm[t] / 32768.0f;
                        audio[t * 2 + 0] = s;
                        audio[t * 2 + 1] = s;
                    } else {
                        audio[t * 2 + 0] = (float) pcm[t * n_channels + 0] / 32768.0f;
                        audio[t * 2 + 1] = (float) pcm[t * n_channels + 1] / 32768.0f;
                    }
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                n_samples          = (int) (data_bytes / ((size_t) n_channels * 4));
                audio              = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                const float * fbuf = (const float *) (data + pos);
                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        audio[t * 2 + 0] = fbuf[t];
                        audio[t * 2 + 1] = fbuf[t];
                    } else {
                        audio[t * 2 + 0] = fbuf[t * n_channels + 0];
                        audio[t * 2 + 1] = fbuf[t * n_channels + 1];
                    }
                }
            } else {
                fprintf(stderr, "[WAV] Unsupported: format=%d bits=%d\n", audio_format, bits_per_sample);
                return NULL;
            }
            break;
        } else {
            pos += (size_t) chunk_size;
        }
    }

    if (!audio) {
        fprintf(stderr, "[WAV] No audio data in buffer\n");
        return NULL;
    }

    *T_audio = n_samples;
    *sr      = sample_rate;
    fprintf(stderr, "[WAV] Read buffer: %d samples, %d Hz, %d ch, %d bit\n", n_samples, sample_rate, n_channels,
            bits_per_sample);
    return audio;
}
