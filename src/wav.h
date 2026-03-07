// wav.h: minimal WAV reader/writer (16-bit PCM stereo)
//
// read_wav:  PCM16 or float32, mono/stereo, any rate -> interleaved [T, 2] float
// write_wav: planar [ch0: T, ch1: T] float -> PCM16 stereo WAV

#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// Read WAV file. Returns interleaved float [T, 2]. Sets *T_audio, *sr.
// Supports PCM16 (format=1) and float32 (format=3), mono or stereo.
// Mono is duplicated to stereo. Caller frees the returned pointer.
static float * read_wav(const char * path, int * T_audio, int * sr) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[WAV] Cannot open %s\n", path);
        return NULL;
    }

    char riff[4];
    fread(riff, 1, 4, f);
    if (memcmp(riff, "RIFF", 4) != 0) {
        fprintf(stderr, "[WAV] Not a RIFF file: %s\n", path);
        fclose(f);
        return NULL;
    }
    fseek(f, 4, SEEK_CUR);
    char wave[4];
    fread(wave, 1, 4, f);
    if (memcmp(wave, "WAVE", 4) != 0) {
        fprintf(stderr, "[WAV] Not a WAVE file: %s\n", path);
        fclose(f);
        return NULL;
    }

    int     n_channels = 0, sample_rate = 0, bits_per_sample = 0;
    short   audio_format = 0;
    float * audio        = NULL;
    int     n_samples    = 0;

    while (!feof(f)) {
        char chunk_id[4];
        int  chunk_size;
        if (fread(chunk_id, 1, 4, f) != 4) {
            break;
        }
        if (fread(&chunk_size, 4, 1, f) != 1) {
            break;
        }

        if (memcmp(chunk_id, "fmt ", 4) == 0) {
            fread(&audio_format, 2, 1, f);
            short nc;
            fread(&nc, 2, 1, f);
            n_channels = nc;
            fread(&sample_rate, 4, 1, f);
            fseek(f, 4, SEEK_CUR);
            fseek(f, 2, SEEK_CUR);
            short bps;
            fread(&bps, 2, 1, f);
            bits_per_sample = bps;
            int consumed    = 16;
            if (chunk_size > consumed) {
                fseek(f, chunk_size - consumed, SEEK_CUR);
            }

        } else if (memcmp(chunk_id, "data", 4) == 0 && n_channels > 0) {
            if (audio_format == 1 && bits_per_sample == 16) {
                n_samples = chunk_size / (n_channels * 2);
                audio     = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                std::vector<short> buf((size_t) n_samples * n_channels);
                fread(buf.data(), 2, (size_t) n_samples * n_channels, f);
                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        float s          = (float) buf[t] / 32768.0f;
                        audio[t * 2 + 0] = s;
                        audio[t * 2 + 1] = s;
                    } else {
                        audio[t * 2 + 0] = (float) buf[t * n_channels + 0] / 32768.0f;
                        audio[t * 2 + 1] = (float) buf[t * n_channels + 1] / 32768.0f;
                    }
                }
            } else if (audio_format == 3 && bits_per_sample == 32) {
                n_samples = chunk_size / (n_channels * 4);
                audio     = (float *) malloc((size_t) n_samples * 2 * sizeof(float));
                std::vector<float> buf((size_t) n_samples * n_channels);
                fread(buf.data(), 4, (size_t) n_samples * n_channels, f);
                for (int t = 0; t < n_samples; t++) {
                    if (n_channels == 1) {
                        audio[t * 2 + 0] = buf[t];
                        audio[t * 2 + 1] = buf[t];
                    } else {
                        audio[t * 2 + 0] = buf[t * n_channels + 0];
                        audio[t * 2 + 1] = buf[t * n_channels + 1];
                    }
                }
            } else {
                fprintf(stderr, "[WAV] Unsupported: format=%d bits=%d\n", audio_format, bits_per_sample);
                fclose(f);
                return NULL;
            }
            break;
        } else {
            fseek(f, chunk_size, SEEK_CUR);
        }
    }
    fclose(f);
    if (!audio) {
        fprintf(stderr, "[WAV] No audio data in %s\n", path);
        return NULL;
    }

    *T_audio = n_samples;
    *sr      = sample_rate;
    fprintf(stderr, "[WAV] Read %s: %d samples, %d Hz, %d ch, %d bit\n", path, n_samples, sample_rate, n_channels,
            bits_per_sample);
    return audio;
}

// Write WAV file. Audio layout: planar [ch0: T_audio floats, ch1: T_audio floats].
// Output: 16-bit PCM stereo.
static bool write_wav(const char * path, const float * audio, int T_audio, int sr) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        return false;
    }
    int n_channels = 2, bits = 16;
    int byte_rate   = sr * n_channels * (bits / 8);
    int block_align = n_channels * (bits / 8);
    int data_size   = T_audio * n_channels * (bits / 8);
    int file_size   = 36 + data_size;
    fwrite("RIFF", 1, 4, f);
    fwrite(&file_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    int fmt_size = 16;
    fwrite(&fmt_size, 4, 1, f);
    short audio_fmt = 1;
    fwrite(&audio_fmt, 2, 1, f);
    short nc = (short) n_channels;
    fwrite(&nc, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    short ba = (short) block_align;
    fwrite(&ba, 2, 1, f);
    short bp = (short) bits;
    fwrite(&bp, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);
    // peak-normalize to 0 dBFS: max amplitude, no clipping, best quality
    float peak = 0.0f;
    for (int i = 0; i < T_audio * 2; i++) {
        float a = audio[i] < 0 ? -audio[i] : audio[i];
        if (a > peak) {
            peak = a;
        }
    }
    float scale = peak > 0.0f ? 32767.0f / peak : 0.0f;

    for (int t = 0; t < T_audio; t++) {
        for (int c = 0; c < 2; c++) {
            short v = (short) (audio[c * T_audio + t] * scale);
            fwrite(&v, 2, 1, f);
        }
    }
    fclose(f);
    return true;
}
