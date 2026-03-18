#pragma once
// audio-io.h: unified audio read/write for WAV and MP3.
// Reads any WAV (PCM16/float32, mono/stereo, any rate) or MP3.
// Writes WAV (16-bit PCM) or MP3 (via mp3enc).
// All functions use planar stereo float: [L: T samples][R: T samples].
// Part of acestep.cpp. MIT license.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>

// wav.h: WAV reader (returns interleaved, we deinterleave below)
#include "wav.h"

// audio-resample.h: sample rate conversion
#include "audio-resample.h"

// minimp3 (CC0): MP3 decoder. Guard against double-implementation.
#ifndef AUDIO_IO_MP3DEC_IMPL
#    define AUDIO_IO_MP3DEC_IMPL
#    define MINIMP3_IMPLEMENTATION
#    ifdef _MSC_VER
#        pragma warning(push, 0)
#    elif defined(__GNUC__)
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wconversion"
#        pragma GCC diagnostic ignored "-Wsign-conversion"
#    endif
#    include "vendor/minimp3/minimp3.h"
#    ifdef _MSC_VER
#        pragma warning(pop)
#    elif defined(__GNUC__)
#        pragma GCC diagnostic pop
#    endif
#    undef MINIMP3_IMPLEMENTATION
#endif

// mp3enc: MP3 encoder
#include "mp3/mp3enc.h"

// case-insensitive extension check
static bool audio_io_ends_with(const char * str, const char * suffix) {
    int slen = (int) strlen(str);
    int xlen = (int) strlen(suffix);
    if (slen < xlen) {
        return false;
    }
    for (int i = 0; i < xlen; i++) {
        char a = str[slen - xlen + i];
        char b = suffix[i];
        if (a >= 'A' && a <= 'Z') {
            a += 32;
        }
        if (b >= 'A' && b <= 'Z') {
            b += 32;
        }
        if (a != b) {
            return false;
        }
    }
    return true;
}

// Load entire file into memory. Caller must free() the returned pointer.
static uint8_t * audio_io_load_file(const char * path, size_t * size_out) {
    *size_out = 0;
    FILE * fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Audio] Cannot open %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t * buf = (uint8_t *) malloc((size_t) fsize);
    if (!buf) {
        fclose(fp);
        return NULL;
    }
    fread(buf, 1, (size_t) fsize, fp);
    fclose(fp);

    *size_out = (size_t) fsize;
    return buf;
}

// Decode MP3 from memory buffer. Returns planar stereo float [L:T][R:T].
static float * audio_io_read_mp3_buf(const uint8_t * data, size_t size, int * T_out, int * sr_out) {
    *T_out  = 0;
    *sr_out = 0;

    mp3dec_t dec;
    mp3dec_init(&dec);

    short * pcm_buf   = NULL;
    int     pcm_cap   = 0;
    int     pcm_count = 0;
    int     out_sr    = 0;
    int     out_nch   = 0;

    size_t offset = 0;
    while (offset < size) {
        mp3dec_frame_info_t info;
        short               pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
        int                 samples = mp3dec_decode_frame(&dec, data + offset, (int) (size - offset), pcm, &info);
        if (info.frame_bytes == 0) {
            break;
        }
        offset += (size_t) info.frame_bytes;

        if (samples > 0) {
            if (out_sr == 0) {
                out_sr  = info.hz;
                out_nch = info.channels;
            }
            int need = pcm_count + samples * out_nch;
            if (need > pcm_cap) {
                pcm_cap = (need < 65536) ? 65536 : need * 2;
                pcm_buf = (short *) realloc(pcm_buf, (size_t) pcm_cap * sizeof(short));
            }
            memcpy(pcm_buf + pcm_count, pcm, (size_t) samples * (size_t) out_nch * sizeof(short));
            pcm_count += samples * out_nch;
        }
    }

    if (pcm_count == 0 || out_sr == 0) {
        fprintf(stderr, "[Audio] No audio decoded from buffer\n");
        free(pcm_buf);
        return NULL;
    }

    int T = pcm_count / out_nch;

    float * planar = (float *) malloc((size_t) T * 2 * sizeof(float));
    if (!planar) {
        free(pcm_buf);
        return NULL;
    }
    for (int t = 0; t < T; t++) {
        float l       = (float) pcm_buf[t * out_nch + 0] / 32768.0f;
        float r       = (out_nch >= 2) ? (float) pcm_buf[t * out_nch + 1] / 32768.0f : l;
        planar[t]     = l;
        planar[T + t] = r;
    }
    free(pcm_buf);

    *T_out  = T;
    *sr_out = out_sr;
    fprintf(stderr, "[MP3] Read buffer: %d samples, %d Hz, %d ch\n", T, out_sr, out_nch);
    return planar;
}

// Decode WAV from memory buffer. Returns planar stereo float [L:T][R:T].
static float * audio_io_read_wav_buf(const uint8_t * data, size_t size, int * T_out, int * sr_out) {
    *T_out  = 0;
    *sr_out = 0;

    int     T = 0, sr = 0;
    float * interleaved = read_wav_buf(data, size, &T, &sr);
    if (!interleaved) {
        return NULL;
    }

    float * planar = (float *) malloc((size_t) T * 2 * sizeof(float));
    if (!planar) {
        free(interleaved);
        return NULL;
    }
    for (int t = 0; t < T; t++) {
        planar[t]     = interleaved[t * 2 + 0];
        planar[T + t] = interleaved[t * 2 + 1];
    }
    free(interleaved);

    *T_out  = T;
    *sr_out = sr;
    return planar;
}

// Decode WAV or MP3 from memory buffer (auto-detect from magic bytes).
// Returns planar stereo float [L:T][R:T]. Caller frees.
static float * audio_read_buf(const uint8_t * data, size_t size, int * T_out, int * sr_out) {
    if (size >= 4 && memcmp(data, "RIFF", 4) == 0) {
        return audio_io_read_wav_buf(data, size, T_out, sr_out);
    }
    return audio_io_read_mp3_buf(data, size, T_out, sr_out);
}

// Decode WAV or MP3 from memory buffer and resample to 48000 Hz stereo.
// Returns planar stereo float [L:T][R:T]. Caller frees.
static float * audio_read_48k_buf(const uint8_t * data, size_t size, int * T_out) {
    int     T = 0, sr = 0;
    float * raw = audio_read_buf(data, size, &T, &sr);
    if (!raw) {
        *T_out = 0;
        return NULL;
    }

    if (sr == 48000) {
        *T_out = T;
        return raw;
    }

    int T_rs = 0;
    fprintf(stderr, "[Audio-Resample] %d Hz -> 48000 Hz, %d samples...\n", sr, T);
    float * resampled = audio_resample(raw, T, sr, 48000, 2, &T_rs);
    free(raw);

    if (!resampled) {
        fprintf(stderr, "[Audio-Resample] Resample failed\n");
        *T_out = 0;
        return NULL;
    }

    fprintf(stderr, "[Audio-Resample] Done: %d -> %d samples\n", T, T_rs);

    *T_out = T_rs;
    return resampled;
}

// File-based wrappers: load file into memory, delegate to buffer functions.
// Used by CLI tools that take file paths as arguments.

static float * audio_io_read_mp3(const char * path, int * T_out, int * sr_out) {
    size_t    size = 0;
    uint8_t * buf  = audio_io_load_file(path, &size);
    if (!buf) {
        return NULL;
    }
    float * result = audio_io_read_mp3_buf(buf, size, T_out, sr_out);
    free(buf);
    return result;
}

static float * audio_io_read_wav(const char * path, int * T_out, int * sr_out) {
    size_t    size = 0;
    uint8_t * buf  = audio_io_load_file(path, &size);
    if (!buf) {
        return NULL;
    }
    float * result = audio_io_read_wav_buf(buf, size, T_out, sr_out);
    free(buf);
    return result;
}

// Read WAV or MP3 (auto-detect from extension).
// Returns planar stereo float [L: T][R: T]. Caller frees.
static float * audio_read(const char * path, int * T_out, int * sr_out) {
    if (audio_io_ends_with(path, ".mp3")) {
        return audio_io_read_mp3(path, T_out, sr_out);
    }
    return audio_io_read_wav(path, T_out, sr_out);
}

// Read WAV or MP3 and resample to 48000 Hz stereo.
// Returns planar stereo float [L: T][R: T]. Caller frees.
static float * audio_read_48k(const char * path, int * T_out) {
    int     T = 0, sr = 0;
    float * raw = audio_read(path, &T, &sr);
    if (!raw) {
        *T_out = 0;
        return NULL;
    }

    if (sr == 48000) {
        *T_out = T;
        return raw;
    }

    int T_rs = 0;
    fprintf(stderr, "[Audio-Resample] %d Hz -> 48000 Hz, %d samples...\n", sr, T);
    float * resampled = audio_resample(raw, T, sr, 48000, 2, &T_rs);
    free(raw);

    if (!resampled) {
        fprintf(stderr, "[Audio-Resample] Resample failed\n");
        *T_out = 0;
        return NULL;
    }

    fprintf(stderr, "[Audio-Resample] Done: %d -> %d samples\n", T, T_rs);

    *T_out = T_rs;
    return resampled;
}

// Write planar stereo float to WAV 16-bit PCM.
// audio: [L: T_audio][R: T_audio], clipped to [-1, 1].
static bool audio_write_wav(const char * path, const float * audio, int T_audio, int sr) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Audio] Cannot open %s for writing\n", path);
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
    int   fmt_size = 16;
    short fmt_tag  = 1;
    short nc       = (short) n_channels;
    short ba       = (short) block_align;
    short bp       = (short) bits;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&fmt_tag, 2, 1, f);
    fwrite(&nc, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&ba, 2, 1, f);
    fwrite(&bp, 2, 1, f);
    fwrite("data", 1, 4, f);
    fwrite(&data_size, 4, 1, f);

    // buffer all samples then write once (avoids millions of fwrite syscalls)
    // audio is pre-normalized to 0 dBFS by audio_write(), no clamp needed
    const float * L   = audio;
    const float * R   = audio + T_audio;
    short *       pcm = (short *) malloc((size_t) T_audio * 2 * sizeof(short));
    if (!pcm) {
        fclose(f);
        return false;
    }
    for (int t = 0; t < T_audio; t++) {
        pcm[t * 2 + 0] = (short) (L[t] * 32767.0f);
        pcm[t * 2 + 1] = (short) (R[t] * 32767.0f);
    }
    fwrite(pcm, 2, (size_t) T_audio * 2, f);
    free(pcm);

    fclose(f);
    fprintf(stderr, "[WAV] Wrote %s: %d samples, %d Hz, stereo\n", path, T_audio, sr);
    return true;
}

// Encode planar stereo float to MP3.
// sr must be 32000, 44100, or 48000 (MP3 MPEG1 rates).
// If sr is unsupported, resamples to 44100 first.
// audio_encode_mp3 is the core: encode planar stereo to MP3 in memory.
// Does NOT normalize - caller is responsible (audio_write does it).
// Returns empty string on failure.
static std::string audio_encode_mp3(const float * audio,
                                    int           T_audio,
                                    int           sr,
                                    int           kbps,
                                    bool (*cancel)(void *) = nullptr,
                                    void * cancel_data     = nullptr) {
    const float * enc_audio = audio;
    int           enc_T     = T_audio;
    int           enc_sr    = sr;
    float *       resampled = NULL;

    // resample to 44100 if sr is not a valid MPEG1 rate
    if (sr != 32000 && sr != 44100 && sr != 48000) {
        int T_rs  = 0;
        resampled = audio_resample(audio, T_audio, sr, 44100, 2, &T_rs);
        if (!resampled) {
            fprintf(stderr, "[Audio-Resample] Resample failed\n");
            return "";
        }
        fprintf(stderr, "[Audio-Resample] %d Hz -> 44100 Hz (%d -> %d samples)\n", sr, T_audio, T_rs);
        enc_audio = resampled;
        enc_T     = T_rs;
        enc_sr    = 44100;
    }

    mp3enc_t * enc = mp3enc_init(enc_sr, 2, kbps);
    if (!enc) {
        fprintf(stderr, "[Audio] mp3enc_init failed: %d Hz, %d kbps\n", enc_sr, kbps);
        free(resampled);
        return "";
    }

    std::string out;
    float       duration = (float) enc_T / (float) enc_sr;
    out.reserve((size_t) ((float) kbps * 128.0f * duration));  // rough: kbps*1000/8*dur

    fprintf(stderr, "[MP3] Encoding %.1fs @ %d kbps, %d Hz stereo\n", duration, kbps, enc_sr);
    clock_t t_start = clock();

    // encode in 1-second chunks
    int chunk = enc_sr;
    for (int pos = 0; pos < enc_T; pos += chunk) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[MP3] Cancelled\n");
            mp3enc_free(enc);
            free(resampled);
            return "";
        }
        int n = (pos + chunk <= enc_T) ? chunk : (enc_T - pos);

        // build planar chunk for this segment
        float * buf = (float *) malloc((size_t) n * 2 * sizeof(float));
        memcpy(buf, enc_audio + pos, (size_t) n * sizeof(float));
        memcpy(buf + n, enc_audio + enc_T + pos, (size_t) n * sizeof(float));

        int             out_size = 0;
        const uint8_t * mp3      = mp3enc_encode(enc, buf, n, &out_size);
        out.append((const char *) mp3, (size_t) out_size);
        free(buf);
    }

    int             flush_size = 0;
    const uint8_t * flush_data = mp3enc_flush(enc, &flush_size);
    out.append((const char *) flush_data, (size_t) flush_size);

    float encode_ms = (float) (clock() - t_start) * 1000.0f / (float) CLOCKS_PER_SEC;
    float realtime  = (encode_ms > 0.0f) ? (duration * 1000.0f / encode_ms) : 0.0f;

    mp3enc_free(enc);
    free(resampled);

    float ratio = (enc_T > 0) ? (float) (enc_T * 2 * 2) / (float) out.size() : 0.0f;
    fprintf(stderr, "[MP3] %zu bytes (%.1f:1), %.0f ms (%.2fx realtime)\n", out.size(), ratio, encode_ms, realtime);
    return out;
}

// Write planar stereo audio to MP3 file. Thin wrapper around audio_encode_mp3.
static bool audio_write_mp3(const char * path, const float * audio, int T_audio, int sr, int kbps) {
    std::string mp3 = audio_encode_mp3(audio, T_audio, sr, kbps);
    if (mp3.empty()) {
        return false;
    }

    FILE * fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[Audio] Cannot open %s for writing\n", path);
        return false;
    }
    fwrite(mp3.data(), 1, mp3.size(), fp);
    fclose(fp);

    fprintf(stderr, "[MP3] Wrote %s\n", path);
    return true;
}

// Write audio, auto-detect format from extension.
// .mp3 -> MP3 encoding at the given kbps (default 128).
// .wav (or anything else) -> WAV 16-bit PCM.
// Peak-normalizes to 0 dBFS in-place before writing (single normalization point).
static bool audio_write(const char * path, float * audio, int T_audio, int sr, int kbps) {
    // 0 dBFS peak normalization: scale so max |sample| = 1.0
    int   n_total = T_audio * 2;
    float peak    = 0.0f;
    for (int i = 0; i < n_total; i++) {
        float a = audio[i] < 0.0f ? -audio[i] : audio[i];
        if (a > peak) {
            peak = a;
        }
    }
    if (peak > 1e-8f && peak != 1.0f) {
        float gain = 1.0f / peak;
        for (int i = 0; i < n_total; i++) {
            audio[i] *= gain;
        }
    }

    if (audio_io_ends_with(path, ".mp3")) {
        return audio_write_mp3(path, audio, T_audio, sr, (kbps > 0) ? kbps : 128);
    }
    return audio_write_wav(path, audio, T_audio, sr);
}
