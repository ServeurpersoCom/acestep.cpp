#pragma once
// audio-io.h: unified audio read/write for WAV and MP3.
// Reads any WAV (PCM16/float32, mono/stereo, any rate) or MP3.
// Writes WAV (16-bit PCM) or MP3 (via mp3enc).
// All functions use planar stereo float: [L: T samples][R: T samples].
// Part of acestep.cpp. MIT license.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <thread>
#include <vector>

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
    size_t nr = fread(buf, 1, (size_t) fsize, fp);
    fclose(fp);
    if (nr != (size_t) fsize) {
        free(buf);
        return NULL;
    }

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

// maximize perceived loudness via percentile normalization.
// peak_clip controls the tradeoff between loudness and clipping:
//   0   = peak normalization (100.0000th percentile, no clipping)
//   10  = default (99.9990th percentile, clips top 0.001%)
//   999 = max (99.9001th percentile, clips top 0.1%)
// the target percentile is 1.0 - peak_clip/1000000.0.
// n_total = number of float samples (both channels combined).
static void audio_normalize(float * audio, int n_total, int peak_clip = 10) {
    if (n_total <= 0) {
        return;
    }

    // clamp to valid range
    if (peak_clip < 0) {
        peak_clip = 0;
    }
    if (peak_clip > 999) {
        peak_clip = 999;
    }

    // collect absolute values
    std::vector<float> absvals((size_t) n_total);
    for (int i = 0; i < n_total; i++) {
        absvals[i] = audio[i] < 0.0f ? -audio[i] : audio[i];
    }

    // partial sort to find the target percentile
    double pct = 1.0 - (double) peak_clip / 1000000.0;
    size_t idx = (size_t) ((double) (n_total - 1) * pct);
    std::nth_element(absvals.begin(), absvals.begin() + idx, absvals.end());
    float ref = absvals[idx];

    if (ref < 1e-6f) {
        return;
    }

    // scale so the target percentile hits 1.0, hard clip the rest
    float gain = 1.0f / ref;
    for (int i = 0; i < n_total; i++) {
        float v = audio[i] * gain;
        if (v > 1.0f) {
            v = 1.0f;
        } else if (v < -1.0f) {
            v = -1.0f;
        }
        audio[i] = v;
    }
}

// convert planar [L:T][R:T] to interleaved [L0,R0,L1,R1,...].
// returns malloc'd buffer of 2*T floats. caller must free().
static float * audio_planar_to_interleaved(const float * planar, int T) {
    float * out = (float *) malloc((size_t) T * 2 * sizeof(float));
    for (int t = 0; t < T; t++) {
        out[t * 2 + 0] = planar[t];
        out[t * 2 + 1] = planar[T + t];
    }
    return out;
}

static_assert(sizeof(float) == 4, "requires 32-bit float");
static_assert(std::numeric_limits<float>::is_iec559, "requires IEEE-754 float");

inline void wav_write_u16le(char *& p, std::uint16_t x) {
    *p++ = (char) (x & 0xff);
    *p++ = (char) ((x >> 8) & 0xff);
}

inline void wav_write_u24le(char *& p, std::uint32_t x) {
    *p++ = (char) (x & 0xff);
    *p++ = (char) ((x >> 8) & 0xff);
    *p++ = (char) ((x >> 16) & 0xff);
}

inline void wav_write_u32le(char *& p, std::uint32_t x) {
    *p++ = (char) (x & 0xff);
    *p++ = (char) ((x >> 8) & 0xff);
    *p++ = (char) ((x >> 16) & 0xff);
    *p++ = (char) ((x >> 24) & 0xff);
}

inline float wav_sanitize(float x) {
    return std::isfinite(x) ? x : 0.0f;
}

inline void wav_write_f32le(char *& p, float x) {
    x = wav_sanitize(x);
    std::uint32_t o;
    std::memcpy(&o, &x, 4);
    wav_write_u32le(p, o);
}
 
inline float wav_clamp1(float x) {
    return x < -1.0f ? -1.0f : (x > 1.0f ? 1.0f : x);
}

inline std::int16_t wav_quantize_s16(float x) {
    x = wav_clamp1(wav_sanitize(x));
    return (std::int16_t) (x * 32767.0f);
}

inline std::int32_t wav_quantize_s24(float x) {
    x = wav_clamp1(wav_sanitize(x));
    return (std::int32_t) (x * 8388607.0f);
}

inline void wav_write_s16le(char *& p, std::int16_t x) {
    wav_write_u16le(p, (std::uint16_t) x);
}

inline void wav_write_s24le(char *& p, std::int32_t x) {
    wav_write_u24le(p, (std::uint32_t) x);
}

inline void wav_write_guid_pcm(char *& p) {
    wav_write_u32le(p, 0x00000001u);
    wav_write_u16le(p, 0x0000u);
    wav_write_u16le(p, 0x0010u);
    *p++ = (char) 0x80;
    *p++ = (char) 0x00;
    *p++ = (char) 0x00;
    *p++ = (char) 0xAA;
    *p++ = (char) 0x00;
    *p++ = (char) 0x38;
    *p++ = (char) 0x9B;
    *p++ = (char) 0x71;
}

inline void wav_write_header_basic(
        char *& p,
        int T_audio,
        int sr,
        int n_channels,
        int bits,
        std::uint16_t fmt_tag) {
    assert(T_audio >= 0);
    assert(sr > 0);
    assert(n_channels > 0);
    assert(bits == 16 || bits == 32);

    const std::uint32_t bytes_per_sample = (std::uint32_t) bits / 8;
    const std::uint32_t byte_rate        = (std::uint32_t) sr * (std::uint32_t) n_channels * bytes_per_sample;
    const std::uint16_t block_align      = (std::uint16_t) (n_channels * (int) bytes_per_sample);
    const std::uint32_t data_size        = (std::uint32_t) T_audio * (std::uint32_t) n_channels * bytes_per_sample;
    const std::uint32_t file_size        = 36 + data_size;

    memcpy(p, "RIFF", 4);
    p += 4;
    wav_write_u32le(p, file_size);
    memcpy(p, "WAVE", 4);
    p += 4;

    memcpy(p, "fmt ", 4);
    p += 4;
    wav_write_u32le(p, 16);
    wav_write_u16le(p, fmt_tag);
    wav_write_u16le(p, (std::uint16_t) n_channels);
    wav_write_u32le(p, (std::uint32_t) sr);
    wav_write_u32le(p, byte_rate);
    wav_write_u16le(p, block_align);
    wav_write_u16le(p, (std::uint16_t) bits);

    memcpy(p, "data", 4);
    p += 4;
    wav_write_u32le(p, data_size);
}

inline void wav_write_header_extensible_pcm_s24(
        char *& p,
        int T_audio,
        int sr,
        int n_channels) {
    assert(T_audio >= 0);
    assert(sr > 0);
    assert(n_channels == 2);

    const int bits = 24;
    const std::uint32_t bytes_per_sample = 3;
    const std::uint32_t byte_rate        = (std::uint32_t) sr * (std::uint32_t) n_channels * bytes_per_sample;
    const std::uint16_t block_align      = (std::uint16_t) (n_channels * (int) bytes_per_sample);
    const std::uint32_t data_size        = (std::uint32_t) T_audio * (std::uint32_t) n_channels * bytes_per_sample;
    const std::uint32_t file_size        = 60 + data_size;

    memcpy(p, "RIFF", 4);
    p += 4;
    wav_write_u32le(p, file_size);
    memcpy(p, "WAVE", 4);
    p += 4;

    memcpy(p, "fmt ", 4);
    p += 4;
    wav_write_u32le(p, 40);
    wav_write_u16le(p, 0xfffe);
    wav_write_u16le(p, (std::uint16_t) n_channels);
    wav_write_u32le(p, (std::uint32_t) sr);
    wav_write_u32le(p, byte_rate);
    wav_write_u16le(p, block_align);
    wav_write_u16le(p, (std::uint16_t) bits);
    wav_write_u16le(p, 22);
    wav_write_u16le(p, (std::uint16_t) bits);
    wav_write_u32le(p, 0x00000003u);
    wav_write_guid_pcm(p);

    memcpy(p, "data", 4);
    p += 4;
    wav_write_u32le(p, data_size);
}

inline std::string audio_encode_wav_pcm_s16(const float * audio, int T_audio, int sr) {
    const int n_channels = 2;
    const int bits       = 16;

    assert(audio != nullptr);
    assert(T_audio >= 0);
    assert(sr > 0);

    const std::uint32_t data_size = (std::uint32_t) T_audio * n_channels * (bits / 8);

    std::string out;
    out.resize(44 + (size_t) data_size);

    char * p = out.data();
    wav_write_header_basic(p, T_audio, sr, n_channels, bits, 1);

    const float * L = audio;
    const float * R = audio + T_audio;

    for (int t = 0; t < T_audio; ++t) {
        wav_write_s16le(p, wav_quantize_s16(L[t]));
        wav_write_s16le(p, wav_quantize_s16(R[t]));
    }

    return out;
}

inline std::string audio_encode_wav_pcm_s24(const float * audio, int T_audio, int sr) {
    const int n_channels = 2;
    const int bits       = 24;

    assert(audio != nullptr);
    assert(T_audio >= 0);
    assert(sr > 0);

    const std::uint32_t data_size = (std::uint32_t) T_audio * n_channels * (bits / 8);

    std::string out;
    out.resize(68 + (size_t) data_size);

    char * p = out.data();
    wav_write_header_extensible_pcm_s24(p, T_audio, sr, n_channels);

    const float * L = audio;
    const float * R = audio + T_audio;

    for (int t = 0; t < T_audio; ++t) {
        wav_write_s24le(p, wav_quantize_s24(L[t]));
        wav_write_s24le(p, wav_quantize_s24(R[t]));
    }

    return out;
}

inline std::string audio_encode_wav_ieee_f32(const float * audio, int T_audio, int sr) {
    const int n_channels = 2;
    const int bits       = 32;

    assert(audio != nullptr);
    assert(T_audio >= 0);
    assert(sr > 0);

    const std::uint32_t data_size = (std::uint32_t) T_audio * n_channels * (bits / 8);

    std::string out;
    out.resize(44 + (size_t) data_size);

    char * p = out.data();
    wav_write_header_basic(p, T_audio, sr, n_channels, bits, 3);

    const float * L = audio;
    const float * R = audio + T_audio;

    for (int t = 0; t < T_audio; ++t) {
        wav_write_f32le(p, L[t]);
        wav_write_f32le(p, R[t]);
    }

    return out;
}

enum AudioFileKind {
    AUDIO_FILE_KIND_MP3,
    AUDIO_FILE_KIND_WAV,
};

enum AudioFileFormat {
    AUDIO_FILE_FORMAT_MP3,
    AUDIO_FILE_FORMAT_WAV_S16,
    AUDIO_FILE_FORMAT_WAV_S24,
    AUDIO_FILE_FORMAT_WAV_IEEE_F32,
};

enum WavFormat {
    WAV_FORMAT_S16,
    WAV_FORMAT_S24,
    WAV_FORMAT_IEEE_F32,
};

inline AudioFileKind convert_audio_file_format_to_kind(AudioFileFormat format) {
    switch (format) {
        case AUDIO_FILE_FORMAT_MP3:
            return AUDIO_FILE_KIND_MP3;
        case AUDIO_FILE_FORMAT_WAV_S16:
            return AUDIO_FILE_KIND_WAV;
        case AUDIO_FILE_FORMAT_WAV_S24:
            return AUDIO_FILE_KIND_WAV;
        case AUDIO_FILE_FORMAT_WAV_IEEE_F32:
            return AUDIO_FILE_KIND_WAV;
    }

    assert(false && "unsupported AudioFileFormat");
    std::terminate();
}

inline bool convert_audio_file_format_to_wav_format(AudioFileFormat format, WavFormat& out) {
    switch (format) {
        case AUDIO_FILE_FORMAT_MP3:
            return false;
        case AUDIO_FILE_FORMAT_WAV_S16:
            out = WAV_FORMAT_S16;
            return true;
        case AUDIO_FILE_FORMAT_WAV_S24:
            out = WAV_FORMAT_S24;
            return true;
        case AUDIO_FILE_FORMAT_WAV_IEEE_F32:
            out = WAV_FORMAT_IEEE_F32;
            return true;
    }

    assert(false && "unsupported AudioFileFormat");
    std::terminate();
}

inline AudioFileFormat convert_wav_format_to_audio_file_format(WavFormat format) {
    switch (format) {
        case WAV_FORMAT_S16:
            return AUDIO_FILE_FORMAT_WAV_S16;
        case WAV_FORMAT_S24:
            return AUDIO_FILE_FORMAT_WAV_S24;
        case WAV_FORMAT_IEEE_F32:
            return AUDIO_FILE_FORMAT_WAV_IEEE_F32;
    }

    assert(false && "unsupported WavFormat");
    std::terminate();
}

static bool parse_audio_file_format(const char* s, AudioFileFormat& out) {
    if (!s) {
        return false;
    }

    AudioFileFormat parsed;
    if (std::strcmp(s, "mp3") == 0) {
        parsed = AUDIO_FILE_FORMAT_MP3;
    } else if (std::strcmp(s, "wav") == 0) {
        parsed = AUDIO_FILE_FORMAT_WAV_S16;
    } else if (std::strcmp(s, "wav16") == 0) {
        parsed = AUDIO_FILE_FORMAT_WAV_S16;
    } else if (std::strcmp(s, "wav24") == 0) {
        parsed = AUDIO_FILE_FORMAT_WAV_S24;
    } else if (std::strcmp(s, "wav32") == 0) {
        parsed = AUDIO_FILE_FORMAT_WAV_IEEE_F32;
    } else {
        return false;
    }

    out = parsed;
    return true;
}

static bool parse_optional_wav_format(const char* s, WavFormat& out) {
    if (!s) {
        return true;
    }

    WavFormat parsed;
    if (std::strcmp(s, "wav") == 0) {
        parsed = WAV_FORMAT_S16;
    } else if (std::strcmp(s, "wav16") == 0) {
        parsed = WAV_FORMAT_S16;
    } else if (std::strcmp(s, "wav24") == 0) {
        parsed = WAV_FORMAT_S24;
    } else if (std::strcmp(s, "wav32") == 0) {
        parsed = WAV_FORMAT_IEEE_F32;
    } else {
        return false;
    }

    out = parsed;
    return true;
}

inline bool should_normalize_audio(AudioFileFormat format) {
    switch (format) {
        case AUDIO_FILE_FORMAT_MP3:
            return true;
        case AUDIO_FILE_FORMAT_WAV_S16:
            return true;
        case AUDIO_FILE_FORMAT_WAV_S24:
            return true;
        case AUDIO_FILE_FORMAT_WAV_IEEE_F32:
            return false;
    }

    assert(false && "unsupported AudioFileFormat");
    std::terminate();
}

inline bool should_normalize_wav_audio(WavFormat format) {
    switch (format) {
        case WAV_FORMAT_S16:
            return true;
        case WAV_FORMAT_S24:
            return true;
        case WAV_FORMAT_IEEE_F32:
            return false;
    }

    assert(false && "unsupported WavFormat");
    std::terminate();
}

// audio_encode_wav is the core: encode planar stereo to WAV 16-bit PCM, 24-bit PCM, or 32-bit floating point in memory.
// 16-bit PCM outputs classic RIFF header and little-endian integer samples.
// 24-bit PCM outputs extensible header and little-endian integer samples.
// 32-bit floating-point outputs classic RIFF header and little-endian IEEE-754 floating-point samples.
// audio is planar [L0..LN, R0..RN], pre-normalized by caller.
// Does NOT normalize - caller is responsible (audio_write does it).
// NaN, -Inf, and +Inf are automatically coerced to zero.
// 16-bit and 24-bit PCM output is automatically clamped to -1/+1 range.
// 32-bit floating-point output may exceed -1/+1 range.
// Returns empty string on failure.
inline std::string audio_encode_wav(const float * audio, int T_audio, int sr, WavFormat wav_format) {
    switch (wav_format) {
        case WAV_FORMAT_S16:
            return audio_encode_wav_pcm_s16(audio, T_audio, sr);
        case WAV_FORMAT_S24:
            return audio_encode_wav_pcm_s24(audio, T_audio, sr);
        case WAV_FORMAT_IEEE_F32:
            return audio_encode_wav_ieee_f32(audio, T_audio, sr);
    }

    assert(false && "unsupported WavFormat");
    std::terminate();
}

// Write planar stereo audio to WAV file. Thin wrapper around audio_encode_wav.
static bool audio_write_wav(const char * path, const float * audio, int T_audio, int sr, WavFormat wav_format) {
    std::string wav = audio_encode_wav(audio, T_audio, sr, wav_format);
    if (wav.empty()) {
        return false;
    }

    FILE * fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[Audio] Cannot open %s for writing\n", path);
        return false;
    }
    fwrite(wav.data(), 1, wav.size(), fp);
    fclose(fp);

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

    float duration = (float) enc_T / (float) enc_sr;
    fprintf(stderr, "[MP3] Encoding %.1fs @ %d kbps, %d Hz stereo\n", duration, kbps, enc_sr);

    // thread count: all logical cores. MP3 is ALU-bound with small working set,
    // hyperthreads help (unlike GGML GEMM which shares SIMD units).
    // minimum ~2s per chunk so filter warmup at boundaries is negligible.
    int n_threads = (int) std::thread::hardware_concurrency();
    if (n_threads < 1) {
        n_threads = 1;
    }
    int total_frames = (enc_T + 1151) / 1152;
    int min_frames   = (enc_sr * 2 + 1151) / 1152;  // ~2s worth of frames
    int max_threads  = total_frames / (min_frames > 0 ? min_frames : 1);
    if (max_threads < 1) {
        max_threads = 1;
    }
    if (n_threads > max_threads) {
        n_threads = max_threads;
    }

    // per-thread sample ranges, aligned to 1152 (MP3 frame boundary).
    // each thread gets its own encoder instance. threads > 0 pre-encode
    // warmup frames from before their start to prime the filterbank, MDCT
    // overlap, and psy state. the warmup output is discarded via flush.
    // after flush, pending_bytes=0 so the first real frame naturally gets
    // main_data_begin=0 (self-contained, no reservoir reference). cost:
    // one frame of lost reservoir per boundary (~26ms of slightly lower
    // quality), but filter/MDCT/psy are fully primed = no audible gap.
    static const int WARMUP_FRAMES = 3;

    struct chunk_range {
        int start;
        int end;
    };

    std::vector<chunk_range> ranges(n_threads);
    {
        int base = total_frames / n_threads;
        int rem  = total_frames % n_threads;
        int f    = 0;
        for (int t = 0; t < n_threads; t++) {
            int nf          = base + (t < rem ? 1 : 0);
            ranges[t].start = f * 1152;
            f += nf;
            ranges[t].end = (t == n_threads - 1) ? enc_T : f * 1152;
        }
    }

    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::string> results(n_threads);

    // worker: encode one chunk with a private encoder instance.
    // feeds 1-second sub-chunks for cancel responsiveness.
    auto worker = [&](int tid) {
        int chunk_start = ranges[tid].start;
        int chunk_end   = ranges[tid].end;
        if (chunk_end <= chunk_start) {
            return;
        }

        mp3enc_t * e = mp3enc_init(enc_sr, 2, kbps);
        if (!e) {
            return;
        }

        // warmup: encode frames from before chunk_start to prime encoder
        // state (filter, sb_prev, psy). output is discarded via flush.
        // warmup length is exact multiple of 1152 so pcm_fill=0 after.
        if (tid > 0 && chunk_start > 0) {
            int n_warm = WARMUP_FRAMES;
            if (n_warm > chunk_start / 1152) {
                n_warm = chunk_start / 1152;
            }
            if (n_warm > 0) {
                int warm_len   = n_warm * 1152;
                int warm_start = chunk_start - warm_len;

                float * buf = (float *) malloc((size_t) warm_len * 2 * sizeof(float));
                memcpy(buf, enc_audio + warm_start, (size_t) warm_len * sizeof(float));
                memcpy(buf + warm_len, enc_audio + enc_T + warm_start, (size_t) warm_len * sizeof(float));

                int sz = 0;
                mp3enc_encode(e, buf, warm_len, &sz);
                free(buf);
            }

            // flush: output and discard warmup frames. after this,
            // pending_bytes=0 so next frame gets main_data_begin=0
            // naturally. filter and sb_prev state are preserved.
            int flush_sz = 0;
            mp3enc_flush(e, &flush_sz);
            e->out_written = 0;
        }

        // encode real chunk
        int chunk_len = chunk_end - chunk_start;
        int sub       = enc_sr;  // ~1 second
        for (int p = 0; p < chunk_len; p += sub) {
            if (cancel && cancel(cancel_data)) {
                break;
            }
            int len = (p + sub <= chunk_len) ? sub : (chunk_len - p);

            float * buf = (float *) malloc((size_t) len * 2 * sizeof(float));
            memcpy(buf, enc_audio + chunk_start + p, (size_t) len * sizeof(float));
            memcpy(buf + len, enc_audio + enc_T + chunk_start + p, (size_t) len * sizeof(float));

            int             sz  = 0;
            const uint8_t * mp3 = mp3enc_encode(e, buf, len, &sz);
            results[tid].append((const char *) mp3, (size_t) sz);
            free(buf);
        }

        int             flush_sz = 0;
        const uint8_t * flush    = mp3enc_flush(e, &flush_sz);
        results[tid].append((const char *) flush, (size_t) flush_sz);
        mp3enc_free(e);
    };

    // fork-join: main thread takes chunk 0, spawn threads for the rest
    if (n_threads == 1) {
        worker(0);
    } else {
        std::vector<std::thread> threads;
        for (int t = 1; t < n_threads; t++) {
            threads.emplace_back(worker, t);
        }
        worker(0);
        for (auto & th : threads) {
            th.join();
        }
    }

    auto  t_end     = std::chrono::steady_clock::now();
    float encode_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();

    free(resampled);

    if (cancel && cancel(cancel_data)) {
        fprintf(stderr, "[MP3] Cancelled\n");
        return "";
    }

    // concatenate thread outputs (order = chunk order = correct MP3 stream)
    size_t total_bytes = 0;
    for (auto & r : results) {
        total_bytes += r.size();
    }
    std::string out;
    out.reserve(total_bytes);
    for (auto & r : results) {
        out.append(r);
    }

    float realtime = (encode_ms > 0.0f) ? (duration * 1000.0f / encode_ms) : 0.0f;
    float ratio    = (enc_T > 0) ? (float) (enc_T * 2 * 2) / (float) out.size() : 0.0f;
    fprintf(stderr, "[MP3] %zu bytes (%.1f:1), %.0f ms (%.2fx realtime), %d threads\n", out.size(), ratio, encode_ms,
            realtime, n_threads);
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
// Normalizes in place before writing (single normalization point).
static bool audio_write(const char * path, float * audio, int T_audio, int sr, int kbps, WavFormat wav_format, int peak_clip = 10) {
    const bool write_mp3 = audio_io_ends_with(path, ".mp3");

    if (write_mp3 || should_normalize_wav_audio(wav_format)) {
        audio_normalize(audio, T_audio * 2, peak_clip);
    }

    if (write_mp3) {
        return audio_write_mp3(path, audio, T_audio, sr, (kbps > 0) ? kbps : 128);
    }
    return audio_write_wav(path, audio, T_audio, sr, wav_format);
}
