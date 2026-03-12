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

// Read an MP3 file via minimp3. Returns planar stereo float.
// Caller must free() the result.
static float * audio_io_read_mp3(const char * path, int * T_out, int * sr_out) {
    *T_out  = 0;
    *sr_out = 0;

    FILE * fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "[Audio] cannot open %s\n", path);
        return NULL;
    }
    fseek(fp, 0, SEEK_END);
    long fsize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char * mp3_buf = (unsigned char *) malloc((size_t) fsize);
    if (!mp3_buf) {
        fclose(fp);
        return NULL;
    }
    fread(mp3_buf, 1, (size_t) fsize, fp);
    fclose(fp);

    mp3dec_t dec;
    mp3dec_init(&dec);

    // decode all frames, accumulate interleaved short samples
    short * pcm_buf   = NULL;
    int     pcm_cap   = 0;
    int     pcm_count = 0;  // total samples (frames * channels)
    int     out_sr    = 0;
    int     out_nch   = 0;

    int offset = 0;
    while (offset < fsize) {
        mp3dec_frame_info_t info;
        short               pcm[MINIMP3_MAX_SAMPLES_PER_FRAME];
        int                 samples = mp3dec_decode_frame(&dec, mp3_buf + offset, (int) (fsize - offset), pcm, &info);
        if (info.frame_bytes == 0) {
            break;
        }
        offset += info.frame_bytes;

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
    free(mp3_buf);

    if (pcm_count == 0 || out_sr == 0) {
        fprintf(stderr, "[Audio] no audio decoded from %s\n", path);
        free(pcm_buf);
        return NULL;
    }

    int T = pcm_count / out_nch;

    // convert to planar stereo float
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

    fprintf(stderr, "[MP3] read %s: %d samples, %d Hz, %d ch\n", path, T, out_sr, out_nch);
    return planar;
}

// Read a WAV file. Returns planar stereo float.
// Wraps read_wav from wav.h (which returns interleaved) and deinterleaves.
static float * audio_io_read_wav(const char * path, int * T_out, int * sr_out) {
    *T_out  = 0;
    *sr_out = 0;

    int     T = 0, sr = 0;
    float * interleaved = read_wav(path, &T, &sr);
    if (!interleaved) {
        return NULL;
    }

    // read_wav always returns stereo interleaved [L0,R0,L1,R1,...]
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
    fprintf(stderr, "[Audio-resample] %d Hz -> 48000 Hz, %d samples...\n", sr, T);
    float * resampled = audio_resample(raw, T, sr, 48000, 2, &T_rs);
    free(raw);

    if (!resampled) {
        fprintf(stderr, "[Audio-resample] resample failed\n");
        *T_out = 0;
        return NULL;
    }

    fprintf(stderr, "[Audio-resample] Done: %d -> %d samples\n", T, T_rs);

    *T_out = T_rs;
    return resampled;
}

// Write planar stereo float to WAV 16-bit PCM.
// audio: [L: T_audio][R: T_audio], clipped to [-1, 1].
static bool audio_write_wav(const char * path, const float * audio, int T_audio, int sr) {
    FILE * f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[Audio] cannot open %s for writing\n", path);
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

    // write interleaved PCM16 from planar float
    const float * L = audio;
    const float * R = audio + T_audio;
    for (int t = 0; t < T_audio; t++) {
        float lf = L[t];
        float rf = R[t];
        if (lf > 1.0f) {
            lf = 1.0f;
        }
        if (lf < -1.0f) {
            lf = -1.0f;
        }
        if (rf > 1.0f) {
            rf = 1.0f;
        }
        if (rf < -1.0f) {
            rf = -1.0f;
        }
        short ls = (short) (lf * 32767.0f);
        short rs = (short) (rf * 32767.0f);
        fwrite(&ls, 2, 1, f);
        fwrite(&rs, 2, 1, f);
    }

    fclose(f);
    fprintf(stderr, "[WAV] wrote %s: %d samples, %d Hz, stereo\n", path, T_audio, sr);
    return true;
}

// Encode planar stereo float to MP3.
// sr must be 32000, 44100, or 48000 (MP3 MPEG1 rates).
// If sr is unsupported, resamples to 44100 first.
static bool audio_write_mp3(const char * path, const float * audio, int T_audio, int sr, int kbps) {
    const float * enc_audio = audio;
    int           enc_T     = T_audio;
    int           enc_sr    = sr;
    float *       resampled = NULL;

    // resample to 44100 if sr is not a valid MPEG1 rate
    if (sr != 32000 && sr != 44100 && sr != 48000) {
        int T_rs  = 0;
        resampled = audio_resample(audio, T_audio, sr, 44100, 2, &T_rs);
        if (!resampled) {
            fprintf(stderr, "[Audio-resample] resample failed\n");
            return false;
        }
        fprintf(stderr, "[Audio-resample] %d Hz -> 44100 Hz (%d -> %d samples)\n", sr, T_audio, T_rs);
        enc_audio = resampled;
        enc_T     = T_rs;
        enc_sr    = 44100;
    }

    mp3enc_t * enc = mp3enc_init(enc_sr, 2, kbps);
    if (!enc) {
        fprintf(stderr, "[Audio] mp3enc_init failed: %d Hz, %d kbps\n", enc_sr, kbps);
        free(resampled);
        return false;
    }

    FILE * fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "[Audio] cannot open %s for writing\n", path);
        mp3enc_free(enc);
        free(resampled);
        return false;
    }

    float duration = (float) enc_T / (float) enc_sr;
    fprintf(stderr, "[MP3] encoding %.1fs @ %d kbps, %d Hz stereo\n", duration, kbps, enc_sr);

    clock_t t_start = clock();

    // encode in 1-second chunks to keep memory usage low
    int chunk   = enc_sr;
    int written = 0;
    for (int pos = 0; pos < enc_T; pos += chunk) {
        int n = (pos + chunk <= enc_T) ? chunk : (enc_T - pos);

        // build planar chunk for this segment
        float * buf = (float *) malloc((size_t) n * 2 * sizeof(float));
        memcpy(buf, enc_audio + pos, (size_t) n * sizeof(float));
        memcpy(buf + n, enc_audio + enc_T + pos, (size_t) n * sizeof(float));

        int             out_size = 0;
        const uint8_t * mp3      = mp3enc_encode(enc, buf, n, &out_size);
        fwrite(mp3, 1, (size_t) out_size, fp);
        written += out_size;
        free(buf);
    }

    int             flush_size = 0;
    const uint8_t * flush_data = mp3enc_flush(enc, &flush_size);
    fwrite(flush_data, 1, (size_t) flush_size, fp);
    written += flush_size;

    float encode_ms = (float) (clock() - t_start) * 1000.0f / (float) CLOCKS_PER_SEC;
    float realtime  = (encode_ms > 0.0f) ? (duration * 1000.0f / encode_ms) : 0.0f;

    fclose(fp);
    mp3enc_free(enc);
    free(resampled);

    float ratio = (enc_T > 0) ? (float) (enc_T * 2 * 2) / (float) written : 0.0f;
    fprintf(stderr, "[MP3] wrote %s: %d bytes (%.1f:1), %.0f ms (%.2fx realtime)\n", path, written, ratio, encode_ms,
            realtime);
    return true;
}

// Write audio, auto-detect format from extension.
// .mp3 -> MP3 encoding at the given kbps (default 128).
// .wav (or anything else) -> WAV 16-bit PCM.
static bool audio_write(const char * path, const float * audio, int T_audio, int sr, int kbps) {
    if (audio_io_ends_with(path, ".mp3")) {
        return audio_write_mp3(path, audio, T_audio, sr, (kbps > 0) ? kbps : 128);
    }
    return audio_write_wav(path, audio, T_audio, sr);
}
