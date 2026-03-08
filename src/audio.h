// audio.h: unified audio reader supporting WAV and MP3 input formats.
//
// Decodes files via header-only libraries, resamples to 48 kHz if needed,
// and always returns interleaved stereo float [T * 2] — the layout required
// by the ACEStep VAE encoder.  Mono input is upmixed (L=R), N-channel input
// uses the first two channels; no other manipulation is performed.
//
// Third-party header-only libraries used (thirdparty/):
//   dr_wav.h  - WAV decoding  (public domain / MIT-0, mackron/dr_libs)
//   dr_mp3.h  - MP3 decoding  (public domain / MIT-0, mackron/dr_libs)
//
// read_audio(path, T_audio, n_channels)
//   Reads a WAV or MP3 file.  Returns a malloc'd interleaved stereo float
//   buffer [T * 2] at 48 kHz.  Sets *T_audio (frame count) and *n_channels
//   (always 2 on success).  Returns NULL on error; caller frees on success.

#pragma once
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// Bring in header-only implementations once per translation unit.
// Safe because audio.h is guarded by #pragma once and each tool is a
// separate binary with its own translation unit.
#define DR_WAV_IMPLEMENTATION
#include "dr_wav.h"

#define DR_MP3_IMPLEMENTATION
#include "dr_mp3.h"

// ---------------------------------------------------------------------------
// Internal: linear sample-rate converter (channel-agnostic)
// ---------------------------------------------------------------------------

// Resample interleaved PCM from sr_src to 48000 Hz using linear interpolation.
// Works for any number of channels (n_ch).
// Returns a malloc'd buffer of (*T_dst * n_ch) floats.  Caller frees.
static float * audio_resample_linear(const float * src, int T_src, int n_ch, int sr_src, int * T_dst) {
    static const int TARGET_SR = 48000;

    if (sr_src == TARGET_SR) {
        size_t  bytes = (size_t) T_src * (size_t) n_ch * sizeof(float);
        float * out   = (float *) malloc(bytes);
        if (out) {
            memcpy(out, src, bytes);
        }
        *T_dst = T_src;
        return out;
    }

    double  ratio = (double) sr_src / TARGET_SR;
    int     T_out = (int) ((double) T_src / ratio);
    float * out   = (float *) malloc((size_t) T_out * (size_t) n_ch * sizeof(float));
    if (!out) {
        *T_dst = 0;
        return NULL;
    }

    for (int t = 0; t < T_out; t++) {
        double pos = (double) t * ratio;
        int    t0  = (int) pos;
        double f   = pos - (double) t0;
        int    t1  = (t0 + 1 < T_src) ? t0 + 1 : t0;
        for (int c = 0; c < n_ch; c++) {
            out[(size_t) t * n_ch + c] =
                (float) ((1.0 - f) * (double) src[(size_t) t0 * n_ch + c] +
                                  f * (double) src[(size_t) t1 * n_ch + c]);
        }
    }

    *T_dst = T_out;
    return out;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// Read a WAV or MP3 file, resample to 48 kHz if needed, and return
// interleaved stereo float [T * 2] at 48 kHz.
// Format is detected by file extension (.mp3 -> MP3, anything else -> WAV).
// Mono input is upmixed to stereo (L = R = the single channel).
// Multi-channel input (>2) uses the first two channels only.
//
// On success:
//   *T_audio    <- number of PCM frames at 48 kHz
//   *n_channels <- always 2 (stereo)
//   return value <- malloc'd buffer [T_audio * 2] floats; caller frees
//
// Returns NULL on failure.
static float * read_audio(const char * path, int * T_audio, int * n_channels) {
    if (!path || !T_audio || !n_channels) {
        return NULL;
    }

    // Determine file extension (lowercase)
    std::string ext;
    {
        const char * dot = strrchr(path, '.');
        if (dot) {
            ext = dot;
            for (char & c : ext) {
                c = (char) tolower((unsigned char) c);
            }
        }
    }

    // -----------------------------------------------------------------------
    // Decode to native interleaved float PCM via dr_libs
    // -----------------------------------------------------------------------
    float *      raw      = NULL;
    unsigned int channels = 0;
    unsigned int sr       = 0;
    long long    n_frames = 0;

    if (ext == ".mp3") {
        drmp3_config cfg        = {};
        drmp3_uint64 mp3_frames = 0;
        float *      mp3_data   = drmp3_open_file_and_read_pcm_frames_f32(path, &cfg, &mp3_frames, NULL);
        if (!mp3_data) {
            fprintf(stderr, "[Audio] Failed to decode MP3: %s\n", path);
            return NULL;
        }
        channels = cfg.channels;
        sr       = cfg.sampleRate;
        n_frames = (long long) mp3_frames;
        raw      = mp3_data;
        fprintf(stderr, "[Audio] Read MP3 %s: %lld frames, %u Hz, %uch\n", path, n_frames, sr, channels);
    } else {
        drwav_uint32 wav_ch = 0, wav_sr = 0;
        drwav_uint64 wav_frames = 0;
        float *      wav_data   = drwav_open_file_and_read_pcm_frames_f32(path, &wav_ch, &wav_sr, &wav_frames, NULL);
        if (!wav_data) {
            fprintf(stderr, "[Audio] Failed to decode WAV: %s\n", path);
            return NULL;
        }
        channels = wav_ch;
        sr       = wav_sr;
        n_frames = (long long) wav_frames;
        raw      = wav_data;
        fprintf(stderr, "[Audio] Read WAV %s: %lld frames, %u Hz, %uch\n", path, n_frames, sr, channels);
    }

    if (channels == 0 || sr == 0 || n_frames <= 0) {
        fprintf(stderr, "[Audio] Empty or invalid audio file: %s\n", path);
        free(raw);
        return NULL;
    }

    // -----------------------------------------------------------------------
    // Resample to 48 kHz if required (channel layout unchanged)
    // -----------------------------------------------------------------------
    int     T_raw = (int) n_frames;
    float * out   = NULL;

    if (sr != 48000u) {
        fprintf(stderr, "[Audio] Resampling %u Hz -> 48000 Hz (%uch)...\n", sr, channels);
        int T_48k = 0;
        out = audio_resample_linear(raw, T_raw, (int) channels, (int) sr, &T_48k);
        free(raw);
        if (!out) {
            fprintf(stderr, "[Audio] Resampling failed (out of memory)\n");
            return NULL;
        }
        T_raw = T_48k;
    } else {
        out = raw;  // ownership transferred to caller
    }

    // -----------------------------------------------------------------------
    // Upmix to stereo: vae-enc.h always reads [T, 2] interleaved stereo.
    // Mono -> duplicate to stereo (L = R).
    // N > 2 channels -> keep first two channels only.
    // -----------------------------------------------------------------------
    if ((int) channels != 2) {
        int    n_ch_src = (int) channels;
        float * stereo  = (float *) malloc((size_t) T_raw * 2 * sizeof(float));
        if (!stereo) {
            fprintf(stderr, "[Audio] Out of memory converting to stereo\n");
            free(out);
            return NULL;
        }
        for (int t = 0; t < T_raw; t++) {
            float L             = out[(size_t) t * n_ch_src + 0];
            float R             = (n_ch_src > 1) ? out[(size_t) t * n_ch_src + 1] : L;
            stereo[t * 2 + 0]  = L;
            stereo[t * 2 + 1]  = R;
        }
        free(out);
        out = stereo;
        fprintf(stderr, "[Audio] Converted %dch -> stereo\n", n_ch_src);
    }

    fprintf(stderr, "[Audio] Ready: %d stereo frames (%.2fs @ 48 kHz)\n", T_raw,
            (float) T_raw / 48000.0f);
    *T_audio    = T_raw;
    *n_channels = 2;
    return out;
}

