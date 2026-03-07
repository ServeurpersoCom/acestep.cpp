// neural-codec.cpp: neural audio codec (Oobleck VAE)
//
// 2 modes:
//   encode: WAV -> latent file (f32, Q8, or Q4)
//   decode: latent file -> WAV (48kHz stereo)
//
// Three latent formats, decode auto-detects:
//
//   f32 (default): flat [T, 64] f32, no header.
//     T = file_size / 256. 25Hz, ~6.4 KB/s, ~51 kbit/s.
//
//   Q8 (--q8): symmetric per-frame int8 quantization.
//     header: "NAC8" magic (4B) + uint32 T_latent (4B)
//     frame:  f16 scale (2B) + int8[64] (64B) = 66B
//     25Hz, ~1.6 KB/s, ~13 kbit/s.
//
//   Q4 (--q4): symmetric per-frame 4-bit quantization.
//     header: "NAC4" magic (4B) + uint32 T_latent (4B)
//     frame:  f16 scale (2B) + nibbles[32] (32B) = 34B
//     25Hz, ~850 B/s, ~6.8 kbit/s.
//
// Usage:
//   neural-codec --vae model.gguf --encode -i song.wav -o song.latent
//   neural-codec --vae model.gguf --encode --q8 -i song.wav -o song.nac8
//   neural-codec --vae model.gguf --encode --q4 -i song.wav -o song.nac4
//   neural-codec --vae model.gguf --decode -i song.nac4 -o song.wav

#include "vae.h"
#include "vae-enc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "wav.h"

// Q8 format constants
static const char NAC8_MAGIC[4] = {'N', 'A', 'C', '8'};
static const int NAC8_HEADER = 8;   // 4B magic + 4B T_latent
static const int NAC8_FRAME  = 66;  // 2B f16 scale + 64B int8

// Write Q8 quantized latent
static bool write_latent_q8(const char * path, const float * data, int T_latent) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;

    fwrite(NAC8_MAGIC, 1, 4, f);
    uint32_t t = (uint32_t)T_latent;
    fwrite(&t, 4, 1, f);

    for (int i = 0; i < T_latent; i++) {
        const float * frame = data + i * 64;

        // find max abs for symmetric quant
        float amax = 0.0f;
        for (int j = 0; j < 64; j++) {
            float a = fabsf(frame[j]);
            if (a > amax) amax = a;
        }
        float scale = amax / 127.0f;
        ggml_fp16_t scale_f16 = ggml_fp32_to_fp16(scale);
        fwrite(&scale_f16, 2, 1, f);

        // quantize
        int8_t q[64];
        float inv = (scale > 0.0f) ? 127.0f / amax : 0.0f;
        for (int j = 0; j < 64; j++) {
            int v = (int)roundf(frame[j] * inv);
            q[j] = (int8_t)(v < -127 ? -127 : (v > 127 ? 127 : v));
        }
        fwrite(q, 1, 64, f);
    }
    fclose(f);

    size_t bytes = NAC8_HEADER + (size_t)T_latent * NAC8_FRAME;
    float duration = (float)T_latent * 1920.0f / 48000.0f;
    float kbps = (float)bytes * 8.0f / (duration * 1000.0f);
    fprintf(stderr, "[Latent] Wrote %s: Q8, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n",
            path, T_latent, duration, (float)bytes / 1024.0f, kbps);
    return true;
}

// Q4 format constants
static const char NAC4_MAGIC[4] = {'N', 'A', 'C', '4'};
static const int NAC4_HEADER = 8;   // 4B magic + 4B T_latent
static const int NAC4_FRAME  = 34;  // 2B f16 scale + 32B packed nibbles

// Write Q4 quantized latent
// Symmetric 4-bit: range [-7, 7], scale = amax / 7.0
// Packing: byte = (low & 0x0F) | (high << 4), two signed nibbles per byte
static bool write_latent_q4(const char * path, const float * data, int T_latent) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;

    fwrite(NAC4_MAGIC, 1, 4, f);
    uint32_t t = (uint32_t)T_latent;
    fwrite(&t, 4, 1, f);

    for (int i = 0; i < T_latent; i++) {
        const float * frame = data + i * 64;

        // find max abs for symmetric quant
        float amax = 0.0f;
        for (int j = 0; j < 64; j++) {
            float a = fabsf(frame[j]);
            if (a > amax) amax = a;
        }
        float scale = amax / 7.0f;
        ggml_fp16_t scale_f16 = ggml_fp32_to_fp16(scale);
        fwrite(&scale_f16, 2, 1, f);

        // quantize and pack pairs into bytes
        float inv = (scale > 0.0f) ? 7.0f / amax : 0.0f;
        uint8_t packed[32];
        for (int j = 0; j < 32; j++) {
            int lo = (int)roundf(frame[j * 2 + 0] * inv);
            int hi = (int)roundf(frame[j * 2 + 1] * inv);
            lo = lo < -7 ? -7 : (lo > 7 ? 7 : lo);
            hi = hi < -7 ? -7 : (hi > 7 ? 7 : hi);
            packed[j] = (uint8_t)((lo & 0x0F) | (hi << 4));
        }
        fwrite(packed, 1, 32, f);
    }
    fclose(f);

    size_t bytes = NAC4_HEADER + (size_t)T_latent * NAC4_FRAME;
    float duration = (float)T_latent * 1920.0f / 48000.0f;
    float kbps = (float)bytes * 8.0f / (duration * 1000.0f);
    fprintf(stderr, "[Latent] Wrote %s: Q4, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n",
            path, T_latent, duration, (float)bytes / 1024.0f, kbps);
    return true;
}

// Write f32 raw latent (no header)
static bool write_latent_f32(const char * path, const float * data, int T_latent) {
    FILE * f = fopen(path, "wb");
    if (!f) return false;
    size_t bytes = (size_t)T_latent * 64 * sizeof(float);
    fwrite(data, 1, bytes, f);
    fclose(f);
    float duration = (float)T_latent * 1920.0f / 48000.0f;
    fprintf(stderr, "[Latent] Wrote %s: f32, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n",
            path, T_latent, duration, (float)bytes / 1024.0f,
            (float)bytes * 8.0f / (duration * 1000.0f));
    return true;
}

// Read latent, auto-detect format (NAC8 -> Q8, NAC4 -> Q4, else f32).
// Returns [T_latent, 64] f32 (dequantized if quantized). Caller frees.
static float * read_latent(const char * path, int * T_latent) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[Latent] Cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    // Check magic
    char magic[4] = {};
    if (fsize >= 8) fread(magic, 1, 4, f);

    if (memcmp(magic, NAC8_MAGIC, 4) == 0) {
        // Q8 format
        uint32_t t;
        fread(&t, 4, 1, f);
        *T_latent = (int)t;

        long expected = NAC8_HEADER + (long)t * NAC8_FRAME;
        if (fsize != expected) {
            fprintf(stderr, "[Latent] Q8 size mismatch: expected %ld, got %ld\n", expected, fsize);
            fclose(f); return NULL;
        }

        float * data = (float *)malloc((size_t)t * 64 * sizeof(float));
        for (int i = 0; i < (int)t; i++) {
            ggml_fp16_t scale_f16;
            fread(&scale_f16, 2, 1, f);
            float scale = ggml_fp16_to_fp32(scale_f16);

            int8_t q[64];
            fread(q, 1, 64, f);

            float * frame = data + i * 64;
            for (int j = 0; j < 64; j++)
                frame[j] = (float)q[j] * scale;
        }
        fclose(f);

        float duration = (float)(*T_latent) * 1920.0f / 48000.0f;
        float kbps = (float)fsize * 8.0f / (duration * 1000.0f);
        fprintf(stderr, "[Latent] Read %s: Q8, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n",
                path, *T_latent, duration, (float)fsize / 1024.0f, kbps);
        return data;
    }

    if (memcmp(magic, NAC4_MAGIC, 4) == 0) {
        // Q4 format
        uint32_t t;
        fread(&t, 4, 1, f);
        *T_latent = (int)t;

        long expected = NAC4_HEADER + (long)t * NAC4_FRAME;
        if (fsize != expected) {
            fprintf(stderr, "[Latent] Q4 size mismatch: expected %ld, got %ld\n", expected, fsize);
            fclose(f); return NULL;
        }

        float * data = (float *)malloc((size_t)t * 64 * sizeof(float));
        for (int i = 0; i < (int)t; i++) {
            ggml_fp16_t scale_f16;
            fread(&scale_f16, 2, 1, f);
            float scale = ggml_fp16_to_fp32(scale_f16);

            uint8_t packed[32];
            fread(packed, 1, 32, f);

            // unpack signed nibbles
            float * frame = data + i * 64;
            for (int j = 0; j < 32; j++) {
                int lo = (int)(packed[j] & 0x0F);
                int hi = (int)(packed[j] >> 4);
                if (lo >= 8) lo -= 16;
                if (hi >= 8) hi -= 16;
                frame[j * 2 + 0] = (float)lo * scale;
                frame[j * 2 + 1] = (float)hi * scale;
            }
        }
        fclose(f);

        float duration = (float)(*T_latent) * 1920.0f / 48000.0f;
        float kbps = (float)fsize * 8.0f / (duration * 1000.0f);
        fprintf(stderr, "[Latent] Read %s: Q4, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n",
                path, *T_latent, duration, (float)fsize / 1024.0f, kbps);
        return data;
    }

    // f32 format (no header, rewind)
    fseek(f, 0, SEEK_SET);
    if (fsize % (64 * (int)sizeof(float)) != 0) {
        fprintf(stderr, "[Latent] File size %ld not a multiple of %d (64 * f32)\n",
                fsize, (int)(64 * sizeof(float)));
        fclose(f); return NULL;
    }

    *T_latent = (int)(fsize / (64 * sizeof(float)));
    float * data = (float *)malloc(fsize);
    fread(data, 1, fsize, f);
    fclose(f);

    float duration = (float)(*T_latent) * 1920.0f / 48000.0f;
    fprintf(stderr, "[Latent] Read %s: f32, %d frames (%.2fs, %.1f KB, %.1f kbit/s)\n",
            path, *T_latent, duration, (float)fsize / 1024.0f,
            (float)fsize * 8.0f / (duration * 1000.0f));
    return data;
}

static void print_usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --vae <gguf> --encode|--decode -i <input> [-o <output>] [--q8|--q4]\n\n"
        "Required:\n"
        "  --vae <path>            VAE GGUF file\n"
        "  --encode | --decode     Encode WAV to latent, or decode latent to WAV\n"
        "  -i <path>               Input (WAV for encode, latent for decode)\n\n"
        "Output:\n"
        "  -o <path>               Output file (auto-named if omitted)\n"
        "  --q8                    Quantize latent to int8 (~13 kbit/s)\n"
        "  --q4                    Quantize latent to int4 (~6.8 kbit/s)\n\n"
        "Output naming: song.wav -> song.latent (f32) or song.nac8 (Q8) or song.nac4 (Q4)\n"
        "               song.latent -> song.wav\n\n"
        "VAE tiling (memory control):\n"
        "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
        "  --vae-overlap <N>       Overlap frames per side (default: 64)\n\n"
        "Latent formats (decode auto-detects):\n"
        "  f32:  flat [T, 64] f32, no header. ~51 kbit/s.\n"
        "  NAC8: header + per-frame Q8. ~13 kbit/s.\n"
        "  NAC4: header + per-frame Q4. ~6.8 kbit/s.\n",
        prog);
}

static std::string auto_output(const char * input, const char * ext) {
    std::string s = input;
    size_t dot = s.rfind('.');
    if (dot != std::string::npos)
        return s.substr(0, dot) + ext;
    return s + ext;
}

int main(int argc, char ** argv) {
    const char * vae_path = NULL;
    const char * input_path = NULL;
    const char * output_path = NULL;
    int chunk_size = 256;
    int overlap = 64;
    int mode = -1;  // 0 = encode, 1 = decode
    int quant = 0;  // 0 = f32, 8 = q8, 4 = q4

    for (int i = 1; i < argc; i++) {
        if      (strcmp(argv[i], "--vae") == 0 && i + 1 < argc)     vae_path = argv[++i];
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc)        input_path = argv[++i];
        else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc)   input_path = argv[++i];
        else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc)        output_path = argv[++i];
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)  output_path = argv[++i];
        else if (strcmp(argv[i], "--vae-chunk") == 0 && i + 1 < argc)   chunk_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--vae-overlap") == 0 && i + 1 < argc) overlap = atoi(argv[++i]);
        else if (strcmp(argv[i], "--encode") == 0)  mode = 0;
        else if (strcmp(argv[i], "--decode") == 0)  mode = 1;
        else if (strcmp(argv[i], "--q8") == 0)      quant = 8;
        else if (strcmp(argv[i], "--q4") == 0)      quant = 4;
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            print_usage(argv[0]); return 1;
        }
    }

    if (!vae_path || !input_path || mode < 0) {
        print_usage(argv[0]); return 1;
    }

    // Auto output names
    std::string out_str;
    if (!output_path) {
        if (mode == 0) {
            const char * ext = ".latent";
            if (quant == 8) ext = ".nac8";
            if (quant == 4) ext = ".nac4";
            out_str = auto_output(input_path, ext);
        } else {
            out_str = auto_output(input_path, ".wav");
        }
        output_path = out_str.c_str();
    }

    const char * quant_str = "";
    if (mode == 0 && quant == 8) quant_str = " (Q8)";
    if (mode == 0 && quant == 4) quant_str = " (Q4)";
    fprintf(stderr, "\n[VAE] Mode: %s%s\n", mode == 0 ? "encode" : "decode", quant_str);
    fprintf(stderr, "[VAE] Input:  %s\n", input_path);
    fprintf(stderr, "[VAE] Output: %s\n\n", output_path);

    // ENCODE
    if (mode == 0) {
        int T_audio = 0, sr = 0;
        float * audio = read_wav(input_path, &T_audio, &sr);
        if (!audio) return 1;
        if (sr != 48000)
            fprintf(stderr, "[WARN] Input is %d Hz, VAE expects 48000. Resample with ffmpeg first.\n", sr);

        VAEEncoder enc = {};
        vae_enc_load(&enc, vae_path);

        int max_T = (T_audio / 1920) + 64;
        std::vector<float> latent((size_t)max_T * 64);

        fprintf(stderr, "\n[VAE] Encoding %d samples (%.2fs)...\n",
                T_audio, (float)T_audio / (float)(sr > 0 ? sr : 48000));
        int T_latent = vae_enc_encode_tiled(&enc, audio, T_audio,
                                             latent.data(), max_T, chunk_size, overlap);
        free(audio);
        if (T_latent < 0) { vae_enc_free(&enc); return 1; }

        if (quant == 8)
            write_latent_q8(output_path, latent.data(), T_latent);
        else if (quant == 4)
            write_latent_q4(output_path, latent.data(), T_latent);
        else
            write_latent_f32(output_path, latent.data(), T_latent);

        vae_enc_free(&enc);
        fprintf(stderr, "[VAE] Done.\n");
        return 0;
    }

    // DECODE (auto-detects f32 vs Q8 vs Q4 from file content)
    {
        int T_latent = 0;
        float * latent = read_latent(input_path, &T_latent);
        if (!latent) return 1;

        VAEGGML dec = {};
        vae_ggml_load(&dec, vae_path);

        int max_T = T_latent * 1920 + 4096;
        std::vector<float> audio((size_t)2 * max_T, 0.0f);

        fprintf(stderr, "\n[VAE] Decoding %d latent frames...\n", T_latent);
        int T_audio = vae_ggml_decode_tiled(&dec, latent, T_latent,
                                             audio.data(), max_T, chunk_size, overlap);
        free(latent);
        if (T_audio < 0) { vae_ggml_free(&dec); return 1; }

        if (write_wav(output_path, audio.data(), T_audio, 48000))
            fprintf(stderr, "\n[VAE] Output: %s (%d samples, %.2fs @ 48kHz)\n",
                    output_path, T_audio, (float)T_audio / 48000.0f);
        else
            fprintf(stderr, "[VAE] FATAL: failed to write %s\n", output_path);

        vae_ggml_free(&dec);
        fprintf(stderr, "[VAE] Done.\n");
        return 0;
    }
}
