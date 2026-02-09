// ace-cuda pipeline: prompt.json -> WAV audio
// Single binary, no PyTorch, no Python runtime.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <random>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include "bpe.h"

using bf16 = __nv_bfloat16;

#include "safetensors.h"
#include "kernels.cuh"
#include "dit.cuh"
#include "transformer.cuh"
#include "text_encoder.cuh"
#include "condition.cuh"
#include "tokenizer.cuh"
#include "vae.cuh"

// Timer for phase tracking
struct Timer {
    std::chrono::steady_clock::time_point t;
    Timer() : t(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t).count();
    }
};

// WAV writer (PCM16 stereo 48kHz)
static void write_wav(const char *path, const float *samples, int n_samples, int channels, int sample_rate) {
    FILE *f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s for writing\n", path); return; }

    int data_size = n_samples * channels * sizeof(int16_t);
    int file_size = 36 + data_size;

    // RIFF header
    fwrite("RIFF", 1, 4, f);
    uint32_t tmp32 = file_size; fwrite(&tmp32, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    // fmt chunk
    fwrite("fmt ", 1, 4, f);
    tmp32 = 16; fwrite(&tmp32, 4, 1, f);
    uint16_t tmp16 = 1; fwrite(&tmp16, 2, 1, f);  // PCM
    tmp16 = channels; fwrite(&tmp16, 2, 1, f);
    tmp32 = sample_rate; fwrite(&tmp32, 4, 1, f);
    tmp32 = sample_rate * channels * sizeof(int16_t); fwrite(&tmp32, 4, 1, f);
    tmp16 = channels * sizeof(int16_t); fwrite(&tmp16, 2, 1, f);
    tmp16 = 16; fwrite(&tmp16, 2, 1, f);

    // data chunk
    fwrite("data", 1, 4, f);
    tmp32 = data_size; fwrite(&tmp32, 4, 1, f);

    std::vector<int16_t> pcm(n_samples * channels);
    for (int i = 0; i < n_samples * channels; i++) {
        float v = fmaxf(-1.0f, fminf(1.0f, samples[i]));
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    fwrite(pcm.data(), sizeof(int16_t), pcm.size(), f);
    fclose(f);
}

// Load silence_latent.bin [15000, 64] float32
static float *load_silence_latent(const char *path, int *out_T) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "FATAL: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n_floats = sz / sizeof(float);
    int T = n_floats / 64;
    if (T * 64 != n_floats) {
        fprintf(stderr, "FATAL: silence_latent.bin size mismatch: %ld bytes\n", sz);
        exit(1);
    }
    float *data = (float *)malloc(sz);
    fread(data, 1, sz, f);
    fclose(f);
    *out_T = T;
    return data;
}

// CUDA kernels (pipeline-specific)
__global__ void fill_ones_bf16_kernel(bf16 *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    data[i] = __float2bfloat16(1.0f);
}

__global__ void transpose_2d_kernel(bf16 *out, const bf16 *in, int rows, int cols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * cols) return;
    int r = idx / cols;
    int c = idx % cols;
    out[c * rows + r] = in[idx];
}

__global__ void interleave_stereo_kernel(float *out, const bf16 *in, int T) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    out[t * 2 + 0] = __bfloat162float(in[t]);
    out[t * 2 + 1] = __bfloat162float(in[T + t]);
}

// Compute timestep schedule: t' = shift * t / (1 + (shift-1) * t)
static void build_schedule(float *out, int steps, float shift) {
    for (int i = 0; i < steps; i++) {
        float t = 1.0f - (float)i / (float)steps;
        out[i] = shift * t / (1.0f + (shift - 1.0f) * t);
    }
}

static void generate_audio(
    const char *text_enc_dir,   // Qwen3-Embedding-0.6B
    const char *dit_dir,        // acestep-v15-turbo
    const char *vae_dir,        // vae
    const int *text_ids, int S_text,
    const int *lyric_ids, int S_lyric,
    float duration_sec,
    int seed,
    const char *output_path,
    const float *t_schedule,
    int num_steps,
    const int *audio_codes = nullptr,
    int n_audio_codes = 0,
    const char *noise_file = nullptr)
{
    Timer t0;

    std::string silence_path = std::string(dit_dir) + "/silence_latent.bin";

    // Compute dimensions
    int T_25Hz = (int)(duration_sec * 25.0f);
    if (T_25Hz % 2 != 0) T_25Hz++;
    int T_audio = T_25Hz * 1920;
    fprintf(stderr, "[Pipeline] %.1fs, T_25Hz=%d, T_audio=%d (%.2fs @ 48kHz)\n",
            duration_sec, T_25Hz, T_audio, (float)T_audio / 48000.0f);

    int max_text_seq = S_text > S_lyric ? S_text : S_lyric;
    if (max_text_seq < 512) max_text_seq = 512;

    // 1) Load TextEncoder
    Timer t_load;
    TextEncoder text_enc;
    text_encoder_load(&text_enc, text_enc_dir, max_text_seq);
    fprintf(stderr, "[Load] TextEncoder (%.0fms)\n", t_load.ms());

    // 2) Load DiT model
    SafeTensors model_st;
    if (!safe_load(model_st, dit_dir)) {
        fprintf(stderr, "FATAL: cannot load %s\n", dit_dir);
        exit(1);
    }

    // 2a) ConditionEncoder
    cublasHandle_t cond_cublas;
    cublasCreate(&cond_cublas);
    ConditionEncoder cond_enc;
    load_condition_encoder(&cond_enc, model_st,
        S_lyric + 16, 768, S_text + 16, cond_cublas);
    fprintf(stderr, "[Load] ConditionEncoder (%.0fms)\n", t_load.ms());

    // 2b) DiT
    DiTConfig dit_cfg = dit_default_config();
    dit_cfg.max_seq_len = T_25Hz / 2 + 16;
    DiTModel dit;
    load_dit_model(&dit, dit_dir, dit_cfg);
    fprintf(stderr, "[Load] DiT (%.0fms)\n", t_load.ms());

    // 2c) FSQ + Detokenizer (for LM audio codes)
    FSQ fsq = {};
    Detokenizer detok = {};
    bool have_audio_decoder = false;
    if (audio_codes && n_audio_codes > 0) {
        cublasHandle_t cublas;
        cublasCreate(&cublas);
        fsq.project_out_w = must_upload(model_st, "tokenizer.quantizer.project_out.weight");
        fsq.project_out_b = must_upload(model_st, "tokenizer.quantizer.project_out.bias");
        int T_5Hz = n_audio_codes;
        cudaMalloc(&fsq.buf_quant, (size_t)T_5Hz * FSQ_NDIMS * sizeof(bf16));
        cudaMalloc(&fsq.buf_6, (size_t)T_5Hz * FSQ_NDIMS * sizeof(bf16));
        load_detokenizer(&detok, model_st, T_5Hz, cublas);
        have_audio_decoder = true;
        fprintf(stderr, "[Load] FSQ + Detokenizer: %d codes -> %d frames (%.0fms)\n",
                T_5Hz, T_5Hz * 5, t_load.ms());
    }

    // 3) Load VAE
    VAEDecoder vae;
    load_vae_decoder(&vae, vae_dir, T_25Hz + 16);
    double load_ms = t_load.ms();
    fprintf(stderr, "[Load] VAE (%.0fms)\n", load_ms);

    // 4) Load silence latent
    int silence_T;
    float *silence_host = load_silence_latent(silence_path.c_str(), &silence_T);
    if (silence_T < T_25Hz) {
        fprintf(stderr, "FATAL: silence_latent too short: %d < %d\n", silence_T, T_25Hz);
        exit(1);
    }

    Timer t_encode;

    // Step 1: TextEncoder
    fprintf(stderr, "[Encode] TextEncoder: %d tokens -> [%d, 1024]\n", S_text, S_text);
    text_encoder_forward(&text_enc, text_ids, S_text);

    // Step 2: Lyric embedding
    bf16 *lyric_embed_gpu;
    cudaMalloc(&lyric_embed_gpu, (size_t)S_lyric * 1024 * sizeof(bf16));
    lyric_embed(&text_enc, lyric_ids, S_lyric, lyric_embed_gpu);

    // Step 3: ConditionEncoder
    int S_ref = 750;  // 30 seconds @ 25Hz
    bf16 *timbre_feats_gpu;
    {
        int n = S_ref * 64;
        float *tmp_f32;
        cudaMalloc(&tmp_f32, n * sizeof(float));
        cudaMalloc(&timbre_feats_gpu, n * sizeof(bf16));
        cudaMemcpy(tmp_f32, silence_host, n * sizeof(float), cudaMemcpyHostToDevice);
        f32_to_bf16_kernel<<<(n + 255) / 256, 256>>>(timbre_feats_gpu, tmp_f32, n);
        cudaFree(tmp_f32);
    }
    condition_encoder_forward(&cond_enc,
                              text_enc.enc.buf_hidden, S_text,
                              lyric_embed_gpu, S_lyric,
                              timbre_feats_gpu, S_ref);
    int S_enc = cond_enc.enc_seq_len;
    fprintf(stderr, "[Encode] ConditionEncoder: [%d, 2048]\n", S_enc);
    double encode_ms = t_encode.ms();
    fprintf(stderr, "[Encode] %.0fms\n", encode_ms);

    // Step 4: context_latents [T_25Hz, 128]
    Timer t_context;
    bf16 *context_latents_gpu;
    cudaMalloc(&context_latents_gpu, (size_t)T_25Hz * 128 * sizeof(bf16));

    {
        int n = T_25Hz * 64;
        bf16 *latent_bf16;
        cudaMalloc(&latent_bf16, n * sizeof(bf16));

        if (have_audio_decoder && audio_codes && n_audio_codes > 0) {
            // Decode LM audio codes -> FSQ -> detokenizer -> [T_25Hz_codes, 64]
            int T_5Hz = n_audio_codes;
            int T_25Hz_codes = T_5Hz * 5;

            int *codes_gpu;
            cudaMalloc(&codes_gpu, T_5Hz * sizeof(int));
            cudaMemcpy(codes_gpu, audio_codes, T_5Hz * sizeof(int), cudaMemcpyHostToDevice);

            bf16 *quantized_gpu;
            cudaMalloc(&quantized_gpu, (size_t)T_5Hz * 2048 * sizeof(bf16));
            fsq_indices_to_quantized(&fsq, codes_gpu, T_5Hz, quantized_gpu, detok.cublas);

            bf16 *decoded_gpu;
            cudaMalloc(&decoded_gpu, (size_t)T_25Hz_codes * 64 * sizeof(bf16));
            detokenizer_forward(&detok, quantized_gpu, T_5Hz, decoded_gpu);

            int copy_T = T_25Hz_codes < T_25Hz ? T_25Hz_codes : T_25Hz;
            cudaMemcpy(latent_bf16, decoded_gpu, (size_t)copy_T * 64 * sizeof(bf16),
                       cudaMemcpyDeviceToDevice);

            // Pad remainder with silence
            if (copy_T < T_25Hz) {
                int pad_n = (T_25Hz - copy_T) * 64;
                float *sil_f32;
                bf16 *sil_bf16;
                cudaMalloc(&sil_f32, pad_n * sizeof(float));
                cudaMalloc(&sil_bf16, pad_n * sizeof(bf16));
                cudaMemcpy(sil_f32, silence_host + copy_T * 64, pad_n * sizeof(float),
                           cudaMemcpyHostToDevice);
                f32_to_bf16_kernel<<<(pad_n + 255) / 256, 256>>>(sil_bf16, sil_f32, pad_n);
                cudaMemcpy(latent_bf16 + (int64_t)copy_T * 64, sil_bf16,
                           pad_n * sizeof(bf16), cudaMemcpyDeviceToDevice);
                cudaFree(sil_f32);
                cudaFree(sil_bf16);
                fprintf(stderr, "[Context] Audio codes: %d decoded + %d silence\n",
                        copy_T, T_25Hz - copy_T);
            } else {
                fprintf(stderr, "[Context] Audio codes: %d decoded (full)\n", copy_T);
            }

            cudaFree(codes_gpu);
            cudaFree(quantized_gpu);
            cudaFree(decoded_gpu);
        } else {
            // Silence fallback
            float *sil_f32;
            cudaMalloc(&sil_f32, n * sizeof(float));
            cudaMemcpy(sil_f32, silence_host, n * sizeof(float), cudaMemcpyHostToDevice);
            f32_to_bf16_kernel<<<(n + 255) / 256, 256>>>(latent_bf16, sil_f32, n);
            cudaFree(sil_f32);
        }

        // Build context_latents: cat(latent[T,64], ones[T,64]) -> [T,128]
        bf16 *chunk_mask_gpu;
        cudaMalloc(&chunk_mask_gpu, n * sizeof(bf16));
        fill_ones_bf16_kernel<<<(n + 255) / 256, 256>>>(chunk_mask_gpu, n);

        int total = T_25Hz * 128;
        concat_channels_kernel<<<(total + 255) / 256, 256>>>(
            context_latents_gpu, latent_bf16, chunk_mask_gpu,
            T_25Hz, 64, 64, 128);

        cudaFree(latent_bf16);
        cudaFree(chunk_mask_gpu);
    }
    fprintf(stderr, "[Context] [%d, 128] Ready\n", T_25Hz);

    // Step 5: Noise [T_25Hz, 64]
    int noise_n = T_25Hz * 64;
    bf16 *noise_gpu;
    cudaMalloc(&noise_gpu, noise_n * sizeof(bf16));
    if (noise_file) {
        // Load pre-generated noise (raw bf16, e.g. from PyTorch)
        FILE *nf = fopen(noise_file, "rb");
        if (!nf) { fprintf(stderr, "ERROR: cannot open noise file %s\n", noise_file); return; }
        std::vector<bf16> noise_cpu(noise_n);
        size_t got = fread(noise_cpu.data(), sizeof(bf16), noise_n, nf);
        fclose(nf);
        if ((int)got != noise_n) {
            fprintf(stderr, "WARNING: noise file has %zu values, expected %d\n", got, noise_n);
        }
        cudaMemcpy(noise_gpu, noise_cpu.data(), noise_n * sizeof(bf16), cudaMemcpyHostToDevice);
        fprintf(stderr, "[Context] Noise: [%d, 64] loaded from %s\n", T_25Hz, noise_file);
    } else {
        std::mt19937 rng(seed);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        std::vector<float> noise_cpu(noise_n);
        for (int i = 0; i < noise_n; i++) noise_cpu[i] = dist(rng);

        float *noise_f32_gpu;
        cudaMalloc(&noise_f32_gpu, noise_n * sizeof(float));
        cudaMemcpy(noise_f32_gpu, noise_cpu.data(), noise_n * sizeof(float), cudaMemcpyHostToDevice);
        f32_to_bf16_kernel<<<(noise_n + 255) / 256, 256>>>(noise_gpu, noise_f32_gpu, noise_n);
        cudaFree(noise_f32_gpu);
        fprintf(stderr, "[Context] Noise: [%d, 64] (seed=%d)\n", T_25Hz, seed);
    }
    double context_ms = t_context.ms();
    fprintf(stderr, "[Context] %.0fms\n", context_ms);

    // Step 6: DiT flow matching
    bf16 *x0_gpu;
    cudaMalloc(&x0_gpu, (size_t)T_25Hz * 64 * sizeof(bf16));

    Timer t_dit;
    fprintf(stderr, "[DiT] Flow Matching: %d steps, T=%d\n", num_steps, T_25Hz);

    dit_generate(&dit, noise_gpu, context_latents_gpu,
                 cond_enc.encoder_hidden_states, S_enc, T_25Hz,
                 t_schedule, num_steps,
                 x0_gpu);
    cudaDeviceSynchronize();

    double dit_ms = t_dit.ms();
    fprintf(stderr, "[DiT] %.0fms (%.1fms/step)\n", dit_ms, dit_ms / num_steps);

    // Step 7: VAE decode
    bf16 *x0_transposed;
    cudaMalloc(&x0_transposed, (size_t)T_25Hz * 64 * sizeof(bf16));
    {
        int n = T_25Hz * 64;
        transpose_2d_kernel<<<(n + 255) / 256, 256>>>(x0_transposed, x0_gpu, T_25Hz, 64);
    }
    Timer t_vae;
    fprintf(stderr, "[VAE] Decode: [64, %d] -> [2, %d]\n", T_25Hz, T_audio);

    bf16 *audio_bf16;
    cudaMalloc(&audio_bf16, (size_t)T_audio * 2 * sizeof(bf16));

    int actual_T_audio = vae_decode(&vae, x0_transposed, T_25Hz, audio_bf16);

    cudaDeviceSynchronize();
    double vae_ms = t_vae.ms();
    fprintf(stderr, "[VAE] %.0fms\n", vae_ms);

    // Step 8: Write WAV
    Timer t_wav;
    int audio_elems = actual_T_audio * 2;
    float *audio_f32_gpu;
    cudaMalloc(&audio_f32_gpu, audio_elems * sizeof(float));
    interleave_stereo_kernel<<<(actual_T_audio + 255) / 256, 256>>>(audio_f32_gpu, audio_bf16, actual_T_audio);

    std::vector<float> audio_cpu(audio_elems);
    cudaMemcpy(audio_cpu.data(), audio_f32_gpu, audio_elems * sizeof(float), cudaMemcpyDeviceToHost);

    write_wav(output_path, audio_cpu.data(), actual_T_audio, 2, 48000);
    double wav_ms = t_wav.ms();
    fprintf(stderr, "[WAV] %s (%.1fs @ 48kHz stereo) %.0fms\n",
            output_path, (float)actual_T_audio / 48000.0f, wav_ms);

    double total_ms = t0.ms();
    fprintf(stderr, "[Pipeline] Load %.0f | Encode %.0f | Context %.0f | DiT %.0f | VAE %.0f | WAV %.0f | Total %.0fms\n",
            load_ms, encode_ms, context_ms, dit_ms, vae_ms, wav_ms, total_ms);

    // Cleanup
    free(silence_host);
    cudaFree(lyric_embed_gpu);
    cudaFree(timbre_feats_gpu);
    cudaFree(context_latents_gpu);
    cudaFree(noise_gpu);
    cudaFree(x0_gpu);
    cudaFree(x0_transposed);
    cudaFree(audio_bf16);
    cudaFree(audio_f32_gpu);
}

// Prompt JSON parser
struct Prompt {
    std::string caption;
    std::string lyrics;
    std::string vocal_language = "unknown";
    std::string timesignature = "4";
    std::string keyscale;
    std::string bpm;
    float duration = 30.0f;
    bool instrumental = false;
    int seed = 42;
};

static std::string json_get_string(const std::string &json, const char *key) {
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    // Skip whitespace after colon
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    // Must start with quote to be a string value
    if (pos >= json.size() || json[pos] != '"') return "";
    pos++; // skip opening quote
    std::string result;
    while (pos < json.size() && json[pos] != '"') {
        if (json[pos] == '\\' && pos + 1 < json.size()) {
            pos++;
            switch (json[pos]) {
                case 'n': result += '\n'; break;
                case 't': result += '\t'; break;
                case '"': result += '"'; break;
                case '\\': result += '\\'; break;
                default: result += json[pos]; break;
            }
        } else {
            result += json[pos];
        }
        pos++;
    }
    return result;
}

// Get JSON value as string (handles both "key": "str" and "key": 123)
static std::string json_get_value(const std::string &json, const char *key) {
    std::string s = json_get_string(json, key);
    if (!s.empty()) return s;
    // Try numeric: find "key": <number>
    std::string needle = std::string("\"") + key + "\"";
    size_t pos = json.find(needle);
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos + needle.size());
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) pos++;
    if (pos >= json.size() || (!isdigit(json[pos]) && json[pos] != '-')) return "";
    size_t start = pos;
    while (pos < json.size() && (isdigit(json[pos]) || json[pos] == '.' || json[pos] == '-')) pos++;
    return json.substr(start, pos - start);
}

static bool load_prompt(const char *path, Prompt *p) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return false; }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string json(sz, '\0');
    fread(&json[0], 1, sz, f);
    fclose(f);

    p->caption = json_get_string(json, "caption");
    p->lyrics = json_get_string(json, "lyrics");
    std::string tmp;
    tmp = json_get_string(json, "vocal_language");
    if (!tmp.empty()) p->vocal_language = tmp;
    tmp = json_get_value(json, "timesignature");
    if (!tmp.empty()) p->timesignature = tmp;
    tmp = json_get_string(json, "keyscale");
    if (!tmp.empty()) p->keyscale = tmp;
    tmp = json_get_value(json, "bpm");
    if (!tmp.empty()) p->bpm = tmp;

    size_t dpos = json.find("\"duration\"");
    if (dpos != std::string::npos) {
        dpos = json.find(':', dpos);
        if (dpos != std::string::npos) p->duration = atof(json.c_str() + dpos + 1);
    }
    size_t spos = json.find("\"seed\"");
    if (spos != std::string::npos) {
        spos = json.find(':', spos);
        if (spos != std::string::npos) p->seed = atoi(json.c_str() + spos + 1);
    }
    size_t ipos = json.find("\"instrumental\"");
    if (ipos != std::string::npos) {
        p->instrumental = (json.find("true", ipos) != std::string::npos &&
                           json.find("true", ipos) < json.find('\n', ipos));
    }
    if (p->caption.empty()) { fprintf(stderr, "ERROR: no caption in %s\n", path); return false; }
    return true;
}

// ACE-Step prompt templates
static const char *INSTRUCTION_TEXT2MUSIC = "Fill the audio semantic mask based on the given conditions:";
static const char *INSTRUCTION_COVER = "Generate audio semantic tokens based on the given conditions:";

static std::string build_text_prompt(const Prompt &p, bool is_cover) {
    const char *instruction = is_cover ? INSTRUCTION_COVER : INSTRUCTION_TEXT2MUSIC;
    std::string bpm_str = p.bpm.empty() ? "N/A" : p.bpm;
    std::string ts_str = p.timesignature.empty() ? "N/A" : p.timesignature;
    std::string ks_str = p.keyscale.empty() ? "N/A" : p.keyscale;
    char metas[512];
    snprintf(metas, sizeof(metas),
             "- bpm: %s\n- timesignature: %s\n- keyscale: %s\n- duration: %d seconds\n",
             bpm_str.c_str(), ts_str.c_str(), ks_str.c_str(), (int)p.duration);
    return std::string("# Instruction\n") + instruction + "\n\n"
         + "# Caption\n" + p.caption + "\n\n"
         + "# Metas\n" + metas + "<|endoftext|>\n";
}

static std::string build_lyric_prompt(const Prompt &p) {
    std::string lyrics = p.instrumental ? "[Instrumental]" : p.lyrics;
    return "# Languages\n" + p.vocal_language + "\n\n# Lyric\n" + lyrics + "<|endoftext|>";
}

// Parse audio codes from file (comma-separated integers)
static std::vector<int> load_audio_codes(const char *path) {
    std::vector<int> codes;
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "ERROR: cannot open %s\n", path); return codes; }
    char buf[65536];
    while (fgets(buf, sizeof(buf), f)) {
        const char *p = buf;
        while (*p) {
            while (*p == ',' || *p == ' ' || *p == '\n' || *p == '\r') p++;
            if (!*p) break;
            codes.push_back(atoi(p));
            while (*p && *p != ',' && *p != ' ' && *p != '\n') p++;
        }
    }
    fclose(f);
    return codes;
}

// Main
static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s [options]\n", prog);
    fprintf(stderr, "  --prompt <json>         Prompt JSON file (required)\n");
    fprintf(stderr, "  --text-encoder <dir>    Qwen3-Embedding-0.6B directory\n");
    fprintf(stderr, "  --dit <dir>             DiT model directory (e.g. acestep-v15-turbo)\n");
    fprintf(stderr, "  --vae <dir>             VAE directory\n");
    fprintf(stderr, "  --input-codes <file>    LM audio codes (from ace-qwen3 --output-codes)\n");
    fprintf(stderr, "  --duration <sec>        Override duration (default: from prompt)\n");
    fprintf(stderr, "  --seed <n>              Override seed (default: from prompt)\n");
    fprintf(stderr, "  --shift <f>             Timestep shift (default: 3.0)\n");
    fprintf(stderr, "  --steps <n>             Euler steps (default: 8)\n");
    fprintf(stderr, "  --output <path>         Output WAV (default: output.wav)\n");
    fprintf(stderr, "\nExample:\n");
    fprintf(stderr, "  %s --prompt prompt.json --text-encoder checkpoints/Qwen3-Embedding-0.6B \\\n", prog);
    fprintf(stderr, "     --dit checkpoints/acestep-v15-turbo --vae checkpoints/vae \\\n");
    fprintf(stderr, "     --input-codes codes.txt --output song.wav\n");
}

int main(int argc, char **argv) {
    const char *prompt_file = nullptr;
    const char *text_enc_dir = nullptr;
    const char *dit_dir = nullptr;
    const char *vae_dir = nullptr;
    const char *audio_codes_file = nullptr;
    const char *noise_file = nullptr;
    const char *output = "output.wav";
    float duration = -1;
    int seed = -1;
    float shift = 3.0f;
    int steps = 8;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--prompt") && i + 1 < argc)
            prompt_file = argv[++i];
        else if (!strcmp(argv[i], "--text-encoder") && i + 1 < argc)
            text_enc_dir = argv[++i];
        else if (!strcmp(argv[i], "--dit") && i + 1 < argc)
            dit_dir = argv[++i];
        else if (!strcmp(argv[i], "--vae") && i + 1 < argc)
            vae_dir = argv[++i];
        else if (!strcmp(argv[i], "--input-codes") && i + 1 < argc)
            audio_codes_file = argv[++i];
        else if (!strcmp(argv[i], "--noise") && i + 1 < argc)
            noise_file = argv[++i];
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc)
            duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shift") && i + 1 < argc)
            shift = atof(argv[++i]);
        else if (!strcmp(argv[i], "--steps") && i + 1 < argc)
            steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--output") && i + 1 < argc)
            output = argv[++i];
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!prompt_file) {
        fprintf(stderr, "ERROR: --prompt required\n");
        usage(argv[0]);
        return 1;
    }
    if (!text_enc_dir || !dit_dir || !vae_dir) {
        fprintf(stderr, "ERROR: --text-encoder, --dit, and --vae are required\n");
        usage(argv[0]);
        return 1;
    }

    // Load prompt
    Prompt prompt;
    if (!load_prompt(prompt_file, &prompt)) return 1;
    if (duration < 0) duration = prompt.duration;
    if (seed < 0) seed = prompt.seed;

    // Load audio codes (before prompt building to detect cover mode)
    std::vector<int> audio_codes;
    bool is_cover = false;
    if (audio_codes_file) {
        audio_codes = load_audio_codes(audio_codes_file);
        if (audio_codes.empty()) {
            fprintf(stderr, "ERROR: no audio codes in %s\n", audio_codes_file);
            return 1;
        }
        is_cover = true;
        fprintf(stderr, "[Pipeline] %zu audio codes from %s (%.1fs @ 5Hz)\n",
                audio_codes.size(), audio_codes_file, audio_codes.size() / 5.0f);
    }

    // Tokenize
    std::string bpe_dir = std::string(text_enc_dir);
    BPETokenizer bpe;
    if (!load_bpe_tokenizer(&bpe, bpe_dir.c_str())) return 1;

    std::string text_str = build_text_prompt(prompt, is_cover);
    std::string lyric_str = build_lyric_prompt(prompt);
    std::vector<int> text_ids = bpe_encode(&bpe, text_str, true);
    std::vector<int> lyric_ids = bpe_encode(&bpe, lyric_str, true);
    fprintf(stderr, "[Pipeline] Text: %zu tokens, Lyrics: %zu tokens\n", text_ids.size(), lyric_ids.size());

    fprintf(stderr, "[Pipeline] %.1fs, Seed: %d, Shift: %.1f, Steps: %d, Output: %s\n",
            duration, seed, shift, steps, output);

    float t_schedule[64];
    build_schedule(t_schedule, steps, shift);
    fprintf(stderr, "[Pipeline] Schedule: [");
    for (int i = 0; i < steps; i++) fprintf(stderr, "%s%.4f", i ? ", " : "", t_schedule[i]);
    fprintf(stderr, "]\n");

    generate_audio(text_enc_dir, dit_dir, vae_dir,
                   text_ids.data(), (int)text_ids.size(),
                   lyric_ids.data(), (int)lyric_ids.size(),
                   duration, seed, output, t_schedule, steps,
                   audio_codes.empty() ? nullptr : audio_codes.data(),
                   (int)audio_codes.size(),
                   noise_file);

    return 0;
}
