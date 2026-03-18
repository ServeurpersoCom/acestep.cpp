// ace-server.cpp - HTTP server for ACE-Step music generation
//
// Single binary, three endpoints (POST /lm, POST /synth, POST /understand),
// one port. Models loaded at boot based on which options are provided:
//
//   --lm                          -> enables /lm + /understand
//   --embedding + --dit + --vae   -> enables /synth
//
// At least one complete group required. Endpoints for missing groups return
// 501. When both are loaded, /synth runs on its own mutex so it can
// overlap with /lm on the GPU (disjoint models). /lm and /understand always
// share mtx_lm (same Qwen3 KV cache). When only one group is loaded,
// everything is serial on mtx_lm.
//
// The understand pipeline shares the Qwen3 LM from pipeline-lm to save
// ~5GB VRAM. Audio input mode (/understand with WAV/MP3) additionally
// needs DiT + VAE from the synth pipeline for FSQ tokenization.

#include "audio-io.h"
#include "pipeline-lm.h"
#include "pipeline-synth.h"
#include "pipeline-understand.h"
#include "request.h"

// suppress warnings in third-party headers
#ifdef __GNUC__
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wshadow"
#endif
#include "httplib.h"
#ifdef __GNUC__
#    pragma GCC diagnostic pop
#endif

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <random>
#include <string>
#include <vector>

// server instance pointer for the signal handler
static httplib::Server * g_svr = nullptr;

static void on_signal(int) {
    if (g_svr) {
        g_svr->stop();
    }
}

// global queue: tracks total requests in the system (waiting + running).
// all three endpoints share the same counter. when full -> 503.
static std::atomic<int> g_queue_n{ 0 };
static int              g_max_queue = 4;

static bool queue_acquire(void) {
    int cur = g_queue_n.fetch_add(1, std::memory_order_relaxed);
    if (cur >= g_max_queue) {
        g_queue_n.fetch_sub(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

static void queue_release(void) {
    g_queue_n.fetch_sub(1, std::memory_order_relaxed);
}

// pipeline mutexes.
// /lm and /understand always lock mtx_lm (shared Qwen3 KV cache).
// /synth locks mtx_synth when both pipelines are loaded (disjoint GPU mem,
// safe to overlap). when synth is the only pipeline, it uses mtx_lm
// (no contention anyway, keeps things simple).
static std::mutex mtx_lm;
static std::mutex mtx_synth;

// pipeline contexts. NULL when the corresponding pipeline was not loaded.
static AceLm *         g_ctx_lm         = nullptr;
static AceSynth *      g_ctx_synth      = nullptr;
static AceUnderstand * g_ctx_understand = nullptr;

// limits
static int g_max_batch = 1;
static int g_mp3_kbps  = 128;

// cancel trampoline: bridges httplib's is_connection_closed to our cancel callback.
// data points to the std::function<bool()> from httplib::Request.
static bool server_cancel(void * data) {
    auto * fn = (const std::function<bool()> *) data;
    return (*fn)();
}

// helper: set a JSON error response
static void json_error(httplib::Response & res, int status, const char * msg) {
    char buf[512];
    snprintf(buf, sizeof(buf), "{\"error\":\"%s\"}", msg);
    res.status = status;
    res.set_content(buf, "application/json");
}

// helper: set a 503 with queue info
static void json_busy(httplib::Response & res) {
    int  n = g_queue_n.load(std::memory_order_relaxed);
    char buf[256];
    snprintf(buf, sizeof(buf), "{\"error\":\"server busy\",\"queue\":%d,\"max_queue\":%d}", n, g_max_queue);
    res.status = 503;
    res.set_header("Retry-After", "5");
    res.set_content(buf, "application/json");
}

// POST /lm
// accepts: AceRequest JSON
// returns: JSON array of enriched AceRequests (batch_size controls count)
static void handle_lm(const httplib::Request & req, httplib::Response & res) {
    if (!g_ctx_lm) {
        json_error(res, 501, "LM pipeline not loaded (requires --lm)");
        return;
    }
    if (!queue_acquire()) {
        json_busy(res);
        return;
    }

    // parse the incoming JSON body
    AceRequest ace_req;
    if (!request_parse_json(&ace_req, req.body.c_str())) {
        queue_release();
        json_error(res, 400, "invalid JSON");
        return;
    }

    // clamp batch_size to [1, max_batch]
    int batch_size = ace_req.batch_size;
    if (batch_size < 1) {
        batch_size = 1;
    }
    if (batch_size > g_max_batch) {
        batch_size = g_max_batch;
    }

    // run the LM pipeline under lock
    std::vector<AceRequest> out(batch_size);
    int                     rc;
    {
        std::lock_guard<std::mutex> lock(mtx_lm);
        rc = ace_lm_generate(g_ctx_lm, &ace_req, batch_size, out.data(), NULL, NULL, server_cancel,
                             (void *) &req.is_connection_closed);
    }
    queue_release();

    if (rc != 0) {
        json_error(res, 500, "LM generation failed");
        return;
    }

    // serialize output as a JSON array
    std::string body = "[";
    for (int i = 0; i < batch_size; i++) {
        if (i > 0) {
            body += ",";
        }
        body += request_to_json(&out[i]);
    }
    body += "]";

    res.set_content(body, "application/json");
}

// POST /synth
// Two modes:
//   application/json body        -> text2music (no source audio)
//   multipart/form-data          -> cover/repaint/lego (with source audio)
//     part "request": JSON text
//     part "audio":   WAV or MP3 file
// returns:
//   batch_size == 1: audio/mpeg with X-Seed, X-Duration, X-Compute-Ms headers
//   batch_size >  1: multipart/mixed, each part is audio/mpeg with per-part headers
static void handle_synth(const httplib::Request & req, httplib::Response & res) {
    if (!g_ctx_synth) {
        json_error(res, 501, "synth pipeline not loaded (requires --embedding --dit --vae)");
        return;
    }
    if (!queue_acquire()) {
        json_busy(res);
        return;
    }

    // parse request: plain JSON or multipart (JSON + audio file)
    AceRequest ace_req;
    float *    src_interleaved = nullptr;
    int        src_len         = 0;

    if (req.is_multipart_form_data()) {
        // multipart mode: "request" part = JSON, optional "audio" part = WAV/MP3

        // extract JSON from "request" part (try file first, then field)
        std::string json_body;
        if (req.form.has_file("request")) {
            json_body = req.form.get_file("request").content;
        } else if (req.form.has_field("request")) {
            json_body = req.form.get_field("request");
        } else {
            queue_release();
            json_error(res, 400, "multipart: missing 'request' part");
            return;
        }
        if (!request_parse_json(&ace_req, json_body.c_str())) {
            queue_release();
            json_error(res, 400, "multipart: invalid JSON in 'request' part");
            return;
        }

        // extract source audio from "audio" part (optional)
        if (req.form.has_file("audio")) {
            auto file = req.form.get_file("audio");
            if (file.content.empty()) {
                queue_release();
                json_error(res, 400, "multipart: empty 'audio' part");
                return;
            }

            // decode directly from multipart buffer (WAV/MP3 auto-detected)
            int     T_audio = 0;
            float * planar  = audio_read_48k_buf((const uint8_t *) file.content.data(), file.content.size(), &T_audio);
            if (!planar || T_audio <= 0) {
                queue_release();
                json_error(res, 400, "failed to decode audio");
                return;
            }

            fprintf(stderr, "[Server] Source audio: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

            // convert planar [L:T][R:T] to interleaved [L0,R0,L1,R1,...] for pipeline
            src_interleaved = (float *) malloc((size_t) T_audio * 2 * sizeof(float));
            for (int t = 0; t < T_audio; t++) {
                src_interleaved[t * 2 + 0] = planar[t];
                src_interleaved[t * 2 + 1] = planar[T_audio + t];
            }
            free(planar);
            src_len = T_audio;
        }
    } else {
        // plain JSON body (backward compatible, no source audio)
        if (!request_parse_json(&ace_req, req.body.c_str())) {
            queue_release();
            json_error(res, 400, "invalid JSON");
            return;
        }
    }

    // clamp batch_size to [1, max_batch]
    int batch_n = ace_req.batch_size;
    if (batch_n < 1) {
        batch_n = 1;
    }
    if (batch_n > g_max_batch) {
        batch_n = g_max_batch;
    }

    // resolve seed so we can report it in response headers.
    // pipeline skips resolution when seed >= 0 (same logic as pipeline-synth.cpp).
    if (ace_req.seed < 0) {
        std::random_device rd;
        ace_req.seed = (int64_t) rd() << 32 | rd();
        if (ace_req.seed < 0) {
            ace_req.seed = -ace_req.seed;
        }
    }

    // synth gets its own mutex when LM is also loaded (disjoint GPU mem).
    // when synth is the only pipeline, just use mtx_lm (no contention).
    std::mutex & mtx = g_ctx_lm ? mtx_synth : mtx_lm;

    // generate N tracks (single graph with N samples)
    std::vector<AceAudio> audio(batch_n);
    int                   rc;
    auto                  t0 = std::chrono::steady_clock::now();
    {
        std::lock_guard<std::mutex> lock(mtx);
        rc = ace_synth_generate(g_ctx_synth, &ace_req, src_interleaved, src_len, batch_n, audio.data(), server_cancel,
                                (void *) &req.is_connection_closed);
    }
    auto t1 = std::chrono::steady_clock::now();
    free(src_interleaved);

    if (rc != 0) {
        for (int b = 0; b < batch_n; b++) {
            ace_audio_free(&audio[b]);
        }
        queue_release();
        json_error(res, 500, "synth generation failed");
        return;
    }

    float compute_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

    // encode each track to MP3 (peak normalize + encode)
    std::vector<std::string> mp3s(batch_n);
    std::vector<float>       durations(batch_n);
    for (int b = 0; b < batch_n; b++) {
        if (!audio[b].samples) {
            durations[b] = 0.0f;
            continue;
        }
        durations[b] = (float) audio[b].n_samples / 48000.0f;

        // peak normalize to 0 dBFS (audio_encode_mp3 does not normalize)
        int   n_total = audio[b].n_samples * 2;
        float peak    = 0.0f;
        for (int i = 0; i < n_total; i++) {
            float a = audio[b].samples[i] < 0.0f ? -audio[b].samples[i] : audio[b].samples[i];
            if (a > peak) {
                peak = a;
            }
        }
        if (peak > 1e-8f && peak != 1.0f) {
            float gain = 1.0f / peak;
            for (int i = 0; i < n_total; i++) {
                audio[b].samples[i] *= gain;
            }
        }

        mp3s[b] = audio_encode_mp3(audio[b].samples, audio[b].n_samples, 48000, g_mp3_kbps, server_cancel,
                                   (void *) &req.is_connection_closed);
        ace_audio_free(&audio[b]);
    }
    queue_release();

    // single track: plain audio/mpeg (backward compatible)
    if (batch_n == 1) {
        if (mp3s[0].empty()) {
            json_error(res, 500, "MP3 encoding failed");
            return;
        }

        char val[64];
        snprintf(val, sizeof(val), "%lld", (long long) ace_req.seed);
        res.set_header("X-Seed", val);
        snprintf(val, sizeof(val), "%.2f", durations[0]);
        res.set_header("X-Duration", val);
        snprintf(val, sizeof(val), "%.0f", compute_ms);
        res.set_header("X-Compute-Ms", val);

        res.set_content(mp3s[0], "audio/mpeg");
        return;
    }

    // multiple tracks: multipart/mixed, each part is audio/mpeg
    // seed per track = base_seed + b (same as CLI philox_randn(seed+b))
    std::string boundary = "ace-batch-boundary";
    std::string body;

    for (int b = 0; b < batch_n; b++) {
        body += "--" + boundary + "\r\n";
        body += "Content-Type: audio/mpeg\r\n";

        char hdr[256];
        snprintf(hdr, sizeof(hdr), "X-Seed: %lld\r\n", (long long) (ace_req.seed + b));
        body += hdr;
        snprintf(hdr, sizeof(hdr), "X-Duration: %.2f\r\n", durations[b]);
        body += hdr;
        snprintf(hdr, sizeof(hdr), "X-Compute-Ms: %.0f\r\n", compute_ms);
        body += hdr;
        body += "\r\n";
        body += mp3s[b];
        body += "\r\n";
    }
    body += "--" + boundary + "--\r\n";

    res.set_content(body, "multipart/mixed; boundary=" + boundary);
}

// POST /understand
// Two modes:
//   multipart/form-data          -> full pipeline (audio + optional JSON params)
//     part "audio":   WAV or MP3 file (required)
//     part "request": JSON text (optional, for sampling params)
//   application/json body        -> codes-only (audio_codes in JSON, skip VAE+FSQ)
// returns: application/json AceRequest with metadata + lyrics + codes
static void handle_understand(const httplib::Request & req, httplib::Response & res) {
    if (!g_ctx_understand) {
        json_error(res, 501, "understand pipeline not loaded (requires --lm)");
        return;
    }
    if (!queue_acquire()) {
        json_busy(res);
        return;
    }

    // parse request: multipart (audio + optional JSON) or plain JSON (codes-only)
    AceRequest ace_req;
    request_init(&ace_req);
    ace_req.lm_temperature = 0.3f;  // understand default: lower than generation
    ace_req.lm_top_p       = 1.0f;  // understand default: no nucleus sampling

    float * src_interleaved = nullptr;
    int     src_len         = 0;

    if (req.is_multipart_form_data()) {
        // multipart: required "audio" part, optional "request" part for sampling params
        if (req.form.has_file("request")) {
            if (!request_parse_json(&ace_req, req.form.get_file("request").content.c_str())) {
                queue_release();
                json_error(res, 400, "multipart: invalid JSON in 'request' part");
                return;
            }
        } else if (req.form.has_field("request")) {
            if (!request_parse_json(&ace_req, req.form.get_field("request").c_str())) {
                queue_release();
                json_error(res, 400, "multipart: invalid JSON in 'request' part");
                return;
            }
        }

        if (!req.form.has_file("audio")) {
            queue_release();
            json_error(res, 400, "multipart: missing 'audio' part");
            return;
        }
        auto file = req.form.get_file("audio");
        if (file.content.empty()) {
            queue_release();
            json_error(res, 400, "multipart: empty 'audio' part");
            return;
        }

        // decode directly from multipart buffer (WAV/MP3 auto-detected)
        int     T_audio = 0;
        float * planar  = audio_read_48k_buf((const uint8_t *) file.content.data(), file.content.size(), &T_audio);
        if (!planar || T_audio <= 0) {
            queue_release();
            json_error(res, 400, "failed to decode audio");
            return;
        }

        fprintf(stderr, "[Server] Understand source: %.2fs @ 48kHz\n", (float) T_audio / 48000.0f);

        // convert planar [L:T][R:T] to interleaved [L0,R0,L1,R1,...] for pipeline
        src_interleaved = (float *) malloc((size_t) T_audio * 2 * sizeof(float));
        for (int t = 0; t < T_audio; t++) {
            src_interleaved[t * 2 + 0] = planar[t];
            src_interleaved[t * 2 + 1] = planar[T_audio + t];
        }
        free(planar);
        src_len = T_audio;
    } else {
        // plain JSON body: codes-only mode
        if (!request_parse_json(&ace_req, req.body.c_str())) {
            queue_release();
            json_error(res, 400, "invalid JSON");
            return;
        }
    }

    // always lock mtx_lm: understand shares the Qwen3 model with /lm,
    // so they must never touch the KV cache concurrently.
    AceRequest out;
    int        rc;
    {
        std::lock_guard<std::mutex> lock(mtx_lm);
        rc = ace_understand_generate(g_ctx_understand, src_interleaved, src_len, &ace_req, &out, server_cancel,
                                     (void *) &req.is_connection_closed);
    }
    free(src_interleaved);
    queue_release();

    if (rc != 0) {
        json_error(res, 500, "understand generation failed");
        return;
    }

    res.set_content(request_to_json(&out), "application/json");
}

// GET /health
static void handle_health(const httplib::Request &, httplib::Response & res) {
    int q = g_queue_n.load(std::memory_order_relaxed);

    // build pipelines list from what was actually loaded
    std::string pipes = "[";
    bool        first = true;
    if (g_ctx_lm) {
        pipes += "\"lm\"";
        first = false;
    }
    if (g_ctx_synth) {
        if (!first) {
            pipes += ",";
        }
        pipes += "\"synth\"";
        first = false;
    }
    if (g_ctx_understand) {
        if (!first) {
            pipes += ",";
        }
        pipes += "\"understand\"";
    }
    pipes += "]";

    char buf[512];
    snprintf(buf, sizeof(buf),
             "{\"status\":\"ok\""
             ",\"queue\":%d"
             ",\"max_queue\":%d"
             ",\"max_batch\":%d"
             ",\"pipelines\":%s}",
             q, g_max_queue, g_max_batch, pipes.c_str());
    res.set_content(buf, "application/json");
}

static void usage(const char * prog) {
    fprintf(stderr,
            "Usage: %s [options]\n"
            "\n"
            "LM (enables /lm + /understand):\n"
            "  --lm <gguf>             Qwen3 LM GGUF file\n"
            "  --max-seq <N>           KV cache size (default: 8192)\n"
            "\n"
            "Synth (enables /synth, all three required):\n"
            "  --embedding <gguf>      Text encoder GGUF file\n"
            "  --dit <gguf>            DiT GGUF file\n"
            "  --vae <gguf>            VAE GGUF file\n"
            "\n"
            "LoRA:\n"
            "  --lora <path>           LoRA safetensors file or directory\n"
            "  --lora-scale <float>    LoRA scaling factor (default: 1.0)\n"
            "\n"
            "VAE tiling (memory control):\n"
            "  --vae-chunk <N>         Latent frames per tile (default: 256)\n"
            "  --vae-overlap <N>       Overlap frames per side (default: 64)\n"
            "\n"
            "Output:\n"
            "  --mp3-bitrate <kbps>    MP3 bitrate (default: 128)\n"
            "\n"
            "Server:\n"
            "  --host <addr>           Listen address (default: 127.0.0.1)\n"
            "  --port <N>              Listen port (default: 8080)\n"
            "  --max-batch <N>         LM/synth batch limit (default: 1)\n"
            "  --max-queue <N>         Global queue depth (default: 4)\n"
            "\n"
            "Debug:\n"
            "  --no-fsm                Disable FSM constrained decoding\n"
            "  --no-fa                 Disable flash attention\n",
            prog);
}

int main(int argc, char ** argv) {
    AceLmParams lm_params;
    ace_lm_default_params(&lm_params);

    AceSynthParams synth_params;
    ace_synth_default_params(&synth_params);

    const char * host = "127.0.0.1";
    int          port = 8080;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        // LM
        if (!strcmp(argv[i], "--lm") && i + 1 < argc) {
            lm_params.model_path = argv[++i];
        } else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc) {
            lm_params.max_seq = atoi(argv[++i]);

            // synth models
        } else if (!strcmp(argv[i], "--embedding") && i + 1 < argc) {
            synth_params.text_encoder_path = argv[++i];
        } else if (!strcmp(argv[i], "--dit") && i + 1 < argc) {
            synth_params.dit_path = argv[++i];
        } else if (!strcmp(argv[i], "--vae") && i + 1 < argc) {
            synth_params.vae_path = argv[++i];

            // lora
        } else if (!strcmp(argv[i], "--lora") && i + 1 < argc) {
            synth_params.lora_path = argv[++i];
        } else if (!strcmp(argv[i], "--lora-scale") && i + 1 < argc) {
            synth_params.lora_scale = (float) atof(argv[++i]);

            // vae tiling
        } else if (!strcmp(argv[i], "--vae-chunk") && i + 1 < argc) {
            synth_params.vae_chunk = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--vae-overlap") && i + 1 < argc) {
            synth_params.vae_overlap = atoi(argv[++i]);

            // output
        } else if (!strcmp(argv[i], "--mp3-bitrate") && i + 1 < argc) {
            g_mp3_kbps = atoi(argv[++i]);

            // server
        } else if (!strcmp(argv[i], "--host") && i + 1 < argc) {
            host = argv[++i];
        } else if (!strcmp(argv[i], "--port") && i + 1 < argc) {
            port = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-batch") && i + 1 < argc) {
            g_max_batch = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--max-queue") && i + 1 < argc) {
            g_max_queue = atoi(argv[++i]);

            // debug
        } else if (!strcmp(argv[i], "--no-fsm")) {
            lm_params.use_fsm = false;
        } else if (!strcmp(argv[i], "--no-fa")) {
            lm_params.use_fa    = false;
            synth_params.use_fa = false;

        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    // at least --lm or the synth trio is required.
    // synth needs all three: --embedding + --dit + --vae.
    bool have_lm_group = (lm_params.model_path != NULL);

    bool have_any_synth   = (synth_params.text_encoder_path || synth_params.dit_path || synth_params.vae_path);
    bool have_synth_group = (synth_params.text_encoder_path && synth_params.dit_path && synth_params.vae_path);

    if (have_any_synth && !have_synth_group) {
        fprintf(stderr, "[Server] ERROR: incomplete synth options\n");
        fprintf(stderr, "  synth requires all three: --embedding --dit --vae\n");
        return 1;
    }
    if (!have_lm_group && !have_synth_group) {
        fprintf(stderr, "[Server] ERROR: no models provided\n");
        usage(argv[0]);
        return 1;
    }

    // clamp max_batch
    if (g_max_batch < 1) {
        g_max_batch = 1;
    }
    if (g_max_batch > 9) {
        g_max_batch = 9;
    }

    // load LM pipeline (optional)
    if (have_lm_group) {
        lm_params.max_batch = g_max_batch;
        fprintf(stderr, "[Server] Loading LM (max_batch=%d, max_seq=%d)...\n", g_max_batch, lm_params.max_seq);
        g_ctx_lm = ace_lm_load(&lm_params);
        if (!g_ctx_lm) {
            fprintf(stderr, "[Server] FATAL: LM load failed\n");
            return 1;
        }
    }

    // load synth pipeline (optional)
    if (have_synth_group) {
        fprintf(stderr, "[Server] Loading synth...\n");
        g_ctx_synth = ace_synth_load(&synth_params);
        if (!g_ctx_synth) {
            fprintf(stderr, "[Server] FATAL: synth load failed\n");
            ace_lm_free(g_ctx_lm);
            return 1;
        }
    }

    // load understand pipeline when LM is available.
    // shares the Qwen3 model from pipeline-lm to save ~5GB VRAM.
    // when synth is also loaded, understand gets DiT + VAE for
    // audio input mode. without synth, only codes-only mode works.
    if (g_ctx_lm) {
        AceUnderstandParams und_params;
        ace_understand_default_params(&und_params);
        und_params.shared_model = ace_lm_get_model(g_ctx_lm);
        und_params.shared_bpe   = ace_lm_get_bpe(g_ctx_lm);
        und_params.dit_path     = synth_params.dit_path;  // NULL when synth not loaded
        und_params.vae_path     = synth_params.vae_path;  // NULL when synth not loaded
        und_params.use_fa       = lm_params.use_fa;
        und_params.use_fsm      = lm_params.use_fsm;
        fprintf(stderr, "[Server] Loading understand%s...\n", have_synth_group ? " (audio + codes)" : " (codes-only)");
        g_ctx_understand = ace_understand_load(&und_params);
        if (!g_ctx_understand) {
            fprintf(stderr, "[Server] FATAL: understand load failed\n");
            ace_synth_free(g_ctx_synth);
            ace_lm_free(g_ctx_lm);
            return 1;
        }
    }

    // setup HTTP server
    httplib::Server svr;
    g_svr = &svr;

    // reject oversized bodies (120 MB: ~10min WAV stereo 48kHz 16-bit)
    svr.set_payload_max_length(120 * 1024 * 1024);

    // thread pool sized for queue depth + margin for /health
    int n_threads      = g_max_queue + 2;
    svr.new_task_queue = [n_threads]() {
        return new httplib::ThreadPool((size_t) n_threads);
    };

    // all endpoints are always registered. handlers return 501 when the
    // backing pipeline was not loaded.
    svr.Post("/lm", handle_lm);
    svr.Post("/synth", handle_synth);
    svr.Post("/understand", handle_understand);
    svr.Get("/health", handle_health);

    // graceful shutdown on SIGINT/SIGTERM
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    fprintf(stderr, "[Server] Listening on %s:%d\n", host, port);
    fprintf(stderr, "[Server] Pipelines:%s%s%s\n", g_ctx_lm ? " /lm" : "", g_ctx_synth ? " /synth" : "",
            g_ctx_understand ? " /understand" : "");
    fprintf(stderr, "[Server] max_batch=%d max_queue=%d mp3_kbps=%d\n", g_max_batch, g_max_queue, g_mp3_kbps);

    if (!svr.listen(host, port)) {
        fprintf(stderr, "[Server] FATAL: cannot bind %s:%d\n", host, port);
    }

    // cleanup (all _free functions handle NULL)
    fprintf(stderr, "\n[Server] Shutting down...\n");
    ace_understand_free(g_ctx_understand);
    ace_synth_free(g_ctx_synth);
    ace_lm_free(g_ctx_lm);
    fprintf(stderr, "[Server] Done\n");
    return 0;
}
