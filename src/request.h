#pragma once
// request.h: AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct AceRequest {
    // text content
    std::string caption;  // ""
    std::string lyrics;   // ""

    // metadata (user-provided or LLM-enriched)
    int         bpm;             // 0 = unset
    float       duration;        // 0 = unset
    std::string keyscale;        // "" = unset
    std::string timesignature;   // "" = unset
    std::string vocal_language;  // "" = unset

    // generation
    int     lm_batch_size;     // 1 (number of LLM variations)
    int     synth_batch_size;  // 1 (synth batch: number of DiT variations per request)
    int64_t seed;              // -1 = random (DiT Philox noise)

    // LM control
    float       lm_temperature;      // 0.85
    float       lm_cfg_scale;        // 2.0
    float       lm_top_p;            // 0.9
    int         lm_top_k;            // 0 = disabled (matches Python None)
    std::string lm_negative_prompt;  // ""
    bool        use_cot_caption;     // true = LLM enriches caption via CoT

    // codes (Python-compatible string: "3101,11837,27514,...")
    // empty = text2music (silence context), non-empty = cover mode
    std::string audio_codes;  // ""

    // DiT control (0 = auto-detect from model: turbo vs base/sft)
    int   inference_steps;  // 0 = auto (turbo: 8, base/sft: 50)
    float guidance_scale;   // 0 = auto (1.0 for all models)
    float shift;            // 0 = auto (turbo: 3.0, base/sft: 1.0)

    // Differential Correction in Wavelet domain (CVPR 2026, arXiv:2604.16044).
    // Sampler-side correction for SNR-t bias in flow matching.
    // dcw_mode = "low"|"high"|"double"|"pix". dcw_scaler applies to the low
    // band in "low" and "double", to all bands in "high" and "pix". In
    // "double", dcw_high_scaler is the independent scaler for the high
    // band. Both scalers are modulated by t_curr. 0.0 disables (bit-perfect
    // master). Paper-recommended starting value: 0.1.
    float       dcw_scaler;       // 0.0 (disabled)
    float       dcw_high_scaler;  // 0.0 (only read in mode "double")
    std::string dcw_mode;         // "low"

    // cover mode (active when source audio is provided)
    float audio_cover_strength;  // 1.0 (0-1, fraction of DiT steps using source context)
    float cover_noise_strength;  // 0.0 (0-1, how close to source: 0=pure noise, 1=source)

    // repaint region (requires source audio)
    // start: seconds offset. 0 = source start. Negative = outpaint before source.
    // end: seconds offset. Negative = source duration (sentinel).
    //      Values beyond source duration outpaint after source.
    float repainting_start;  // 0
    float repainting_end;    // -1

    // task type: one of text2music, cover, cover-nofsq, repaint, lego, extract, complete.
    // Default: text2music.
    std::string task_type;  // "text2music"

    // track name for lego/extract/complete (e.g. "vocals", "drums", "guitar")
    std::string track;  // ""

    // inference method: "ode" = ODE Euler, "sde" = SDE Stochastic. Default: ode.
    std::string infer_method;  // "ode"

    // LM mode: "generate" (full: metadata + lyrics + codes),
    // "inspire" (short query -> metadata + lyrics, no codes),
    // "format" (caption + lyrics -> metadata + lyrics, no codes). Default: generate.
    std::string lm_mode;  // "generate"

    // Audio output format: "mp3", "wav16", "wav24", "wav32". Default: mp3.
    std::string output_format;  // "mp3"

    // model selection. synth_model and lm_model are resolved through the
    // registry scanned from --models <dir>, by both the HTTP server and the
    // CLI binaries. An empty value falls to the first matching entry of the
    // registry. adapter and adapter_scale are read by server and ace-synth
    // and resolved against --adapters <dir> when set.
    std::string synth_model;    // ""
    std::string lm_model;       // ""
    std::string adapter;        // ""
    float       adapter_scale;  // 1.0

    // audio output: peak clip via percentile normalization.
    // 0 = peak normalization (100.0000th percentile, no clipping).
    // 10 = default (99.9990th percentile, clips top 0.001%).
    // 999 = max (99.9001th percentile, clips top 0.1%).
    int peak_clip;  // 10

    // audio output: MP3 encoder bitrate in kbps. Applies only when
    // output_format is "mp3". WAV outputs ignore this field.
    int mp3_bitrate;  // 128
};

// Initialize all fields to defaults (matches Python GenerationParams defaults)
void request_init(AceRequest * r);

// Parse JSON file into struct. Missing fields keep their defaults.
// Returns false on file error or malformed JSON.
bool request_parse(AceRequest * r, const char * path);

// Parse JSON string into struct. Missing fields keep their defaults.
// Returns false on malformed JSON.
bool request_parse_json(AceRequest * r, const char * json);

// Write struct to JSON file (overwrites). Returns false on file error.
bool request_write(const AceRequest * r, const char * path);

// Serialize struct to JSON string.
// sparse=true: omit fields at their default value (for cards and exports).
// sparse=false: serialize all fields (for /props documentation).
std::string request_to_json(const AceRequest * r, bool sparse = true);

// Parse JSON: single object {} or array [{}, ...] into a vector.
// Returns false on malformed JSON or empty result.
bool request_parse_json_array(const char * json, std::vector<AceRequest> * out);

// Dump human-readable summary to stream (debug)
void request_dump(const AceRequest * r, FILE * f);

// Resolve seed: if negative, replace with a hardware random value.
void request_resolve_seed(AceRequest * r);
