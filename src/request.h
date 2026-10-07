#pragma once
// request.h: AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

// One adapter of a DiT adapter stack: a name under --adapters and its scale.
struct AceAdapterRef {
    std::string name;
    float       scale = 1.0f;
};

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
    int64_t seed;              // -1 = random. DiT Philox noise consumes the low
                               // 32 bits. Stored in int64_t to land positive
                               // after rd(). Range [0, UINT32_MAX], matches the
                               // Python ACE-Step reference (random.randint(0, 2**32 - 1)).

    // LM control
    float       lm_temperature;      // 0.85
    float       lm_cfg_scale;        // 2.0
    float       lm_top_p;            // 0.9
    int         lm_top_k;            // 0 = disabled (matches Python None)
    std::string lm_negative_prompt;  // ""
    int64_t     lm_seed;             // -1 = random. mt19937 consumes the low 32
                                     // bits. Same int64_t storage trick as seed
                                     // above: rd() lands positive, no -1
                                     // collision, no mask needed.
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

    // Latent post-processing applied after DiT sampling, before VAE decode:
    //   pred = pred * latent_rescale + latent_shift
    // Defaults are no-op.
    float latent_shift;    // 0.0
    float latent_rescale;  // 1.0

    // Custom flow matching schedule: comma-separated floats,
    // e.g. "0.97,0.76,0.615,0.5,0.395,0.28,0.18,0.085,0". When non-empty,
    // overrides inference_steps and shift. The trailing endpoint is the x0
    // target dropped by the sampler, so a CSV of N values yields N-1 steps.
    std::string custom_timesteps;  // ""

    // task type: one of text2music, cover, cover-nofsq, repaint, lego, extract, complete.
    // Default: text2music.
    std::string task_type;  // "text2music"

    // track name for lego/extract/complete (e.g. "vocals", "drums", "guitar")
    std::string track;  // ""

    // Solver name resolved by solver_lookup() (see src/solvers).
    // Accepted values: "euler", "sde", "dpm3m", "stork4".
    std::string solver;          // "euler"
    int         stork_substeps;  // 10, only used by the "stork2" and "stork4" solvers

    // JKASS Fast momentum blending (see src/solvers/solver-jkass.h).
    float jkass_beat_stability;      // 0.25
    float jkass_frequency_damping;   // 0.4
    float jkass_temporal_smoothing;  // 0.13

    // Timestep spacing, resolved by scheduler_build() (see src/schedulers).
    // "linear" is the original shift schedule. Also "power:<p>",
    // "beta:<a>:<b>" and "composite:<A>+<B>:<crossover>:<split>".
    // custom_timesteps, when set, overrides it.
    std::string scheduler;  // "linear"

    // Guidance mode for CFG (see src/guidance.h): apg, cfg_pp, dynamic_cfg,
    // rescaled_cfg, cfg_zero_star, smc_cfg, cfg_mp. Only active when the
    // resolved guidance_scale is above 1.
    std::string guidance;             // "apg"
    float       apg_momentum;         // 0.75
    float       apg_norm_threshold;   // 2.5, 0 disables the clip
    int         cfg_zero_init_steps;  // 1
    float       smc_lambda;           // 0.5
    float       smc_k;                // 0.1
    int         cfg_mp_iterations;    // 1
    // CFG only while t_curr is inside [cfg_interval_start, cfg_interval_end],
    // the conditional prediction alone outside it.
    float       cfg_interval_start;  // 0.0
    float       cfg_interval_end;    // 1.0

    // Retake: blend the initial noise with a second draw,
    // noise = cos(v*pi/2) * noise(seed) + sin(v*pi/2) * noise(retake_seed).
    // 0 keeps the take, 1 is the take retake_seed would give. -1 derives the
    // retake seed from seed.
    int64_t retake_seed;      // -1
    float   retake_variance;  // 0.0

    // LM mode: "generate" (full: metadata + lyrics + codes),
    // "inspire" (short query -> metadata + lyrics, no codes),
    // "format" (caption + lyrics -> metadata + lyrics, no codes). Default: generate.
    std::string lm_mode;  // "generate"

    // Audio output format: "mp3", "wav16", "wav24", "wav32". Default: mp3.
    std::string output_format;  // "mp3"

    // model selection. synth_model, lm_model and vae are resolved through
    // the registry scanned from --models <dir>, by both the HTTP server and
    // the CLI binaries. An empty value falls to the first matching entry of
    // the registry. adapter and adapter_scale are read by server and
    // ace-synth and resolved against --adapters <dir> when set.
    std::string                synth_model;    // ""
    std::string                lm_model;       // ""
    std::string                adapter;        // ""
    float                      adapter_scale;  // 1.0
    // Planner LM adapter (LoRA, LoKr, DoRA), resolved against --adapters
    // <dir> like adapter, applied at runtime by /lm and ace-lm.
    std::string                lm_adapter;        // ""
    float                      lm_adapter_scale;  // 1.0
    // DiT adapter stack, merged in order. When non-empty it replaces
    // adapter / adapter_scale. JSON: [{"name": "...", "scale": 1.0}, ...]
    std::vector<AceAdapterRef> adapters;
    // Per group multipliers on every DiT adapter delta, all 1.0 by default.
    // JSON: {"self_attn", "cross_attn", "mlp", "cond_embed", "time_embed", "proj_in"}
    float                      adapter_group_self_attn;
    float                      adapter_group_cross_attn;
    float                      adapter_group_mlp;
    float                      adapter_group_cond_embed;
    float                      adapter_group_time_embed;
    float                      adapter_group_proj_in;
    std::string                vae;  // ""

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

// Resolve LM seed: if negative, replace with a hardware random value.
void request_resolve_lm_seed(AceRequest * r);
