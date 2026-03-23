#pragma once
//
// request.h - AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.
//

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
    int     batch_size;  // 1 (number of variations, matches Python GenerationConfig.batch_size)
    int64_t seed;        // -1 = random

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

    // cover mode (active when --src-audio is provided on CLI)
    float audio_cover_strength;  // 0.5 (0-1, fraction of DiT steps using source context)

    // repaint mode (requires --src-audio)
    // Both -1 = no repaint (plain cover). One or both >= 0 activates repaint.
    // -1 on start means 0s, -1 on end means source duration.
    float repainting_start;  // -1
    float repainting_end;    // -1

    // lego mode (requires --src-audio, base model only)
    // Track name from TRACK_NAMES: vocals, backing_vocals, drums, bass, guitar,
    // keyboard, percussion, strings, synth, fx, brass, woodwinds.
    // Empty = not lego. Sets instruction, cover path, strength=1.0.
    std::string lego;  // ""
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
std::string request_to_json(const AceRequest * r);

// Parse JSON: single object {} or array [{}, ...] into a vector.
// Returns false on malformed JSON or empty result.
bool request_parse_json_array(const char * json, std::vector<AceRequest> * out);

// Dump human-readable summary to stream (debug)
void request_dump(const AceRequest * r, FILE * f);
