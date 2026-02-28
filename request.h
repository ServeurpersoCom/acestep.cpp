#pragma once
//
// request.h - AceStep generation request (JSON serialization)
//
// Pure data container + JSON read/write. Zero business logic.
// Aligned with Python GenerationParams and ACE-Step 1.5 Tutorial:
// https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/Tutorial.md
//

#include <string>
#include <cstdio>

struct AceRequest {
    // --- Task & audio inputs (Tutorial: Input Control) ---
    // task_type: "text2music" | "cover" | "repaint" | "lego" | "extract" | "complete"
    // (lego/extract/complete require Base DiT; we only have Turbo/SFT)
    std::string task_type;         // "text2music"

    // text content
    std::string caption;            // ""
    std::string lyrics;             // ""
    bool        instrumental;       // false

    // metadata (user-provided or LLM-enriched)
    int         bpm;                // 0 = unset
    float       duration;           // -1 = unset
    std::string keyscale;           // "" = unset
    std::string timesignature;      // "" = unset
    std::string vocal_language;     // "unknown"

    // generation
    int64_t     seed;               // -1 = random

    // LM control
    float       lm_temperature;     // 0.85
    float       lm_cfg_scale;       // 2.0
    float       lm_top_p;           // 0.9
    int         lm_top_k;           // 0 = disabled (matches Python None)
    std::string lm_negative_prompt; // ""

    // Audio control (Tutorial: reference_audio, src_audio, audio_codes)
    // reference_audio: path to WAV for global timbre/style (VAE encode → CondEncoder timbre)
    std::string reference_audio;   // ""
    // src_audio: path to WAV for Cover (encode → codes) or Repaint context
    std::string src_audio;          // ""
    // codes (Python-compatible string: "3101,11837,27514,...")
    // empty = text2music (silence context), non-empty = cover mode
    std::string audio_codes;        // ""
    // 0.0–1.0: how strongly generation follows reference/codes (Tutorial: audio_cover_strength)
    float       audio_cover_strength; // 1.0

    // Repaint interval (seconds). Only used when task_type == "repaint".
    float       repainting_start;   // 0.0
    float       repainting_end;     // 0.0

    // DiT control (Python: inference_steps, guidance_scale, shift)
    int         inference_steps;    // 8
    float       guidance_scale;     // 7.0
    float       shift;              // 1.0
};

// Initialize all fields to defaults (matches Python GenerationParams defaults)
void request_init(AceRequest * r);

// Parse JSON file into struct. Missing fields keep their defaults.
// Returns false on file error or malformed JSON.
bool request_parse(AceRequest * r, const char * path);

// Write struct to JSON file (overwrites). Returns false on file error.
bool request_write(const AceRequest * r, const char * path);

// Dump human-readable summary to stream (debug)
void request_dump(const AceRequest * r, FILE * f);
