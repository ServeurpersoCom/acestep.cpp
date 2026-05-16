#pragma once
// task-types.h: ACE-Step task types, instructions, and track names.
//
// All instruction strings and task identifiers live here.
// Pipeline order: LM -> DiT -> Understand.

#include <string>

// task type identifiers
inline constexpr const char * TASK_TEXT2MUSIC  = "text2music";
inline constexpr const char * TASK_COVER       = "cover";
inline constexpr const char * TASK_COVER_NOFSQ = "cover-nofsq";
inline constexpr const char * TASK_REPAINT     = "repaint";
inline constexpr const char * TASK_LEGO        = "lego";
inline constexpr const char * TASK_EXTRACT     = "extract";
inline constexpr const char * TASK_COMPLETE    = "complete";

// Solver identifiers (DiT diffusion sampler, see src/solvers).
// The string is the canonical solver name resolved by solver_lookup().
inline constexpr const char * SOLVER_EULER  = "euler";
inline constexpr const char * SOLVER_SDE    = "sde";
inline constexpr const char * SOLVER_DPM3M  = "dpm3m";
inline constexpr const char * SOLVER_STORK4 = "stork4";

// STORK4 sub stepping count (Chebyshev recurrence depth). Higher values
// stabilize stiff trajectories at the cost of arithmetic, halved on NaN.
inline constexpr int STORK_SUBSTEPS_DEFAULT = 10;

// DCW mode identifiers (Differential Correction in Wavelet domain).
// "low":    correct low-frequency sub-band only.
// "high":   correct high-frequency sub-band only.
// "double": correct both bands with independent scalers.
// "pix":    no wavelet transform, pixel/latent-space correction.
inline constexpr const char * DCW_MODE_LOW    = "low";
inline constexpr const char * DCW_MODE_HIGH   = "high";
inline constexpr const char * DCW_MODE_DOUBLE = "double";
inline constexpr const char * DCW_MODE_PIX    = "pix";

// LM generation mode names (JSON field lm_mode)
inline constexpr const char * LM_MODE_NAME_GENERATE = "generate";
inline constexpr const char * LM_MODE_NAME_INSPIRE  = "inspire";
inline constexpr const char * LM_MODE_NAME_FORMAT   = "format";

// Audio output format names (JSON field output_format)
inline constexpr const char * OUTPUT_FORMAT_MP3   = "mp3";
inline constexpr const char * OUTPUT_FORMAT_WAV16 = "wav16";
inline constexpr const char * OUTPUT_FORMAT_WAV24 = "wav24";
inline constexpr const char * OUTPUT_FORMAT_WAV32 = "wav32";

// LM system instruction (Composer Agent: text -> metadata + audio codes)
inline constexpr const char * LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:";

// LM inspire instruction (short query -> metadata + lyrics, no codes)
inline constexpr const char * LM_INSPIRE_INSTRUCTION =
    "Expand the user's input into a more detailed and specific musical description:";

// LM format instruction (caption + lyrics -> metadata + lyrics, no codes)
inline constexpr const char * LM_FORMAT_INSTRUCTION =
    "Format the user's input into a more detailed and specific musical description:";

// Understand system instruction (Listener: audio codes -> metadata + lyrics)
inline constexpr const char * LM_UNDERSTAND_INSTRUCTION =
    "Understand the given musical conditions and describe the audio semantics accordingly:";

// LM generation modes
#define LM_MODE_GENERATE 0  // full: metadata + lyrics + audio codes
#define LM_MODE_INSPIRE  1  // inspire: metadata + lyrics only (no codes)
#define LM_MODE_FORMAT   2  // format: metadata + lyrics only (no codes)

// DiT instructions (injected into text encoder cross-attention).
// Fixed instructions for tasks without track name.
inline constexpr const char * DIT_INSTR_TEXT2MUSIC = "Fill the audio semantic mask based on the given conditions:";
inline constexpr const char * DIT_INSTR_COVER      = "Generate audio semantic tokens based on the given conditions:";
inline constexpr const char * DIT_INSTR_REPAINT    = "Repaint the mask area based on the given conditions:";

// DiT instructions with track name (UPPERCASE). Inline helpers return std::string.
static inline std::string dit_instr_lego(const std::string & track) {
    if (track.empty()) {
        return "Generate the track based on the audio context:";
    }
    return "Generate the " + track + " track based on the audio context:";
}

static inline std::string dit_instr_extract(const std::string & track) {
    if (track.empty()) {
        return "Extract the track from the audio:";
    }
    return "Extract the " + track + " track from the audio:";
}

static inline std::string dit_instr_complete(const std::string & track) {
    if (track.empty()) {
        return "Complete the input track:";
    }
    return "Complete the input track with " + track + ":";
}

// valid track names for lego/extract/complete
inline constexpr const char * TRACK_NAMES[] = {
    "vocals",     "backing_vocals", "drums", "bass", "guitar", "keyboard",
    "percussion", "strings",        "synth", "fx",   "brass",  "woodwinds",
};
inline constexpr int TRACK_NAMES_COUNT = 12;

// validate track names in a " | " separated string (complete supports multi-track).
// warns per invalid name; does nothing when track is empty.
static inline void validate_track_names(const std::string & track, const char * label) {
    if (track.empty()) {
        return;
    }
    size_t pos = 0;
    while (pos < track.size()) {
        size_t      sep  = track.find(" | ", pos);
        size_t      len  = (sep == std::string::npos) ? std::string::npos : sep - pos;
        std::string name = track.substr(pos, len);
        pos              = (sep == std::string::npos) ? track.size() : sep + 3;

        bool valid = false;
        for (int k = 0; k < TRACK_NAMES_COUNT; k++) {
            if (name == TRACK_NAMES[k]) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            fprintf(stderr, "[%s] WARNING: '%s' is not a standard track name\n", label, name.c_str());
        }
    }
}
