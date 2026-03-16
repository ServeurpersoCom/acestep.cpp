#pragma once
// prompt.h: ACE-Step prompt building and CoT parsing
//
// AcePrompt struct, Qwen3 chat template formatting,
// CoT (Chain-of-Thought) metadata extraction, YAML builders.

#include "bpe.h"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Qwen3 special token IDs (ACE-Step LM vocabulary)
#define TOKEN_IM_START   151644
#define TOKEN_IM_END     151645
#define TOKEN_THINK      151667
#define TOKEN_THINK_END  151668
#define AUDIO_CODE_BASE  151669
#define AUDIO_CODE_COUNT 65535

// LM system instruction (same for all 4 prompt variants, the LM always generates audio tokens)
static const char * LM_INSTRUCTION = "Generate audio semantic tokens based on the given conditions:";

// ACE-Step prompt
struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float       duration;
    int         bpm;
    std::string keyscale;
    std::string timesignature;
    std::string vocal_language;
};

// CoT parsing (extract metadata + lyrics from LLM Phase1 output)
static bool parse_cot_and_lyrics(const std::string & text, AcePrompt * out) {
    // Extract CoT content between <think>...</think>
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");

    std::string cot;
    std::string lyrics_after;

    if (ts != std::string::npos && te != std::string::npos) {
        cot          = text.substr(ts + 7, te - ts - 7);
        lyrics_after = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot          = text.substr(0, te);
        lyrics_after = text.substr(te + 8);
    } else {
        cot = text;
    }

    // Parse YAML-like fields from CoT
    auto get_field = [&](const std::string & key) -> std::string {
        std::string needle = key + ":";
        size_t      p      = cot.find(needle);
        if (p == std::string::npos) {
            return "";
        }
        p += needle.size();
        while (p < cot.size() && (cot[p] == ' ' || cot[p] == '\'')) {
            p++;
        }
        size_t end = cot.find('\n', p);
        if (end == std::string::npos) {
            end = cot.size();
        }
        std::string val = cot.substr(p, end - p);
        // Strip trailing whitespace and quotes
        while (!val.empty() && (val.back() == ' ' || val.back() == '\'' || val.back() == '\r')) {
            val.pop_back();
        }
        return val;
    };

    std::string bpm_s = get_field("bpm");
    if (!bpm_s.empty()) {
        out->bpm = atoi(bpm_s.c_str());
    }

    std::string dur_s = get_field("duration");
    if (!dur_s.empty()) {
        out->duration = (float) atof(dur_s.c_str());
    }

    std::string ks = get_field("keyscale");
    if (!ks.empty()) {
        out->keyscale = ks;
    }

    std::string ts_s = get_field("timesignature");
    if (!ts_s.empty()) {
        out->timesignature = ts_s;
    }

    std::string lang = get_field("language");
    if (!lang.empty()) {
        out->vocal_language = lang;
    }

    std::string cap = get_field("caption");
    if (!cap.empty()) {
        // Caption may span multiple lines (YAML word-wrap), read until next field
        size_t cp = cot.find("caption:");
        if (cp != std::string::npos) {
            cp += 8;
            size_t end = cot.find("\nduration:", cp);
            if (end == std::string::npos) {
                end = cot.find("\nkeyscale:", cp);
            }
            if (end == std::string::npos) {
                end = cot.size();
            }
            std::string full_cap = cot.substr(cp, end - cp);
            // Trim and collapse whitespace
            std::string cleaned;
            bool        in_space = true;
            for (char ch : full_cap) {
                if (ch == '\n' || ch == '\r') {
                    ch = ' ';
                }
                if (ch == ' ') {
                    if (!in_space) {
                        cleaned += ' ';
                    }
                    in_space = true;
                } else {
                    cleaned += ch;
                    in_space = false;
                }
            }
            while (!cleaned.empty() && cleaned.back() == ' ') {
                cleaned.pop_back();
            }
            while (!cleaned.empty() && cleaned.front() == ' ') {
                cleaned.erase(cleaned.begin());
            }
            if (!cleaned.empty()) {
                out->caption = cleaned;
            }
        }
    }

    // Lyrics after </think>
    if (!lyrics_after.empty()) {
        // Trim leading whitespace
        size_t s = lyrics_after.find_first_not_of(" \t\n\r");
        if (s != std::string::npos) {
            lyrics_after = lyrics_after.substr(s);
        }
        // Strip "# Lyric\n" header the LM may echo back
        // (may be preceded by "# Languages\n...\n\n")
        size_t lp = lyrics_after.find("# Lyric\n");
        if (lp != std::string::npos && lp < 64) {
            lyrics_after = lyrics_after.substr(lp + 8);
        }
        // Trim trailing whitespace
        while (!lyrics_after.empty() &&
               (lyrics_after.back() == ' ' || lyrics_after.back() == '\n' || lyrics_after.back() == '\r')) {
            lyrics_after.pop_back();
        }
        if (!lyrics_after.empty()) {
            out->lyrics = lyrics_after;
        }
    }

    return (out->bpm > 0 || out->duration > 0);
}

// Prompt building (Qwen3 chat template)
static std::vector<int> build_lm_prompt(BPETokenizer & bpe, const AcePrompt & prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

static std::vector<int> build_lm_prompt_uncond(BPETokenizer &    bpe,
                                               const AcePrompt & prompt,
                                               const char *      negative_prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0;
    if (has_neg) {
        append("user\n# Caption\n" + std::string(negative_prompt) + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    } else {
        append("user\n# Lyric\n" + prompt.lyrics + "\n");
    }
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

// Build CoT YAML content (matching Python yaml.dump sort_keys=True)
static std::string build_cot_yaml(const AcePrompt & prompt) {
    // Matches Python yaml.dump(allow_unicode=True, sort_keys=True) wrapping:
    // line break + 2-space indent when current column exceeds 80.
    auto yaml_wrap = [](const std::string & key, const std::string & val) -> std::string {
        std::string result = key + ":";
        int         col    = (int) (key.size() + 1);
        size_t      i      = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) {
                end = val.size();
            }
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int) word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (prompt.bpm > 0) {
        yaml += "bpm: " + std::to_string(prompt.bpm) + "\n";
    }
    if (!prompt.caption.empty()) {
        yaml += yaml_wrap("caption", prompt.caption);
    }
    if (prompt.duration > 0) {
        yaml += "duration: " + std::to_string((int) prompt.duration) + "\n";
    }
    if (!prompt.keyscale.empty()) {
        yaml += "keyscale: " + prompt.keyscale + "\n";
    }
    if (!prompt.vocal_language.empty()) {
        yaml += "language: " + prompt.vocal_language + "\n";
    }
    if (!prompt.timesignature.empty()) {
        yaml += "timesignature: " + prompt.timesignature + "\n";
    }
    return yaml;
}

// Prompt with injected CoT (Phase 2: all metas known)
static std::vector<int> build_lm_prompt_with_cot(BPETokenizer &      bpe,
                                                 const AcePrompt &   prompt,
                                                 const std::string & cot_yaml) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n# Caption\n" + prompt.caption + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n" + cot_yaml);
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    return ids;
}

// Unconditional prompt with empty CoT for CFG (Phase 2)
static std::vector<int> build_lm_prompt_uncond_with_cot(BPETokenizer &    bpe,
                                                        const AcePrompt & prompt,
                                                        const char *      negative_prompt) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    bool        has_neg = negative_prompt && strlen(negative_prompt) > 0;
    std::string cap     = has_neg ? std::string(negative_prompt) : prompt.caption;
    append("user\n# Caption\n" + cap + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    ids.push_back(TOKEN_THINK);
    append("\n\n");
    ids.push_back(TOKEN_THINK_END);
    append("\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    return ids;
}

// Build Qwen3 chat prompt: <|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...<|im_end|>\n<|im_start|>assistant\n
static std::vector<int> build_custom_prompt(BPETokenizer & bpe, const char * sys, const char * user) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n" + std::string(sys) + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n" + std::string(user) + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

// Understand system instruction (reverse: audio codes -> metadata + lyrics)
static const char * LM_UNDERSTAND_INSTRUCTION =
    "Understand the given musical conditions and describe the audio semantics accordingly:";

// Build understand prompt: system instruction + raw audio code tokens as user input.
// codes[] are FSQ indices (from tok_ggml_encode or parsed from audio_codes string).
// They become token IDs AUDIO_CODE_BASE + codes[i] in the user turn.
static std::vector<int> build_understand_prompt(BPETokenizer & bpe, const int * codes, int n_codes) {
    std::vector<int> ids;
    auto             append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append(std::string("system\n# Instruction\n") + LM_UNDERSTAND_INSTRUCTION + "\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("user\n");
    for (int i = 0; i < n_codes; i++) {
        ids.push_back(AUDIO_CODE_BASE + codes[i]);
    }
    append("\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}
