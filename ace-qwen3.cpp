// ace-qwen3.cpp : ACE-Step 5Hz LM inference (GGML)
// Qwen3 causal LM: CoT reasoning + audio code generation
// ace-qwen3: Qwen3 causal LM for ACE-Step music generation (GGML backend)
#include "qwen3-lm.h"
#include "bpe.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <sstream>
#include <map>
#include <unordered_map>

// Timer
struct Timer {
    std::chrono::steady_clock::time_point t;
    Timer() : t(std::chrono::steady_clock::now()) {}
    double ms() const {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t).count();
    }
};

// Special token IDs (Qwen3 extended vocab)
#define TOKEN_IM_START      151644
#define TOKEN_IM_END        151645
#define TOKEN_THINK         151667
#define TOKEN_THINK_END     151668
#define AUDIO_CODE_BASE     151669
#define AUDIO_CODE_COUNT    65535

//
// Sampling
//

static std::mt19937 g_rng(42);

struct TokenProb {
    int id;
    float prob;
};

static int sample_top_p(float * logits, int vocab_size, float temperature, float top_p) {
    for (int i = 0; i < vocab_size; i++)
        logits[i] /= temperature;

    float max_val = *std::max_element(logits, logits + vocab_size);
    float sum = 0.0f;
    for (int i = 0; i < vocab_size; i++) {
        logits[i] = expf(logits[i] - max_val);
        sum += logits[i];
    }
    float inv_sum = 1.0f / sum;
    for (int i = 0; i < vocab_size; i++)
        logits[i] *= inv_sum;

    static std::vector<TokenProb> candidates;
    candidates.clear();

    float threshold = 1.0f / (float)vocab_size * 0.01f;
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > threshold)
            candidates.push_back({i, logits[i]});
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const TokenProb & a, const TokenProb & b) { return a.prob > b.prob; });

    float cum = 0.0f;
    int n_keep = 0;
    for (size_t i = 0; i < candidates.size(); i++) {
        cum += candidates[i].prob;
        n_keep = (int)i + 1;
        if (cum >= top_p) break;
    }

    float renorm_sum = 0.0f;
    for (int i = 0; i < n_keep; i++)
        renorm_sum += candidates[i].prob;

    std::uniform_real_distribution<float> dist(0.0f, renorm_sum);
    float r = dist(g_rng);
    float acc = 0.0f;
    for (int i = 0; i < n_keep; i++) {
        acc += candidates[i].prob;
        if (acc >= r) return candidates[i].id;
    }
    return candidates[0].id;
}

//
// BPE decode (token IDs -> text)
//

static std::string bpe_decode(const BPETokenizer & bpe, const std::vector<int> & ids) {
    static std::unordered_map<int, uint8_t> byte_dec;
    static bool init = false;
    if (!init) {
        for (int b = 0; b < 256; b++) {
            int adv;
            int cp = utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t)b;
        }
        init = true;
    }

    std::string result;
    for (int id : ids) {
        if (id == TOKEN_THINK)     { result += "<think>";  continue; }
        if (id == TOKEN_THINK_END) { result += "</think>"; continue; }
        if (id == TOKEN_IM_START || id == TOKEN_IM_END) continue;
        if (id >= AUDIO_CODE_BASE) continue;
        if (id < 0 || id >= (int)bpe.id_to_str.size()) continue;
        const std::string & s = bpe.id_to_str[id];
        if (s.empty()) continue;
        const char * p = s.c_str();
        while (*p) {
            int adv;
            int cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) result += (char)it->second;
            p += adv;
        }
    }
    return result;
}

//
// ACE-Step prompt
//

struct AcePrompt {
    std::string caption;
    std::string lyrics;
    float duration;
    int bpm;
    std::string keyscale;
    std::string timesignature;
    std::string vocal_language;
};

//
// CoT parsing (extract metadata + lyrics from LLM Phase1 output)
//

static bool parse_cot_and_lyrics(const std::string & text, AcePrompt * out) {
    // Extract CoT content between <think>...</think>
    size_t ts = text.find("<think>");
    size_t te = text.find("</think>");

    std::string cot;
    std::string lyrics_after;

    if (ts != std::string::npos && te != std::string::npos) {
        cot = text.substr(ts + 7, te - ts - 7);
        lyrics_after = text.substr(te + 8);
    } else if (te != std::string::npos) {
        cot = text.substr(0, te);
        lyrics_after = text.substr(te + 8);
    } else {
        cot = text;
    }

    // Parse YAML-like fields from CoT
    auto get_field = [&](const std::string & key) -> std::string {
        std::string needle = key + ":";
        size_t p = cot.find(needle);
        if (p == std::string::npos) return "";
        p += needle.size();
        while (p < cot.size() && (cot[p] == ' ' || cot[p] == '\'')) p++;
        size_t end = cot.find('\n', p);
        if (end == std::string::npos) end = cot.size();
        std::string val = cot.substr(p, end - p);
        // Strip trailing whitespace and quotes
        while (!val.empty() && (val.back() == ' ' || val.back() == '\'' || val.back() == '\r'))
            val.pop_back();
        return val;
    };

    std::string bpm_s = get_field("bpm");
    if (!bpm_s.empty()) out->bpm = atoi(bpm_s.c_str());

    std::string dur_s = get_field("duration");
    if (!dur_s.empty()) out->duration = (float)atof(dur_s.c_str());

    std::string ks = get_field("keyscale");
    if (!ks.empty()) out->keyscale = ks;

    std::string ts_s = get_field("timesignature");
    if (!ts_s.empty()) out->timesignature = ts_s;

    std::string lang = get_field("language");
    if (!lang.empty()) out->vocal_language = lang;

    std::string cap = get_field("caption");
    if (!cap.empty()) {
        // Caption may span multiple lines (yaml word-wrap)
        size_t cp = cot.find("caption:");
        if (cp != std::string::npos) {
            cp += 8;
            size_t end = cot.find("\nduration:", cp);
            if (end == std::string::npos) end = cot.find("\nkeyscale:", cp);
            if (end == std::string::npos) end = cot.size();
            std::string full_cap = cot.substr(cp, end - cp);
            // Trim and collapse whitespace
            std::string cleaned;
            bool in_space = true;
            for (char ch : full_cap) {
                if (ch == '\n' || ch == '\r') ch = ' ';
                if (ch == ' ') {
                    if (!in_space) cleaned += ' ';
                    in_space = true;
                } else {
                    cleaned += ch;
                    in_space = false;
                }
            }
            while (!cleaned.empty() && cleaned.back() == ' ') cleaned.pop_back();
            while (!cleaned.empty() && cleaned.front() == ' ') cleaned.erase(cleaned.begin());
            if (!cleaned.empty()) out->caption = cleaned;
        }
    }

    // Lyrics after </think>
    if (!lyrics_after.empty()) {
        // Trim leading whitespace
        size_t s = lyrics_after.find_first_not_of(" \t\n\r");
        if (s != std::string::npos)
            lyrics_after = lyrics_after.substr(s);
        // Trim trailing whitespace
        while (!lyrics_after.empty() &&
               (lyrics_after.back() == ' ' || lyrics_after.back() == '\n' || lyrics_after.back() == '\r'))
            lyrics_after.pop_back();
        if (!lyrics_after.empty())
            out->lyrics = lyrics_after;
    }

    return (out->bpm > 0 || out->duration > 0);
}

//
// Prompt building (Qwen3 chat template)
//

static std::vector<int> build_lm_prompt(BPETokenizer & bpe, const AcePrompt & prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
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

static std::vector<int> build_lm_prompt_uncond(BPETokenizer & bpe, const AcePrompt & prompt,
                                                const char * negative_prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    if (has_neg)
        append("user\n# Caption\n" + std::string(negative_prompt) + "\n\n# Lyric\n" + prompt.lyrics + "\n");
    else
        append("user\n# Lyric\n" + prompt.lyrics + "\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    append("assistant\n");
    return ids;
}

// Build CoT YAML content (matching Python yaml.dump sort_keys=True)
static std::string build_cot_yaml(const AcePrompt & prompt) {
    auto yaml_wrap = [](const std::string & key, const std::string & val) -> std::string {
        std::string result = key + ":";
        int col = (int)(key.size() + 1);
        size_t i = 0;
        while (i < val.size()) {
            size_t end = val.find(' ', i);
            if (end == std::string::npos) end = val.size();
            std::string word = val.substr(i, end - i);
            if (col > 80) {
                result += "\n  ";
                col = 2;
            } else {
                result += " ";
                col += 1;
            }
            result += word;
            col += (int)word.size();
            i = (end < val.size()) ? end + 1 : val.size();
        }
        result += "\n";
        return result;
    };

    std::string yaml;
    if (prompt.bpm > 0)
        yaml += "bpm: " + std::to_string(prompt.bpm) + "\n";
    if (!prompt.caption.empty())
        yaml += yaml_wrap("caption", prompt.caption);
    if (prompt.duration > 0)
        yaml += "duration: " + std::to_string((int)prompt.duration) + "\n";
    if (!prompt.keyscale.empty())
        yaml += "keyscale: " + prompt.keyscale + "\n";
    if (!prompt.vocal_language.empty())
        yaml += "language: " + prompt.vocal_language + "\n";
    if (!prompt.timesignature.empty())
        yaml += "timesignature: " + prompt.timesignature + "\n";
    return yaml;
}

// Prompt with injected CoT (Phase 2: all metas known)
static std::vector<int> build_lm_prompt_with_cot(BPETokenizer & bpe, const AcePrompt & prompt,
                                                   const std::string & cot_yaml) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
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
static std::vector<int> build_lm_prompt_uncond_with_cot(BPETokenizer & bpe, const AcePrompt & prompt,
                                                          const char * negative_prompt) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
        auto t = bpe_encode(&bpe, text, false);
        ids.insert(ids.end(), t.begin(), t.end());
    };
    ids.push_back(TOKEN_IM_START);
    append("system\n# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    ids.push_back(TOKEN_IM_END);
    append("\n");
    ids.push_back(TOKEN_IM_START);
    bool has_neg = negative_prompt && strlen(negative_prompt) > 0
                   && strcmp(negative_prompt, "NO USER INPUT") != 0;
    std::string cap = has_neg ? std::string(negative_prompt) : prompt.caption;
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

// Custom prompt (raw system + user)
static std::vector<int> build_custom_prompt(BPETokenizer & bpe, const char * sys, const char * user) {
    std::vector<int> ids;
    auto append = [&](const std::string & text) {
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

//
// Prefix tree for FSM constrained decoding
//

struct PrefixTree {
    // Maps prefix (token sequence) to set of valid next tokens
    std::map<std::vector<int>, std::vector<int>> nodes;

    void add(const std::vector<int> & seq) {
        for (size_t i = 0; i < seq.size(); i++) {
            std::vector<int> prefix(seq.begin(), seq.begin() + i);
            int next = seq[i];
            auto & vec = nodes[prefix];
            if (std::find(vec.begin(), vec.end(), next) == vec.end())
                vec.push_back(next);
        }
    }

    const std::vector<int> * get(const std::vector<int> & prefix) const {
        auto it = nodes.find(prefix);
        return it != nodes.end() ? &it->second : nullptr;
    }
};

//
// Metadata FSM (constrained decoding for CoT fields)
//

struct MetadataFSM {
    enum State {
        BPM_NAME, BPM_VALUE,
        CAPTION_NAME, CAPTION_VALUE,
        DURATION_NAME, DURATION_VALUE,
        KEYSCALE_NAME, KEYSCALE_VALUE,
        LANGUAGE_NAME, LANGUAGE_VALUE,
        TIMESIG_NAME, TIMESIG_VALUE,
        THINK_END,
        CODES,
        DISABLED
    };

    State state = DISABLED;
    int name_pos = 0;
    std::vector<int> value_acc;
    bool enabled = false;

    std::vector<int> bpm_name, caption_name, duration_name;
    std::vector<int> keyscale_name, language_name, timesig_name;
    PrefixTree bpm_tree, duration_tree, keyscale_tree, language_tree, timesig_tree;
    int newline_tok = -1;
    int think_end_tok = TOKEN_THINK_END;
    int vocab_size = 0;

    static std::vector<int> tokenize_strip(BPETokenizer & bpe,
                                            const std::string & full,
                                            const std::string & prefix) {
        std::vector<int> full_tok = bpe_encode(&bpe, full, false);
        std::vector<int> pre_tok  = bpe_encode(&bpe, prefix, false);
        if (full_tok.size() >= pre_tok.size() &&
            std::equal(pre_tok.begin(), pre_tok.end(), full_tok.begin()))
            return std::vector<int>(full_tok.begin() + pre_tok.size(), full_tok.end());
        return full_tok;
    }

    void build_value_tree(BPETokenizer & bpe, PrefixTree & tree,
                          const std::string & field_prefix,
                          const std::vector<std::string> & values) {
        for (auto & val : values) {
            std::string full = field_prefix + val + "\n";
            std::vector<int> vtok = tokenize_strip(bpe, full, field_prefix);
            tree.add(vtok);
        }
    }

    void init(BPETokenizer & bpe, int vsize) {
        vocab_size = vsize;
        auto nl = bpe_encode(&bpe, "\n", false);
        newline_tok = nl.empty() ? -1 : nl[0];

        bpm_name      = bpe_encode(&bpe, "bpm:", false);
        caption_name  = bpe_encode(&bpe, "caption:", false);
        duration_name = bpe_encode(&bpe, "duration:", false);
        keyscale_name = bpe_encode(&bpe, "keyscale:", false);
        language_name = bpe_encode(&bpe, "language:", false);
        timesig_name  = bpe_encode(&bpe, "timesignature:", false);

        // BPM 30-300
        {
            std::vector<std::string> vals;
            for (int v = 30; v <= 300; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, bpm_tree, "bpm:", vals);
        }
        // Duration 10-600
        {
            std::vector<std::string> vals;
            for (int v = 10; v <= 600; v++) vals.push_back(std::to_string(v));
            build_value_tree(bpe, duration_tree, "duration:", vals);
        }
        // Keyscale
        {
            const char * notes[] = {"A","B","C","D","E","F","G"};
            const char * accs[]  = {"","b","#"};
            const char * modes[] = {
                "major","minor","dorian","phrygian","lydian","mixolydian",
                "aeolian","locrian","chromatic","blues","pentatonic",
                "harmonic minor","melodic minor"
            };
            std::vector<std::string> vals;
            for (auto n : notes)
                for (auto a : accs)
                    for (auto m : modes)
                        vals.push_back(std::string(n) + a + " " + m);
            build_value_tree(bpe, keyscale_tree, "keyscale:", vals);
        }
        // Language
        {
            std::vector<std::string> vals = {
                "en","zh","ja","ko","es","fr","de","uk","ru","pt",
                "it","ar","tr","pl","sv","nl","unknown"
            };
            build_value_tree(bpe, language_tree, "language:", vals);
        }
        // Time signature
        {
            std::vector<std::string> vals = {"2","3","4","6"};
            build_value_tree(bpe, timesig_tree, "timesignature:", vals);
        }

        fprintf(stderr, "[FSM] Prefix trees: bpm=%zu, dur=%zu, key=%zu, lang=%zu, tsig=%zu nodes\n",
                bpm_tree.nodes.size(), duration_tree.nodes.size(),
                keyscale_tree.nodes.size(), language_tree.nodes.size(),
                timesig_tree.nodes.size());
        enabled = true;
        state = BPM_NAME;
        name_pos = 0;
        value_acc.clear();
    }

    void reset() {
        state = BPM_NAME;
        name_pos = 0;
        value_acc.clear();
    }

    const std::vector<int> * current_name_tokens() const {
        switch (state) {
            case BPM_NAME:      return &bpm_name;
            case CAPTION_NAME:  return &caption_name;
            case DURATION_NAME: return &duration_name;
            case KEYSCALE_NAME: return &keyscale_name;
            case LANGUAGE_NAME: return &language_name;
            case TIMESIG_NAME:  return &timesig_name;
            default: return nullptr;
        }
    }

    const PrefixTree * current_value_tree() const {
        switch (state) {
            case BPM_VALUE:      return &bpm_tree;
            case DURATION_VALUE: return &duration_tree;
            case KEYSCALE_VALUE: return &keyscale_tree;
            case LANGUAGE_VALUE: return &language_tree;
            case TIMESIG_VALUE:  return &timesig_tree;
            default: return nullptr;
        }
    }

    State next_name_state() const {
        switch (state) {
            case BPM_NAME:      case BPM_VALUE:      return CAPTION_NAME;
            case CAPTION_NAME:  case CAPTION_VALUE:   return DURATION_NAME;
            case DURATION_NAME: case DURATION_VALUE:  return KEYSCALE_NAME;
            case KEYSCALE_NAME: case KEYSCALE_VALUE:  return LANGUAGE_NAME;
            case LANGUAGE_NAME: case LANGUAGE_VALUE:   return TIMESIG_NAME;
            case TIMESIG_NAME:  case TIMESIG_VALUE:   return THINK_END;
            default: return CODES;
        }
    }

    void apply_mask(float * logits) {
        if (!enabled || state == CODES || state == DISABLED) return;

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int)name->size()) {
            int forced = (*name)[name_pos];
            for (int v = 0; v < vocab_size; v++)
                if (v != forced) logits[v] = -1e9f;
            return;
        }

        const PrefixTree * tree = current_value_tree();
        if (tree) {
            const std::vector<int> * allowed = tree->get(value_acc);
            if (allowed && !allowed->empty()) {
                std::vector<float> saved(allowed->size());
                for (size_t i = 0; i < allowed->size(); i++)
                    saved[i] = logits[(*allowed)[i]];
                for (int v = 0; v < vocab_size; v++) logits[v] = -1e9f;
                for (size_t i = 0; i < allowed->size(); i++)
                    logits[(*allowed)[i]] = saved[i];
            } else {
                if (newline_tok >= 0) {
                    for (int v = 0; v < vocab_size; v++)
                        if (v != newline_tok) logits[v] = -1e9f;
                }
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            for (int v = AUDIO_CODE_BASE; v < AUDIO_CODE_BASE + AUDIO_CODE_COUNT; v++)
                if (v < vocab_size) logits[v] = -1e9f;
            return;
        }

        if (state == THINK_END) {
            for (int v = 0; v < vocab_size; v++)
                if (v != think_end_tok) logits[v] = -1e9f;
            return;
        }
    }

    void update(int token) {
        if (!enabled || state == CODES || state == DISABLED) return;

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int)name->size()) {
            name_pos++;
            if (name_pos >= (int)name->size()) {
                switch (state) {
                    case BPM_NAME:      state = BPM_VALUE; break;
                    case CAPTION_NAME:  state = CAPTION_VALUE; break;
                    case DURATION_NAME: state = DURATION_VALUE; break;
                    case KEYSCALE_NAME: state = KEYSCALE_VALUE; break;
                    case LANGUAGE_NAME: state = LANGUAGE_VALUE; break;
                    case TIMESIG_NAME:  state = TIMESIG_VALUE; break;
                    default: break;
                }
                value_acc.clear();
            }
            return;
        }

        if (current_value_tree()) {
            if (token == newline_tok) {
                state = next_name_state();
                name_pos = 0;
                value_acc.clear();
            } else {
                value_acc.push_back(token);
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            if (token == newline_tok) {
                state = DURATION_NAME;
                name_pos = 0;
                value_acc.clear();
            }
            return;
        }

        if (state == THINK_END) {
            state = CODES;
            return;
        }
    }
};

//
// Generation
//

// Main generation loop (codes + optional CFG + optional FSM)
static void generate(Qwen3LM * m, BPETokenizer * bpe,
                     const std::vector<int> & prompt_tokens,
                     int max_new_tokens, float temperature, float top_p, int seed,
                     std::vector<int> * out_audio_codes = nullptr,
                     float cfg_scale = 1.0f,
                     const std::vector<int> * uncond_tokens = nullptr,
                     bool cot_injected = false,
                     double * out_prefill_ms = nullptr,
                     double * out_decode_ms = nullptr,
                     bool stop_at_reasoning = false,
                     std::vector<int> * out_generated_tokens = nullptr,
                     MetadataFSM * fsm = nullptr) {

    int V = m->cfg.vocab_size;
    int total_tokens = 0;
    int audio_code_count = 0;
    bool use_cfg = cfg_scale > 1.0f && uncond_tokens != nullptr;
    bool codes_phase = cot_injected;

    // Reset KV caches
    qw3lm_reset_kv(m, 0);
    if (use_cfg) qw3lm_reset_kv(m, 1);

    fprintf(stderr, "[Prefill] Cond: %zu tokens", prompt_tokens.size());
    if (use_cfg) fprintf(stderr, ", Uncond: %zu tokens", uncond_tokens->size());
    fprintf(stderr, "\n");

    std::vector<float> logits_cond(V);
    std::vector<float> logits_uncond(V);

    // Prefill: batch all prompt tokens at once (faster than token-by-token)
    Timer t_prefill;
    qw3lm_forward(m, prompt_tokens.data(), (int)prompt_tokens.size(), 0, logits_cond.data());

    if (use_cfg)
        qw3lm_forward(m, uncond_tokens->data(), (int)uncond_tokens->size(), 1, logits_uncond.data());

    size_t total_prefill = prompt_tokens.size() + (use_cfg ? uncond_tokens->size() : 0);
    double prefill_ms = t_prefill.ms();
    fprintf(stderr, "[Prefill] %.0fms (%.1f tok/s, %zu tokens)\n",
            prefill_ms, total_prefill / (prefill_ms / 1000.0), total_prefill);
    if (out_prefill_ms) *out_prefill_ms = prefill_ms;

    // Decode loop
    Timer t_decode;
    g_rng.seed(seed);

    for (int i = 0; i < max_new_tokens; i++) {
        float * sample_logits = logits_cond.data();

        // CFG combination
        if (use_cfg) {
            for (int v = 0; v < V; v++)
                logits_cond[v] = logits_uncond[v] + cfg_scale * (logits_cond[v] - logits_uncond[v]);
            sample_logits = logits_cond.data();
        }

        // FSM constrained decoding (before </think>)
        if (fsm && fsm->enabled && !codes_phase)
            fsm->apply_mask(sample_logits);

        // After </think>: only audio codes + EOS
        if (codes_phase && !stop_at_reasoning) {
            for (int v = 0; v < AUDIO_CODE_BASE; v++)
                if (v != TOKEN_IM_END) sample_logits[v] = -1e9f;
        }

        int next_token = sample_top_p(sample_logits, V, temperature, top_p);

        // FSM update
        if (fsm && fsm->enabled && !codes_phase)
            fsm->update(next_token);

        // EOS
        if (next_token == TOKEN_IM_END) {
            total_tokens++;
            break;
        }

        if (out_generated_tokens) out_generated_tokens->push_back(next_token);

        // Detect </think> -> codes phase
        if (next_token == TOKEN_THINK_END && !codes_phase) {
            if (stop_at_reasoning) {
                total_tokens++;
                break;
            }
            codes_phase = true;
        }

        // Collect audio codes
        if (next_token >= AUDIO_CODE_BASE && next_token < AUDIO_CODE_BASE + AUDIO_CODE_COUNT) {
            if (out_audio_codes) out_audio_codes->push_back(next_token - AUDIO_CODE_BASE);
            audio_code_count++;
        }

        // Forward next token (decode: 1 token)
        qw3lm_forward(m, &next_token, 1, 0, logits_cond.data());

        // CFG: forward unconditional too
        if (use_cfg) {
            qw3lm_forward(m, &next_token, 1, 1, logits_uncond.data());
        }

        total_tokens++;

        if (codes_phase && (total_tokens % 50 == 0)) {
            double elapsed = t_decode.ms() / 1000.0;
            fprintf(stderr, "[Decode] %d/%d codes, %.1f tok/s\n",
                    audio_code_count, max_new_tokens, total_tokens / elapsed);
        }
    }

    if (total_tokens > 0) {
        double decode_ms = t_decode.ms();
        double tok_s = total_tokens / (decode_ms / 1000.0);
        fprintf(stderr, "[Decode] %d/%d codes, %.1f tok/s\n",
                audio_code_count, max_new_tokens, tok_s);
        fprintf(stderr, "[Decode] %.0fms (%d tokens, %d audio codes)\n",
                decode_ms, total_tokens, audio_code_count);
        if (out_decode_ms) *out_decode_ms = decode_ms;
    }
}

// Text-only generation (Phase 1: no CFG, stops at EOS)
static std::vector<int> generate_text(Qwen3LM * m, BPETokenizer * bpe,
                                       const std::vector<int> & prompt_tokens,
                                       int max_new_tokens, float temperature, float top_p, int seed,
                                       MetadataFSM * fsm = nullptr) {
    int V = m->cfg.vocab_size;
    std::vector<int> generated;

    qw3lm_reset_kv(m, 0);

    fprintf(stderr, "[Phase1-Prefill] %zu tokens\n", prompt_tokens.size());
    Timer t_prefill;
    std::vector<float> logits(V);
    qw3lm_forward(m, prompt_tokens.data(), (int)prompt_tokens.size(), 0, logits.data());
    fprintf(stderr, "[Phase1-Prefill] %.0fms\n", t_prefill.ms());

    Timer t_decode;
    g_rng.seed(seed);

    for (int i = 0; i < max_new_tokens; i++) {
        if (fsm && fsm->enabled) fsm->apply_mask(logits.data());
        int next = sample_top_p(logits.data(), V, temperature, top_p);
        if (fsm && fsm->enabled) fsm->update(next);
        if (next == TOKEN_IM_END) break;
        generated.push_back(next);
        qw3lm_forward(m, &next, 1, 0, logits.data());
    }

    fprintf(stderr, "[Phase1-Decode] %.0fms (%zu tokens)\n", t_decode.ms(), generated.size());
    return generated;
}

static bool write_codes(const char * dir, const std::vector<int> & codes);

// Phase 2: run audio code generation with all metas known
static void run_phase2(Qwen3LM * m, BPETokenizer & bpe, const AcePrompt & ace,
                       const std::vector<int> & /* p1_prompt (unused, KV reset) */,
                       float temperature, float top_p, int seed,
                       float cfg_scale, const char * negative_prompt,
                       const char * output_dir) {
    std::string cot = build_cot_yaml(ace);
    fprintf(stderr, "[Phase2] CoT:\n%s", cot.c_str());
    std::vector<int> prompt = build_lm_prompt_with_cot(bpe, ace, cot);
    std::vector<int> uncond;
    if (cfg_scale > 1.0f)
        uncond = build_lm_prompt_uncond_with_cot(bpe, ace, negative_prompt);

    int max_tokens = (int)(ace.duration * 5) + 100;

    fprintf(stderr, "[Phase2] %zu tokens, max: %d, CFG: %.2f\n",
            prompt.size(), max_tokens, cfg_scale);

    double prefill_ms = 0, decode_ms = 0;
    std::vector<int> audio_codes;
    generate(m, &bpe, prompt, max_tokens, temperature, top_p, seed,
             output_dir ? &audio_codes : nullptr,
             cfg_scale, uncond.empty() ? nullptr : &uncond,
             true, &prefill_ms, &decode_ms);

    if (output_dir && !audio_codes.empty())
        write_codes(output_dir, audio_codes);

    fprintf(stderr, "[Phase2] Prefill %.0f | Decode %.0fms\n", prefill_ms, decode_ms);
}

//
// Write helpers
//

// Write audio codes as CSV (compatible with dit-vae --input-dir)
static bool write_codes(const char * dir, const std::vector<int> & codes) {
    std::string path = std::string(dir) + "/codes";
    FILE * f = fopen(path.c_str(), "w");
    if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return false; }
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) fprintf(f, ",");
        fprintf(f, "%d", codes[i]);
    }
    fprintf(f, "\n");
    fclose(f);
    fprintf(stderr, "[Output] %s (%zu audio codes)\n", path.c_str(), codes.size());
    return true;
}

static void write_output_dir(const char * dir, const AcePrompt & ace) {
    std::string d(dir);
    auto write_file = [&](const char * name, const std::string & content) {
        std::string path = d + "/" + name;
        FILE * f = fopen(path.c_str(), "w");
        if (!f) { fprintf(stderr, "ERROR: cannot write %s\n", path.c_str()); return; }
        fwrite(content.data(), 1, content.size(), f);
        fclose(f);
    };
    write_file("caption", ace.caption);
    write_file("lyrics", ace.lyrics);
    if (ace.bpm > 0) write_file("bpm", std::to_string(ace.bpm));
    if (ace.duration > 0) write_file("duration", std::to_string((int)ace.duration));
    if (!ace.keyscale.empty()) write_file("keyscale", ace.keyscale);
    if (!ace.timesignature.empty()) write_file("timesig", ace.timesignature);
    if (!ace.vocal_language.empty()) write_file("language", ace.vocal_language);
    fprintf(stderr, "[Output] metadata -> %s/\n", dir);
}

//
// CLI
//

static void usage(const char * prog) {
    fprintf(stderr,
        "Usage: %s --model <dir> [options]\n"
        "\n"
        "Model:\n"
        "  --model <dir>          Model directory (safetensors + config.json)\n"
        "\n"
        "Simple mode (inspiration):\n"
        "  --query <text>         Natural language music description\n"
        "  --instrumental         Generate instrumental (no vocals)\n"
        "\n"
        "Custom mode:\n"
        "  --caption <text>       Music description\n"
        "  --lyrics <text>        Lyrics (default: empty)\n"
        "  --bpm <N>              BPM (0=LLM decides)\n"
        "  --duration <N>         Duration in seconds (0=LLM decides)\n"
        "  --keyscale <text>      Key/scale (e.g. 'C major')\n"
        "  --timesignature <N>    Time signature (2,3,4,6)\n"
        "  --language <code>      Vocal language (en,fr,zh,...)\n"
        "\n"
        "Raw mode (advanced):\n"
        "  --system <text>        Custom system message\n"
        "  --user <text>          Custom user message\n"
        "\n"
        "Generation:\n"
        "  --max-tokens <N>       Max new tokens (default: 256)\n"
        "  --max-seq <N>          KV cache size (default: 8192)\n"
        "  --temperature <f>      Sampling temperature (default: 0.85)\n"
        "  --top-p <f>            Top-p sampling (default: 0.9, 1.0=disabled)\n"
        "  --seed <N>             RNG seed (default: random)\n"
        "  --cfg-scale <f>        CFG scale for Phase 2 (default: 2.0, 1.0=disabled)\n"
        "  --negative-prompt <t>  Negative prompt for CFG\n"
        "  --no-fsm               Disable FSM constrained decoding\n"
        "  --no-codes             Phase 1 only (no audio codes)\n"
        "\n"
        "Output:\n"
        "  --output-dir <dir>     Write codes + metadata for dit-vae\n"
        "\n"
        "Debug:\n"
        "  --dump-logits <path>   Dump prefill logits (binary f32)\n"
        "  --dump-tokens <path>   Dump prompt token IDs (CSV)\n"
        "\n", prog);
}

int main(int argc, char ** argv) {
    const char * model_dir = nullptr;
    const char * cli_caption = nullptr;
    const char * cli_lyrics = nullptr;
    int cli_bpm = 0;
    float cli_duration = 0;
    const char * cli_keyscale = nullptr;
    const char * cli_timesig = nullptr;
    const char * cli_language = nullptr;
    const char * cli_query = nullptr;
    bool cli_instrumental = false;
    const char * system_msg = nullptr;
    const char * user_msg = nullptr;
    int max_tokens = 256;
    int max_seq = 8192;
    float temperature = 0.85f;
    float top_p = 0.9f;
    int seed = -1;
    float cfg_scale = 2.0f;
    const char * negative_prompt = nullptr;
    bool use_fsm = true;
    bool no_codes = false;
    const char * output_dir = nullptr;
    const char * dump_logits = nullptr;
    const char * dump_tokens = nullptr;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_dir = argv[++i];
        else if (!strcmp(argv[i], "--caption") && i + 1 < argc)
            cli_caption = argv[++i];
        else if (!strcmp(argv[i], "--lyrics") && i + 1 < argc)
            cli_lyrics = argv[++i];
        else if (!strcmp(argv[i], "--bpm") && i + 1 < argc)
            cli_bpm = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc)
            cli_duration = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--keyscale") && i + 1 < argc)
            cli_keyscale = argv[++i];
        else if (!strcmp(argv[i], "--timesignature") && i + 1 < argc)
            cli_timesig = argv[++i];
        else if (!strcmp(argv[i], "--language") && i + 1 < argc)
            cli_language = argv[++i];
        else if (!strcmp(argv[i], "--system") && i + 1 < argc)
            system_msg = argv[++i];
        else if (!strcmp(argv[i], "--user") && i + 1 < argc)
            user_msg = argv[++i];
        else if (!strcmp(argv[i], "--query") && i + 1 < argc)
            cli_query = argv[++i];
        else if (!strcmp(argv[i], "--instrumental"))
            cli_instrumental = true;
        else if (!strcmp(argv[i], "--max-tokens") && i + 1 < argc)
            max_tokens = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--max-seq") && i + 1 < argc)
            max_seq = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--temperature") && i + 1 < argc)
            temperature = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--top-p") && i + 1 < argc)
            top_p = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cfg-scale") && i + 1 < argc)
            cfg_scale = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--negative-prompt") && i + 1 < argc)
            negative_prompt = argv[++i];
        else if (!strcmp(argv[i], "--no-fsm"))
            use_fsm = false;
        else if (!strcmp(argv[i], "--no-codes"))
            no_codes = true;
        else if (!strcmp(argv[i], "--output-dir") && i + 1 < argc)
            output_dir = argv[++i];
        else if (!strcmp(argv[i], "--dump-logits") && i + 1 < argc)
            dump_logits = argv[++i];
        else if (!strcmp(argv[i], "--dump-tokens") && i + 1 < argc)
            dump_tokens = argv[++i];
        else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        }
        else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }

    if (!model_dir) {
        fprintf(stderr, "ERROR: --model required\n");
        usage(argv[0]);
        return 1;
    }
    int n_modes = (cli_query ? 1 : 0) + (cli_caption ? 1 : 0) + (system_msg ? 1 : 0);
    if (n_modes == 0) {
        fprintf(stderr, "ERROR: provide --query (simple), --caption (custom), or --system + --user (raw)\n");
        usage(argv[0]);
        return 1;
    }
    if (n_modes > 1) {
        fprintf(stderr, "ERROR: --query, --caption, and --system are mutually exclusive\n");
        return 1;
    }
    if (system_msg && !user_msg) {
        fprintf(stderr, "ERROR: --system requires --user\n");
        return 1;
    }

    if (seed < 0) {
        std::random_device rd;
        seed = (int)(rd() & 0x7FFFFFFF);
    }

    Timer t_total;

    // Load BPE tokenizer
    BPETokenizer bpe;
    if (!load_bpe_tokenizer(&bpe, model_dir)) return 1;

    // Load model
    int n_kv_sets = (cfg_scale > 1.0f) ? 2 : 1;
    Timer t_load;
    Qwen3LM model;
    if (!qw3lm_load(&model, model_dir, max_seq, n_kv_sets)) return 1;
    double load_ms = t_load.ms();

    // FSM
    MetadataFSM fsm;
    if (use_fsm) fsm.init(bpe, model.cfg.vocab_size);

    if (cli_query) {
        // Simple/Inspiration mode: query -> metadata + lyrics -> codes
        fprintf(stderr, "[Simple] query: %s, instrumental: %s\n",
                cli_query, cli_instrumental ? "true" : "false");

        // Build inspiration prompt (matches Python create_sample_from_query)
        std::string instr_user = std::string(cli_query) + "\n\ninstrumental: "
                               + (cli_instrumental ? "true" : "false");
        std::vector<int> p1_prompt = build_custom_prompt(bpe,
            "# Instruction\nExpand the user's input into a more detailed and specific musical description:\n",
            instr_user.c_str());
        fprintf(stderr, "[Phase1] %zu tokens, seed: %d\n", p1_prompt.size(), seed);

        // Phase 1: inspiration (no CFG, top_p disabled, matches Python)
        fsm.reset();
        std::vector<int> gen_tokens = generate_text(&model, &bpe, p1_prompt, 2048,
                                                     temperature, 1.0f, seed,
                                                     use_fsm ? &fsm : nullptr);
        std::string gen_text = bpe_decode(bpe, gen_tokens);
        fprintf(stderr, "[Phase1] %zu tokens decoded, %zuB text\n", gen_tokens.size(), gen_text.size());
        fprintf(stderr, "[Phase1] output:\n%s\n", gen_text.c_str());

        AcePrompt ace = {};
        if (!parse_cot_and_lyrics(gen_text, &ace)) {
            fprintf(stderr, "ERROR: failed to parse Phase 1 output\n");
            return 1;
        }
        if (ace.duration <= 0) ace.duration = 120.0f;
        if (ace.duration > 600) ace.duration = 600.0f;

        if (output_dir) write_output_dir(output_dir, ace);

        if (!no_codes) {
            run_phase2(&model, bpe, ace, p1_prompt, temperature, top_p, seed,
                       cfg_scale, negative_prompt, output_dir);
        }

        fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());

    } else if (system_msg) {
        // Raw mode (advanced)
        fprintf(stderr, "[Raw] system: %.60s...\n", system_msg);

        std::vector<int> p1_prompt = build_custom_prompt(bpe, system_msg, user_msg);
        fprintf(stderr, "[Phase1] %zu tokens, seed: %d\n", p1_prompt.size(), seed);

        fsm.reset();
        std::vector<int> gen_tokens = generate_text(&model, &bpe, p1_prompt, 2048,
                                                     temperature, top_p, seed,
                                                     use_fsm ? &fsm : nullptr);
        std::string gen_text = bpe_decode(bpe, gen_tokens);
        fprintf(stderr, "[Phase1] %zu tokens decoded, %zuB text\n", gen_tokens.size(), gen_text.size());
        fprintf(stderr, "[Phase1] output:\n%s\n", gen_text.c_str());

        AcePrompt ace = {};
        if (!parse_cot_and_lyrics(gen_text, &ace)) {
            fprintf(stderr, "ERROR: failed to parse Phase 1 output\n");
            return 1;
        }
        if (ace.duration <= 0) ace.duration = 120.0f;
        if (ace.duration > 600) ace.duration = 600.0f;

        if (output_dir) write_output_dir(output_dir, ace);

        if (!no_codes) {
            run_phase2(&model, bpe, ace, p1_prompt, temperature, top_p, seed,
                       cfg_scale, negative_prompt, output_dir);
        }

        fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());

    } else {
        // Custom mode (--caption with optional metadata)
        AcePrompt ace = {};
        ace.caption = cli_caption;
        ace.lyrics = cli_lyrics ? cli_lyrics : "";
        ace.duration = cli_duration;
        ace.bpm = cli_bpm;
        ace.keyscale = cli_keyscale ? cli_keyscale : "";
        ace.timesignature = cli_timesig ? cli_timesig : "";
        ace.vocal_language = cli_language ? cli_language : "";

        bool has_all_metas = (ace.bpm > 0 && ace.duration > 0 &&
                              !ace.keyscale.empty() && !ace.timesignature.empty());

        std::vector<int> prompt;

        if (no_codes && has_all_metas) {
            if (output_dir) write_output_dir(output_dir, ace);
            fprintf(stderr, "[All-metas] No codes requested, metadata written\n");
            fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());

        } else if (has_all_metas) {
            // All metas known: inject CoT, generate codes directly
            std::string cot = build_cot_yaml(ace);
            fprintf(stderr, "[All-metas] CoT:\n%s", cot.c_str());
            prompt = build_lm_prompt_with_cot(bpe, ace, cot);
            std::vector<int> uncond;
            if (cfg_scale > 1.0f)
                uncond = build_lm_prompt_uncond_with_cot(bpe, ace, negative_prompt);
            if (max_tokens == 256) max_tokens = (int)(ace.duration * 5) + 100;

            fprintf(stderr, "[All-metas] %zu tokens, max: %d, CFG: %.2f, seed: %d\n",
                    prompt.size(), max_tokens, cfg_scale, seed);

            // Debug: dump tokens for Python comparison
            if (dump_tokens) {
                FILE * f = fopen(dump_tokens, "w");
                if (f) {
                    for (size_t i = 0; i < prompt.size(); i++) {
                        if (i > 0) fprintf(f, ",");
                        fprintf(f, "%d", prompt[i]);
                    }
                    fprintf(f, "\n");
                    fclose(f);
                    fprintf(stderr, "[Debug] Tokens -> %s (%zu tokens)\n", dump_tokens, prompt.size());
                }
            }

            // Debug: dump prefill logits for comparison
            if (dump_logits) {
                qw3lm_reset_kv(&model, 0);
                std::vector<float> dbg_logits(model.cfg.vocab_size);
                qw3lm_forward(&model, prompt.data(), (int)prompt.size(), 0, dbg_logits.data());
                FILE * f = fopen(dump_logits, "wb");
                if (f) {
                    fwrite(dbg_logits.data(), sizeof(float), model.cfg.vocab_size, f);
                    fclose(f);
                    fprintf(stderr, "[Debug] Logits -> %s (%d floats, argmax=%d)\n",
                            dump_logits, model.cfg.vocab_size,
                            (int)(std::max_element(dbg_logits.begin(), dbg_logits.end()) - dbg_logits.begin()));
                }
                qw3lm_reset_kv(&model, 0);
            }

            double prefill_ms = 0, decode_ms = 0;
            std::vector<int> audio_codes;
            generate(&model, &bpe, prompt, max_tokens, temperature, top_p, seed,
                     output_dir ? &audio_codes : nullptr,
                     cfg_scale, uncond.empty() ? nullptr : &uncond,
                     true, &prefill_ms, &decode_ms);

            if (output_dir && !audio_codes.empty())
                write_codes(output_dir, audio_codes);
            if (output_dir) write_output_dir(output_dir, ace);

            fprintf(stderr, "[Ace-Qwen3] Load %.0f | Prefill %.0f | Decode %.0f | Total %.0fms\n",
                    load_ms, prefill_ms, decode_ms, t_total.ms());

        } else {
            // Partial metas: two-phase generation
            fprintf(stderr, "[Partial-metas] Two-phase generation\n");

            prompt = build_lm_prompt(bpe, ace);
            std::vector<int> uncond;
            if (cfg_scale > 1.0f)
                uncond = build_lm_prompt_uncond(bpe, ace, negative_prompt);

            fprintf(stderr, "[Phase1] %zu tokens, CFG: %.2f, seed: %d\n",
                    prompt.size(), cfg_scale, seed);

            std::vector<int> gen_tokens;
            double p1_prefill = 0, p1_decode = 0;
            fsm.reset();
            generate(&model, &bpe, prompt, 2048, temperature, top_p, seed,
                     nullptr, cfg_scale, uncond.empty() ? nullptr : &uncond,
                     false, &p1_prefill, &p1_decode,
                     true, &gen_tokens, use_fsm ? &fsm : nullptr);

            std::string gen_text = bpe_decode(bpe, gen_tokens);
            fprintf(stderr, "[Phase1] %zu tokens decoded, %zuB text\n", gen_tokens.size(), gen_text.size());
            fprintf(stderr, "[Partial-metas] CoT:\n%s\n", gen_text.c_str());

            AcePrompt parsed = ace;
            if (!parse_cot_and_lyrics(gen_text, &parsed)) {
                fprintf(stderr, "WARNING: CoT parse incomplete, using available fields\n");
            }
            if (parsed.bpm > 0) ace.bpm = parsed.bpm;
            if (parsed.duration > 0) ace.duration = parsed.duration;
            if (!parsed.keyscale.empty()) ace.keyscale = parsed.keyscale;
            if (!parsed.timesignature.empty()) ace.timesignature = parsed.timesignature;
            if (!parsed.vocal_language.empty()) ace.vocal_language = parsed.vocal_language;
            if (!parsed.caption.empty()) ace.caption = parsed.caption;

            if (ace.duration <= 0) ace.duration = 120.0f;
            if (ace.duration > 600) ace.duration = 600.0f;

            if (output_dir) write_output_dir(output_dir, ace);

            if (!no_codes) {
                run_phase2(&model, bpe, ace, prompt, temperature, top_p, seed,
                           cfg_scale, negative_prompt, output_dir);
            }

            fprintf(stderr, "[Ace-Qwen3] Load %.0f | Total %.0fms\n", load_ms, t_total.ms());
        }
    }

    qw3lm_free(&model);
    return 0;
}
