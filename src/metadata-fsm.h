#pragma once
// metadata-fsm.h: constrained decoding FSM for ACE-Step metadata
//
// PrefixTree for token-level constraints, MetadataFSM for structured
// YAML metadata generation (BPM, duration, key, time signature, language).
// Also: audio code parsing and Phase 1 output merging.

#include "bpe.h"
#include "prompt.h"

#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <cstdio>

// Prefix tree for FSM constrained decoding
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

// Metadata FSM (constrained decoding for CoT fields)
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

    // Force FSM to only allow a specific language value
    void force_language(BPETokenizer & bpe, const std::string & lang) {
        language_tree = PrefixTree();
        build_value_tree(bpe, language_tree, "language:", {lang});
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
            case CAPTION_NAME:  case CAPTION_VALUE:  return DURATION_NAME;
            case DURATION_NAME: case DURATION_VALUE: return KEYSCALE_NAME;
            case KEYSCALE_NAME: case KEYSCALE_VALUE: return LANGUAGE_NAME;
            case LANGUAGE_NAME: case LANGUAGE_VALUE: return TIMESIG_NAME;
            case TIMESIG_NAME:  case TIMESIG_VALUE:  return THINK_END;
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

// Generation
// Text-only generation (Phase 1: no CFG, stops at EOS)
static std::string codes_to_string(const std::vector<int> & codes);

// Convert audio codes vector to comma-separated string (Python-compatible)
static std::string codes_to_string(const std::vector<int> & codes) {
    std::string s;
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) s += ',';
        s += std::to_string(codes[i]);
    }
    return s;
}

// Phase 2: run audio code generation with all metas known
// Returns comma-separated codes string (empty on failure)

// Parse N Phase 1 outputs into N AcePrompts, merging into base.
// merge_lyrics: true for simple mode (Phase 1 generates lyrics),
//               false for partial mode (user provided lyrics).
static void parse_phase1_into_aces(
        const std::vector<std::string> & texts, const AcePrompt & base,
        std::vector<AcePrompt> & aces, long long base_seed,
        const char * label, bool merge_lyrics) {
    int N = (int)texts.size();
    aces.resize(N);
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[%s Batch%d] seed=%lld:\n%s\n", label, i, base_seed + i, texts[i].c_str());
        AcePrompt parsed = {};
        if (!parse_cot_and_lyrics(texts[i], &parsed))
            fprintf(stderr, "WARNING: batch %d CoT parse incomplete\n", i);
        aces[i] = base;
        // gap fill: only write fields the user left empty
        if (parsed.bpm > 0 && base.bpm <= 0) aces[i].bpm = parsed.bpm;
        if (parsed.duration > 0 && base.duration <= 0) aces[i].duration = parsed.duration;
        if (!parsed.keyscale.empty() && base.keyscale.empty()) aces[i].keyscale = parsed.keyscale;
        if (!parsed.timesignature.empty() && base.timesignature.empty()) aces[i].timesignature = parsed.timesignature;
        if (!parsed.vocal_language.empty() && base.vocal_language.empty()) aces[i].vocal_language = parsed.vocal_language;
        if (!parsed.caption.empty()) aces[i].caption = parsed.caption;
        // lyrics: only generated when user had none
        if (merge_lyrics && !parsed.lyrics.empty()) aces[i].lyrics = parsed.lyrics;
        if (aces[i].duration <= 0) aces[i].duration = 120.0f;
        if (aces[i].duration > 600) aces[i].duration = 600.0f;
    }
}
