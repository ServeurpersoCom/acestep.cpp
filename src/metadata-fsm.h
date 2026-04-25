#pragma once
// metadata-fsm.h: constrained decoding FSM for ACE-Step metadata
//
// PrefixTree for token-level constraints, MetadataFSM for structured
// YAML metadata generation (BPM, duration, key, time signature, language).
// Also: audio code parsing and Phase 1 output merging.

#include "bpe.h"
#include "prompt.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

// Prefix tree for FSM constrained decoding
struct PrefixTree {
    // Maps prefix (token sequence) to set of valid next tokens
    std::map<std::vector<int>, std::vector<int>> nodes;

    void add(const std::vector<int> & seq) {
        for (size_t i = 0; i < seq.size(); i++) {
            std::vector<int> prefix(seq.begin(), seq.begin() + i);
            int              next = seq[i];
            auto &           vec  = nodes[prefix];
            if (std::find(vec.begin(), vec.end(), next) == vec.end()) {
                vec.push_back(next);
            }
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
        BPM_NAME,
        BPM_VALUE,
        CAPTION_NAME,
        CAPTION_VALUE,
        DURATION_NAME,
        DURATION_VALUE,
        KEYSCALE_NAME,
        KEYSCALE_VALUE,
        LANGUAGE_NAME,
        LANGUAGE_VALUE,
        TIMESIG_NAME,
        TIMESIG_VALUE,
        THINK_END,
        CODES,
        DISABLED
    };

    State            state    = DISABLED;
    int              name_pos = 0;
    std::vector<int> value_acc;
    bool             enabled                 = false;
    bool             caption_pending_newline = false;

    // Caption lock: when true, the FSM skips the caption field entirely.
    // The user-provided caption is already in the prompt context; the LM
    // does not regenerate it and the lyrics/codes stay conditioned on it.
    bool skip_caption = false;

    std::vector<int> bpm_name, caption_name, duration_name;
    std::vector<int> keyscale_name, language_name, timesig_name;
    PrefixTree       bpm_tree, duration_tree, keyscale_tree, language_tree, timesig_tree;
    int              newline_tok   = -1;
    int              think_end_tok = TOKEN_THINK_END;
    int              vocab_size    = 0;

    // Token decoding (for caption transition detection)
    BPETokenizer *                   bpe_ptr = nullptr;
    std::unordered_map<int, uint8_t> byte_dec;  // BPE codepoint -> raw byte

    // Caption transition: text-based field detection (matches Python FSM)
    // When LLM generates a newline in caption followed by a non-space char,
    // we enter caption_ending mode and accumulate text until we see ":".
    // Then we parse the field name and jump directly to the VALUE state.
    bool        caption_ending = false;
    std::string pending_field_name;

    // Token injection for user-provided field values.
    // When active, apply_mask whitelists only the next token, update consumes it.
    // After the queue is exhausted, we transition to the next field.
    std::vector<int> inject_queue;

    // Forced values: set by force_field(), injected when FSM reaches the VALUE state.
    // Conditions the KV cache on the user's values so lyrics and codes match.
    std::string forced_bpm;
    std::string forced_duration;
    std::string forced_keyscale;
    std::string forced_language;
    std::string forced_timesig;

    static std::vector<int> tokenize_strip(BPETokenizer & bpe, const std::string & full, const std::string & prefix) {
        std::vector<int> full_tok = bpe_encode(&bpe, full, false);
        std::vector<int> pre_tok  = bpe_encode(&bpe, prefix, false);
        if (full_tok.size() >= pre_tok.size() && std::equal(pre_tok.begin(), pre_tok.end(), full_tok.begin())) {
            return std::vector<int>(full_tok.begin() + pre_tok.size(), full_tok.end());
        }

        // BPE context mismatch: "keyscale:" tokenizes differently alone vs
        // in "keyscale: C# minor\n" (colon+space merge). Tokenize the value
        // part directly to avoid injecting the field name as part of the value.
        return bpe_encode(&bpe, full.substr(prefix.size()), false);
    }

    void build_value_tree(BPETokenizer &                   bpe,
                          PrefixTree &                     tree,
                          const std::string &              field_prefix,
                          const std::vector<std::string> & values) {
        for (auto & val : values) {
            std::string      full = field_prefix + val + "\n";
            std::vector<int> vtok = tokenize_strip(bpe, full, field_prefix);
            tree.add(vtok);
        }
    }

    // Decode a single token ID to its raw text representation.
    // Uses the byte_dec reverse map built during init().
    std::string decode_token(int id) const {
        if (!bpe_ptr || id < 0 || id >= (int) bpe_ptr->id_to_str.size()) {
            return "";
        }
        const std::string & s = bpe_ptr->id_to_str[id];
        std::string         result;
        const char *        p = s.c_str();
        while (*p) {
            int  adv;
            int  cp = utf8_codepoint(p, &adv);
            auto it = byte_dec.find(cp);
            if (it != byte_dec.end()) {
                result += (char) it->second;
            }
            p += adv;
        }
        return result;
    }

    void init(BPETokenizer & bpe, int vsize) {
        vocab_size = vsize;
        bpe_ptr    = &bpe;

        // Build reverse byte map for decode_token()
        for (int b = 0; b < 256; b++) {
            int adv;
            int cp       = utf8_codepoint(bpe.byte2str[b].c_str(), &adv);
            byte_dec[cp] = (uint8_t) b;
        }

        auto nl     = bpe_encode(&bpe, "\n", false);
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
            for (int v = 30; v <= 300; v++) {
                vals.push_back(std::to_string(v));
            }
            build_value_tree(bpe, bpm_tree, "bpm:", vals);
        }
        // Duration 10-600
        {
            std::vector<std::string> vals;
            for (int v = 10; v <= 600; v++) {
                vals.push_back(std::to_string(v));
            }
            build_value_tree(bpe, duration_tree, "duration:", vals);
        }
        // Keyscale
        {
            // Keyscale: 7 notes x 5 accidentals x 2 modes = 70 values
            // Matches Python constants.py: KEYSCALE_NOTES x KEYSCALE_ACCIDENTALS x KEYSCALE_MODES
            const char * notes[] = { "A", "B", "C", "D", "E", "F", "G" };
            const char * accs[]  = { "", "#", "b", "\xe2\x99\xaf", "\xe2\x99\xad" };  // empty, #, b, U+266F, U+266D
            const char * modes[] = { "major", "minor" };
            std::vector<std::string> vals;
            for (auto n : notes) {
                for (auto a : accs) {
                    for (auto m : modes) {
                        vals.push_back(std::string(n) + a + " " + m);
                    }
                }
            }
            build_value_tree(bpe, keyscale_tree, "keyscale:", vals);
        }
        // Language: 51 codes matching Python constants.py VALID_LANGUAGES
        {
            std::vector<std::string> vals = {
                "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",  "es", "fa",      "fi",
                "fr", "he", "hi", "hr", "ht", "hu", "id", "is", "it", "ja",  "ko", "la",      "lt",
                "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro", "ru", "sa",  "sk", "sr",      "sv",
                "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh", "unknown",
            };
            build_value_tree(bpe, language_tree, "language:", vals);
        }
        // Time signature
        {
            std::vector<std::string> vals = { "2", "3", "4", "6" };
            build_value_tree(bpe, timesig_tree, "timesignature:", vals);
        }

        fprintf(stderr, "[FSM] Prefix trees: bpm=%zu, dur=%zu, key=%zu, lang=%zu, tsig=%zu nodes\n",
                bpm_tree.nodes.size(), duration_tree.nodes.size(), keyscale_tree.nodes.size(),
                language_tree.nodes.size(), timesig_tree.nodes.size());
        enabled  = true;
        state    = BPM_NAME;
        name_pos = 0;
        value_acc.clear();
    }

    void reset() {
        state                   = BPM_NAME;
        name_pos                = 0;
        caption_pending_newline = false;
        caption_ending          = false;
        pending_field_name.clear();
        value_acc.clear();
        inject_queue.clear();
        // forced_* values are NOT cleared: they persist across resets
        // (set once at setup, apply to every generate call)
    }

    // Force FSM to inject a specific value for a metadata field.
    // The value is tokenized and force-injected when the FSM reaches
    // the VALUE state, conditioning the KV cache on the user's value.
    void force_field(BPETokenizer & bpe, State value_state, const std::string & val) {
        switch (value_state) {
            case BPM_VALUE:
                forced_bpm = val;
                bpm_tree   = PrefixTree();
                build_value_tree(bpe, bpm_tree, "bpm:", { val });
                break;
            case DURATION_VALUE:
                forced_duration = val;
                duration_tree   = PrefixTree();
                build_value_tree(bpe, duration_tree, "duration:", { val });
                break;
            case KEYSCALE_VALUE:
                forced_keyscale = val;
                keyscale_tree   = PrefixTree();
                build_value_tree(bpe, keyscale_tree, "keyscale:", { val });
                break;
            case LANGUAGE_VALUE:
                forced_language = val;
                language_tree   = PrefixTree();
                build_value_tree(bpe, language_tree, "language:", { val });
                break;
            case TIMESIG_VALUE:
                forced_timesig = val;
                timesig_tree   = PrefixTree();
                build_value_tree(bpe, timesig_tree, "timesignature:", { val });
                break;
            default:
                break;
        }
    }

    const std::vector<int> * current_name_tokens() const {
        switch (state) {
            case BPM_NAME:
                return &bpm_name;
            case CAPTION_NAME:
                return &caption_name;
            case DURATION_NAME:
                return &duration_name;
            case KEYSCALE_NAME:
                return &keyscale_name;
            case LANGUAGE_NAME:
                return &language_name;
            case TIMESIG_NAME:
                return &timesig_name;
            default:
                return nullptr;
        }
    }

    const PrefixTree * current_value_tree() const {
        switch (state) {
            case BPM_VALUE:
                return &bpm_tree;
            case DURATION_VALUE:
                return &duration_tree;
            case KEYSCALE_VALUE:
                return &keyscale_tree;
            case LANGUAGE_VALUE:
                return &language_tree;
            case TIMESIG_VALUE:
                return &timesig_tree;
            default:
                return nullptr;
        }
    }

    State next_name_state() const {
        switch (state) {
            case BPM_NAME:
            case BPM_VALUE:
                return skip_caption ? DURATION_NAME : CAPTION_NAME;
            case CAPTION_NAME:
            case CAPTION_VALUE:
                return DURATION_NAME;
            case DURATION_NAME:
            case DURATION_VALUE:
                return KEYSCALE_NAME;
            case KEYSCALE_NAME:
            case KEYSCALE_VALUE:
                return LANGUAGE_NAME;
            case LANGUAGE_NAME:
            case LANGUAGE_VALUE:
                return TIMESIG_NAME;
            case TIMESIG_NAME:
            case TIMESIG_VALUE:
                return THINK_END;
            default:
                return CODES;
        }
    }

    void apply_mask(float * logits) {
        if (!enabled || state == CODES || state == DISABLED) {
            return;
        }

        // Token injection active: whitelist only the next queued token.
        // This is the C++ equivalent of Python's user_field_token_queue.
        if (!inject_queue.empty()) {
            int forced = inject_queue[0];
            for (int v = 0; v < vocab_size; v++) {
                if (v != forced) {
                    logits[v] = -1e9f;
                }
            }
            return;
        }

        // Force name tokens for the current field
        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int) name->size()) {
            int forced = (*name)[name_pos];
            for (int v = 0; v < vocab_size; v++) {
                if (v != forced) {
                    logits[v] = -1e9f;
                }
            }
            return;
        }

        // Value injection: force user-provided values into the KV cache.
        // Conditions the LM on the correct metadata so lyrics and codes match.
        const char *        prefix = nullptr;
        const std::string * forced = nullptr;
        switch (state) {
            case BPM_VALUE:
                prefix = "bpm:";
                forced = &forced_bpm;
                break;
            case DURATION_VALUE:
                prefix = "duration:";
                forced = &forced_duration;
                break;
            case KEYSCALE_VALUE:
                prefix = "keyscale:";
                forced = &forced_keyscale;
                break;
            case LANGUAGE_VALUE:
                prefix = "language:";
                forced = &forced_language;
                break;
            case TIMESIG_VALUE:
                prefix = "timesignature:";
                forced = &forced_timesig;
                break;
            default:
                break;
        }
        if (prefix && forced && !forced->empty() && bpe_ptr) {
            std::string      text  = std::string(prefix) + *forced + "\n";
            std::vector<int> vtoks = tokenize_strip(*bpe_ptr, text, prefix);
            if (!vtoks.empty()) {
                inject_queue = vtoks;
                int ftok     = inject_queue[0];
                for (int v = 0; v < vocab_size; v++) {
                    if (v != ftok) {
                        logits[v] = -1e9f;
                    }
                }
                return;
            }
        }

        // Prefix tree constrained values (bpm, duration, keyscale, language, timesig)
        const PrefixTree * tree = current_value_tree();
        if (tree) {
            const std::vector<int> * allowed = tree->get(value_acc);
            if (allowed && !allowed->empty()) {
                std::vector<float> saved(allowed->size());
                for (size_t i = 0; i < allowed->size(); i++) {
                    saved[i] = logits[(*allowed)[i]];
                }
                for (int v = 0; v < vocab_size; v++) {
                    logits[v] = -1e9f;
                }
                for (size_t i = 0; i < allowed->size(); i++) {
                    logits[(*allowed)[i]] = saved[i];
                }
            } else {
                if (newline_tok >= 0) {
                    for (int v = 0; v < vocab_size; v++) {
                        if (v != newline_tok) {
                            logits[v] = -1e9f;
                        }
                    }
                }
            }
            return;
        }

        // Caption value: block audio codes, allow everything else.
        // Also applies when caption_ending (LLM generating next field name).
        if (state == CAPTION_VALUE) {
            for (int v = AUDIO_CODE_BASE; v < AUDIO_CODE_BASE + AUDIO_CODE_COUNT; v++) {
                if (v < vocab_size) {
                    logits[v] = -1e9f;
                }
            }
            return;
        }

        if (state == THINK_END) {
            for (int v = 0; v < vocab_size; v++) {
                if (v != think_end_tok) {
                    logits[v] = -1e9f;
                }
            }
            return;
        }
    }

    void update(int token) {
        if (!enabled || state == CODES || state == DISABLED) {
            return;
        }

        // Inject queue: consume the forced token, transition when exhausted.
        // Matches Python: user_field_token_queue.pop(0), then next_state.
        if (!inject_queue.empty()) {
            inject_queue.erase(inject_queue.begin());
            if (inject_queue.empty()) {
                // Injection complete: transition to next field
                state    = next_name_state();
                name_pos = 0;
                value_acc.clear();
            }
            return;
        }

        // Caption YAML continuation: after a \n in caption, decode the token
        // text and check if it starts with space/tab (continuation) or not
        // (new field name). Matches Python: caption_after_newline logic.
        if (caption_pending_newline) {
            caption_pending_newline = false;
            std::string tok_text    = decode_token(token);
            if (!tok_text.empty() && tok_text[0] != ' ' && tok_text[0] != '\t') {
                // Non-indented: LLM is generating a field name (like "duration:")
                caption_ending     = true;
                pending_field_name = tok_text;
                // Check if we already have a colon (short token like "d:")
                if (tok_text.find(':') != std::string::npos) {
                    // Already got the full field name, parse it
                    std::string field = pending_field_name.substr(0, pending_field_name.find(':'));
                    // Trim whitespace
                    while (!field.empty() && field.back() == ' ') {
                        field.pop_back();
                    }
                    State target = field_name_to_value_state(field);
                    if (target != DISABLED) {
                        state    = target;
                        name_pos = 0;
                        value_acc.clear();
                        caption_ending = false;
                        pending_field_name.clear();
                    }
                }
                return;
            }
            // Continuation line (YAML wrap with leading spaces), stay in CAPTION_VALUE
            return;
        }

        // Caption ending: accumulate decoded text until ":" detected,
        // then parse field name and jump to VALUE state.
        // Matches Python: caption_ending + pending_field_name logic.
        if (caption_ending) {
            std::string tok_text = decode_token(token);
            pending_field_name += tok_text;
            if (tok_text.find(':') != std::string::npos || pending_field_name.find(':') != std::string::npos) {
                std::string field = pending_field_name.substr(0, pending_field_name.find(':'));
                while (!field.empty() && field.back() == ' ') {
                    field.pop_back();
                }
                State target = field_name_to_value_state(field);
                if (target != DISABLED) {
                    state    = target;
                    name_pos = 0;
                    value_acc.clear();
                    caption_ending = false;
                    pending_field_name.clear();
                }
            }
            return;
        }

        const std::vector<int> * name = current_name_tokens();
        if (name && name_pos < (int) name->size()) {
            name_pos++;
            if (name_pos >= (int) name->size()) {
                switch (state) {
                    case BPM_NAME:
                        state = BPM_VALUE;
                        break;
                    case CAPTION_NAME:
                        state = CAPTION_VALUE;
                        break;
                    case DURATION_NAME:
                        state = DURATION_VALUE;
                        break;
                    case KEYSCALE_NAME:
                        state = KEYSCALE_VALUE;
                        break;
                    case LANGUAGE_NAME:
                        state = LANGUAGE_VALUE;
                        break;
                    case TIMESIG_NAME:
                        state = TIMESIG_VALUE;
                        break;
                    default:
                        break;
                }
                value_acc.clear();
            }
            return;
        }

        if (current_value_tree()) {
            if (token == newline_tok) {
                state    = next_name_state();
                name_pos = 0;
                value_acc.clear();
            } else {
                value_acc.push_back(token);
            }
            return;
        }

        if (state == CAPTION_VALUE) {
            if (token == newline_tok) {
                // Don't transition yet: the caption may wrap (YAML continuation
                // lines start with spaces). Peek at next token via a flag.
                caption_pending_newline = true;
            }
            return;
        }

        if (state == THINK_END) {
            state = CODES;
            return;
        }
    }

    // Map field name text to the corresponding VALUE state.
    // Used by caption_ending to jump directly when ":" is detected.
    State field_name_to_value_state(const std::string & field) const {
        if (field == "duration") {
            return DURATION_VALUE;
        }
        if (field == "keyscale") {
            return KEYSCALE_VALUE;
        }
        if (field == "language") {
            return LANGUAGE_VALUE;
        }
        if (field == "timesignature") {
            return TIMESIG_VALUE;
        }
        return DISABLED;
    }
};

// Generation
// Text-only generation (Phase 1: no CFG, stops at EOS)
static std::string codes_to_string(const std::vector<int> & codes);

// Convert audio codes vector to comma-separated string (Python-compatible)
static std::string codes_to_string(const std::vector<int> & codes) {
    std::string s;
    for (size_t i = 0; i < codes.size(); i++) {
        if (i > 0) {
            s += ',';
        }
        s += std::to_string(codes[i]);
    }
    return s;
}

// Phase 2: run audio code generation with all metas known
// Returns comma-separated codes string (empty on failure)

// Parse N Phase 1 outputs into N AcePrompts, merging into base.
// merge_lyrics: true for simple mode (Phase 1 generates lyrics),
//               false for partial mode (user provided lyrics).
static void parse_phase1_into_aces(const std::vector<std::string> & texts,
                                   const AcePrompt &                base,
                                   std::vector<AcePrompt> &         aces,
                                   long long                        base_seed,
                                   const char *                     label,
                                   bool                             merge_lyrics,
                                   bool                             use_cot_caption = true) {
    int N = (int) texts.size();
    aces.resize(N);
    for (int i = 0; i < N; i++) {
        fprintf(stderr, "[%s Batch%d] seed=%lld:\n%s\n", label, i, base_seed + i, texts[i].c_str());
        AcePrompt parsed = {};
        if (!parse_cot_and_lyrics(texts[i], &parsed)) {
            fprintf(stderr, "WARNING: batch %d CoT parse incomplete\n", i);
        }
        aces[i] = base;
        // gap fill: only write fields the user left empty
        if (parsed.bpm > 0 && base.bpm <= 0) {
            aces[i].bpm = parsed.bpm;
        }
        if (parsed.duration > 0 && base.duration <= 0) {
            aces[i].duration = parsed.duration;
        }
        if (!parsed.keyscale.empty() && base.keyscale.empty()) {
            aces[i].keyscale = parsed.keyscale;
        }
        if (!parsed.timesignature.empty() && base.timesignature.empty()) {
            aces[i].timesignature = parsed.timesignature;
        }
        if (!parsed.vocal_language.empty() && (base.vocal_language.empty() || base.vocal_language == "unknown")) {
            aces[i].vocal_language = parsed.vocal_language;
        }
        if (!parsed.caption.empty() && use_cot_caption) {
            aces[i].caption = parsed.caption;
        }
        // lyrics: only generated when user had none
        if (merge_lyrics && !parsed.lyrics.empty()) {
            aces[i].lyrics = parsed.lyrics;
        }
        if (aces[i].duration <= 0) {
            aces[i].duration = 120.0f;
        }
        if (aces[i].duration > 600) {
            aces[i].duration = 600.0f;
        }
    }
}
