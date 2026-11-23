// test-lm-prompt.cpp: byte for byte check of Phase 2 LM prompt builders
// against the Qwen3 chat template reference.
//
// Runs the three Phase 2 builders (cond, uncond with empty neg, uncond with
// real neg), decodes their token streams back to text and dumps each one on
// stderr tagged with a [LABEL] so the output is directly diffable against a
// reference produced by `transformers.apply_chat_template` or llama.cpp's
// `common_chat_templates_apply` on the Qwen3 chat template.
//
// Usage:
//   ./test-lm-prompt --models <dir>
//
// Picks the first LM entry of the registry, shares its BPE through the store
// exactly like the production pipelines do.

#include "bpe.h"
#include "model-registry.h"
#include "model-store.h"
#include "prompt.h"
#include "task-types.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Reverse of byte_level_encode: map a Qwen BPE piece back to raw bytes.
static std::string piece_to_bytes(const BPETokenizer & tok, const std::string & piece) {
    static int  str2byte[0x10000];
    static bool init = false;
    if (!init) {
        for (int i = 0; i < 0x10000; i++) {
            str2byte[i] = -1;
        }
        for (int b = 0; b < 256; b++) {
            const std::string & s = tok.byte2str[b];
            if (s.size() == 1) {
                str2byte[(unsigned char) s[0]] = b;
            } else if (s.size() == 2) {
                unsigned key  = ((unsigned char) s[0] << 8) | (unsigned char) s[1];
                str2byte[key] = b;
            }
        }
        init = true;
    }
    std::string out;
    size_t      i = 0;
    while (i < piece.size()) {
        unsigned char c0 = (unsigned char) piece[i];
        int           b  = str2byte[c0];
        if (b >= 0) {
            out.push_back((char) b);
            i += 1;
            continue;
        }
        if (i + 1 < piece.size()) {
            unsigned char c1  = (unsigned char) piece[i + 1];
            unsigned      key = (c0 << 8) | c1;
            int           b2  = str2byte[key];
            if (b2 >= 0) {
                out.push_back((char) b2);
                i += 2;
                continue;
            }
        }
        out.push_back(piece[i]);
        i += 1;
    }
    return out;
}

// Decode a token id stream to the exact string it represents.
static std::string decode_ids(const BPETokenizer & tok, const std::vector<int> & ids) {
    std::string out;
    for (int id : ids) {
        if (id == TOKEN_IM_START) {
            out += "<|im_start|>";
            continue;
        }
        if (id == TOKEN_IM_END) {
            out += "<|im_end|>";
            continue;
        }
        if (id == TOKEN_THINK) {
            out += "<think>";
            continue;
        }
        if (id == TOKEN_THINK_END) {
            out += "</think>";
            continue;
        }
        if (id >= 0 && id < (int) tok.id_to_str.size()) {
            out += piece_to_bytes(tok, tok.id_to_str[id]);
        }
    }
    return out;
}

// One line repr dump: [LABEL] '...' with control chars escaped.
static void dump_repr(const char * label, const std::string & s) {
    fprintf(stderr, "[%s] '", label);
    for (char c : s) {
        if (c == '\n') {
            fprintf(stderr, "\\n");
        } else if (c == '\\') {
            fprintf(stderr, "\\\\");
        } else if (c == '\'') {
            fprintf(stderr, "\\'");
        } else if ((unsigned char) c < 32 || (unsigned char) c == 127) {
            fprintf(stderr, "\\x%02x", (unsigned char) c);
        } else {
            fputc(c, stderr);
        }
    }
    fprintf(stderr, "'\n");
}

static void usage(const char * prog) {
    fprintf(stderr, "Usage: %s --models <dir>\n", prog);
}

int main(int argc, char ** argv) {
    const char * models_dir = nullptr;

    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--models") && i + 1 < argc) {
            models_dir = argv[++i];
        } else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
            usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            usage(argv[0]);
            return 1;
        }
    }
    if (!models_dir) {
        fprintf(stderr, "[CLI] ERROR: --models required\n");
        usage(argv[0]);
        return 1;
    }

    ModelRegistry registry;
    if (!registry_scan(&registry, models_dir)) {
        fprintf(stderr, "[Test] FATAL: cannot scan --models %s\n", models_dir);
        return 1;
    }
    if (registry.lm.empty()) {
        fprintf(stderr, "[Test] FATAL: registry needs an LM under %s\n", models_dir);
        return 1;
    }
    const char * lm_path = registry.lm[0].path.c_str();
    fprintf(stderr, "[Test] LM=%s\n", lm_path);

    ModelStore *   s   = store_create(EVICT_STRICT);
    BPETokenizer * bpe = store_bpe(s, lm_path);
    if (!bpe) {
        fprintf(stderr, "[Test] FAIL: BPE load\n");
        store_free(s);
        return 1;
    }

    AcePrompt p;
    p.caption       = "chill lofi beats";
    p.lyrics        = "la la la";
    p.bpm           = 90;
    p.duration      = 30;
    std::string cot = build_cot_yaml(p);

    auto id_cond     = build_lm_prompt_with_cot(*bpe, p, cot);
    auto id_uncond_e = build_lm_prompt_uncond_with_cot(*bpe, "");
    auto id_uncond_n = build_lm_prompt_uncond_with_cot(*bpe, "low quality, distorted");

    dump_repr("COND", decode_ids(*bpe, id_cond));
    dump_repr("UNCOND_EMPTY", decode_ids(*bpe, id_uncond_e));
    dump_repr("UNCOND_NEG", decode_ids(*bpe, id_uncond_n));

    std::string inspire_sys = std::string("# Instruction\n") + LM_INSPIRE_INSTRUCTION + "\n";
    auto        id_inspire  = build_custom_prompt(*bpe, inspire_sys.c_str(), "a cat who loves the rain\n\ninstrumental: false");
    dump_repr("INSPIRE", decode_ids(*bpe, id_inspire));
    store_free(s);
    fprintf(stderr, "[Test] PASS\n");
    return 0;
}
