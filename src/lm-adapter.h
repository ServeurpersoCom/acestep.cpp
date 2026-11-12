#pragma once
// lm-adapter.h: runtime LoRA loading for the 5Hz planner LM.
//
// Loads a PEFT LoRA adapter (directory with adapter_model.safetensors +
// adapter_config.json, or a bare .safetensors file) and stages the A/B
// pairs on the LM's backend.  Applied at graph-build time via
// qwen3_linear_lora() — never merged, so the base LM can be any quant
// (Q8_0/Q5_K/NVFP4/...) with zero requantization loss.  Per-token cost at
// rank 16 is negligible (two rank-r matmuls per adapted projection).
//
// Trained by Side-Step's `sidestep lm-train` (sidestep_engine/lm/), which
// replicates the engine's exact prompt format (engine/src/prompt.h).
//
// PEFT tensor naming handled:
//   base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight
//   (any prefix before "model.layers." is ignored)
// Supported slots: q/k/v/o_proj, gate/up/down_proj.  Anything else
// (embed_tokens, lm_head) is skipped with a warning.
//
// Ported from HOT-Step-CPP (scragnog).

#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "qwen3-enc.h"  // Qwen3Layer + lm_slot_weight (the DoRA norm pass)
#include "qwen3-lora.h"
#include "safetensors.h"
#include "yyjson.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct LMLora {
    struct ggml_context * ctx = nullptr;
    ggml_backend_buffer_t buf = nullptr;
    QwLoraLayer           layers[QWEN3_LORA_MAX_LAYERS];
    int                   n_tensors = 0;
    int                   max_layer = -1;
    std::string           path;
    float                 user_scale = 1.0f;

    // ── DoRA (2026-09-05) ────────────────────────────────────────────────────
    //
    // The magnitudes come off disk into QwLoraPair::m. `nrm` cannot: it is
    // ||W + s*BA||_col, which needs the BASE WEIGHTS this loader has never
    // seen. lm_adapter_dora_prepare() fills it once, after model-store.cpp has
    // both halves, and `dora_pending` is what says the adapter is not usable
    // until it has run — a DoRA pair with m but no nrm would apply nothing and
    // look like a working adapter.
    int  dora_n       = 0;
    bool dora_pending = false;
};

// Map a module substring to a slot. Order matters: check longer names first.
static bool lm_adapter_slot_for(const std::string & name, QwLoraSlot * out) {
    struct {
        const char * pat;
        QwLoraSlot   slot;
    } map[] = {
        { ".self_attn.q_proj.", QW_LORA_Q    },
        { ".self_attn.k_proj.", QW_LORA_K    },
        { ".self_attn.v_proj.", QW_LORA_V    },
        { ".self_attn.o_proj.", QW_LORA_O    },
        { ".mlp.gate_proj.",    QW_LORA_GATE },
        { ".mlp.up_proj.",      QW_LORA_UP   },
        { ".mlp.down_proj.",    QW_LORA_DOWN },
    };

    for (auto & m : map) {
        if (name.find(m.pat) != std::string::npos) {
            *out = m.slot;
            return true;
        }
    }
    return false;
}

// Parse "…model.layers.<N>…" → layer index, or -1.
static int lm_adapter_layer_for(const std::string & name) {
    size_t p = name.find("model.layers.");
    if (p == std::string::npos) {
        return -1;
    }
    p += strlen("model.layers.");
    if (p >= name.size() || name[p] < '0' || name[p] > '9') {
        return -1;
    }
    return atoi(name.c_str() + p);
}

// What adapter_config.json says about a PEFT adapter. `ratio` is -1 when the
// file is missing or unparseable (caller falls back to alpha == r, i.e. 1.0,
// with a warning). peft_type distinguishes LORA / HIRA / LOHA — the DiT's
// exporter and now the LM's write the same key set, so one convention serves
// both models.
struct LmAdapterCfgInfo {
    float       ratio     = -1.0f;
    bool        use_dora  = false;
    std::string peft_type = "LORA";
    /** Whether adapter_config.json existed at all. Everything above is a
     *  DEFAULT when it did not, and "the config says LORA" is a different claim
     *  from "there is no config" the moment the file's own marker is checked
     *  against it. */
    bool        present   = false;
};

static LmAdapterCfgInfo lm_adapter_read_cfg(const std::string & dir) {
    LmAdapterCfgInfo out;
    std::string      cfg_path = dir + "/adapter_config.json";
    yyjson_doc *     doc      = yyjson_read_file(cfg_path.c_str(), 0, NULL, NULL);
    if (!doc) {
        return out;
    }
    yyjson_val * root  = yyjson_doc_get_root(doc);
    double       alpha = 0.0, r = 0.0;
    bool         rslora = false;
    if (root && yyjson_is_obj(root)) {
        yyjson_val * a  = yyjson_obj_get(root, "lora_alpha");
        yyjson_val * rv = yyjson_obj_get(root, "r");
        yyjson_val * rs = yyjson_obj_get(root, "use_rslora");
        yyjson_val * dv = yyjson_obj_get(root, "use_dora");
        yyjson_val * pt = yyjson_obj_get(root, "peft_type");
        if (a && yyjson_is_num(a)) {
            alpha = yyjson_get_num(a);
        }
        if (rv && yyjson_is_num(rv)) {
            r = yyjson_get_num(rv);
        }
        if (rs && yyjson_is_true(rs)) {
            rslora = true;
        }
        if (dv && yyjson_is_true(dv)) {
            out.use_dora = true;
        }
        if (pt && yyjson_is_str(pt)) {
            out.peft_type = yyjson_get_str(pt);
        }
        out.present = true;
    }
    yyjson_doc_free(doc);
    // rsLoRA (use_rslora): the trainer applied alpha/sqrt(r) in-graph, so the
    // runtime must too, or the adapter comes in at sqrt(r) times the wrong
    // strength — 11x at r128 — with nothing in the log to say so.
    if (alpha > 0.0 && r > 0.0) {
        out.ratio = (float) (rslora ? alpha / sqrt(r) : alpha / r);
    }
    return out;
}

// Kept for callers that only want the scale factor.
static float lm_adapter_read_alpha_ratio(const std::string & dir) {
    return lm_adapter_read_cfg(dir).ratio;
}

static void lm_adapter_free(LMLora * l) {
    if (!l) {
        return;
    }
    if (l->buf) {
        ggml_backend_buffer_free(l->buf);
    }
    if (l->ctx) {
        ggml_free(l->ctx);
    }
    delete l;
}

// Load a PEFT LoRA for the LM and stage its tensors on `backend`.
// ─── LoKr support (2026-07-30) ──────────────────────────────────────────────
//
// Reads the LyCORIS layout the LM trainer writes (train/lm-export.h), which is
// byte-layout identical to the DiT's:
//
//   lycoris_layers_<L>_<site>.alpha        [1]
//   lycoris_layers_<L>_<site>.lokr_w1      [out_l, in_m]
//   lycoris_layers_<L>_<site>.lokr_w2      [out_k, in_n]   monolithic
//   lycoris_layers_<L>_<site>.lokr_w2_a    [out_k, dim]    factorized
//   lycoris_layers_<L>_<site>.lokr_w2_b    [dim,   in_n]
//
// NO METADATA IS REQUIRED, deliberately: safetensors.h skips __metadata__, and
// everything needed is derivable from the shapes plus one LyCORIS rule —
// alpha is FORCED to dim when both factors are monolithic, so the scale is
// exactly 1 there, and a factorized site gets dim from w2_a's own ne0.
static bool lm_adapter_lokr_slot(const std::string & site, QwLoraSlot * out) {
    if (site == "self_attn_q_proj") {
        *out = QW_LORA_Q;
        return true;
    }
    if (site == "self_attn_k_proj") {
        *out = QW_LORA_K;
        return true;
    }
    if (site == "self_attn_v_proj") {
        *out = QW_LORA_V;
        return true;
    }
    if (site == "self_attn_o_proj") {
        *out = QW_LORA_O;
        return true;
    }
    if (site == "mlp_gate_proj") {
        *out = QW_LORA_GATE;
        return true;
    }
    if (site == "mlp_up_proj") {
        *out = QW_LORA_UP;
        return true;
    }
    if (site == "mlp_down_proj") {
        *out = QW_LORA_DOWN;
        return true;
    }
    return false;
}

// "lycoris_layers_12_self_attn_q_proj.lokr_w2" -> layer 12, slot Q, "lokr_w2".
// Returns false for anything that is not a LoKr key at all. `site_out` is
// filled even when the slot is unknown, so the caller can tell a DiT adapter
// (which carries cross_attn sites) from a corrupt one.
static bool lm_adapter_lokr_parse(const std::string & name,
                                  int *               layer_out,
                                  QwLoraSlot *        slot_out,
                                  std::string *       suffix_out,
                                  std::string *       site_out) {
    static const std::string PFX = "lycoris_layers_";
    if (name.compare(0, PFX.size(), PFX) != 0) {
        return false;
    }
    size_t i     = PFX.size();
    int    layer = 0;
    if (i >= name.size() || !isdigit((unsigned char) name[i])) {
        return false;
    }
    while (i < name.size() && isdigit((unsigned char) name[i])) {
        layer = layer * 10 + (name[i] - '0');
        i++;
    }
    if (i >= name.size() || name[i] != '_') {
        return false;
    }
    const size_t dot = name.find('.', i);
    if (dot == std::string::npos) {
        return false;
    }
    *site_out   = name.substr(i + 1, dot - i - 1);
    *suffix_out = name.substr(dot + 1);
    *layer_out  = layer;
    return lm_adapter_lokr_slot(*site_out, slot_out);
}

// Read a [1] alpha tensor, whatever dtype it was written in.
static float lm_adapter_read_alpha(const STFile & st, const STEntry & e) {
    const void * d = st_data(st, e);
    if (e.dtype == "F32") {
        float v = 0.0f;
        memcpy(&v, d, sizeof(float));
        return v;
    }
    if (e.dtype == "BF16") {
        uint16_t b = 0;
        memcpy(&b, d, sizeof(uint16_t));
        const uint32_t u = (uint32_t) b << 16;
        float          v = 0.0f;
        memcpy(&v, &u, sizeof(float));
        return v;
    }
    return 0.0f;
}

// Returns nullptr on failure (caller must treat this as a load failure —
// never fall back silently to the base LM under an adapter-bearing key).
// HOT-Step's LM trainer can store a learned artist token and a KV prefix in
// the adapter file (hot_step.artist_token.*, hot_step.prefix.*). This runtime
// does not apply them, and an adapter trained with them does not behave as
// trained without them, so such a file is refused by name.
static bool lm_adapter_has_soft_prompt(const STFile & st) {
    for (const STEntry & e : st.entries) {
        if (e.name.rfind("hot_step.artist_token.", 0) == 0 || e.name.rfind("hot_step.prefix.", 0) == 0) {
            return true;
        }
    }
    return false;
}

static LMLora * lm_adapter_load(const char * path, float user_scale, ggml_backend_t backend) {
    std::string p = path;
    std::string dir;
    std::string sf_path;
    bool        is_file = p.size() > 12 && p.compare(p.size() - 12, 12, ".safetensors") == 0;
    if (is_file) {
        sf_path      = p;
        size_t slash = p.find_last_of("/\\");
        dir          = (slash == std::string::npos) ? "." : p.substr(0, slash);
    } else {
        dir          = p;
        sf_path      = p + "/adapter_model.safetensors";
        // A LoKr adapter dir has no PEFT file at all — the weights live in
        // lokr_weights.safetensors and there is deliberately no
        // adapter_config.json (alpha rides the per-module tensors instead).
        FILE * probe = fopen(sf_path.c_str(), "rb");
        if (probe) {
            fclose(probe);
        } else {
            sf_path = p + "/lokr_weights.safetensors";
        }
    }

    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        fprintf(stderr, "[LM-Adapter] FATAL: cannot open %s\n", sf_path.c_str());
        return nullptr;
    }

    if (lm_adapter_has_soft_prompt(st)) {
        fprintf(stderr,
                "[LM-Adapter] FATAL: %s carries an artist token or KV prefix, which this runtime does not apply\n",
                sf_path.c_str());
        st_close(&st);
        return nullptr;
    }

    const LmAdapterCfgInfo cfg_info       = lm_adapter_read_cfg(dir);
    float                  alpha_ratio    = cfg_info.ratio;
    bool                   ratio_from_cfg = alpha_ratio > 0.0f;

    // ── Parameterizations with no low-rank runtime path ──────────────────
    //
    // HiRA's delta is W (.) (s*BA) and LoHa's is (A1B1) (.) (A2B2). Neither is
    // low-rank, so applying either means materialising a full [in, out] tensor
    // per adapted module PER TOKEN. The DiT refuses both in runtime mode and
    // defers to merge; the AS1.5 planner LM has no merge mode at all, so this
    // is a hard refusal. It is NOT a silent skip: an adapter whose tensors are
    // ignored is the turbo8 no-op, and the cache-key law here already forbids
    // a base-only fallback under an adapter-bearing key.
    {
        bool has_hada = false;
        for (const STEntry & e : st.entries) {
            if (e.name.find(".hada_w") != std::string::npos) {
                has_hada = true;
                break;
            }
        }
        // HiRA has no tensor of its own — it exports ordinary lora_A/lora_B — so
        // a checkpoint that lost its adapter_config.json used to load here as a
        // plain LoRA and apply the wrong delta silently. train/lm-export.h now
        // writes what the file IS as hot_step.param_method (0=lora 1=dora
        // 2=hira 3=loha): read it, refuse a disagreement with the config, and
        // believe it when there is no config. Adapters that predate the marker
        // carry none and behave exactly as before.
        std::string marker;
        {
            const STEntry * mk = st_find(st, "hot_step.param_method");
            if (mk && mk->dtype == "F32" && mk->n_dims == 1 && mk->shape[0] == 1) {
                const int code = (int) *(const float *) st_data(st, *mk);
                if (code == 4) {
                    // PiSSA DELTA form (2026-09-09): lora_A/lora_B are A - A0 and
                    // s(B - B0), meaningless without the residual file, which only
                    // the MM3 loader (minimax/mm3-lm-adapter.h) knows how to pair.
                    // Applying them as a plain LoRA would add (B - B0)(A - A0): a
                    // wrong delta, silently.
                    fprintf(stderr,
                            "[LM-Adapter] FATAL: %s is a PiSSA delta adapter (adapter-only, needs its residual file).\n"
                            "             This loader cannot apply it; it was trained for the MM3 LM.\n",
                            sf_path.c_str());
                    st_close(&st);
                    return nullptr;
                }
                marker = code == 1 ? "dora" : code == 2 ? "hira" : code == 3 ? "loha" : "lora";
            }
        }
        if (!marker.empty() && cfg_info.present) {
            const std::string from_cfg = cfg_info.peft_type == "HIRA" ? "hira" :
                                         cfg_info.peft_type == "LOHA" ? "loha" :
                                         cfg_info.use_dora            ? "dora" :
                                                                        "lora";
            if (marker != from_cfg) {
                fprintf(stderr,
                        "[LM-Adapter] FATAL: %s carries a %s marker but adapter_config.json describes %s.\n"
                        "             Refusing rather than guessing which one the weights were trained as.\n",
                        sf_path.c_str(), marker.c_str(), from_cfg.c_str());
                st_close(&st);
                return nullptr;
            }
        }
        if (has_hada || marker == "loha" || marker == "hira" || cfg_info.peft_type == "LOHA" ||
            cfg_info.peft_type == "HIRA") {
            const std::string kind = has_hada                     ? "LoHa" :
                                     cfg_info.peft_type != "LORA" ? cfg_info.peft_type :
                                     marker == "hira"             ? "HiRA" :
                                                                    "LoHa";
            fprintf(stderr,
                    "[LM-Adapter] FATAL: %s is a %s adapter. Its delta is not low-rank, so the planner LM's\n"
                    "             runtime path cannot apply it, and this model has no merge mode to fall back\n"
                    "             on. Train --dora or --rslora instead, or use the MM3 LM (merge mode).\n",
                    sf_path.c_str(), kind.c_str());
            st_close(&st);
            return nullptr;
        }
    }

    LMLora * l    = new LMLora();
    l->path       = p;
    l->user_scale = user_scale;

    // ── LoKr? ────────────────────────────────────────────────────────────
    int  skipped  = 0;
    bool has_lokr = false, has_cross_attn = false;
    for (const STEntry & e : st.entries) {
        if (e.name.find(".lokr_w1") != std::string::npos) {
            has_lokr = true;
        }
        if (e.name.find("cross_attn") != std::string::npos) {
            has_cross_attn = true;
        }
    }
    if (has_lokr) {
        // A DiT LoKr uses the SAME lycoris_layers_<L>_<site> stems and differs
        // only in its site set, so without this a DiT adapter would load into
        // the LM's slots and quietly compute nonsense. cross_attn exists only on
        // the DiT.
        if (has_cross_attn) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s carries cross_attn sites - that is a DiT LoKr, not an LM one\n",
                    sf_path.c_str());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }

        struct LkPending {
            const STEntry * e;
            ggml_tensor **  dst;
        };

        std::vector<LkPending> lkp;
        std::vector<float>     alphas((size_t) QWEN3_LORA_MAX_LAYERS * QW_LORA_NSLOTS, -1.0f);
        int                    n_ten = 0;
        for (const STEntry & e : st.entries) {
            int         layer = -1;
            QwLoraSlot  slot;
            std::string sfx, site;
            if (lm_adapter_lokr_parse(e.name, &layer, &slot, &sfx, &site) && layer >= 0 &&
                layer < QWEN3_LORA_MAX_LAYERS && sfx != "alpha") {
                n_ten++;
            }
        }
        struct ggml_init_params gp2 = { ggml_tensor_overhead() * (size_t) (n_ten + 8), NULL, true };
        l->ctx                      = ggml_init(gp2);

        for (const STEntry & e : st.entries) {
            int         layer = -1;
            QwLoraSlot  slot;
            std::string sfx, site;
            if (!lm_adapter_lokr_parse(e.name, &layer, &slot, &sfx, &site)) {
                skipped++;
                continue;
            }
            if (layer < 0 || layer >= QWEN3_LORA_MAX_LAYERS) {
                skipped++;
                continue;
            }
            QwLoraPair & pr = l->layers[layer].p[slot];
            if (sfx == "alpha") {
                alphas[(size_t) layer * QW_LORA_NSLOTS + (size_t) slot] = lm_adapter_read_alpha(st, e);
                continue;
            }
            if (e.n_dims != 2) {
                fprintf(stderr, "[LM-Adapter] FATAL: %s has %d dims (want 2)\n", e.name.c_str(), e.n_dims);
                st_close(&st);
                lm_adapter_free(l);
                return nullptr;
            }
            const ggml_type ty = st_ggml_type(e);
            if (ty == GGML_TYPE_COUNT) {
                fprintf(stderr, "[LM-Adapter] FATAL: %s has unsupported dtype %s\n", e.name.c_str(), e.dtype.c_str());
                st_close(&st);
                lm_adapter_free(l);
                return nullptr;
            }
            // torch [rows, cols] row-major -> ggml ne0=cols, ne1=rows
            ggml_tensor *  t   = ggml_new_tensor_2d(l->ctx, ty, e.shape[1], e.shape[0]);
            ggml_tensor ** dst = nullptr;
            if (sfx == "lokr_w1") {
                dst = &pr.w1;
            } else if (sfx == "lokr_w2") {
                dst = &pr.w2;
            } else if (sfx == "lokr_w2_a") {
                dst = &pr.w2_a;
            } else if (sfx == "lokr_w2_b") {
                dst = &pr.w2_b;
            }
            if (!dst) {
                skipped++;
                continue;
            }
            *dst = t;
            if (layer > l->max_layer) {
                l->max_layer = layer;
            }
            lkp.push_back({ &e, dst });
        }

        l->buf = ggml_backend_alloc_ctx_tensors(l->ctx, backend);
        if (!l->buf) {
            fprintf(stderr, "[LM-Adapter] FATAL: backend alloc failed for %d LoKr tensors\n", (int) lkp.size());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        for (auto & pd : lkp) {
            ggml_backend_tensor_set(*pd.dst, st_data(st, *pd.e), 0, ggml_nbytes(*pd.dst));
        }
        l->n_tensors = (int) lkp.size();
        st_close(&st);

        int sites = 0, mono = 0, fact = 0;
        for (int i = 0; i <= l->max_layer; i++) {
            for (int sl = 0; sl < QW_LORA_NSLOTS; sl++) {
                QwLoraPair & pr = l->layers[i].p[sl];
                if (!pr.w1) {
                    continue;
                }
                if (!pr.w2 && !(pr.w2_a && pr.w2_b)) {
                    fprintf(stderr, "[LM-Adapter] FATAL: layer %d slot %d has lokr_w1 but no usable w2\n", i, sl);
                    lm_adapter_free(l);
                    return nullptr;
                }
                // Factor dims come straight from the tensors: w1 is ggml
                // [in_m, out_l], w2 [in_n, out_k], w2_a [dim, out_k],
                // w2_b [in_n, dim].
                pr.in_m  = pr.w1->ne[0];
                pr.out_l = pr.w1->ne[1];
                if (pr.w2) {
                    pr.in_n       = pr.w2->ne[0];
                    pr.out_k      = pr.w2->ne[1];
                    // LyCORIS forces alpha == dim when both factors are
                    // monolithic, so the scale is exactly 1 and no metadata is
                    // needed to recover it.
                    pr.lokr_scale = 1.0f * user_scale;
                    mono++;
                } else {
                    pr.in_n         = pr.w2_b->ne[0];
                    pr.out_k        = pr.w2_a->ne[1];
                    const float dim = (float) pr.w2_a->ne[0];
                    const float a   = alphas[(size_t) i * QW_LORA_NSLOTS + (size_t) sl];
                    pr.lokr_scale   = ((a > 0.0f && dim > 0.0f) ? (a / dim) : 1.0f) * user_scale;
                    fact++;
                }
                sites++;
            }
        }
        if (sites == 0) {
            fprintf(stderr, "[LM-Adapter] FATAL: no usable LoKr sites in %s\n", sf_path.c_str());
            lm_adapter_free(l);
            return nullptr;
        }
        fprintf(stderr,
                "[LM-Adapter] Loaded %s: LoKr, %d sites across %d layers (%d monolithic / %d factorized), "
                "user scale=%.2f%s\n",
                p.c_str(), sites, l->max_layer + 1, mono, fact, user_scale,
                skipped ? " (some non-LoKr tensors skipped)" : "");
        return l;
    }

    // Pass 1: count usable tensors. A DoRA magnitude costs TWO context slots:
    // the m it loads plus the nrm lm_adapter_dora_prepare fills later.
    int usable = 0, n_mag = 0;
    for (const STEntry & e : st.entries) {
        QwLoraSlot slot;
        int        layer = lm_adapter_layer_for(e.name);
        bool       is_ab = e.name.find(".lora_A.") != std::string::npos || e.name.find(".lora_B.") != std::string::npos;
        bool       is_mag = e.name.find(".lora_magnitude_vector.") != std::string::npos;
        if (layer >= 0 && layer < QWEN3_LORA_MAX_LAYERS && (is_ab || is_mag) && lm_adapter_slot_for(e.name, &slot)) {
            usable++;
            if (is_mag) {
                n_mag++;
            }
        } else {
            skipped++;
        }
    }
    if (usable == 0) {
        fprintf(stderr, "[LM-Adapter] FATAL: no usable lora_A/lora_B tensors in %s (%d entries)\n", sf_path.c_str(),
                (int) st.entries.size());
        st_close(&st);
        lm_adapter_free(l);
        return nullptr;
    }

    struct ggml_init_params gp = { ggml_tensor_overhead() * (size_t) (usable + n_mag + 8), NULL, true };
    l->ctx                     = ggml_init(gp);

    // Pass 2: create tensors
    struct Pending {
        const STEntry *      e;
        struct ggml_tensor * t;
    };

    std::vector<Pending> pending;
    pending.reserve((size_t) usable);
    for (const STEntry & e : st.entries) {
        QwLoraSlot slot;
        int        layer = lm_adapter_layer_for(e.name);
        bool       is_a  = e.name.find(".lora_A.") != std::string::npos;
        bool       is_b  = e.name.find(".lora_B.") != std::string::npos;
        bool       is_m  = e.name.find(".lora_magnitude_vector.") != std::string::npos;
        if (layer < 0 || layer >= QWEN3_LORA_MAX_LAYERS || (!is_a && !is_b && !is_m) ||
            !lm_adapter_slot_for(e.name, &slot)) {
            continue;
        }
        // PEFT writes the magnitude as [out]; a 2-D [out,1] / [1,out] form is
        // accepted too, since the element count is what matters.
        const int want_dims = is_m ? 0 : 2;
        if (!is_m && e.n_dims != want_dims) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s has %d dims (want 2)\n", e.name.c_str(), e.n_dims);
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        ggml_type ty = st_ggml_type(e);
        if (ty == GGML_TYPE_COUNT) {
            fprintf(stderr, "[LM-Adapter] FATAL: %s has unsupported dtype %s\n", e.name.c_str(), e.dtype.c_str());
            st_close(&st);
            lm_adapter_free(l);
            return nullptr;
        }
        QwLoraPair &         pair = l->layers[layer].p[slot];
        struct ggml_tensor * t    = nullptr;
        if (is_m) {
            int64_t n = 1;
            for (int d = 0; d < e.n_dims; d++) {
                n *= e.shape[d];
            }
            // F32 regardless of the file's dtype: nrm is computed in F32 and the
            // two are divided elementwise in-graph, so keeping them the same
            // type removes a cast node and a rounding question.
            t        = ggml_new_tensor_1d(l->ctx, GGML_TYPE_F32, n);
            pair.m   = t;
            pair.nrm = ggml_new_tensor_1d(l->ctx, GGML_TYPE_F32, n);
            l->dora_n++;
        } else {
            // torch [rows, cols] row-major -> ggml ne0=cols, ne1=rows
            t = ggml_new_tensor_2d(l->ctx, ty, e.shape[1], e.shape[0]);
            if (is_a) {
                pair.A = t;
            } else {
                pair.B = t;
            }
        }
        if (layer > l->max_layer) {
            l->max_layer = layer;
        }
        pending.push_back({ &e, t });
    }

    l->buf = ggml_backend_alloc_ctx_tensors(l->ctx, backend);
    if (!l->buf) {
        fprintf(stderr, "[LM-Adapter] FATAL: backend alloc failed for %d tensors\n", (int) pending.size());
        st_close(&st);
        lm_adapter_free(l);
        return nullptr;
    }
    for (auto & pd : pending) {
        // The magnitude vector is forced to F32 above, so a BF16/F16 file needs
        // widening rather than a raw copy — the byte counts do not match.
        if (pd.t->type == GGML_TYPE_F32 && pd.e->dtype != "F32") {
            const int64_t      n = ggml_nelements(pd.t);
            std::vector<float> f((size_t) n);
            if (pd.e->dtype == "BF16") {
                const uint16_t * u = (const uint16_t *) st_data(st, *pd.e);
                for (int64_t i = 0; i < n; i++) {
                    const uint32_t bits = (uint32_t) u[i] << 16;
                    memcpy(&f[(size_t) i], &bits, 4);
                }
            } else if (pd.e->dtype == "F16") {
                ggml_fp16_to_fp32_row((const ggml_fp16_t *) st_data(st, *pd.e), f.data(), n);
            } else {
                fprintf(stderr, "[LM-Adapter] FATAL: %s has unsupported dtype %s for a magnitude vector\n",
                        pd.e->name.c_str(), pd.e->dtype.c_str());
                st_close(&st);
                lm_adapter_free(l);
                return nullptr;
            }
            ggml_backend_tensor_set(pd.t, f.data(), 0, (size_t) n * sizeof(float));
            continue;
        }
        ggml_backend_tensor_set(pd.t, st_data(st, *pd.e), 0, ggml_nbytes(pd.t));
    }
    l->n_tensors = (int) pending.size();
    st_close(&st);

    // Finalize pairs: both halves present, per-pair scale = ratio * user_scale
    // (ratio from adapter_config.json, else alpha=r fallback -> 1.0).
    if (!ratio_from_cfg) {
        fprintf(stderr,
                "[LM-Adapter] WARNING: no adapter_config.json next to %s — "
                "assuming lora_alpha == r (scale factor 1.0)\n",
                sf_path.c_str());
        alpha_ratio = 1.0f;
    }
    int pairs = 0;
    for (int i = 0; i <= l->max_layer; i++) {
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = l->layers[i].p[s];
            if ((pr.A == nullptr) != (pr.B == nullptr)) {
                fprintf(stderr, "[LM-Adapter] FATAL: layer %d slot %d has only one of lora_A/lora_B\n", i, s);
                lm_adapter_free(l);
                return nullptr;
            }
            if (pr.m && !pr.A) {
                fprintf(stderr, "[LM-Adapter] FATAL: layer %d slot %d has a DoRA magnitude but no lora_A/lora_B\n", i,
                        s);
                lm_adapter_free(l);
                return nullptr;
            }
            if (pr.m && pr.m->ne[0] != pr.B->ne[1]) {
                fprintf(stderr,
                        "[LM-Adapter] FATAL: layer %d slot %d magnitude is %lld long but lora_B has %lld outputs\n", i,
                        s, (long long) pr.m->ne[0], (long long) pr.B->ne[1]);
                lm_adapter_free(l);
                return nullptr;
            }
            if (pr.A) {
                pr.scale = alpha_ratio * user_scale;
                pairs++;
            }
        }
    }

    // use_dora in adapter_config.json and the tensors on disk must agree. Either
    // one alone means the file was written by something that half-understood
    // DoRA, and guessing which half is right is how an adapter silently renders
    // at the wrong strength.
    if (cfg_info.use_dora != (l->dora_n > 0)) {
        fprintf(stderr,
                "[LM-Adapter] FATAL: adapter_config.json says use_dora=%s but the file carries %d "
                "lora_magnitude_vector tensor(s)\n",
                cfg_info.use_dora ? "true" : "false", l->dora_n);
        lm_adapter_free(l);
        return nullptr;
    }
    // The norms need the base weights, which this loader does not have. Until
    // lm_adapter_dora_prepare() runs, every DoRA pair carries an UNINITIALISED
    // nrm and must not reach a graph.
    l->dora_pending = l->dora_n > 0;

    fprintf(stderr, "[LM-Adapter] Loaded %s: %d pairs across %d layers, alpha/r=%.3f, user scale=%.2f%s%s\n", p.c_str(),
            pairs, l->max_layer + 1, alpha_ratio, user_scale, l->dora_n ? " (DoRA)" : "",
            skipped ? " (some non-projection tensors skipped)" : "");
    return l;
}

// ─── DoRA: the one-time norm pass (2026-09-05) ──────────────────────────────
//
// nrm = ||W + s*BA||_col for every DoRA site. The trainer refreshes this once
// per optimizer window because A/B move; at inference W, A and B are ALL
// FROZEN, so it is exact after one pass and reused for every token. That is
// what makes DoRA correct in the unmerged runtime path, where the DiT's
// adapter-runtime.h only warns and defers to merge mode.
//
// Runs on the model's own backend with a plain gallocr — the LoRA tensors were
// staged on that same backend by lm_adapter_load, so there is nothing to
// schedule across devices. One graph per layer, for the same node-budget reason
// DitAdapterLora::preWindow gives.
static bool lm_adapter_dora_prepare(LMLora * l, Qwen3Layer * layers, ggml_backend_t backend) {
    if (!l || l->dora_n == 0) {
        return true;
    }
    const size_t         n_nodes = (size_t) QW_LORA_NSLOTS * 12 + 64;
    const size_t         need    = ggml_tensor_overhead() * n_nodes + ggml_graph_overhead_custom(n_nodes, false);
    std::vector<uint8_t> arena(need);
    ggml_gallocr_t       ga = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ga) {
        fprintf(stderr, "[LM-Adapter] FATAL: DoRA norm pass cannot allocate\n");
        return false;
    }
    bool ok = true;
    for (int layer = 0; layer <= l->max_layer && ok; layer++) {
        ggml_init_params ip  = { arena.size(), arena.data(), true };
        ggml_context *   ctx = ggml_init(ip);
        if (!ctx) {
            ok = false;
            break;
        }
        ggml_cgraph * gf   = ggml_new_graph_custom(ctx, n_nodes, false);
        int           want = 0;
        for (int s = 0; s < QW_LORA_NSLOTS; s++) {
            QwLoraPair & pr = l->layers[layer].p[s];
            if (!pr.m || !pr.nrm || !pr.A || !pr.B) {
                continue;
            }
            ggml_tensor * w = lm_slot_weight(&layers[layer], s);
            if (!w) {
                fprintf(stderr, "[LM-Adapter] FATAL: DoRA site layer %d slot %d has no base weight\n", layer, s);
                ok = false;
                break;
            }
            // ggml_cast dequantizes: the base LM here is routinely Q8_0/Q5_K,
            // which is the whole reason the runtime path exists.
            ggml_tensor * wf    = (w->type == GGML_TYPE_F32) ? w : ggml_cast(ctx, w, GGML_TYPE_F32);
            ggml_tensor * a32   = (pr.A->type == GGML_TYPE_F32) ? pr.A : ggml_cast(ctx, pr.A, GGML_TYPE_F32);
            ggml_tensor * b32   = (pr.B->type == GGML_TYPE_F32) ? pr.B : ggml_cast(ctx, pr.B, GGML_TYPE_F32);
            // delta[in, out] = A[in, r] . B[r, out]: mul_mat contracts ne0, so
            // the left operand is A transposed to [r, in].
            ggml_tensor * delta = ggml_mul_mat(ctx, ggml_cont(ctx, ggml_transpose(ctx, a32)), b32);
            // pr.scale already carries user_scale. The norm has to describe the
            // delta this render will actually apply, so the dial belongs in it.
            ggml_tensor * wd    = ggml_add(ctx, wf, ggml_scale(ctx, delta, pr.scale));
            ggml_tensor * nr    = ggml_sqrt(ctx, ggml_sum_rows(ctx, ggml_sqr(ctx, wd)));  // [1, out]
            ggml_build_forward_expand(gf, ggml_cpy(ctx, nr, ggml_reshape_2d(ctx, pr.nrm, 1, pr.nrm->ne[0])));
            want++;
        }
        if (ok && want > 0) {
            ok = ggml_gallocr_alloc_graph(ga, gf) && ggml_backend_graph_compute(backend, gf) == GGML_STATUS_SUCCESS;
            if (!ok) {
                fprintf(stderr, "[LM-Adapter] FATAL: DoRA norm pass failed at layer %d\n", layer);
            }
        }
        ggml_free(ctx);
    }
    ggml_gallocr_free(ga);
    if (ok) {
        l->dora_pending = false;
        fprintf(stderr, "[LM-Adapter] DoRA: ||W + s*BA||_col computed for %d site(s)\n", l->dora_n);
    }
    return ok;
}
