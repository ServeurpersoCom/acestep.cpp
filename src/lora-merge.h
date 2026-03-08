#pragma once
// lora-merge.h: runtime LoRA merge into GGUF weights before QKV fusion
//
// Merges (alpha / rank) * scale * B @ A deltas into base weights in place.
// Called after individual GGUF projection tensors are loaded into WeightCtx
// but BEFORE wctx_alloc uploads to GPU and BEFORE QKV fusion concatenation.
//
// Each projection (q_proj, k_proj, v_proj, o_proj) has its own PendingCopy
// even when destined for a fused QKV tensor. We patch each one separately,
// so fusion proceeds normally on already merged data.

#include "ggml.h"
#include "gguf-weights.h"
#include "safetensors.h"
#include "weight-ctx.h"

#include <sys/stat.h>
#ifdef _WIN32
#    ifndef S_ISDIR
#        define S_ISDIR(m) (((m) & _S_IFMT) == _S_IFDIR)
#    endif
#endif

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Convert safetensors tensor data to F32 based on dtype string.
// Handles "F32", "BF16", "F16". Returns false for unknown dtypes.
static bool lora_to_f32(const void * src, float * dst, int64_t n, const std::string & dtype) {
    if (dtype == "F32") {
        memcpy(dst, src, (size_t) n * sizeof(float));
    } else if (dtype == "BF16") {
        ggml_bf16_to_fp32_row((const ggml_bf16_t *) src, dst, n);
    } else if (dtype == "F16") {
        ggml_fp16_to_fp32_row((const ggml_fp16_t *) src, dst, n);
    } else {
        return false;
    }
    return true;
}

// Map a PEFT LoRA safetensors key to the GGUF base tensor name.
//
// PEFT keys look like:
//   base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
//   base_model.model.layers.0.self_attn.q_proj.lora_A.weight
//   layers.0.self_attn.q_proj.lora_B.weight   (some exports omit prefix)
//
// GGUF names look like:
//   decoder.layers.0.self_attn.q_proj.weight
//
// Steps: strip "base_model.model.", extract module path before ".lora_",
// prepend "decoder." if needed, append ".weight".
static std::string lora_base_name(const std::string & key) {
    std::string s = key;

    // strip PEFT wrapper prefix
    const char * pfx     = "base_model.model.";
    size_t       pfx_len = strlen(pfx);
    if (s.compare(0, pfx_len, pfx) == 0) {
        s = s.substr(pfx_len);
    }

    // everything before ".lora_" is the module path
    size_t pos = s.find(".lora_");
    if (pos == std::string::npos) {
        return "";
    }
    s = s.substr(0, pos);

    // ensure decoder prefix (PEFT wraps the decoder directly,
    // so the internal path starts at "layers." not "decoder.layers.")
    if (s.compare(0, 8, "decoder.") != 0) {
        s = "decoder." + s;
    }

    return s + ".weight";
}

// Check whether a safetensors key is a lora_A or lora_B weight
static bool lora_is_a(const std::string & key) {
    return key.find(".lora_A.") != std::string::npos;
}

static bool lora_is_b(const std::string & key) {
    return key.find(".lora_B.") != std::string::npos;
}

// Read adapter_config.json for alpha. Returns alpha or 0 if not found.
// Rank is always read from the actual tensor shapes (more reliable).
static int lora_read_alpha(const char * dir) {
    std::string path = std::string(dir) + "/adapter_config.json";

    FILE * f = fopen(path.c_str(), "rb");
    if (!f) {
        return 0;
    }

    fseek(f, 0, SEEK_END);
    long len = ftell(f);
    fseek(f, 0, SEEK_SET);

    std::vector<char> buf((size_t) len + 1);
    fread(buf.data(), 1, (size_t) len, f);
    buf[(size_t) len] = '\0';
    fclose(f);

    const char * json  = buf.data();
    int          alpha = 0;

    // look for "lora_alpha": <int>
    const char * p = strstr(json, "\"lora_alpha\"");
    if (p) {
        p = strchr(p + 12, ':');
        if (p) {
            alpha = atoi(p + 1);
        }
    }
    // fallback: try "alpha": <int> (some configs use this)
    if (alpha == 0) {
        p = strstr(json, "\"alpha\"");
        if (p) {
            p = strchr(p + 7, ':');
            if (p) {
                alpha = atoi(p + 1);
            }
        }
    }

    if (alpha > 0) {
        fprintf(stderr, "[LoRA] adapter_config.json: alpha=%d\n", alpha);
    }
    return alpha;
}

// Dequant a GGUF tensor buffer to F32 using ggml type traits.
// Works for all types: F32, BF16, F16, Q4_K, Q8_0, etc.
static void lora_dequant(const void * src, float * dst, int64_t nel, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        memcpy(dst, src, (size_t) nel * sizeof(float));
        return;
    }
    const struct ggml_type_traits * traits = ggml_get_type_traits(type);
    if (traits->to_float) {
        traits->to_float(src, dst, nel);
    } else {
        fprintf(stderr, "[LoRA] WARNING: no dequant for type %d, zeroing\n", type);
        memset(dst, 0, (size_t) nel * sizeof(float));
    }
}

// Requant F32 data back to original type. Writes into dst buffer.
// Returns the number of bytes written.
static size_t lora_requant(const float * src, void * dst, int64_t nel, int64_t n_per_row, enum ggml_type type) {
    if (type == GGML_TYPE_F32) {
        size_t nb = (size_t) nel * sizeof(float);
        memcpy(dst, src, nb);
        return nb;
    }

    const struct ggml_type_traits * traits = ggml_get_type_traits(type);

    if (traits->is_quantized) {
        // quantized types: use ggml_quantize_chunk (handles block alignment)
        int64_t nrows = nel / n_per_row;
        size_t  qsize = ggml_row_size(type, n_per_row) * (size_t) nrows;
        ggml_quantize_chunk(type, src, dst, 0, nrows, n_per_row, NULL);
        return qsize;
    }

    // non quantized (BF16, F16): use from_float_ref
    if (traits->from_float_ref) {
        size_t nb = (size_t) nel * traits->type_size;
        traits->from_float_ref(src, dst, nel);
        return nb;
    }

    fprintf(stderr, "[LoRA] WARNING: no requant for type %d\n", type);
    return 0;
}

// Main LoRA merge function.
//
// Call after all GGUF tensors are loaded into wctx->pending but before wctx_alloc.
// For each LoRA pair found in the safetensors file:
//   1. Map PEFT key to GGUF tensor name
//   2. Find the matching PendingCopy by comparing src pointers
//   3. Dequant base weight to F32
//   4. Add (alpha/rank) * scale * B @ A
//   5. Requant back to original type into a staging buffer
//   6. Patch PendingCopy.src to point to the merged data
//
// The lora_path can be a .safetensors file or a directory containing
// adapter_model.safetensors and adapter_config.json.
static bool lora_merge(WeightCtx * wctx, const GGUFModel & gf, const char * lora_path, float scale) {
    // resolve paths: if lora_path is a directory, look for adapter_model.safetensors
    std::string sf_path = lora_path;
    std::string dir     = lora_path;

    struct stat sb;
    if (stat(lora_path, &sb) == 0 && S_ISDIR(sb.st_mode)) {
        sf_path = std::string(lora_path) + "/adapter_model.safetensors";
    } else {
        size_t sep = dir.find_last_of("/\\");
        dir        = (sep != std::string::npos) ? dir.substr(0, sep) : ".";
    }

    // open safetensors
    STFile st = {};
    if (!st_open(&st, sf_path.c_str())) {
        return false;
    }

    // read alpha from adapter_config.json (0 means not found)
    int alpha_cfg = lora_read_alpha(dir.c_str());

    // group lora_A and lora_B entries by their GGUF base tensor name
    std::map<std::string, const STEntry *> a_map, b_map;
    for (const auto & e : st.entries) {
        std::string base = lora_base_name(e.name);
        if (base.empty()) {
            continue;
        }
        if (lora_is_a(e.name)) {
            a_map[base] = &e;
        } else if (lora_is_b(e.name)) {
            b_map[base] = &e;
        }
    }

    int merged  = 0;
    int skipped = 0;

    for (const auto & kv : a_map) {
        const std::string & gguf_name = kv.first;
        const STEntry *     ea        = kv.second;

        // find matching lora_B
        auto it = b_map.find(gguf_name);
        if (it == b_map.end()) {
            fprintf(stderr, "[LoRA] WARNING: no lora_B for %s, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        const STEntry * eb = it->second;

        // look up the GGUF tensor to get its type and shape
        int64_t tidx = gguf_find_tensor(gf.gguf, gguf_name.c_str());
        if (tidx < 0) {
            fprintf(stderr, "[LoRA] WARNING: tensor %s not in GGUF, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        struct ggml_tensor * tmeta = ggml_get_tensor(gf.meta, gguf_name.c_str());
        enum ggml_type       ttype = tmeta->type;
        int64_t              ne0   = tmeta->ne[0];  // in_features (contiguous dim)
        int64_t              ne1   = tmeta->ne[1];  // out_features
        int64_t              nel   = ne0 * ne1;

        // find the PendingCopy whose src matches this tensor's mmap data
        size_t       toff     = gguf_get_tensor_offset(gf.gguf, tidx);
        const void * base_ptr = gf.mapping + gf.data_offset + toff;

        WeightCtx::PendingCopy * pc = nullptr;
        for (auto & p : wctx->pending) {
            if (p.src == base_ptr) {
                pc = &p;
                break;
            }
        }
        if (!pc) {
            fprintf(stderr, "[LoRA] WARNING: no PendingCopy for %s, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }

        // LoRA shapes (safetensors/PyTorch convention, row major):
        //   A: [rank, in_features]  shape[0]=rank, shape[1]=in_features
        //   B: [out_features, rank] shape[0]=out_features, shape[1]=rank
        int64_t rank     = ea->shape[0];
        int64_t in_feat  = ea->shape[1];
        int64_t out_feat = eb->shape[0];

        // sanity checks
        if (eb->shape[1] != rank) {
            fprintf(stderr, "[LoRA] WARNING: rank mismatch A=%lld vs B=%lld for %s\n", (long long) rank,
                    (long long) eb->shape[1], gguf_name.c_str());
            skipped++;
            continue;
        }
        if (in_feat != ne0 || out_feat != ne1) {
            fprintf(stderr, "[LoRA] WARNING: shape mismatch for %s: LoRA [%lld,%lld] vs GGUF [%lld,%lld]\n",
                    gguf_name.c_str(), (long long) out_feat, (long long) in_feat, (long long) ne1, (long long) ne0);
            skipped++;
            continue;
        }

        // alpha: prefer config, fallback to rank (scaling = 1.0)
        int   alpha   = (alpha_cfg > 0) ? alpha_cfg : (int) rank;
        float scaling = ((float) alpha / (float) rank) * scale;

        // convert LoRA A and B to F32
        int64_t            a_nel = rank * in_feat;
        int64_t            b_nel = out_feat * rank;
        std::vector<float> a_f32((size_t) a_nel);
        std::vector<float> b_f32((size_t) b_nel);

        if (!lora_to_f32(st_data(st, *ea), a_f32.data(), a_nel, ea->dtype)) {
            fprintf(stderr, "[LoRA] WARNING: unsupported dtype %s for lora_A\n", ea->dtype.c_str());
            skipped++;
            continue;
        }
        if (!lora_to_f32(st_data(st, *eb), b_f32.data(), b_nel, eb->dtype)) {
            fprintf(stderr, "[LoRA] WARNING: unsupported dtype %s for lora_B\n", eb->dtype.c_str());
            skipped++;
            continue;
        }

        // PEFT casts LoRA weights to the model dtype (typically BF16) before
        // computing the delta. We must do the same BF16 round trip so the
        // matmul B@A matches PEFT's merge_and_unload exactly. Do this for all
        // base types because the LoRA was always trained against BF16 weights.
        {
            std::vector<ggml_bf16_t> tmp_a((size_t) a_nel);
            std::vector<ggml_bf16_t> tmp_b((size_t) b_nel);
            ggml_fp32_to_bf16_row_ref(a_f32.data(), tmp_a.data(), a_nel);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) tmp_a.data(), a_f32.data(), a_nel);
            ggml_fp32_to_bf16_row_ref(b_f32.data(), tmp_b.data(), b_nel);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) tmp_b.data(), b_f32.data(), b_nel);
        }

        // dequant base weight to F32
        std::vector<float> w_f32((size_t) nel);
        lora_dequant(base_ptr, w_f32.data(), nel, ttype);

        // merge: W += scaling * B @ A
        // A is [rank, in_feat], B is [out_feat, rank], W is [out_feat, in_feat]
        // Compute delta in F32 then round to BF16 to match PEFT's GPU behavior
        // (BF16 matmul output -> BF16 += BF16). Without this intermediate rounding,
        // the diffusion model diverges because it's extremely sensitive to weight values.
        std::vector<float> delta((size_t) nel, 0.0f);
        for (int64_t i = 0; i < out_feat; i++) {
            for (int64_t k = 0; k < rank; k++) {
                float b_ik = b_f32[(size_t) (i * rank + k)] * scaling;
                for (int64_t j = 0; j < in_feat; j++) {
                    delta[(size_t) (i * in_feat + j)] += b_ik * a_f32[(size_t) (k * in_feat + j)];
                }
            }
        }

        // round delta through BF16 to match PEFT's intermediate precision
        {
            std::vector<ggml_bf16_t> delta_bf16((size_t) nel);
            ggml_fp32_to_bf16_row_ref(delta.data(), delta_bf16.data(), nel);
            ggml_bf16_to_fp32_row((const ggml_bf16_t *) delta_bf16.data(), delta.data(), nel);
        }

        // add rounded delta to base weight
        for (size_t i = 0; i < (size_t) nel; i++) {
            w_f32[i] += delta[i];
        }

        // requant back to original type into a staging buffer.
        // we stash the buffer in wctx->staging so it stays alive until wctx_alloc.
        // for quantized types the output is smaller than F32, but we allocate as
        // float vector to reuse the existing staging infrastructure.
        size_t max_bytes = (size_t) nel * sizeof(float);  // upper bound
        size_t n_floats  = (max_bytes + sizeof(float) - 1) / sizeof(float);
        wctx->staging.emplace_back(n_floats);
        void * merged_buf = wctx->staging.back().data();

        size_t merged_bytes = lora_requant(w_f32.data(), merged_buf, nel, ne0, ttype);
        if (merged_bytes == 0) {
            skipped++;
            continue;
        }

        // patch the PendingCopy to use our merged data instead of the mmap
        pc->src    = merged_buf;
        pc->nbytes = merged_bytes;
        merged++;
    }

    st_close(&st);
    fprintf(stderr, "[LoRA] Merged %d pairs (skipped %d), scale=%.2f\n", merged, skipped, scale);
    return merged > 0;
}
