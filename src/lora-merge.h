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
//
// Performance: the matmul B @ A dispatches to the best available backend
// via ggml_backend_graph_compute. On CUDA this uses cuBLAS, on CPU it uses
// ggml's threaded SIMD kernels. PendingCopy lookup is O(1) via hashmap.
// Base weight dequant happens row by row to halve peak memory.

#include "ggml-alloc.h"
#include "ggml-backend.h"
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
#include <unordered_map>
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

// Map a LoRA safetensors key to the GGUF base tensor name.
//
// Supported key formats (all map to GGUF "decoder.layers.0.self_attn.q_proj.weight"):
//
//   PEFT adapter_model.safetensors:
//     base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight
//     base_model.model.layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (no prefix):
//     layers.0.self_attn.q_proj.lora_A.weight
//
//   ComfyUI single-file (diffusion_model prefix):
//     diffusion_model.layers.0.self_attn.q_proj.lora_A.weight
//
// Steps: strip known prefix, extract module path before ".lora_",
// prepend "decoder." if needed, append ".weight".
static std::string lora_base_name(const std::string & key) {
    std::string s = key;

    // strip known prefixes (PEFT, ComfyUI)
    static const char * prefixes[] = {
        "base_model.model.",  // PEFT
        "diffusion_model.",   // ComfyUI official ACE-Step format
    };
    for (const char * pfx : prefixes) {
        size_t pfx_len = strlen(pfx);
        if (s.compare(0, pfx_len, pfx) == 0) {
            s = s.substr(pfx_len);
            break;
        }
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

// Check whether a safetensors key is a lora_A/down or lora_B/up weight.
// PEFT uses .lora_A. / .lora_B., ComfyUI single-file uses .lora_down. / .lora_up.
static bool lora_is_a(const std::string & key) {
    return key.find(".lora_A.") != std::string::npos || key.find(".lora_down.") != std::string::npos;
}

static bool lora_is_b(const std::string & key) {
    return key.find(".lora_B.") != std::string::npos || key.find(".lora_up.") != std::string::npos;
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

// Round F32 data through BF16 in place to match PEFT's intermediate precision.
// Processes in fixed chunks to avoid large stack allocations.
static void lora_bf16_round(float * data, int64_t n) {
    const int64_t chunk = 4096;
    ggml_bf16_t   tmp[4096];
    for (int64_t i = 0; i < n; i += chunk) {
        int64_t len = (n - i < chunk) ? n - i : chunk;
        ggml_fp32_to_bf16_row_ref(data + i, tmp, len);
        ggml_bf16_to_fp32_row(tmp, data + i, len);
    }
}

// Compute delta = scaling * B @ A via the best available backend.
//
// On CUDA: tensors are uploaded to GPU, cuBLAS does the GEMM, result
// is downloaded back. On CPU: ggml's threaded SIMD kernels run locally.
// The backend abstraction handles all memory management transparently.
//
// A is [rank, in_feat] row-major (safetensors convention).
// B is [out_feat, rank] row-major.
// Writes out_feat * in_feat floats into delta.
static void lora_gemm(const float *  a,
                      const float *  b,
                      float *        delta,
                      int64_t        out_feat,
                      int64_t        rank,
                      int64_t        in_feat,
                      float          scaling,
                      ggml_backend_t backend) {
    int64_t a_nel = rank * in_feat;
    int64_t b_nel = out_feat * rank;
    int64_t c_nel = out_feat * in_feat;

    // metadata context: tensor descriptors + graph only, no data
    // (the backend allocates tensor memory via ggml_backend_alloc_ctx_tensors)
    size_t                  meta   = ggml_tensor_overhead() * 6 + ggml_graph_overhead() + 4096;
    struct ggml_init_params params = { meta, NULL, true };
    struct ggml_context *   ctx    = ggml_init(params);

    // input tensors store raw row-major data from safetensors.
    // ggml is column-major, so row-major A[rank, in_feat] maps to ggml [in_feat, rank].
    struct ggml_tensor * ta = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_feat, rank);
    struct ggml_tensor * tb = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, rank, out_feat);

    // graph: transpose A so rank becomes ne[0] (contraction dim), then GEMM + scale.
    // ggml_cont materializes the transposed view into a contiguous tensor.
    // on CUDA, all three ops run as GPU kernels (transpose copy + cuBLAS + scale).
    struct ggml_tensor * ta_t   = ggml_cont(ctx, ggml_transpose(ctx, ta));
    struct ggml_tensor * result = ggml_scale(ctx, ggml_mul_mat(ctx, ta_t, tb), scaling);

    struct ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    // allocate all tensor data on the backend (GPU VRAM or CPU RAM)
    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // upload: host to device (noop when backend is CPU)
    ggml_backend_tensor_set(ta, a, 0, (size_t) a_nel * sizeof(float));
    ggml_backend_tensor_set(tb, b, 0, (size_t) b_nel * sizeof(float));

    // compute: transpose + matmul + scale, all on the backend
    ggml_backend_graph_compute(backend, graph);

    // download: device to host (noop when backend is CPU)
    ggml_backend_tensor_get(result, delta, 0, (size_t) c_nel * sizeof(float));

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

// Main LoRA merge function.
//
// Call after all GGUF tensors are loaded into wctx->pending but before wctx_alloc.
// For each LoRA pair found in the safetensors file:
//   1. Map PEFT key to GGUF tensor name
//   2. Find the matching PendingCopy via hashmap (O(1) lookup)
//   3. Compute delta = (alpha/rank) * scale * B @ A via backend GEMM
//   4. Round delta through BF16 to match PEFT intermediate precision
//   5. Dequant base weight row by row, add to delta
//   6. Requant into staging buffer, patch PendingCopy.src
//
// The lora_path can be a .safetensors file or a directory containing
// adapter_model.safetensors and adapter_config.json.
static bool lora_merge(WeightCtx *       wctx,
                       const GGUFModel & gf,
                       const char *      lora_path,
                       float             scale,
                       ggml_backend_t    backend) {
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

    // group lora_A and lora_B entries by their GGUF base tensor name.
    // also collect per-tensor alpha scalars (ComfyUI baked format).
    std::map<std::string, const STEntry *> a_map, b_map;
    std::map<std::string, float>           alpha_map;
    for (const auto & e : st.entries) {
        // per-tensor alpha: "base_model.model.layers.0.self_attn.q_proj.alpha"
        // scalar F32 with shape [] containing the baked alpha value
        const char * alpha_suffix = ".alpha";
        size_t       slen         = strlen(alpha_suffix);
        if (e.name.size() > slen && e.name.compare(e.name.size() - slen, slen, alpha_suffix) == 0 && e.dtype == "F32" &&
            e.n_dims == 0) {
            // build GGUF base name from the alpha key (strip suffix, reuse prefix logic)
            std::string fake_key = e.name.substr(0, e.name.size() - slen) + ".lora_.x";
            std::string base     = lora_base_name(fake_key);
            if (!base.empty()) {
                float val = 0.0f;
                memcpy(&val, st_data(st, e), sizeof(float));
                alpha_map[base] = val;
            }
            continue;
        }

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

    // O(1) PendingCopy lookup: map each src pointer to its index in wctx->pending
    std::unordered_map<const void *, size_t> pending_idx;
    pending_idx.reserve(wctx->pending.size());
    for (size_t i = 0; i < wctx->pending.size(); i++) {
        pending_idx[wctx->pending[i].src] = i;
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

        auto pc_it = pending_idx.find(base_ptr);
        if (pc_it == pending_idx.end()) {
            fprintf(stderr, "[LoRA] WARNING: no PendingCopy for %s, skipping\n", gguf_name.c_str());
            skipped++;
            continue;
        }
        WeightCtx::PendingCopy * pc = &wctx->pending[pc_it->second];

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

        // alpha: prefer per-tensor (ComfyUI baked), then config, fallback to rank
        float alpha;
        auto  alpha_it = alpha_map.find(gguf_name);
        if (alpha_it != alpha_map.end()) {
            alpha = alpha_it->second;
        } else if (alpha_cfg > 0) {
            alpha = (float) alpha_cfg;
        } else {
            alpha = (float) rank;
        }
        float scaling = (alpha / (float) rank) * scale;

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

        // PEFT casts LoRA weights to BF16 before computing the delta.
        // We replicate this round trip so B @ A matches merge_and_unload exactly.
        lora_bf16_round(a_f32.data(), a_nel);
        lora_bf16_round(b_f32.data(), b_nel);

        // delta = scaling * B @ A via backend GEMM (cuBLAS on CUDA, SIMD on CPU)
        std::vector<float> delta((size_t) nel);
        lora_gemm(a_f32.data(), b_f32.data(), delta.data(), out_feat, rank, in_feat, scaling, backend);

        // round delta through BF16 to match PEFT's intermediate precision.
        // without this, the diffusion model diverges (extremely sensitive to weight values).
        lora_bf16_round(delta.data(), nel);

        // add base weight row by row: dequant each row into a small scratch buffer,
        // accumulate into delta, then discard. Avoids a second full-size F32 allocation.
        size_t             row_bytes = ggml_row_size(ttype, ne0);
        std::vector<float> row_buf((size_t) ne0);
        for (int64_t i = 0; i < ne1; i++) {
            const void * row_src   = (const uint8_t *) base_ptr + i * row_bytes;
            float *      delta_row = delta.data() + i * ne0;
            lora_dequant(row_src, row_buf.data(), ne0, ttype);
            for (int64_t j = 0; j < ne0; j++) {
                delta_row[j] += row_buf[j];
            }
        }

        // requant merged weight into a staging buffer.
        // we stash the buffer in wctx->staging so it stays alive until wctx_alloc.
        size_t max_bytes = (size_t) nel * sizeof(float);
        size_t n_floats  = (max_bytes + sizeof(float) - 1) / sizeof(float);
        wctx->staging.emplace_back(n_floats);
        void * merged_buf = wctx->staging.back().data();

        size_t merged_bytes = lora_requant(delta.data(), merged_buf, nel, ne0, ttype);
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
