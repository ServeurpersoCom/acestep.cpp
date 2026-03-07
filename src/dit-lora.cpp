// dit-lora.cpp: Load LoRA adapters from safetensors into DiT (ACE-Step).
// Compatible with PEFT adapter_model.safetensors (lora_A / lora_B per target layer).

#include "dit.h"
#include "safetensors.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>

// Normalize adapter key to base name: decoder.layers.N.<proj>
// Handles: base_model.model.model., base_model.model.; decoder.layers. or layers.; .lora_A.default/.lora_B.default or .lora_A.weight/.lora_B.weight
static std::string lora_key_to_base(const std::string & key) {
    std::string s = key;
    const char * prefixes[] = { "base_model.model.model.", "base_model.model." };
    for (const char * p : prefixes) {
        size_t pl = strlen(p);
        if (s.size() >= pl && s.compare(0, pl, p) == 0) {
            s = s.substr(pl);
            break;
        }
    }
    // PEFT-style suffix
    if (s.size() > 14 && s.compare(s.size() - 14, 14, ".lora_A.default") == 0)
        s = s.substr(0, s.size() - 14);
    else if (s.size() > 14 && s.compare(s.size() - 14, 14, ".lora_B.default") == 0)
        s = s.substr(0, s.size() - 14);
    else if (s.size() > 14 && s.compare(s.size() - 14, 14, ".lora_A.weight") == 0)
        s = s.substr(0, s.size() - 14);
    else if (s.size() > 14 && s.compare(s.size() - 14, 14, ".lora_B.weight") == 0)
        s = s.substr(0, s.size() - 14);
    else if (s.size() > 7 && s.compare(s.size() - 7, 7, ".lora_A") == 0)
        s = s.substr(0, s.size() - 7);
    else if (s.size() > 7 && s.compare(s.size() - 7, 7, ".lora_B") == 0)
        s = s.substr(0, s.size() - 7);
    // HuggingFace adapter: layers.N -> decoder.layers.N for our DiT naming
    if (s.size() >= 7 && s.compare(0, 7, "layers.") == 0)
        s = "decoder." + s;
    return s;
}

static bool is_lora_a(const std::string & key) {
    return key.find("lora_A") != std::string::npos;
}

// Slot index for layer: 0=sa_q, 1=sa_k, 2=sa_v, 3=sa_o, 4=ca_q, 5=ca_k, 6=ca_v, 7=ca_o, 8=gate, 9=up, 10=down
enum LoraSlot {
    SA_Q, SA_K, SA_V, SA_O, CA_Q, CA_K, CA_V, CA_O, GATE, UP, DOWN, N_SLOTS
};

static bool parse_base_name(const std::string & base, int * layer_idx, LoraSlot * slot) {
    int L = -1;
    if (sscanf(base.c_str(), "decoder.layers.%d.self_attn.q_proj", &L) == 1) { *layer_idx = L; *slot = SA_Q; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.self_attn.k_proj", &L) == 1) { *layer_idx = L; *slot = SA_K; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.self_attn.v_proj", &L) == 1) { *layer_idx = L; *slot = SA_V; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.self_attn.o_proj", &L) == 1) { *layer_idx = L; *slot = SA_O; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.cross_attn.q_proj", &L) == 1) { *layer_idx = L; *slot = CA_Q; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.cross_attn.k_proj", &L) == 1) { *layer_idx = L; *slot = CA_K; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.cross_attn.v_proj", &L) == 1) { *layer_idx = L; *slot = CA_V; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.cross_attn.o_proj", &L) == 1) { *layer_idx = L; *slot = CA_O; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.mlp.gate_proj", &L) == 1) { *layer_idx = L; *slot = GATE; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.mlp.up_proj", &L) == 1) { *layer_idx = L; *slot = UP; return true; }
    if (sscanf(base.c_str(), "decoder.layers.%d.mlp.down_proj", &L) == 1) { *layer_idx = L; *slot = DOWN; return true; }
    return false;
}

static struct ggml_tensor ** slot_to_ptr(DiTGGMLLayer * ly, LoraSlot slot, bool is_b) {
    if (is_b) {
        switch (slot) {
            case SA_Q: return &ly->lora_sa_q_b; case SA_K: return &ly->lora_sa_k_b; case SA_V: return &ly->lora_sa_v_b; case SA_O: return &ly->lora_sa_o_b;
            case CA_Q: return &ly->lora_ca_q_b; case CA_K: return &ly->lora_ca_k_b; case CA_V: return &ly->lora_ca_v_b; case CA_O: return &ly->lora_ca_o_b;
            case GATE: return &ly->lora_gate_b; case UP: return &ly->lora_up_b; case DOWN: return &ly->lora_down_b;
            default: return nullptr;
        }
    } else {
        switch (slot) {
            case SA_Q: return &ly->lora_sa_q_a; case SA_K: return &ly->lora_sa_k_a; case SA_V: return &ly->lora_sa_v_a; case SA_O: return &ly->lora_sa_o_a;
            case CA_Q: return &ly->lora_ca_q_a; case CA_K: return &ly->lora_ca_k_a; case CA_V: return &ly->lora_ca_v_a; case CA_O: return &ly->lora_ca_o_a;
            case GATE: return &ly->lora_gate_a; case UP: return &ly->lora_up_a; case DOWN: return &ly->lora_down_a;
            default: return nullptr;
        }
    }
}

bool dit_ggml_load_lora(DiTGGML * m, const char * lora_path, float scale) {
    FILE * fp = fopen(lora_path, "rb");
    if (!fp) {
        fprintf(stderr, "[LoRA] cannot open %s\n", lora_path);
        return false;
    }
    std::unordered_map<std::string, SafeTensorInfo> tensors;
    if (fseek(fp, 0, SEEK_SET) != 0) {
        fclose(fp);
        return false;
    }
    int n = safetensors_parse_lora(fp, &tensors);
    uint64_t data_section_start = (uint64_t)ftell(fp);
    if (n == 0) {
        fclose(fp);
        fprintf(stderr, "[LoRA] no LoRA tensors found in %s\n", lora_path);
        return false;
    }

    // Count pairs we will load: for each lora_A key, find the matching lora_B (same base name)
    std::unordered_map<std::string, std::pair<std::string, std::string>> pairs;  // base -> (key_a, key_b)
    std::unordered_map<std::string, std::string> base_to_b;
    for (const auto & kv : tensors) {
        std::string base = lora_key_to_base(kv.first);
        if (base.empty()) continue;
        if (is_lora_a(kv.first))
            base_to_b[base] = "";  // mark base as having A; we'll find B next
    }
    for (const auto & kv : tensors) {
        std::string base = lora_key_to_base(kv.first);
        if (base.empty()) continue;
        if (base_to_b.count(base) && kv.first.find("lora_B") != std::string::npos)
            base_to_b[base] = kv.first;
    }
    for (const auto & kv : tensors) {
        if (!is_lora_a(kv.first)) continue;
        std::string base = lora_key_to_base(kv.first);
        auto it = base_to_b.find(base);
        if (it != base_to_b.end() && !it->second.empty())
            pairs[base] = { kv.first, it->second };
    }

    int n_pairs = (int)pairs.size();
    wctx_init(&m->lora_wctx, n_pairs * 2);  // A and B per pair

    fseek(fp, (long)data_section_start, SEEK_SET);

    for (const auto & p : pairs) {
        const std::string & base = p.first;
        const std::string & key_a = p.second.first;
        const std::string & key_b = p.second.second;
        int layer_idx = 0;
        LoraSlot slot = N_SLOTS;
        if (!parse_base_name(base, &layer_idx, &slot) || layer_idx < 0 || layer_idx >= m->cfg.n_layers) continue;

        DiTGGMLLayer * ly = &m->layers[layer_idx];
        SafeTensorInfo & info_a = tensors[key_a];
        SafeTensorInfo & info_b = tensors[key_b];
        if (info_a.n_dims != 2 || info_b.n_dims != 2) continue;
        // A_pt [r, in], B_pt [out, r]. We need A_ggml [r, in] for mul_mat(A,x)=[r,S], B_ggml [out, r] for mul_mat(B, Ax)=[out,S].
        // ggml layout: ne[0]=cols, ne[1]=rows. So A: [r, in] -> ne[0]=in, ne[1]=r. B: [out, r] -> ne[0]=r, ne[1]=out.
        int64_t r = info_a.shape[0], in_dim = info_a.shape[1];
        int64_t out_dim = info_b.shape[0];
        if (info_b.shape[1] != r) continue;

        struct ggml_tensor * ta = ggml_new_tensor_2d(m->lora_wctx.ctx, GGML_TYPE_F32, (int64_t)in_dim, (int64_t)r);
        struct ggml_tensor * tb = ggml_new_tensor_2d(m->lora_wctx.ctx, GGML_TYPE_F32, (int64_t)r, (int64_t)out_dim);
        ggml_set_name(ta, key_a.c_str());
        ggml_set_name(tb, key_b.c_str());

        // Copy A: file is row-major [r, in], we need ggml col-major [in, r] (transpose)
        size_t na = (size_t)(r * in_dim);
        m->lora_wctx.staging.emplace_back(na);
        float * buf_a = m->lora_wctx.staging.back().data();
        if (!safetensors_read_tensor_data(fp, data_section_start, info_a.data_start, info_a.data_end, buf_a)) {
            fclose(fp);
            wctx_free(&m->lora_wctx);
            return false;
        }
        m->lora_wctx.staging.emplace_back(na);
        float * transposed_a = m->lora_wctx.staging.back().data();
        for (int64_t i = 0; i < r; i++)
            for (int64_t j = 0; j < in_dim; j++)
                transposed_a[(size_t)(j * r + i)] = buf_a[(size_t)(i * in_dim + j)];
        m->lora_wctx.pending.push_back({ ta, transposed_a, na * sizeof(float), 0 });

        size_t nb = (size_t)(out_dim * r);
        m->lora_wctx.staging.emplace_back(nb);
        float * buf_b = m->lora_wctx.staging.back().data();
        if (!safetensors_read_tensor_data(fp, data_section_start, info_b.data_start, info_b.data_end, buf_b)) {
            fclose(fp);
            wctx_free(&m->lora_wctx);
            return false;
        }
        m->lora_wctx.staging.emplace_back(nb);
        float * transposed_b = m->lora_wctx.staging.back().data();
        for (int64_t i = 0; i < out_dim; i++)
            for (int64_t j = 0; j < r; j++)
                transposed_b[(size_t)(j * out_dim + i)] = buf_b[(size_t)(i * r + j)];
        m->lora_wctx.pending.push_back({ tb, transposed_b, nb * sizeof(float), 0 });

        struct ggml_tensor ** pa = slot_to_ptr(ly, slot, false);
        struct ggml_tensor ** pb = slot_to_ptr(ly, slot, true);
        if (pa) *pa = ta;
        if (pb) *pb = tb;
    }
    fclose(fp);
    fp = nullptr;

    if (!wctx_alloc(&m->lora_wctx, m->backend)) {
        fprintf(stderr, "[LoRA] failed to allocate LoRA tensors on backend\n");
        wctx_free(&m->lora_wctx);
        return false;
    }
    m->lora_scale = scale;
    fprintf(stderr, "[LoRA] loaded %d adapter pairs from %s (scale=%.4f)\n", n_pairs, lora_path, scale);
    return true;
}
