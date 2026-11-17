#pragma once
// adapter-resolve.h: turn a request's DiT adapter fields into the spec
// dit_ggml_load merges (adapter-stack.h), names resolved in the registry.

#include "adapter-merge.h"
#include "adapter-stack.h"
#include "model-registry.h"
#include "request.h"

#include <string>

// Fills *spec with "" (no adapter), a single adapter path, or an encoded
// stack. Returns false with *missing set when a name is not in the registry.
static bool request_adapter_spec(const AceRequest &    req,
                                 const ModelRegistry & reg,
                                 std::string *         spec,
                                 float *               scale,
                                 std::string *         missing) {
    spec->clear();
    *scale = 1.0f;

    AdapterStack stack;
    stack.groups.self_attn  = req.adapter_group_self_attn;
    stack.groups.cross_attn = req.adapter_group_cross_attn;
    stack.groups.mlp        = req.adapter_group_mlp;
    stack.groups.cond_embed = req.adapter_group_cond_embed;
    stack.groups.time_embed = req.adapter_group_time_embed;
    stack.groups.proj_in    = req.adapter_group_proj_in;

    std::vector<AceAdapterRef> refs = req.adapters;
    if (refs.empty() && !req.adapter.empty()) {
        refs.push_back({ req.adapter, req.adapter_scale });
    }
    for (const AceAdapterRef & ref : refs) {
        const AdapterEntry * e = registry_find_adapter(reg, ref.name.c_str());
        if (!e) {
            *missing = ref.name;
            return false;
        }
        stack.items.push_back({ e->path, ref.scale });
    }
    if (stack.items.empty()) {
        return true;
    }
    // one adapter at unit group scales keeps the plain path form
    if (stack.items.size() == 1 && stack.groups.identity()) {
        *spec  = stack.items[0].path;
        *scale = stack.items[0].scale;
        return true;
    }
    *spec = adapter_stack_encode(stack);
    return true;
}

// What half of the model an adapter changes, read from its tensor names:
// the DiT (decoder, cross attention) or the planner LM (model.layers without
// cross attention). False for both when the weights cannot be read.
static void adapter_classify(const std::string & path, bool * dit, bool * lm) {
    *dit             = false;
    *lm              = false;
    std::string file = path;
    struct stat sb;
    if (stat(path.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
        const char * names[] = { "/adapter_model.safetensors", "/lokr_weights.safetensors" };
        file.clear();
        for (const char * n : names) {
            if (stat((path + n).c_str(), &sb) == 0) {
                file = path + n;
                break;
            }
        }
        if (file.empty()) {
            file = adapter_single_weights(path.c_str());
        }
    }
    STFile st = {};
    if (file.empty() || !st_open(&st, file.c_str())) {
        return;
    }
    bool cross = false, decoder = false, lm_layers = false;
    for (const STEntry & e : st.entries) {
        cross |= e.name.find("cross_attn") != std::string::npos;
        decoder |= e.name.find("decoder.") != std::string::npos || e.name.find("diffusion_model.") != std::string::npos;
        lm_layers |=
            e.name.find("model.layers.") != std::string::npos || e.name.find("lycoris_layers_") != std::string::npos;
    }
    st_close(&st);
    *dit = cross || decoder;
    *lm  = !*dit && lm_layers;
}
