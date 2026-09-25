#pragma once
// adapter-stack.h: several DiT adapters merged in order, with per group
// strength multipliers.
//
// Merged deltas add, so a stack is the adapters merged one after another,
// each at its own scale. Group scales multiply the delta of every tensor in
// their group on top of that:
//   self_attn   decoder.layers.N.self_attn.*
//   cross_attn  decoder.layers.N.cross_attn.*
//   mlp         decoder.layers.N.mlp.*
//   cond_embed  decoder.condition_embedder.*
//   time_embed  decoder.time_embed*.*
//   proj_in     decoder.proj_in.*
//
// The stack travels as one canonical string (the DiT cache key), one line
// per adapter "<scale>\t<path>" and an optional "groups\t<6 floats>" line.
//
// Ported from HOT-Step-CPP (scragnog).

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

struct AdapterGroupScales {
    float self_attn  = 1.0f;
    float cross_attn = 1.0f;
    float mlp        = 1.0f;
    float cond_embed = 1.0f;
    float time_embed = 1.0f;
    float proj_in    = 1.0f;

    bool identity() const {
        return self_attn == 1.0f && cross_attn == 1.0f && mlp == 1.0f && cond_embed == 1.0f && time_embed == 1.0f &&
               proj_in == 1.0f;
    }

    // Scale for a GGUF tensor name. cross_attn is checked before self_attn.
    float for_tensor(const std::string & name) const {
        if (name.find("cross_attn") != std::string::npos) {
            return cross_attn;
        }
        if (name.find("self_attn") != std::string::npos) {
            return self_attn;
        }
        if (name.find(".mlp.") != std::string::npos || name.find(".ff.") != std::string::npos) {
            return mlp;
        }
        if (name.find("time_embed") != std::string::npos) {
            return time_embed;
        }
        if (name.find("condition_embed") != std::string::npos) {
            return cond_embed;
        }
        if (name.find("proj_in") != std::string::npos) {
            return proj_in;
        }
        return (self_attn + cross_attn + mlp + cond_embed + time_embed + proj_in) / 6.0f;
    }
};

struct AdapterStackItem {
    std::string path;
    float       scale = 1.0f;
};

struct AdapterStack {
    std::vector<AdapterStackItem> items;
    AdapterGroupScales            groups;
};

static std::string adapter_stack_encode(const AdapterStack & st) {
    std::string out;
    char        buf[256];
    for (const auto & it : st.items) {
        snprintf(buf, sizeof(buf), "%.9g\t", it.scale);
        out += buf;
        out += it.path;
        out += "\n";
    }
    if (!st.groups.identity()) {
        const AdapterGroupScales & g = st.groups;
        snprintf(buf, sizeof(buf), "groups\t%.9g %.9g %.9g %.9g %.9g %.9g\n", g.self_attn, g.cross_attn, g.mlp,
                 g.cond_embed, g.time_embed, g.proj_in);
        out += buf;
    }
    return out;
}

// A spec without a tab is a single adapter path at scale 1 (legacy form).
static AdapterStack adapter_stack_decode(const std::string & spec, float legacy_scale = 1.0f) {
    AdapterStack st;
    if (spec.find('\t') == std::string::npos) {
        if (!spec.empty()) {
            st.items.push_back({ spec, legacy_scale });
        }
        return st;
    }
    size_t pos = 0;
    while (pos < spec.size()) {
        size_t      nl   = spec.find('\n', pos);
        std::string line = spec.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        pos              = nl == std::string::npos ? spec.size() : nl + 1;
        size_t tab       = line.find('\t');
        if (tab == std::string::npos) {
            continue;
        }
        std::string head = line.substr(0, tab);
        std::string rest = line.substr(tab + 1);
        if (head == "groups") {
            AdapterGroupScales & g = st.groups;
            sscanf(rest.c_str(), "%f %f %f %f %f %f", &g.self_attn, &g.cross_attn, &g.mlp, &g.cond_embed, &g.time_embed,
                   &g.proj_in);
        } else {
            st.items.push_back({ rest, (float) atof(head.c_str()) });
        }
    }
    return st;
}
