#pragma once
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml.h"

struct StaticGraph {
    ggml_gallocr_t galloc          = nullptr;
    bool           direct          = false;
    bool           sched_allocated = false;
};

static bool static_graph_backend_supports(ggml_backend_t backend, struct ggml_cgraph * gf) {
    for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i))) {
            return false;
        }
    }
    return true;
}

static void static_graph_release(StaticGraph * sg, ggml_backend_sched_t sched) {
    if (sg->galloc) {
        ggml_gallocr_free(sg->galloc);
    }
    if (sg->sched_allocated && sched) {
        ggml_backend_sched_reset(sched);
    }
    *sg = {};
}

static bool static_graph_alloc(StaticGraph *        sg,
                               ggml_backend_t       backend,
                               ggml_backend_sched_t sched,
                               struct ggml_cgraph * gf) {
    if (static_graph_backend_supports(backend, gf)) {
        sg->galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        if (sg->galloc && ggml_gallocr_alloc_graph(sg->galloc, gf)) {
            sg->direct = true;
            return true;
        }
        if (sg->galloc) {
            ggml_gallocr_free(sg->galloc);
            sg->galloc = nullptr;
        }
    }
    ggml_backend_sched_reset(sched);
    if (!ggml_backend_sched_alloc_graph(sched, gf)) {
        return false;
    }
    sg->sched_allocated = true;
    return true;
}

static enum ggml_status static_graph_compute(const StaticGraph *  sg,
                                             ggml_backend_t       backend,
                                             ggml_backend_sched_t sched,
                                             struct ggml_cgraph * gf) {
    return sg->direct ? ggml_backend_graph_compute(backend, gf) : ggml_backend_sched_graph_compute(sched, gf);
}
