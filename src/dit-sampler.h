#pragma once
// dit-sampler.h: DiT sampling loop with APG (Adaptive Projected Guidance)
//
// Euler flow matching sampler with CFG and APG momentum.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

#include "dit.h"
#include "dit-graph.h"
#include "debug.h"


#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// APG (Adaptive Projected Guidance) for DiT CFG
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py
struct APGMomentumBuffer {
    double momentum;
    std::vector<double> running_average;
    bool initialized;

    APGMomentumBuffer(double m = -0.75) : momentum(m), initialized(false) {}

    void update(const double * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++)
                running_average[i] = values[i] + momentum * running_average[i];
        }
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// All math in double precision matching Python .double() calls.
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(
        const double * v0, const double * v1,
        double * out_par, double * out_orth,
        int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        double norm2 = 0.0;
        for (int t = 0; t < T; t++)
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        double inv_norm = (norm2 > 1e-60) ? (1.0 / sqrt(norm2)) : 0.0;

        double dot = 0.0;
        for (int t = 0; t < T; t++)
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);

        for (int t = 0; t < T; t++) {
            int idx = t * Oc + c;
            double v1n = v1[idx] * inv_norm;
            out_par[idx]  = dot * v1n;
            out_orth[idx] = v0[idx] - out_par[idx];
        }
    }
}

// APG forward matching Python apg_forward() exactly:
//   1. diff = cond - uncond
//   2. momentum.update(diff); diff = running_average
//   3. norm clip: per-channel L2 over T (dims=[1]), clip to norm_threshold=2.5
//   4. project(diff, pred_COND) -> (parallel, orthogonal)
//   5. result = pred_cond + (scale - 1) * orthogonal
// Internal computation in double precision (Python uses .double()).
static void apg_forward(
        const float * pred_cond, const float * pred_uncond,
        float guidance_scale, APGMomentumBuffer & mbuf,
        float * result, int Oc, int T,
        float norm_threshold = 2.5f) {
    int n = Oc * T;

    // 1. diff = cond - uncond (promote to double)
    std::vector<double> diff(n);
    for (int i = 0; i < n; i++)
        diff[i] = (double)pred_cond[i] - (double)pred_uncond[i];

    // 2. momentum update, then use smoothed diff
    mbuf.update(diff.data(), n);
    memcpy(diff.data(), mbuf.running_average.data(), n * sizeof(double));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            double norm2 = 0.0;
            for (int t = 0; t < T; t++)
                norm2 += diff[t * Oc + c] * diff[t * Oc + c];
            double norm = sqrt(norm2 > 0.0 ? norm2 : 0.0);
            double s = (norm > 1e-60) ? fmin(1.0, (double)norm_threshold / norm) : 1.0;
            if (s < 1.0) {
                for (int t = 0; t < T; t++)
                    diff[t * Oc + c] *= s;
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component (double precision)
    std::vector<double> pred_cond_d(n), par(n), orth(n);
    for (int i = 0; i < n; i++) pred_cond_d[i] = (double)pred_cond[i];
    apg_project(diff.data(), pred_cond_d.data(), par.data(), orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal (back to float)
    double w = (double)guidance_scale - 1.0;
    for (int i = 0; i < n; i++)
        result[i] = (float)((double)pred_cond[i] + w * orth[i]);
}

// Flow matching generation loop (batched)
// Runs num_steps euler steps to denoise N latent samples in parallel.
//
// noise:            [N * T * Oc]  N contiguous [T, Oc] noise blocks
// context_latents:  [N * T * ctx_ch]  N contiguous context blocks
// enc_hidden:       [enc_S * H]  SINGLE encoder output (shared, will be broadcast to N)
// schedule:         array of num_steps timestep values
// output:           [N * T * Oc]  generated latents (caller-allocated)
static void dit_ggml_generate(
        DiTGGML * model,
        const float * noise,
        const float * context_latents,
        const float * enc_hidden_data,
        int enc_S,
        int T,
        int N,
        int num_steps,
        const float * schedule,
        float * output,
        float guidance_scale = 1.0f,
        const DebugDumper * dbg = nullptr) {

    DiTGGMLConfig & c = model->cfg;
    int Oc    = c.out_channels;      // 64
    int ctx_ch = c.in_channels - Oc; // 128
    int in_ch = c.in_channels;       // 192
    int S     = T / c.patch_size;
    int n_per = T * Oc;              // elements per sample
    int n_total = N * n_per;         // total output elements
    int H     = c.hidden_size;

    fprintf(stderr, "[DiT] Batch N=%d, T=%d, S=%d, enc_S=%d\n", N, T, S, enc_S);

    // Graph context (generous fixed allocation, shapes are constant across steps)
    size_t ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> ctx_buf(ctx_size);

    struct ggml_init_params gparams = {
        /*.mem_size   =*/ ctx_size,
        /*.mem_buffer =*/ ctx_buf.data(),
        /*.no_alloc   =*/ true,
    };
    struct ggml_context * ctx = ggml_init(gparams);

    struct ggml_tensor * t_input  = NULL;
    struct ggml_tensor * t_output = NULL;
    struct ggml_cgraph * gf = dit_ggml_build_graph(model, ctx, T, enc_S, N,
                                                    &t_input, &t_output);

    fprintf(stderr, "[DiT] Graph: %d nodes\n", ggml_graph_n_nodes(gf));

    struct ggml_tensor * t_enc = ggml_graph_get_tensor(gf, "enc_hidden");

    // Allocate compute buffers.
    // Critical: reset FIRST (clears old state), THEN force inputs to GPU, THEN alloc.
    // Without GPU forcing, inputs default to CPU where the scheduler aliases their
    // buffers with intermediates. enc_hidden is read at every cross-attn layer (24x),
    // so CPU aliasing corrupts it mid-graph. With N>1 the larger buffers trigger
    // more aggressive aliasing, causing batch sample 1+ to produce noise.
    ggml_backend_sched_reset(model->sched);
    if (model->backend != model->cpu_backend) {
        const char * input_names[] = {"enc_hidden", "input_latents", "t", "t_r", "positions", "sw_mask"};
        for (const char * iname : input_names) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, iname);
            if (t) ggml_backend_sched_set_tensor_backend(model->sched, t, model->backend);
        }
    }
    if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
        fprintf(stderr, "FATAL: failed to allocate graph\n");
        ggml_free(ctx);
        return;
    }

    // Encoder hidden states: upload once (re-uploaded per step only when CFG swaps to null)
    // t_enc was declared above for backend forcing

    // t_r is set per-step in the loop (= t_curr, same as Python reference)
    struct ggml_tensor * t_tr = ggml_graph_get_tensor(gf, "t_r");

    // Positions: [0, 1, ..., S-1] repeated N times for batch rope indexing
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos_data(S * N);
    for (int b = 0; b < N; b++)
        for (int i = 0; i < S; i++)
            pos_data[b * S + i] = i;
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));

    // Sliding window mask: [S, S, 1, N] fp16 - N identical copies
    struct ggml_tensor * t_mask = ggml_graph_get_tensor(gf, "sw_mask");
    std::vector<uint16_t> mask_data;
    if (t_mask) {
        int win = c.sliding_window;
        mask_data.resize(S * S * N);
        // fill first copy
        for (int qi = 0; qi < S; qi++)
            for (int ki = 0; ki < S; ki++) {
                int dist = (qi > ki) ? (qi - ki) : (ki - qi);
                float v = (dist <= win) ? 0.0f : -INFINITY;
                mask_data[ki * S + qi] = ggml_fp32_to_fp16(v);
            }
        // replicate for batch elements 1..N-1
        for (int b = 1; b < N; b++)
            memcpy(mask_data.data() + b * S * S, mask_data.data(), S * S * sizeof(uint16_t));
        ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * N * sizeof(uint16_t));
    }

    // CFG setup
    bool do_cfg = guidance_scale > 1.0f;
    std::vector<float> null_enc_buf;
    std::vector<APGMomentumBuffer> apg_mbufs;

    if (do_cfg) {
        if (!model->null_condition_emb) {
            fprintf(stderr, "[DiT] WARNING: guidance_scale=%.1f but null_condition_emb not found. Disabling CFG.\n", guidance_scale);
            do_cfg = false;
        } else {
            int emb_n = (int)ggml_nelements(model->null_condition_emb);
            std::vector<float> null_emb(emb_n);

            if (model->null_condition_emb->type == GGML_TYPE_BF16) {
                std::vector<uint16_t> bf16_buf(emb_n);
                ggml_backend_tensor_get(model->null_condition_emb, bf16_buf.data(), 0, emb_n * sizeof(uint16_t));
                for (int i = 0; i < emb_n; i++) {
                    uint32_t w = (uint32_t)bf16_buf[i] << 16;
                    memcpy(&null_emb[i], &w, 4);
                }
            } else {
                ggml_backend_tensor_get(model->null_condition_emb, null_emb.data(), 0, emb_n * sizeof(float));
            }

            // Broadcast [H] to [enc_S, H] then to N copies [H, enc_S, N]
            std::vector<float> null_enc_single(H * enc_S);
            for (int s = 0; s < enc_S; s++)
                memcpy(&null_enc_single[s * H], null_emb.data(), H * sizeof(float));
            null_enc_buf.resize(H * enc_S * N);
            for (int b = 0; b < N; b++)
                memcpy(null_enc_buf.data() + b * enc_S * H, null_enc_single.data(), enc_S * H * sizeof(float));

            if (dbg && dbg->enabled) {
                debug_dump_1d(dbg, "null_condition_emb", null_emb.data(), emb_n);
                debug_dump_2d(dbg, "null_enc_hidden", null_enc_single.data(), enc_S, H);
            }

            apg_mbufs.resize(N);

            fprintf(stderr, "[DiT] CFG enabled: guidance_scale=%.1f, 2x forward per step, N=%d\n", guidance_scale, N);
        }
    }

    // Prepare host buffers (all N samples contiguous)
    std::vector<float> xt(noise, noise + n_total);
    std::vector<float> vt(n_total);

    std::vector<float> vt_cond;
    std::vector<float> vt_uncond;
    if (do_cfg) {
        vt_cond.resize(n_total);
        vt_uncond.resize(n_total);
    }

    // input_buf: [in_ch, T, N] - pre-fill context_latents (constant across all steps)
    std::vector<float> input_buf(in_ch * T * N);
    for (int b = 0; b < N; b++)
        for (int t = 0; t < T; t++)
            memcpy(&input_buf[b * T * in_ch + t * in_ch],
                   &context_latents[b * T * ctx_ch + t * ctx_ch],
                   ctx_ch * sizeof(float));

    // Pre-allocate enc_buf once (avoids heap alloc per step)
    std::vector<float> enc_buf(H * enc_S * N);
    for (int b = 0; b < N; b++)
        memcpy(enc_buf.data() + b * enc_S * H, enc_hidden_data, enc_S * H * sizeof(float));
    ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));

    struct ggml_tensor * t_t = ggml_graph_get_tensor(gf, "t");

    // Flow matching loop
    for (int step = 0; step < num_steps; step++) {
        float t_curr = schedule[step];

        // Set timestep (changes each step)
        if (t_t) {
            ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
        }
        if (t_tr) {
            ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
        }

        // Re-upload constants (scheduler may reuse input buffers as scratch between computes)
        ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
        ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
        if (t_mask) ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * N * sizeof(uint16_t));

        // Update xt portion of input: [in_ch, T, N] (context_latents pre-filled)
        for (int b = 0; b < N; b++)
            for (int t = 0; t < T; t++)
                memcpy(&input_buf[b * T * in_ch + t * in_ch + ctx_ch],
                       &xt[b * n_per + t * Oc],
                       Oc * sizeof(float));
        ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N * sizeof(float));

        // compute forward pass (conditional)
        ggml_backend_sched_graph_compute(model->sched, gf);

        // dump intermediate tensors on step 0 (sample 0 only for batch)
        if (step == 0 && dbg && dbg->enabled) {
            auto dump_named = [&](const char *name) {
                struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                if (t) {
                    // For batched tensors, dump only sample 0 (first slice)
                    int64_t n0 = t->ne[0];
                    int64_t n1 = t->ne[1];
                    int64_t sample_elems = n0 * n1;  // [ne0, ne1] of first sample
                    std::vector<float> buf(sample_elems);
                    ggml_backend_tensor_get(t, buf.data(), 0, sample_elems * sizeof(float));
                    if (n1 <= 1) {
                        debug_dump_1d(dbg, name, buf.data(), (int)n0);
                    } else {
                        debug_dump_2d(dbg, name, buf.data(), (int)n0, (int)n1);
                    }
                }
            };
            dump_named("tproj");
            dump_named("temb");
            dump_named("temb_t");
            dump_named("temb_r");
            dump_named("sinusoidal_t");
            dump_named("sinusoidal_r");
            dump_named("temb_lin1_t");
            dump_named("temb_lin1_r");
            dump_named("hidden_after_proj_in");
            dump_named("proj_in_input");
            dump_named("enc_after_cond_emb");
            dump_named("layer0_sa_input");
            dump_named("layer0_q_after_rope");
            dump_named("layer0_k_after_rope");
            dump_named("layer0_sa_output");
            dump_named("layer0_attn_out");
            dump_named("layer0_after_self_attn");
            dump_named("layer0_after_cross_attn");
            dump_named("hidden_after_layer0");
            dump_named("hidden_after_layer6");
            dump_named("hidden_after_layer12");
            dump_named("hidden_after_layer18");
            dump_named("hidden_after_layer23");
        }

        // read velocity output: [Oc, T, N]
        ggml_backend_tensor_get(t_output, vt.data(), 0, n_total * sizeof(float));

        // CFG: unconditional pass + APG per sample
        if (do_cfg) {
            memcpy(vt_cond.data(), vt.data(), n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_cond", step);
                debug_dump_2d(dbg, name, vt_cond.data(), T, Oc);
            }

            // Unconditional pass: re-upload all inputs (scheduler clobbers input buffers during compute)
            ggml_backend_tensor_set(t_enc, null_enc_buf.data(), 0, H * enc_S * N * sizeof(float));
            ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N * sizeof(float));
            if (t_t) ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
            if (t_tr) ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
            ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
            if (t_mask) ggml_backend_tensor_set(t_mask, mask_data.data(), 0, S * S * N * sizeof(uint16_t));

            ggml_backend_sched_graph_compute(model->sched, gf);
            ggml_backend_tensor_get(t_output, vt_uncond.data(), 0, n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_uncond", step);
                debug_dump_2d(dbg, name, vt_uncond.data(), T, Oc);
            }

            // APG per sample
            for (int b = 0; b < N; b++) {
                apg_forward(vt_cond.data() + b * n_per,
                            vt_uncond.data() + b * n_per,
                            guidance_scale, apg_mbufs[b],
                            vt.data() + b * n_per, Oc, T);
            }
        }

        if (dbg && dbg->enabled) {
            char name[64];
            snprintf(name, sizeof(name), "dit_step%d_vt", step);
            debug_dump_2d(dbg, name, vt.data(), T, Oc);
        }

        // euler step (all N samples)
        if (step == num_steps - 1) {
            for (int i = 0; i < n_total; i++)
                output[i] = xt[i] - vt[i] * t_curr;
        } else {
            float dt = t_curr - schedule[step + 1];
            for (int i = 0; i < n_total; i++)
                xt[i] -= vt[i] * dt;
        }

        // debug dump (sample 0 only)
        if (dbg && dbg->enabled) {
            char name[64];
            if (step == num_steps - 1) {
                snprintf(name, sizeof(name), "dit_x0");
                debug_dump_2d(dbg, name, output, T, Oc);
            } else {
                snprintf(name, sizeof(name), "dit_step%d_xt", step);
                debug_dump_2d(dbg, name, xt.data(), T, Oc);
            }
        }

        fprintf(stderr, "[DiT] step %d/%d t=%.3f\n", step + 1, num_steps, t_curr);
    }

    // Batch diagnostic: report per-sample stats to catch corruption
    if (N >= 2) {
        for (int b = 0; b < N; b++) {
            const float * s = output + b * n_per;
            float mn = s[0], mx = s[0], sum = 0.0f;
            int n_nan = 0;
            for (int i = 0; i < n_per; i++) {
                float v = s[i];
                if (v != v) { n_nan++; continue; }
                if (v < mn) mn = v;
                if (v > mx) mx = v;
                sum += v;
            }
            fprintf(stderr, "[DiT] Batch%d output: min=%.4f max=%.4f mean=%.6f nan=%d\n",
                    b, mn, mx, sum / (float)n_per, n_nan);
        }
    }

    ggml_free(ctx);
}
