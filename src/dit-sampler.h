#pragma once
// dit-sampler.h: DiT sampling loop with APG (Adaptive Projected Guidance)
//
// Euler flow matching sampler with CFG and APG momentum.
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py

#include "debug.h"
#include "dit-graph.h"
#include "dit.h"
#include "dwt-haar.h"
#include "solvers/solver-registry.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// APG (Adaptive Projected Guidance) for DiT CFG
// Matches Python ACE-Step-1.5 acestep/models/base/apg_guidance.py
struct APGMomentumBuffer {
    double              momentum;
    std::vector<double> running_average;
    bool                initialized;

    APGMomentumBuffer(double m = -0.75) : momentum(m), initialized(false) {}

    void update(const double * values, int n) {
        if (!initialized) {
            running_average.assign(values, values + n);
            initialized = true;
        } else {
            for (int i = 0; i < n; i++) {
                running_average[i] = values[i] + momentum * running_average[i];
            }
        }
    }
};

// project(v0, v1, dims=[1]): decompose v0 into parallel + orthogonal w.r.t. v1
// All math in double precision matching Python .double() calls.
// Layout: memory [T, Oc] time-major (ggml ne=[Oc, T]).
// Python dims=[1] on [B,T,C] = normalize/project per channel over T dimension.
// In memory [T, Oc] layout: for each channel c, operate over all T time frames.
static void apg_project(const double * v0, const double * v1, double * out_par, double * out_orth, int Oc, int T) {
    for (int c = 0; c < Oc; c++) {
        double norm2 = 0.0;
        for (int t = 0; t < T; t++) {
            norm2 += v1[t * Oc + c] * v1[t * Oc + c];
        }
        double inv_norm = (norm2 > 1e-60) ? (1.0 / sqrt(norm2)) : 0.0;

        double dot = 0.0;
        for (int t = 0; t < T; t++) {
            dot += v0[t * Oc + c] * (v1[t * Oc + c] * inv_norm);
        }

        for (int t = 0; t < T; t++) {
            int    idx    = t * Oc + c;
            double v1n    = v1[idx] * inv_norm;
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
static void apg_forward(const float *       pred_cond,
                        const float *       pred_uncond,
                        float               guidance_scale,
                        APGMomentumBuffer & mbuf,
                        float *             result,
                        int                 Oc,
                        int                 T,
                        float               norm_threshold = 2.5f) {
    int n = Oc * T;

    // 1. diff = cond - uncond (promote to double)
    std::vector<double> diff(n);
    for (int i = 0; i < n; i++) {
        diff[i] = (double) pred_cond[i] - (double) pred_uncond[i];
    }

    // 2. momentum update, then use smoothed diff
    mbuf.update(diff.data(), n);
    memcpy(diff.data(), mbuf.running_average.data(), n * sizeof(double));

    // 3. norm clipping: per-channel L2 over T (dims=[1]), clip to threshold
    if (norm_threshold > 0.0f) {
        for (int c = 0; c < Oc; c++) {
            double norm2 = 0.0;
            for (int t = 0; t < T; t++) {
                norm2 += diff[t * Oc + c] * diff[t * Oc + c];
            }
            double norm = sqrt(norm2 > 0.0 ? norm2 : 0.0);
            double s    = (norm > 1e-60) ? fmin(1.0, (double) norm_threshold / norm) : 1.0;
            if (s < 1.0) {
                for (int t = 0; t < T; t++) {
                    diff[t * Oc + c] *= s;
                }
            }
        }
    }

    // 4. project(diff, pred_COND) -> orthogonal component (double precision)
    std::vector<double> pred_cond_d(n), par(n), orth(n);
    for (int i = 0; i < n; i++) {
        pred_cond_d[i] = (double) pred_cond[i];
    }
    apg_project(diff.data(), pred_cond_d.data(), par.data(), orth.data(), Oc, T);

    // 5. result = pred_cond + (scale - 1) * orthogonal (back to float)
    double w = (double) guidance_scale - 1.0;
    for (int i = 0; i < n; i++) {
        result[i] = (float) ((double) pred_cond[i] + w * orth[i]);
    }
}

// Flow matching generation loop (batched)
// Runs num_steps euler steps to denoise N latent samples in parallel.
//
// noise:            [N * T * Oc]  N contiguous [T, Oc] noise blocks
// context_latents:  [N * T * ctx_ch]  N contiguous context blocks
// enc_hidden:       [enc_S * H_enc * N]  per-batch encoder outputs (caller-stacked)
// schedule:         array of num_steps timestep values
// output:           [N * T * Oc]  generated latents (caller-allocated)
static int dit_ggml_generate(DiTGGML *           model,
                             const float *       noise,
                             const float *       context_latents,
                             const float *       enc_hidden_data,
                             int                 enc_S,
                             int                 T,
                             int                 N,
                             int                 num_steps,
                             const float *       schedule,
                             float *             output,
                             float               guidance_scale = 1.0f,
                             const DebugDumper * dbg            = nullptr,
                             const float *       context_switch = nullptr,
                             int                 cover_steps    = -1,
                             bool (*cancel)(void *)             = nullptr,
                             void *          cancel_data        = nullptr,
                             const int *     real_S             = nullptr,
                             const int *     real_enc_S         = nullptr,
                             const float *   enc_switch         = nullptr,
                             const int *     real_enc_S_switch  = nullptr,
                             const int64_t * seeds              = nullptr,
                             bool            use_batch_cfg      = true,
                             float           dcw_scaler         = 0.0f,
                             float           dcw_high_scaler    = 0.0f,
                             const char *    dcw_mode           = "low",
                             const char *    solver_name        = "euler",
                             int             stork_substeps     = 10) {
    DiTGGMLConfig & c       = model->cfg;
    int             Oc      = c.out_channels;      // 64
    int             ctx_ch  = c.in_channels - Oc;  // 128
    int             in_ch   = c.in_channels;       // 192
    int             S       = T / c.patch_size;
    int             n_per   = T * Oc;              // elements per sample
    int             n_total = N * n_per;           // total output elements

    // CFG batching: pack conditional + unconditional into one graph of size 2*N.
    // Slots [0, N): conditional (real encoder states).
    // Slots [N, 2*N): unconditional (null encoder states).
    // Single forward produces both predictions, halving DiT compute per step.
    // When batch_cfg is false, two separate forwards per step (saves activation memory).
    bool do_cfg    = (guidance_scale > 1.0f) && model->null_condition_emb;
    bool batch_cfg = do_cfg && use_batch_cfg;
    int  N_graph   = batch_cfg ? 2 * N : N;

    if (guidance_scale > 1.0f && !model->null_condition_emb) {
        fprintf(stderr, "[DiT] WARNING: guidance_scale=%.1f but null_condition_emb not found. Disabling CFG.\n",
                guidance_scale);
    }

    fprintf(stderr, "[DiT] Batch N=%d, T=%d, S=%d, enc_S=%d%s\n", N, T, S, enc_S,
            batch_cfg ? ", CFG batched 2N" : (do_cfg ? ", CFG 2-pass" : ""));

    // Graph context (generous fixed allocation, shapes are constant across steps)
    size_t               ctx_size = ggml_tensor_overhead() * 8192 + ggml_graph_overhead_custom(8192, false);
    std::vector<uint8_t> ctx_buf(ctx_size);

    struct ggml_init_params gparams = {
        /*.mem_size   =*/ctx_size,
        /*.mem_buffer =*/ctx_buf.data(),
        /*.no_alloc   =*/true,
    };
    struct ggml_context * ctx = ggml_init(gparams);

    struct ggml_tensor * t_input  = NULL;
    struct ggml_tensor * t_output = NULL;
    struct ggml_cgraph * gf       = dit_ggml_build_graph(model, ctx, T, enc_S, N_graph, &t_input, &t_output);

    fprintf(stderr, "[DiT] Graph: %d nodes\n", ggml_graph_n_nodes(gf));

    struct ggml_tensor * t_enc = ggml_graph_get_tensor(gf, "enc_hidden");
    int                  H_enc = (int) t_enc->ne[0];  // encoder hidden size (from condition_embedder)

    // Allocate compute buffers.
    // Critical: reset FIRST (clears old state), THEN force inputs to GPU, THEN alloc.
    // Without GPU forcing, inputs default to CPU where the scheduler aliases their
    // buffers with intermediates. enc_hidden is read at every cross-attn layer (24x),
    // so CPU aliasing corrupts it mid-graph. With N>1 the larger buffers trigger
    // more aggressive aliasing, causing batch sample 1+ to produce noise.
    ggml_backend_sched_reset(model->sched);
    if (model->backend != model->cpu_backend) {
        const char * input_names[] = { "enc_hidden", "input_latents", "t",           "t_r",
                                       "positions",  "sa_mask_sw",    "sa_mask_pad", "ca_mask" };
        for (const char * iname : input_names) {
            struct ggml_tensor * t = ggml_graph_get_tensor(gf, iname);
            if (t) {
                ggml_backend_sched_set_tensor_backend(model->sched, t, model->backend);
            }
        }
    }
    if (!ggml_backend_sched_alloc_graph(model->sched, gf)) {
        fprintf(stderr, "[DiT] FATAL: failed to allocate graph\n");
        ggml_free(ctx);
        return -1;
    }

    // Encoder hidden states: re-uploaded per step (scheduler clobbers input buffers).
    // When CFG batched, slots [0,N) hold real encoder states, [N,2N) hold null.
    // t_enc was declared above for backend forcing

    // t_r is set per-step in the loop (= t_curr, same as Python reference)
    struct ggml_tensor * t_tr = ggml_graph_get_tensor(gf, "t_r");

    // Positions: [0, 1, ..., S-1] repeated N_graph times for batch rope indexing
    struct ggml_tensor * t_pos = ggml_graph_get_tensor(gf, "positions");
    std::vector<int32_t> pos_data(S * N_graph);
    for (int b = 0; b < N_graph; b++) {
        for (int i = 0; i < S; i++) {
            pos_data[b * S + i] = i;
        }
    }
    ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N_graph * sizeof(int32_t));

    // Self-attention masks: per-batch, combines sliding window and padding.
    // GGML flash_attn_ext mask layout: [ne0=KV_len, ne1=Q_len, 1, N_graph]
    // Linear element offset: ki + qi*ne0 + b*ne0*ne1
    //   sa_mask_sw  [S, S, 1, N_graph]: layer_type=0 (sliding window + padding)
    //   sa_mask_pad [S, S, 1, N_graph]: layer_type=1 (full attention, padding only)
    // When real_S is NULL, all positions are real (mask is all 0.0).
    struct ggml_tensor * t_sa_mask_sw  = ggml_graph_get_tensor(gf, "sa_mask_sw");
    struct ggml_tensor * t_sa_mask_pad = ggml_graph_get_tensor(gf, "sa_mask_pad");

    int                   win = c.sliding_window;
    std::vector<uint16_t> sa_sw_data(S * S * N_graph);
    std::vector<uint16_t> sa_pad_data(S * S * N_graph);

    // Fill masks for real samples, then duplicate for uncond slots
    for (int b = 0; b < N; b++) {
        int rs = real_S ? real_S[b] : S;  // real sequence length for this batch element
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < S; ki++) {
                bool real_pos = (qi < rs) && (ki < rs);
                int  dist     = (qi > ki) ? (qi - ki) : (ki - qi);
                bool in_win   = (win <= 0) || (S <= win) || (dist <= win);

                // offset = ki + qi*S + b*S*S  (ne0=S indexed by ki, ne1=S indexed by qi)
                int off = b * S * S + qi * S + ki;

                float sw_val    = (real_pos && in_win) ? 0.0f : -INFINITY;
                sa_sw_data[off] = ggml_fp32_to_fp16(sw_val);

                float pad_val    = real_pos ? 0.0f : -INFINITY;
                sa_pad_data[off] = ggml_fp32_to_fp16(pad_val);
            }
        }
        if (batch_cfg) {
            memcpy(&sa_sw_data[(N + b) * S * S], &sa_sw_data[b * S * S], S * S * sizeof(uint16_t));
            memcpy(&sa_pad_data[(N + b) * S * S], &sa_pad_data[b * S * S], S * S * sizeof(uint16_t));
        }
    }
    ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
    ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N_graph * sizeof(uint16_t));

    // Cross-attention mask: per-batch encoder padding.
    // [ne0=enc_S (KV), ne1=S (Q), 1, N_graph] blocks padding in enc_hidden.
    // Value depends only on ki (encoder position), independent of qi.
    // Linear offset for element (ki, qi, 0, b) = ki + qi*enc_S + b*enc_S*S
    struct ggml_tensor *  t_ca_mask = ggml_graph_get_tensor(gf, "ca_mask");
    std::vector<uint16_t> ca_data(enc_S * S * N_graph);

    for (int b = 0; b < N; b++) {
        int re = real_enc_S ? real_enc_S[b] : enc_S;
        for (int qi = 0; qi < S; qi++) {
            for (int ki = 0; ki < enc_S; ki++) {
                // offset = ki + qi*enc_S + b*enc_S*S  (ne0=enc_S indexed by ki)
                float v                                  = (ki < re) ? 0.0f : -INFINITY;
                ca_data[b * enc_S * S + qi * enc_S + ki] = ggml_fp32_to_fp16(v);
            }
        }
        if (batch_cfg) {
            memcpy(&ca_data[(N + b) * enc_S * S], &ca_data[b * enc_S * S], enc_S * S * sizeof(uint16_t));
        }
    }
    ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N_graph * sizeof(uint16_t));

    // CFG: prepare null encoder states and APG buffers
    std::vector<APGMomentumBuffer> apg_mbufs;
    std::vector<float>             null_enc_buf;

    if (do_cfg) {
        apg_mbufs.resize(N);
        fprintf(stderr, "[DiT] CFG enabled: guidance_scale=%.1f, %s, N_graph=%d\n", guidance_scale,
                batch_cfg ? "batched" : "2-pass", N_graph);
    }

    // Prepare host buffers (all N real samples contiguous)
    std::vector<float> xt(noise, noise + n_total);
    std::vector<float> vt(n_total);

    std::vector<float> vt_cond;
    std::vector<float> vt_uncond;
    if (do_cfg) {
        vt_cond.resize(n_total);
        vt_uncond.resize(n_total);
    }

    // input_buf: [in_ch, T, N_graph]
    // Pre-fill context_latents for slots [0, N). Uncond slots [N, 2N) are duplicated.
    // The xt portion (noisy latent) is updated per step in the loop.
    std::vector<float> input_buf(in_ch * T * N_graph);
    for (int b = 0; b < N; b++) {
        for (int t = 0; t < T; t++) {
            memcpy(&input_buf[b * T * in_ch + t * in_ch], &context_latents[b * T * ctx_ch + t * ctx_ch],
                   ctx_ch * sizeof(float));
        }
        if (batch_cfg) {
            memcpy(&input_buf[(N + b) * T * in_ch], &input_buf[b * T * in_ch], T * in_ch * sizeof(float));
        }
    }

    // enc_buf: [H_enc, enc_S, N_graph]
    // Slots [0, N): real encoder hidden states from caller.
    // Batched CFG: slots [N, 2N) hold null_condition_emb broadcast to [H_enc, enc_S].
    // 2-pass CFG: null_enc_buf is a separate buffer uploaded before the uncond forward.
    std::vector<float> enc_buf(H_enc * enc_S * N_graph);
    memcpy(enc_buf.data(), enc_hidden_data, H_enc * enc_S * N * sizeof(float));
    if (do_cfg) {
        int                emb_n = (int) ggml_nelements(model->null_condition_emb);
        std::vector<float> null_emb(emb_n);

        if (model->null_condition_emb->type == GGML_TYPE_BF16) {
            std::vector<uint16_t> bf16_buf(emb_n);
            ggml_backend_tensor_get(model->null_condition_emb, bf16_buf.data(), 0, emb_n * sizeof(uint16_t));
            for (int i = 0; i < emb_n; i++) {
                uint32_t w = (uint32_t) bf16_buf[i] << 16;
                memcpy(&null_emb[i], &w, 4);
            }
        } else {
            ggml_backend_tensor_get(model->null_condition_emb, null_emb.data(), 0, emb_n * sizeof(float));
        }

        if (dbg && dbg->enabled) {
            debug_dump_1d(dbg, "null_condition_emb", null_emb.data(), emb_n);
        }

        // Broadcast [H_enc] to [H_enc, enc_S] then fill uncond destination
        std::vector<float> null_enc_single(H_enc * enc_S);
        for (int s = 0; s < enc_S; s++) {
            memcpy(&null_enc_single[s * H_enc], null_emb.data(), H_enc * sizeof(float));
        }
        if (dbg && dbg->enabled) {
            debug_dump_2d(dbg, "null_enc_hidden", null_enc_single.data(), enc_S, H_enc);
        }
        if (batch_cfg) {
            // Pack null into graph slots [N, 2N)
            for (int b = 0; b < N; b++) {
                memcpy(enc_buf.data() + (N + b) * enc_S * H_enc, null_enc_single.data(), enc_S * H_enc * sizeof(float));
            }
        } else {
            // Separate buffer for 2-pass re-upload
            null_enc_buf.resize(H_enc * enc_S * N);
            for (int b = 0; b < N; b++) {
                memcpy(null_enc_buf.data() + b * enc_S * H_enc, null_enc_single.data(), enc_S * H_enc * sizeof(float));
            }
        }
    }
    ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));

    struct ggml_tensor * t_t = ggml_graph_get_tensor(gf, "t");

    // Solver dispatch. solver_name is the canonical key, resolved via the registry.
    const SolverInfo * solver_info = solver_lookup(solver_name);
    if (!solver_info) {
        fprintf(stderr, "[DiT] WARNING: unknown solver '%s', falling back to euler\n", solver_name);
        solver_info = solver_lookup("euler");
    }
    fprintf(stderr, "[DiT] Solver: %s (%d NFE/step, order %d)\n", solver_info->display_name, solver_info->nfe,
            solver_info->order);

    SolverState solver_state;
    solver_state.seeds          = seeds;
    solver_state.batch_n        = N;
    solver_state.n_per          = n_per;
    solver_state.stork_substeps = stork_substeps;

    // Flow matching loop
    bool switched_cover = false;
    for (int step = 0; step < num_steps; step++) {
        if (cancel && cancel(cancel_data)) {
            fprintf(stderr, "[DiT] Cancelled at step %d/%d\n", step, num_steps);
            ggml_free(ctx);
            return -1;
        }
        float t_curr = schedule[step];

        // Cover mode: at cover_steps, swap context to silence and enc_hidden to text2music
        if (context_switch && cover_steps >= 0 && step >= cover_steps && !switched_cover) {
            switched_cover = true;
            // Swap context latents for cond slots [0, N)
            for (int b = 0; b < N; b++) {
                for (int t = 0; t < T; t++) {
                    memcpy(&input_buf[b * T * in_ch + t * in_ch], &context_switch[b * T * ctx_ch + t * ctx_ch],
                           ctx_ch * sizeof(float));
                }
                // Batched CFG: uncond slot mirrors cond context
                if (batch_cfg) {
                    memcpy(&input_buf[(N + b) * T * in_ch], &input_buf[b * T * in_ch], T * in_ch * sizeof(float));
                }
            }
            // Swap encoder hidden states to text2music-encoded version.
            // Only cond slots [0, N) are updated; uncond slots [N, 2N) keep null.
            if (enc_switch) {
                memcpy(enc_buf.data(), enc_switch, H_enc * enc_S * N * sizeof(float));

                // Update cross-attention mask for text2music encoder lengths
                if (real_enc_S_switch) {
                    for (int b = 0; b < N; b++) {
                        int re = real_enc_S_switch[b];
                        for (int qi = 0; qi < S; qi++) {
                            for (int ki = 0; ki < enc_S; ki++) {
                                float v                                  = (ki < re) ? 0.0f : -INFINITY;
                                ca_data[b * enc_S * S + qi * enc_S + ki] = ggml_fp32_to_fp16(v);
                            }
                        }
                    }
                }
            }
            fprintf(stderr, "[DiT] Cover: switched to non-cover context at step %d/%d\n", step, num_steps);
        }

        // Set timestep (changes each step)
        if (t_t) {
            ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
        }
        if (t_tr) {
            ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
        }

        // Re-upload constants (scheduler may reuse input buffers as scratch between computes)
        ggml_backend_tensor_set(t_enc, enc_buf.data(), 0, enc_buf.size() * sizeof(float));
        ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N_graph * sizeof(int32_t));
        ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N_graph * sizeof(uint16_t));
        ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N_graph * sizeof(uint16_t));

        // Update xt portion of input: [in_ch, T, N_graph] (context_latents pre-filled)
        // Both cond and uncond slots receive the same noisy latent
        for (int b = 0; b < N; b++) {
            for (int t = 0; t < T; t++) {
                memcpy(&input_buf[b * T * in_ch + t * in_ch + ctx_ch], &xt[b * n_per + t * Oc], Oc * sizeof(float));
            }
            if (batch_cfg) {
                for (int t = 0; t < T; t++) {
                    memcpy(&input_buf[(N + b) * T * in_ch + t * in_ch + ctx_ch], &xt[b * n_per + t * Oc],
                           Oc * sizeof(float));
                }
            }
        }
        ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N_graph * sizeof(float));

        // Conditional forward pass
        ggml_backend_sched_graph_compute(model->sched, gf);

        // dump intermediate tensors on step 0 (sample 0 only for batch)
        if (step == 0 && dbg && dbg->enabled) {
            auto dump_named = [&](const char * name) {
                struct ggml_tensor * t = ggml_graph_get_tensor(gf, name);
                if (t) {
                    // For batched tensors, dump only sample 0 (first slice)
                    int64_t            n0           = t->ne[0];
                    int64_t            n1           = t->ne[1];
                    int64_t            sample_elems = n0 * n1;  // [ne0, ne1] of first sample
                    std::vector<float> buf(sample_elems);
                    ggml_backend_tensor_get(t, buf.data(), 0, sample_elems * sizeof(float));
                    if (n1 <= 1) {
                        debug_dump_1d(dbg, name, buf.data(), (int) n0);
                    } else {
                        debug_dump_2d(dbg, name, buf.data(), (int) n0, (int) n1);
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
            char last_layer_name[64];
            snprintf(last_layer_name, sizeof(last_layer_name), "hidden_after_layer%d", c.n_layers - 1);
            dump_named(last_layer_name);
        }

        // Read velocity output and apply CFG
        if (batch_cfg) {
            // Output is [Oc, T, 2N]: first N = conditional, last N = unconditional
            std::vector<float> full_output(n_per * N_graph);
            ggml_backend_tensor_get(t_output, full_output.data(), 0, n_per * N_graph * sizeof(float));
            memcpy(vt_cond.data(), full_output.data(), n_total * sizeof(float));
            memcpy(vt_uncond.data(), full_output.data() + n_total, n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_cond", step);
                debug_dump_2d(dbg, name, vt_cond.data(), T, Oc);
                snprintf(name, sizeof(name), "dit_step%d_vt_uncond", step);
                debug_dump_2d(dbg, name, vt_uncond.data(), T, Oc);
            }

            // APG per sample
            for (int b = 0; b < N; b++) {
                apg_forward(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per, guidance_scale, apg_mbufs[b],
                            vt.data() + b * n_per, Oc, T);
            }
        } else if (do_cfg) {
            // 2-pass: conditional output already computed, read it
            ggml_backend_tensor_get(t_output, vt_cond.data(), 0, n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_cond", step);
                debug_dump_2d(dbg, name, vt_cond.data(), T, Oc);
            }

            // Unconditional pass: re-upload null encoder + all inputs (scheduler clobbers buffers)
            ggml_backend_tensor_set(t_enc, null_enc_buf.data(), 0, H_enc * enc_S * N * sizeof(float));
            ggml_backend_tensor_set(t_input, input_buf.data(), 0, in_ch * T * N * sizeof(float));
            if (t_t) {
                ggml_backend_tensor_set(t_t, &t_curr, 0, sizeof(float));
            }
            if (t_tr) {
                ggml_backend_tensor_set(t_tr, &t_curr, 0, sizeof(float));
            }
            ggml_backend_tensor_set(t_pos, pos_data.data(), 0, S * N * sizeof(int32_t));
            ggml_backend_tensor_set(t_sa_mask_sw, sa_sw_data.data(), 0, S * S * N * sizeof(uint16_t));
            ggml_backend_tensor_set(t_sa_mask_pad, sa_pad_data.data(), 0, S * S * N * sizeof(uint16_t));
            ggml_backend_tensor_set(t_ca_mask, ca_data.data(), 0, enc_S * S * N * sizeof(uint16_t));

            ggml_backend_sched_graph_compute(model->sched, gf);
            ggml_backend_tensor_get(t_output, vt_uncond.data(), 0, n_total * sizeof(float));

            if (dbg && dbg->enabled) {
                char name[64];
                snprintf(name, sizeof(name), "dit_step%d_vt_uncond", step);
                debug_dump_2d(dbg, name, vt_uncond.data(), T, Oc);
            }

            // APG per sample
            for (int b = 0; b < N; b++) {
                apg_forward(vt_cond.data() + b * n_per, vt_uncond.data() + b * n_per, guidance_scale, apg_mbufs[b],
                            vt.data() + b * n_per, Oc, T);
            }
        } else {
            // read velocity output: [Oc, T, N]
            ggml_backend_tensor_get(t_output, vt.data(), 0, n_total * sizeof(float));
        }

        if (dbg && dbg->enabled) {
            char name[64];
            snprintf(name, sizeof(name), "dit_step%d_vt", step);
            debug_dump_2d(dbg, name, vt.data(), T, Oc);
        }

        // step update (all N samples)
        if (step == num_steps - 1) {
            // Final step: predict x0 directly from the velocity field.
            for (int i = 0; i < n_total; i++) {
                output[i] = xt[i] - vt[i] * t_curr;
            }
        } else {
            float t_next = schedule[step + 1];

            // Modular solver dispatch. The solver mutates xt in place.
            // 1 NFE solvers ignore the model_fn callback (passed empty).
            solver_state.step_index = step;
            solver_info->step_fn(xt.data(), vt.data(), t_curr, t_next, n_total, solver_state, {}, vt.data());

            // DCW: Differential Correction in Wavelet domain (CVPR 2026).
            // Sampler-side correction for SNR-t bias in flow matching.
            // 4 modes with per-mode t-modulation, conformant to the reference
            // code AMAP-ML/DCW (generate.py, FlowMatchEulerDiscreteScheduler.py):
            //   low:    s = t_curr * dcw_scaler          (strong at high noise)
            //   high:   s = (1 - t_curr) * dcw_scaler    (strong near clean)
            //   double: low = t_curr * dcw_scaler, high = (1 - t_curr) * dcw_high_scaler
            //   pix:    s = dcw_scaler                   (constant, no modulation)
            // Rationale: diffusion models reconstruct low-freq before high-freq,
            // so the correction tracks the reconstruction timeline per band.
            // ODE-only: denoised = xt_after - vt * t_next (reconstructed from
            // post-step xt, since xt_before = xt + vt * dt for Euler ODE so
            // denoised = xt_before - vt * t_curr = xt - vt * t_next).
            bool dcw_active = (dcw_scaler > 0.0f || dcw_high_scaler > 0.0f) && !solver_info->injects_noise;
            if (dcw_active) {
                int                Tl = (T + 1) / 2;
                std::vector<float> denoised(n_per);
                std::vector<float> tmp_xL(Tl * Oc), tmp_xH(Tl * Oc);
                std::vector<float> tmp_yL(Tl * Oc), tmp_yH(Tl * Oc);
                bool               is_low    = (strcmp(dcw_mode, "low") == 0);
                bool               is_high   = (strcmp(dcw_mode, "high") == 0);
                bool               is_double = (strcmp(dcw_mode, "double") == 0);
                bool               is_pix    = (strcmp(dcw_mode, "pix") == 0);
                if (!is_low && !is_high && !is_double && !is_pix) {
                    // Unknown mode: fall back to "low" (safest, paper default)
                    is_low = true;
                }
                // per-mode effective scaler, modulated as the paper prescribes
                float s_low      = t_curr * dcw_scaler;           // low-band coefficient
                float s_high     = (1.0f - t_curr) * dcw_scaler;  // high-only uses dcw_scaler with inverse modulation
                float s_double_h = (1.0f - t_curr) * dcw_high_scaler;  // double mode high coefficient
                float s_pix      = dcw_scaler;                         // constant per paper
                for (int b = 0; b < N; b++) {
                    const float * xt_b = xt.data() + b * n_per;
                    const float * vt_b = vt.data() + b * n_per;
                    for (int i = 0; i < n_per; i++) {
                        denoised[i] = xt_b[i] - vt_b[i] * t_next;
                    }
                    float * xt_bw = xt.data() + b * n_per;
                    if (is_low) {
                        dcw_haar_low_inplace(xt_bw, denoised.data(), T, Oc, s_low, tmp_xL.data(), tmp_xH.data(),
                                             tmp_yL.data(), tmp_yH.data());
                    } else if (is_high) {
                        dcw_haar_high_inplace(xt_bw, denoised.data(), T, Oc, s_high, tmp_xL.data(), tmp_xH.data(),
                                              tmp_yL.data(), tmp_yH.data());
                    } else if (is_double) {
                        dcw_haar_double_inplace(xt_bw, denoised.data(), T, Oc, s_low, s_double_h, tmp_xL.data(),
                                                tmp_xH.data(), tmp_yL.data(), tmp_yH.data());
                    } else {  // is_pix
                        dcw_pix_inplace(xt_bw, denoised.data(), T, Oc, s_pix);
                    }
                }
                fprintf(stderr, "[DiT] DCW step %d/%d mode=%s (t_curr=%.3f)\n", step + 1, num_steps, dcw_mode, t_curr);
            }
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

        fprintf(stderr, "[DiT] Step %d/%d t=%.3f\n", step + 1, num_steps, t_curr);
    }

    // Batch diagnostic: report per-sample stats to catch corruption
    if (N >= 2) {
        for (int b = 0; b < N; b++) {
            const float * s  = output + b * n_per;
            float         mn = s[0], mx = s[0], sum = 0.0f;
            int           n_nan = 0;
            for (int i = 0; i < n_per; i++) {
                float v = s[i];
                if (v != v) {
                    n_nan++;
                    continue;
                }
                if (v < mn) {
                    mn = v;
                }
                if (v > mx) {
                    mx = v;
                }
                sum += v;
            }
            fprintf(stderr, "[DiT] Batch%d output: min=%.4f max=%.4f mean=%.6f nan=%d\n", b, mn, mx,
                    sum / (float) n_per, n_nan);
        }
    }

    ggml_free(ctx);
    return 0;
}
