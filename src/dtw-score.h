#pragma once
// dtw-score.h: DTW pathfinding + lyric alignment scoring (pure C++)
//
// Direct port of the Python ACE-Step scoring modules:
//   acestep/core/scoring/_dtw.py    — DTW + median filter (numba-jitted)
//   acestep/core/scoring/dit_score.py — MusicLyricScorer (coverage, monotonicity, confidence)
// Integration parity follows generation/handler/lyric_score.py and
// lyric_alignment_common.py for latent states, lyric slicing, and model heads.
//
// Pinned to ace-step/ACE-Step-1.5 commit 82252c24 (2026-07-09).
// If the Python scoring algorithm changes upstream, this port will need
// re-evaluation. The file:line references in comments below map to that commit.
//
// No external dependencies beyond <vector>, <cmath>, <algorithm>, <cstdint>.
// All compute is CPU-side on small matrices (tokens x frames), matching the
// Python reference which forces CPU ("the scoring matrices are small and this
// avoids occupying GPU VRAM that DiT / VAE / LM need" — dit_score.py:294).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

// ============================================================================
// DTW (Dynamic Time Warping)
// Ported from acestep/core/scoring/_dtw.py:dtw_cpu + _backtrace
// ============================================================================

// DTW backtrace: walk the trace matrix from (N, M) to (0, 0).
// Returns path as two parallel vectors: text_indices[i], time_indices[i].
// Python returns shape (2, path_len); here we return a struct for clarity.
//
// trace values: 0 = diagonal, 1 = up (i-1), 2 = left (j-1)
struct DTWPath {
    std::vector<int> text_idx;  // row index into cost matrix (lyric tokens)
    std::vector<int> time_idx;  // col index into cost matrix (audio frames)
};

static DTWPath dtw_backtrace(std::vector<float> & trace, int N, int M) {
    // Boundary handling (matches _dtw.py:61-62)
    // trace is (N+1) x (M+1), row-major
    auto T = [&](int i, int j) -> float & {
        return trace[(size_t) i * (M + 1) + j];
    };
    for (int j = 0; j <= M; j++) {
        T(0, j) = 2;
    }
    for (int i = 0; i <= N; i++) {
        T(i, 0) = 1;
    }

    // Pre-allocate (max path length = N + M), fill from the end
    int              max_path_len = N + M;
    std::vector<int> path_text(max_path_len, 0);
    std::vector<int> path_time(max_path_len, 0);

    int i = N, j = M;
    int path_idx = max_path_len - 1;

    while (i > 0 || j > 0) {
        path_text[path_idx] = i - 1;  // text index
        path_time[path_idx] = j - 1;  // time index
        path_idx--;

        float t = T(i, j);
        if (t == 0) {
            i--;
            j--;
        } else if (t == 1) {
            i--;
        } else if (t == 2) {
            j--;
        } else {
            break;
        }
    }

    int     start = path_idx + 1;
    DTWPath result;
    result.text_idx.assign(path_text.begin() + start, path_text.end());
    result.time_idx.assign(path_time.begin() + start, path_time.end());
    return result;
}

// DTW forward: compute cost matrix and backtrace the optimal path.
// x: cost matrix of shape [N, M] (row-major: x[i * M + j]).
// Returns the alignment path.
//
// Ported from acestep/core/scoring/_dtw.py:dtw_cpu (numba-jitted).
// The caller passes -calc_matrix (negated) so DTW finds the maximum-energy
// path (matches dit_score.py:255: dtw_cpu(-calc_matrix.astype(np.float32))).
static DTWPath dtw_cpu(const float * x, int N, int M) {
    // cost and trace are (N+1) x (M+1), row-major
    std::vector<float> cost((size_t) (N + 1) * (M + 1), std::numeric_limits<float>::infinity());
    std::vector<float> trace((size_t) (N + 1) * (M + 1), -1.0f);

    auto C = [&](int i, int j) -> float & {
        return cost[(size_t) i * (M + 1) + j];
    };

    C(0, 0) = 0.0f;

    for (int j = 1; j <= M; j++) {
        for (int i = 1; i <= N; i++) {
            float c0 = C(i - 1, j - 1);  // diagonal
            float c1 = C(i - 1, j);      // up
            float c2 = C(i, j - 1);      // left

            float c;
            float t;
            if (c0 < c1 && c0 < c2) {
                c = c0;
                t = 0;
            } else if (c1 < c0 && c1 < c2) {
                c = c1;
                t = 1;
            } else {
                c = c2;
                t = 2;
            }

            C(i, j)                         = x[(size_t) (i - 1) * M + (j - 1)] + c;
            trace[(size_t) i * (M + 1) + j] = t;
        }
    }

    return dtw_backtrace(trace, N, M);
}

// ============================================================================
// Median Filter
// Ported from acestep/core/scoring/_dtw.py:median_filter
// ============================================================================

// 1D median filter with reflect padding.
// x: input data, filter_width: window size (must be odd for symmetric padding).
// Returns filtered vector of the same length.
//
// Python uses F.pad(mode="reflect") then unfold + sort + pick middle.
// We replicate with a sliding window sort.
static std::vector<float> median_filter_1d(const std::vector<float> & x, int filter_width) {
    int n   = (int) x.size();
    int pad = filter_width / 2;
    if (n <= pad) {
        return x;
    }

    // Reflect padding (matches torch F.pad mode="reflect")
    std::vector<float> padded(n + 2 * pad);
    for (int i = 0; i < pad; i++) {
        padded[i] = x[pad - i];  // reflect from left
    }
    for (int i = 0; i < n; i++) {
        padded[pad + i] = x[i];
    }
    for (int i = 0; i < pad; i++) {
        padded[pad + n + i] = x[n - 2 - i];  // reflect from right
    }

    std::vector<float> result(n);
    std::vector<float> window(filter_width);
    int                mid = filter_width / 2;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < filter_width; j++) {
            window[j] = padded[i + j];
        }
        std::nth_element(window.begin(), window.begin() + mid, window.end());
        result[i] = window[mid];
    }
    return result;
}

// 2D median filter applied along the last dimension (columns / frames).
// input: [rows, cols] row-major. filter_width applied along cols.
// Returns [rows, cols] row-major.
//
// Matches median_filter in _dtw.py:90-110 which operates on the last dim.
static std::vector<float> median_filter_2d_rows(const std::vector<float> & input,
                                                int                        rows,
                                                int                        cols,
                                                int                        filter_width) {
    std::vector<float> result((size_t) rows * cols);
    for (int r = 0; r < rows; r++) {
        std::vector<float> row(input.begin() + (size_t) r * cols, input.begin() + (size_t) r * cols + cols);
        std::vector<float> filtered = median_filter_1d(row, filter_width);
        memcpy(result.data() + (size_t) r * cols, filtered.data(), (size_t) cols * sizeof(float));
    }
    return result;
}

// ============================================================================
// Lyric Alignment Scorer
// Ported from acestep/core/scoring/dit_score.py:MusicLyricScorer
// ============================================================================

// Default 2B layer/head configuration for cross-attention extraction.
// Matches the Python default: dit_score.py uses custom_config with
//   {2: [6], 3: [10, 11], 4: [3], 5: [8, 9], 6: [8]}
// 5 layers, 7 heads total. These are the cross-attention heads that
// best track lyric-to-audio alignment. Layer values are absolute DiT layer
// numbers. Callers that compact captured layers must remap them before
// preprocess_attention() indexes the compact attention buffer.
struct ScoreLayerHeadConfig {
    int layer;
    int head;
};

static const ScoreLayerHeadConfig DEFAULT_2B_SCORE_HEADS[] = {
    { 2, 6  },
    { 3, 10 },
    { 3, 11 },
    { 4, 3  },
    { 5, 8  },
    { 5, 9  },
    { 6, 8  },
};
static const int DEFAULT_2B_SCORE_HEADS_COUNT = 7;

// Remap absolute DiT layer numbers to compact slots in captured_layers.
// Returns false when any configured layer/head is unavailable; silently
// scoring a partial or model-incompatible configuration produces misleading
// alignment values.
static bool remap_score_heads(const ScoreLayerHeadConfig *        config,
                              int                                 config_count,
                              const int *                         captured_layers,
                              int                                 captured_count,
                              int                                 n_heads,
                              std::vector<ScoreLayerHeadConfig> & remapped) {
    remapped.clear();
    if (!config || config_count <= 0 || !captured_layers || captured_count <= 0 || n_heads <= 0) {
        return false;
    }

    for (int c = 0; c < config_count; c++) {
        if (config[c].head < 0 || config[c].head >= n_heads) {
            remapped.clear();
            return false;
        }

        int compact_layer = -1;
        for (int i = 0; i < captured_count; i++) {
            if (captured_layers[i] == config[c].layer) {
                compact_layer = i;
                break;
            }
        }
        if (compact_layer < 0) {
            remapped.clear();
            return false;
        }
        remapped.push_back({ compact_layer, config[c].head });
    }
    return !remapped.empty();
}

// Extract one batch item's pure-lyric rows from a GGML attention output.
// source layout is [tokens, frames, heads, batches] with tokens (ne[0])
// contiguous. destination layout is [heads, lyric_tokens, frames], matching
// the per-layer slice expected by preprocess_attention().
static bool extract_attention_slice(const float *        source,
                                    size_t               source_size,
                                    int                  tokens,
                                    int                  frames,
                                    int                  heads,
                                    int                  batches,
                                    int                  batch_index,
                                    int                  token_start,
                                    int                  token_count,
                                    std::vector<float> & destination) {
    destination.clear();
    if (!source || tokens <= 0 || frames <= 0 || heads <= 0 || batches <= 0 || batch_index < 0 ||
        batch_index >= batches || token_start < 0 || token_count <= 0 || token_start + token_count > tokens) {
        return false;
    }

    size_t expected = (size_t) tokens * frames * heads * batches;
    if (source_size < expected) {
        return false;
    }

    destination.resize((size_t) heads * token_count * frames);
    for (int h = 0; h < heads; h++) {
        for (int t = 0; t < token_count; t++) {
            for (int f = 0; f < frames; f++) {
                size_t src_idx = (size_t) (token_start + t) +
                                 (size_t) tokens * (f + (size_t) frames * (h + (size_t) heads * batch_index));
                size_t dst_idx       = ((size_t) h * token_count + t) * frames + f;
                destination[dst_idx] = source[src_idx];
            }
        }
    }
    return true;
}

struct LyricTokenSegment {
    int              start;
    std::vector<int> token_ids;
};

// Match Python _extract_lyric_segment(): strip the generated prompt header and
// stop at the first end-of-text token. The returned start is also the row
// offset into encoder_hidden_states because the condition encoder packs lyric
// tokens first.
static LyricTokenSegment extract_lyric_token_segment(const std::vector<int> & raw_ids, int header_tokens, int eos_id) {
    LyricTokenSegment result = { header_tokens, {} };
    if (header_tokens < 0 || header_tokens > (int) raw_ids.size()) {
        result.start = -1;
        return result;
    }

    int end = (int) raw_ids.size();
    for (int i = header_tokens; i < end; i++) {
        if (raw_ids[i] == eos_id) {
            end = i;
            break;
        }
    }
    result.token_ids.assign(raw_ids.begin() + header_tokens, raw_ids.begin() + end);
    return result;
}

// Token type mask: 1 = lyric token, 0 = structural tag (inside [...]).
// Ported from dit_score.py:_generate_token_type_mask (lines 32-55).
//
// The Python version uses tokenizer.decode([tid]) to get the string for each
// token. Here we accept a pre-decoded vector of token strings from the caller
// (the BPE tokenizer is available in the C++ pipeline but not in this header).
static std::vector<int> generate_token_type_mask(const std::vector<std::string> & decoded_tokens) {
    int              n = (int) decoded_tokens.size();
    std::vector<int> mask(n, 1);
    bool             in_bracket = false;

    for (int i = 0; i < n; i++) {
        const std::string & s = decoded_tokens[i];
        if (s.find('[') != std::string::npos) {
            in_bracket = true;
        }
        if (in_bracket) {
            mask[i] = 0;
        }
        if (s.find(']') != std::string::npos) {
            in_bracket = false;
            mask[i]    = 0;
        }
    }
    return mask;
}

// Preprocess attention matrix: select heads, average, median filter, min-max
// normalize, square for DTW pathfinding.
// Ported from dit_score.py:_preprocess_attention (lines 57-115).
//
// attention: [n_layers, n_heads, tokens, frames] row-major
//   = layers * n_heads * tokens * frames floats
// config: array of (layer, head) pairs to select
// config_count: number of entries in config
// medfilt_width: median filter window (1 = no filter)
//
// Outputs (via pointers):
//   calc_matrix: squared energy for DTW pathfinding [tokens, frames]
//   energy_matrix: normalized energy for scoring [tokens, frames]
// Returns false if no valid heads were found.
static bool preprocess_attention(const float *                attention,
                                 int                          n_layers,
                                 int                          n_heads,
                                 int                          tokens,
                                 int                          frames,
                                 const ScoreLayerHeadConfig * config,
                                 int                          config_count,
                                 int                          medfilt_width,
                                 std::vector<float> &         calc_matrix,
                                 std::vector<float> &         energy_matrix) {
    // 1. Select heads and stack (matches dit_score.py:84-93)
    std::vector<std::vector<float>> selected;
    for (int c = 0; c < config_count; c++) {
        int layer = config[c].layer;
        int head  = config[c].head;
        if (layer < n_layers && head < n_heads) {
            const float * ptr =
                attention + (size_t) layer * n_heads * tokens * frames + (size_t) head * tokens * frames;
            selected.emplace_back(ptr, ptr + (size_t) tokens * frames);
        }
    }

    if (selected.empty()) {
        return false;
    }

    // 2. Average across selected heads (dit_score.py:96)
    std::vector<float> avg_weights((size_t) tokens * frames, 0.0f);
    for (const auto & s : selected) {
        for (size_t i = 0; i < avg_weights.size(); i++) {
            avg_weights[i] += s[i];
        }
    }
    float inv_count = 1.0f / (float) selected.size();
    for (auto & v : avg_weights) {
        v *= inv_count;
    }

    // 3. Median filter (dit_score.py:101)
    if (medfilt_width > 1) {
        energy_matrix = median_filter_2d_rows(avg_weights, tokens, frames, medfilt_width);
    } else {
        energy_matrix = avg_weights;
    }

    // 4. Min-Max normalization (dit_score.py:104-109)
    float e_min = std::numeric_limits<float>::max();
    float e_max = std::numeric_limits<float>::lowest();
    for (float v : energy_matrix) {
        if (v < e_min) {
            e_min = v;
        }
        if (v > e_max) {
            e_max = v;
        }
    }

    if (e_max - e_min > 1e-9f) {
        float range = e_max - e_min;
        for (auto & v : energy_matrix) {
            v = (v - e_min) / range;
        }
    } else {
        std::fill(energy_matrix.begin(), energy_matrix.end(), 0.0f);
    }

    // 5. Contrast enhancement for DTW (dit_score.py:113)
    calc_matrix.resize((size_t) tokens * frames);
    for (size_t i = 0; i < energy_matrix.size(); i++) {
        calc_matrix[i] = energy_matrix[i] * energy_matrix[i];
    }

    return true;
}

// Alignment metrics: coverage, monotonicity, path confidence.
// Ported from dit_score.py:_compute_alignment_metrics (lines 117-214).
//
// energy_matrix: [rows, cols] normalized energy (row-major)
// path: DTW path (text_idx, time_idx pairs)
// type_mask: [rows] — 1 = lyric token, 0 = structural tag
// time_weight: minimum energy threshold for centroid computation (default 0.01)
// overlap_frames: allowed backward movement for monotonicity (default 9.0)
// instrumental_weight: weight for non-lyric path steps (default 1.0)
struct AlignmentMetrics {
    double coverage;
    double monotonicity;
    double confidence;
};

static AlignmentMetrics compute_alignment_metrics(const std::vector<float> & energy_matrix,
                                                  int                        rows,
                                                  int                        cols,
                                                  const DTWPath &            path,
                                                  const std::vector<int> &   type_mask,
                                                  double                     time_weight         = 0.01,
                                                  double                     overlap_frames      = 9.0,
                                                  double                     instrumental_weight = 1.0) {
    auto E = [&](int i, int j) -> double {
        return (double) energy_matrix[(size_t) i * cols + j];
    };

    // ================= A. Coverage Score (dit_score.py:150-161) =================
    int          total_sung_rows    = 0;
    int          valid_sung_rows    = 0;
    const double coverage_threshold = 0.1;

    for (int i = 0; i < rows; i++) {
        if (type_mask[i] == 1) {
            total_sung_rows++;
            double row_max = 0.0;
            for (int j = 0; j < cols; j++) {
                double e = E(i, j);
                if (e > row_max) {
                    row_max = e;
                }
            }
            if (row_max > coverage_threshold) {
                valid_sung_rows++;
            }
        }
    }

    double coverage = (total_sung_rows > 0) ? (double) valid_sung_rows / total_sung_rows : 1.0;

    // ================= B. Monotonicity Score (dit_score.py:163-191) =================
    // Compute centroid (energy-weighted mean column) for each lyric row.
    std::vector<double> centroids(rows, -1.0);
    for (int i = 0; i < rows; i++) {
        double sum_w = 0.0;
        double sum_t = 0.0;
        for (int j = 0; j < cols; j++) {
            double e = E(i, j);
            if (e > time_weight) {
                sum_w += e;
                sum_t += e * (double) j;
            }
        }
        if (sum_w > 1e-9) {
            centroids[i] = sum_t / sum_w;
        }
    }

    // Collect centroids for lyric rows with valid centroid
    std::vector<double> sung_centroids;
    for (int i = 0; i < rows; i++) {
        if (type_mask[i] == 1 && centroids[i] >= 0.0) {
            sung_centroids.push_back(centroids[i]);
        }
    }

    double monotonicity;
    int    cnt = (int) sung_centroids.size();
    if (cnt > 1) {
        double non_decreasing = 0.0;
        for (int k = 0; k < cnt - 1; k++) {
            if (sung_centroids[k + 1] >= (sung_centroids[k] - overlap_frames)) {
                non_decreasing += 1.0;
            }
        }
        monotonicity = non_decreasing / (double) (cnt - 1);
    } else {
        monotonicity = 1.0;
    }

    // ================= C. Path Confidence (dit_score.py:193-212) =================
    double path_confidence;
    int    path_len = (int) path.text_idx.size();
    if (path_len > 0) {
        double total_energy = 0.0;
        double total_steps  = 0.0;
        for (int k = 0; k < path_len; k++) {
            int    r  = path.text_idx[k];
            int    c  = path.time_idx[k];
            double pe = E(r, c);
            double sw = (type_mask[r] == 0) ? instrumental_weight : 1.0;
            total_energy += pe * sw;
            total_steps += sw;
        }
        path_confidence = (total_steps > 0.0) ? total_energy / total_steps : 0.0;
    } else {
        path_confidence = 0.0;
    }

    return { coverage, monotonicity, path_confidence };
}

// Full scoring pipeline: preprocess attention -> DTW -> metrics -> final score.
// Ported from dit_score.py:lyrics_alignment_info + calculate_score.
//
// attention: [n_layers, n_heads, tokens, frames] row-major
// decoded_tokens: per-token decoded strings for type mask generation
// config / config_count: which (layer, head) pairs to use
// medfilt_width: median filter window (1 = disabled)
//
// Returns the final lyrics_score = cov^2 * mono^2 * conf, clipped to [0, 1]
// and rounded to four decimal places like the Python API.
struct LyricScoreResult {
    double  coverage;
    double  monotonicity;
    double  confidence;
    double  lyrics_score;  // cov^2 * mono^2 * conf, clipped [0,1]
    DTWPath path;
};

struct LyricScoreComparison {
    LyricScoreResult lm;
    LyricScoreResult dit;
};

static LyricScoreResult calculate_lyric_score(const float *                    attention,
                                              int                              n_layers,
                                              int                              n_heads,
                                              int                              tokens,
                                              int                              frames,
                                              const std::vector<std::string> & decoded_tokens,
                                              const ScoreLayerHeadConfig *     config,
                                              int                              config_count,
                                              int                              medfilt_width = 1) {
    LyricScoreResult result = { 0, 0, 0, 0, {} };

    // 1. Preprocess attention (dit_score.py:237-239)
    std::vector<float> calc_matrix;
    std::vector<float> energy_matrix;
    if (!preprocess_attention(attention, n_layers, n_heads, tokens, frames, config, config_count, medfilt_width,
                              calc_matrix, energy_matrix)) {
        return result;
    }

    // 2. Generate token type mask (dit_score.py:248)
    std::vector<int> type_mask = generate_token_type_mask(decoded_tokens);

    // Safety check for shape mismatch (dit_score.py:251-252)
    if ((int) type_mask.size() != tokens) {
        type_mask.assign(tokens, 1);
    }

    // 3. DTW pathfinding on negated calc_matrix (dit_score.py:255)
    //    Negate so DTW finds the maximum-energy path (minimum cost = maximum energy)
    std::vector<float> neg_calc((size_t) tokens * frames);
    for (size_t i = 0; i < calc_matrix.size(); i++) {
        neg_calc[i] = -calc_matrix[i];
    }
    result.path = dtw_cpu(neg_calc.data(), tokens, frames);

    // 4. Compute metrics (dit_score.py:313-320)
    AlignmentMetrics m  = compute_alignment_metrics(energy_matrix, tokens, frames, result.path, type_mask);
    result.coverage     = m.coverage;
    result.monotonicity = m.monotonicity;
    result.confidence   = m.confidence;

    // 5. Final score: cov^2 * mono^2 * conf (dit_score.py:324-325)
    double final_score = (m.coverage * m.coverage) * (m.monotonicity * m.monotonicity) * m.confidence;
    if (final_score < 0.0) {
        final_score = 0.0;
    }
    if (final_score > 1.0) {
        final_score = 1.0;
    }
    result.lyrics_score = std::round(final_score * 10000.0) / 10000.0;

    return result;
}
