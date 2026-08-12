// test-dtw-score.cpp: unit tests for DTW pathfinding + lyric alignment scoring
//
// Verifies the C++ port against known-good values from the Python reference
// (acestep/core/scoring/_dtw.py and dit_score.py). No external dependencies
// beyond dtw-score.h itself — all test cases are self-contained.
//
// Usage:
//   ./test-dtw-score
//
// Exit code 0 = all tests passed, 1 = at least one failed.

#include "dtw-score.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static int g_pass = 0;
static int g_fail = 0;

#define CHECK(cond, msg)                                            \
    do {                                                            \
        if (cond) {                                                 \
            g_pass++;                                               \
        } else {                                                    \
            g_fail++;                                               \
            fprintf(stderr, "FAIL: %s (line %d)\n", msg, __LINE__); \
        }                                                           \
    } while (0)

#define CHECK_NEAR(a, b, eps, msg)                                                                             \
    do {                                                                                                       \
        if (fabs((double) (a) - (double) (b)) < (eps)) {                                                       \
            g_pass++;                                                                                          \
        } else {                                                                                               \
            g_fail++;                                                                                          \
            fprintf(stderr, "FAIL: %s — expected %.8g, got %.8g (line %d)\n", msg, (double) (b), (double) (a), \
                    __LINE__);                                                                                 \
        }                                                                                                      \
    } while (0)

// ============================================================================
// Test 1: DTW on a simple diagonal cost matrix
// A diagonal matrix should produce a path that goes straight from (0,0) to (N-1,M-1)
// ============================================================================
static void test_dtw_diagonal() {
    // 3x3 cost matrix where the diagonal is cheapest
    // Cost:  [[0, 9, 9],
    //         [9, 0, 9],
    //         [9, 9, 0]]
    // DTW on this should find the diagonal path.
    float   cost[] = { 0, 9, 9, 9, 0, 9, 9, 9, 0 };
    DTWPath path   = dtw_cpu(cost, 3, 3);

    // Path should visit (0,0), (1,1), (2,2)
    CHECK(path.text_idx.size() == 3, "diagonal DTW path length == 3");
    for (size_t k = 0; k < path.text_idx.size(); k++) {
        CHECK(path.text_idx[k] == (int) k, "diagonal DTW text_idx[k] == k");
        CHECK(path.time_idx[k] == (int) k, "diagonal DTW time_idx[k] == k");
    }
}

// ============================================================================
// Test 2: DTW on a matrix that forces a horizontal step
// ============================================================================
static void test_dtw_horizontal_step() {
    // 2x3 cost matrix:
    // [[0, 0, 0],
    //  [9, 9, 0]]
    // The optimal path should go right along row 0, then diagonal to (1,2).
    float   cost[] = { 0, 0, 0, 9, 9, 0 };
    DTWPath path   = dtw_cpu(cost, 2, 3);

    // Path must start at (0,0) and end at (1,2)
    CHECK(!path.text_idx.empty(), "horizontal DTW path non-empty");
    CHECK(path.text_idx.front() == 0, "horizontal DTW starts at text=0");
    CHECK(path.time_idx.front() == 0, "horizontal DTW starts at time=0");
    CHECK(path.text_idx.back() == 1, "horizontal DTW ends at text=1");
    CHECK(path.time_idx.back() == 2, "horizontal DTW ends at time=2");

    // Diagonal-first tie-breaking selects the minimum-cost three-step path:
    // (0,0) -> (0,1) -> (1,2). The old strict comparisons instead selected
    // the expensive four-step path through (1,0) and (1,1).
    CHECK(path.text_idx.size() == 3, "horizontal DTW path length == 3");
    if (path.text_idx.size() == 3) {
        CHECK(path.text_idx[0] == 0 && path.time_idx[0] == 0, "horizontal DTW step 0");
        CHECK(path.text_idx[1] == 0 && path.time_idx[1] == 1, "horizontal DTW step 1");
        CHECK(path.text_idx[2] == 1 && path.time_idx[2] == 2, "horizontal DTW step 2");
    }
}

// ============================================================================
// Test 3: Median filter — simple cases
// ============================================================================
static void test_median_filter() {
    // Window=3 on a smooth ramp: [1, 2, 3, 4, 5] -> [1, 2, 3, 4, 5]
    // (reflect padding keeps edges intact for monotonic data)
    std::vector<float> ramp     = { 1, 2, 3, 4, 5 };
    auto               filtered = median_filter_1d(ramp, 3);
    CHECK(filtered.size() == 5, "median filter preserves length");
    CHECK_NEAR(filtered[2], 3.0f, 1e-6, "median filter middle of ramp");

    // Window=3 removes a single spike: [1, 2, 100, 4, 5]
    // Reflect padding: [2,1,2,100,4,5,4]
    // Windows: [2,1,2]=2, [1,2,100]=2, [2,100,4]=4, [100,4,5]=5, [4,5,4]=4
    std::vector<float> spike = { 1, 2, 100, 4, 5 };
    auto               fs    = median_filter_1d(spike, 3);
    CHECK_NEAR(fs[0], 2.0f, 1e-6, "median filter spike removal [0]");
    CHECK_NEAR(fs[1], 2.0f, 1e-6, "median filter spike removal [1]");
    CHECK_NEAR(fs[2], 4.0f, 1e-6, "median filter spike removal [2]");
    CHECK_NEAR(fs[3], 5.0f, 1e-6, "median filter spike removal [3]");
    CHECK_NEAR(fs[4], 4.0f, 1e-6, "median filter spike removal [4]");

    // Window=1 = no-op
    auto noop = median_filter_1d(ramp, 1);
    CHECK(noop == ramp, "median filter width=1 is no-op");
}

// ============================================================================
// Test 4: Token type mask generation
// ============================================================================
static void test_token_type_mask() {
    // Tokens: "hello", "[", "Verse", "]", "world"
    // Mask:   1,       0,    0,       0,    1
    std::vector<std::string> tokens = { "hello", "[", "Verse", "]", "world" };
    auto                     mask   = generate_token_type_mask(tokens);
    CHECK(mask.size() == 5, "token type mask size");
    CHECK(mask[0] == 1, "token type mask: lyric before bracket");
    CHECK(mask[1] == 0, "token type mask: opening bracket");
    CHECK(mask[2] == 0, "token type mask: inside bracket");
    CHECK(mask[3] == 0, "token type mask: closing bracket");
    CHECK(mask[4] == 1, "token type mask: lyric after bracket");

    // Multi-token bracket: "[", "Intro", "Guitar", "]"
    std::vector<std::string> tokens2 = { "[", "Intro", "Guitar", "]", "sing" };
    auto                     mask2   = generate_token_type_mask(tokens2);
    CHECK(mask2[0] == 0, "multi-token bracket: open");
    CHECK(mask2[1] == 0, "multi-token bracket: inside 1");
    CHECK(mask2[2] == 0, "multi-token bracket: inside 2");
    CHECK(mask2[3] == 0, "multi-token bracket: close");
    CHECK(mask2[4] == 1, "multi-token bracket: after");
}

// ============================================================================
// Test 5: Full scoring pipeline with a synthetic attention matrix
// A perfect diagonal attention pattern should yield a high score.
// ============================================================================
static void test_scoring_perfect_alignment() {
    // Create a 4-token, 4-frame attention matrix where the diagonal has
    // high energy and off-diagonal is near zero. With 1 layer, 1 head.
    // This simulates perfect lyric-to-audio alignment.
    int                tokens = 4;
    int                frames = 4;
    std::vector<float> attn((size_t) tokens * frames, 0.01f);
    // Diagonal high energy
    for (int i = 0; i < tokens; i++) {
        attn[(size_t) i * frames + i] = 1.0f;
    }

    // All tokens are lyrics (no brackets)
    std::vector<std::string> decoded(tokens, "lyric");

    // Use a custom config: just layer 0, head 0
    ScoreLayerHeadConfig config[] = {
        { 0, 0 }
    };

    LyricScoreResult result = calculate_lyric_score(attn.data(), 1, 1, tokens, frames, decoded, config, 1, 1);

    // With perfect diagonal alignment:
    // - Coverage: all lyric rows have max energy 1.0 > 0.1, so coverage = 1.0
    // - Monotonicity: centroids are [0, 1, 2, 3], strictly increasing, so mono = 1.0
    // - Diagonal-first tie-breaking keeps the four-step path on the diagonal.
    // - Confidence and lyrics_score are therefore both 1.0.
    CHECK_NEAR(result.coverage, 1.0, 1e-6, "perfect alignment coverage");
    CHECK_NEAR(result.monotonicity, 1.0, 1e-6, "perfect alignment monotonicity");
    CHECK_NEAR(result.confidence, 1.0, 1e-6, "perfect alignment confidence");
    CHECK_NEAR(result.lyrics_score, 1.0, 1e-6, "perfect alignment final score");
    CHECK(result.path.text_idx.size() == 4, "perfect alignment path length == 4");
    for (size_t i = 0; i < result.path.text_idx.size(); i++) {
        CHECK(result.path.text_idx[i] == (int) i && result.path.time_idx[i] == (int) i,
              "perfect alignment path stays diagonal");
    }
}

// ============================================================================
// Test 6: Full scoring pipeline with poor alignment
// Uniform attention should yield low confidence and low score.
// ============================================================================
static void test_scoring_uniform_attention() {
    int                tokens = 4;
    int                frames = 8;
    // Uniform attention — no structure
    std::vector<float> attn((size_t) tokens * frames, 0.5f);

    std::vector<std::string> decoded(tokens, "lyric");
    ScoreLayerHeadConfig     config[] = {
        { 0, 0 }
    };

    LyricScoreResult result = calculate_lyric_score(attn.data(), 1, 1, tokens, frames, decoded, config, 1, 1);

    // With uniform attention, min-max normalization zeros everything out
    // (e_max - e_min < 1e-9), so energy_matrix = 0, confidence = 0, score = 0
    CHECK_NEAR(result.lyrics_score, 0.0, 1e-6, "uniform attention score is 0");
}

// ============================================================================
// Test 7: Scoring with structural tags (non-lyric tokens)
// ============================================================================
static void test_scoring_with_tags() {
    // 5 tokens: lyric, [Intro], lyric, lyric, lyric
    // 5 frames: perfect diagonal for lyric tokens, tag token gets low energy
    int                tokens = 5;
    int                frames = 5;
    std::vector<float> attn((size_t) tokens * frames, 0.01f);

    // Diagonal energy for all tokens (including tag)
    for (int i = 0; i < tokens; i++) {
        attn[(size_t) i * frames + i] = 1.0f;
    }

    // Token 1 is a structural tag
    std::vector<std::string> decoded  = { "sing", "[Intro]", "more", "singing", "here" };
    ScoreLayerHeadConfig     config[] = {
        { 0, 0 }
    };

    std::vector<int> type_mask = generate_token_type_mask(decoded);
    CHECK(type_mask == std::vector<int>({ 1, 0, 1, 1, 1 }), "tagged alignment mask excludes only the tag");

    LyricScoreResult result = calculate_lyric_score(attn.data(), 1, 1, tokens, frames, decoded, config, 1, 1);

    // Coverage: 4 lyric tokens, all with max energy 1.0 > 0.1 -> coverage = 1.0
    // Monotonicity: lyric centroids [0, 2, 3, 4] strictly increasing -> mono = 1.0
    // Diagonal-first tie-breaking keeps all five path steps on energy 1.0,
    // so confidence and lyrics_score are both 1.0.
    CHECK_NEAR(result.coverage, 1.0, 1e-6, "tagged alignment coverage");
    CHECK_NEAR(result.monotonicity, 1.0, 1e-6, "tagged alignment monotonicity");
    CHECK_NEAR(result.confidence, 1.0, 1e-6, "tagged alignment confidence");
    CHECK_NEAR(result.lyrics_score, 1.0, 1e-6, "tagged alignment final score");
}

// ============================================================================
// Test 8: Multi-head attention averaging
// ============================================================================
static void test_multi_head_averaging() {
    int tokens   = 3;
    int frames   = 3;
    int n_layers = 2;
    int n_heads  = 2;

    // 2 layers x 2 heads x 3 tokens x 3 frames
    std::vector<float> attn((size_t) n_layers * n_heads * tokens * frames, 0.0f);

    // Layer 0, Head 0: diagonal
    // Layer 0, Head 1: anti-diagonal
    // Layer 1, Head 0: diagonal
    // Layer 1, Head 1: diagonal
    auto A = [&](int l, int h, int t, int f) -> float & {
        return attn[(size_t) ((l * n_heads + h) * tokens + t) * frames + f];
    };

    // 3 of 4 heads have diagonal, 1 has anti-diagonal
    for (int t = 0; t < tokens; t++) {
        A(0, 0, t, t)              = 1.0f;
        A(0, 1, t, tokens - 1 - t) = 1.0f;  // anti-diagonal
        A(1, 0, t, t)              = 1.0f;
        A(1, 1, t, t)              = 1.0f;
    }

    // Select all 4 heads
    ScoreLayerHeadConfig config[] = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };

    std::vector<float> calc_matrix, energy_matrix;
    bool               ok =
        preprocess_attention(attn.data(), n_layers, n_heads, tokens, frames, config, 4, 1, calc_matrix, energy_matrix);
    CHECK(ok, "multi-head preprocess succeeded");

    // Average: 3/4 diagonal + 1/4 anti-diagonal
    // avg = [[0.75, 0, 0.25], [0, 1, 0], [0.25, 0, 0.75]]
    // min=0, max=1, range=1 -> normalization is identity
    //   energy[0,0] = 0.75, energy[0,2] = 0.25
    CHECK_NEAR(energy_matrix[0], 0.75f, 1e-5, "multi-head averaged energy [0,0]");
    CHECK_NEAR(energy_matrix[2], 0.25f, 1e-5, "multi-head averaged energy [0,2]");

    ScoreLayerHeadConfig negative_layer[] = {
        { -1, 0 }
    };
    ScoreLayerHeadConfig negative_head[] = {
        { 0, -1 }
    };
    CHECK(!preprocess_attention(attn.data(), n_layers, n_heads, tokens, frames, negative_layer, 1, 1, calc_matrix,
                                energy_matrix),
          "preprocess rejects a negative layer");
    CHECK(!preprocess_attention(attn.data(), n_layers, n_heads, tokens, frames, negative_head, 1, 1, calc_matrix,
                                energy_matrix),
          "preprocess rejects a negative head");
}

// ============================================================================
// Test 9: Remap absolute model layers to compact captured-layer slots
// ============================================================================
static void test_score_head_remap() {
    int                               captured_layers[] = { 2, 3, 4, 5, 6 };
    std::vector<ScoreLayerHeadConfig> remapped;
    bool ok = remap_score_heads(DEFAULT_2B_SCORE_HEADS, DEFAULT_2B_SCORE_HEADS_COUNT, captured_layers, 5, 16, remapped);
    CHECK(ok, "score head remap succeeds");
    CHECK(remapped.size() == 7, "score head remap preserves every configured head");
    CHECK(remapped[0].layer == 0 && remapped[0].head == 6, "absolute layer 2 maps to compact layer 0");
    CHECK(remapped[1].layer == 1 && remapped[1].head == 10, "absolute layer 3 maps to compact layer 1");
    CHECK(remapped[6].layer == 4 && remapped[6].head == 8, "absolute layer 6 maps to compact layer 4");

    int missing_layers[] = { 2, 3, 4 };
    CHECK(!remap_score_heads(DEFAULT_2B_SCORE_HEADS, DEFAULT_2B_SCORE_HEADS_COUNT, missing_layers, 3, 16, remapped),
          "score head remap rejects a partial capture");
}

// ============================================================================
// Test 10: GGML attention layout and per-batch pure-lyric slicing
// ============================================================================
static void test_attention_slice_layout() {
    const int          tokens  = 4;
    const int          frames  = 2;
    const int          heads   = 2;
    const int          batches = 2;
    std::vector<float> source((size_t) tokens * frames * heads * batches);

    for (int b = 0; b < batches; b++) {
        for (int h = 0; h < heads; h++) {
            for (int f = 0; f < frames; f++) {
                for (int t = 0; t < tokens; t++) {
                    size_t index  = (size_t) t + (size_t) tokens * (f + frames * (h + heads * b));
                    source[index] = (float) (1000 * b + 100 * h + 10 * f + t);
                }
            }
        }
    }

    std::vector<float> slice;
    bool ok = extract_attention_slice(source.data(), source.size(), tokens, frames, heads, batches, 1, 1, 2, slice);
    CHECK(ok, "attention slice succeeds");
    CHECK(slice.size() == 8, "attention slice has heads*lyric_tokens*frames elements");
    CHECK_NEAR(slice[0], 1001.0f, 1e-6, "attention slice batch 1, head 0, token 1, frame 0");
    CHECK_NEAR(slice[1], 1011.0f, 1e-6, "attention slice batch 1, head 0, token 1, frame 1");
    CHECK_NEAR(slice[6], 1102.0f, 1e-6, "attention slice batch 1, head 1, token 2, frame 0");
    CHECK_NEAR(slice[7], 1112.0f, 1e-6, "attention slice batch 1, head 1, token 2, frame 1");
}

// ============================================================================
// Test 11: Strip lyric prompt headers and trailing end-of-text tokens
// ============================================================================
static void test_lyric_token_segment() {
    std::vector<int>  raw     = { 10, 11, 20, 21, 151643, 151643 };
    LyricTokenSegment segment = extract_lyric_token_segment(raw, 2, 151643);
    CHECK(segment.start == 2, "lyric segment preserves encoder row offset");
    CHECK(segment.token_ids == std::vector<int>({ 20, 21 }), "lyric segment excludes header and end tokens");

    LyricTokenSegment invalid = extract_lyric_token_segment(raw, 10, 151643);
    CHECK(invalid.start == -1 && invalid.token_ids.empty(), "lyric segment rejects an invalid header length");
}

// ============================================================================
// Test 12: DTW path monotonicity (paths must be non-decreasing in both dims)
// ============================================================================
static void test_dtw_path_monotonic() {
    // Random-ish cost matrix
    int                N = 5, M = 7;
    std::vector<float> cost((size_t) N * M);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < M; j++) {
            // Cheaper near the diagonal (i/N * j/M)
            float di                 = (float) i / N;
            float dj                 = (float) j / M;
            cost[(size_t) i * M + j] = fabsf(di - dj) + 0.01f * ((i * 7 + j * 3) % 5);
        }
    }

    DTWPath path = dtw_cpu(cost.data(), N, M);

    // Path must be non-decreasing in both text and time
    for (size_t k = 1; k < path.text_idx.size(); k++) {
        CHECK(path.text_idx[k] >= path.text_idx[k - 1], "DTW path text non-decreasing");
        CHECK(path.time_idx[k] >= path.time_idx[k - 1], "DTW path time non-decreasing");
    }

    // Path must start at (0,0) and end at (N-1, M-1)
    CHECK(path.text_idx.front() == 0, "DTW path starts at text 0");
    CHECK(path.time_idx.front() == 0, "DTW path starts at time 0");
    CHECK(path.text_idx.back() == N - 1, "DTW path ends at text N-1");
    CHECK(path.time_idx.back() == M - 1, "DTW path ends at time M-1");
}

/// Run the self-contained DTW and lyric-alignment regression suite.
int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    fprintf(stderr, "=== DTW + Lyric Score Tests ===\n\n");

    test_dtw_diagonal();
    fprintf(stderr, "test_dtw_diagonal: done\n");

    test_dtw_horizontal_step();
    fprintf(stderr, "test_dtw_horizontal_step: done\n");

    test_median_filter();
    fprintf(stderr, "test_median_filter: done\n");

    test_token_type_mask();
    fprintf(stderr, "test_token_type_mask: done\n");

    test_scoring_perfect_alignment();
    fprintf(stderr, "test_scoring_perfect_alignment: done\n");

    test_scoring_uniform_attention();
    fprintf(stderr, "test_scoring_uniform_attention: done\n");

    test_scoring_with_tags();
    fprintf(stderr, "test_scoring_with_tags: done\n");

    test_multi_head_averaging();
    fprintf(stderr, "test_multi_head_averaging: done\n");

    test_score_head_remap();
    fprintf(stderr, "test_score_head_remap: done\n");

    test_attention_slice_layout();
    fprintf(stderr, "test_attention_slice_layout: done\n");

    test_lyric_token_segment();
    fprintf(stderr, "test_lyric_token_segment: done\n");

    test_dtw_path_monotonic();
    fprintf(stderr, "test_dtw_path_monotonic: done\n");

    fprintf(stderr, "\n=== Results: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail > 0 ? 1 : 0;
}
