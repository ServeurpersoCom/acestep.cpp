#pragma once
// lyric-timing.h: compact ACE-Step lyric timing from DiT cross-attention.
//
// Mirrors the ACE-Step Python MusicStampsAligner path: selected attention
// heads -> bidirectional consensus denoising -> DTW -> line timestamps. This
// header is CPU-only and intentionally small so the server can emit an opt-in
// JSON sidecar without depending on Python.

#include "bpe.h"
#include "sampling.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

static float lyric_timing_median(std::vector<float> & tmp) {
    if (tmp.empty()) {
        return 0.0f;
    }
    size_t mid = tmp.size() / 2;
    std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
    float hi = tmp[mid];
    if ((tmp.size() & 1) != 0) {
        return hi;
    }
    std::nth_element(tmp.begin(), tmp.begin() + mid - 1, tmp.begin() + mid);
    return 0.5f * (tmp[mid - 1] + hi);
}

static std::string lyric_timing_trim(const std::string & s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) {
        a++;
    }
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) {
        b--;
    }
    return s.substr(a, b - a);
}

static std::string lyric_timing_json_escape(const std::string & s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[7];
                    snprintf(buf, sizeof(buf), "\\u%04x", (unsigned int) c);
                    out += buf;
                } else {
                    out += (char) c;
                }
                break;
        }
    }
    return out;
}

static std::string lyric_timing_ts(float seconds) {
    if (seconds < 0.0f) {
        seconds = 0.0f;
    }
    int   minutes = (int) (seconds / 60.0f);
    float sec     = seconds - (float) minutes * 60.0f;
    char  buf[32];
    snprintf(buf, sizeof(buf), "[%02d:%05.2f]", minutes, sec);
    return std::string(buf);
}

static void lyric_timing_decode_incremental(const BPETokenizer &     bpe,
                                            const std::vector<int> & token_ids,
                                            std::vector<std::string> & token_texts) {
    token_texts.clear();
    token_texts.reserve(token_ids.size());

    std::vector<int> prefix;
    prefix.reserve(token_ids.size());
    std::string prev;
    for (int id : token_ids) {
        prefix.push_back(id);
        std::string curr = bpe_decode(bpe, prefix);
        std::string piece;
        if (curr.size() >= prev.size()) {
            piece = curr.substr(prev.size());
        }
        token_texts.push_back(piece);
        prev = curr;
    }
}

static bool lyric_timing_calc_matrix(const std::vector<float> & heads,
                                     int                        head_count,
                                     int                        token_count,
                                     int                        frame_count,
                                     std::vector<float> &       calc,
                                     std::string &              error,
                                     float                      violence_level = 2.0f) {
    if (head_count <= 0 || token_count <= 0 || frame_count <= 0) {
        error = "empty attention capture";
        return false;
    }
    const size_t expected = (size_t) head_count * token_count * frame_count;
    if (heads.size() != expected) {
        error = "attention capture shape mismatch";
        return false;
    }

    std::vector<float> processed(expected, 0.0f);

    for (int h = 0; h < head_count; h++) {
        const size_t h_off = (size_t) h * token_count * frame_count;

        std::vector<float> row_prob((size_t) token_count * frame_count);
        std::vector<float> col_prob((size_t) token_count * frame_count);

        for (int tok = 0; tok < token_count; tok++) {
            const size_t row = (size_t) tok * frame_count;
            float        mx  = -std::numeric_limits<float>::infinity();
            for (int fr = 0; fr < frame_count; fr++) {
                mx = (std::max)(mx, heads[h_off + row + fr]);
            }
            double sum = 0.0;
            for (int fr = 0; fr < frame_count; fr++) {
                float v          = expf(heads[h_off + row + fr] - mx);
                row_prob[row + fr] = v;
                sum += v;
            }
            float inv = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
            for (int fr = 0; fr < frame_count; fr++) {
                row_prob[row + fr] *= inv;
            }
        }

        for (int fr = 0; fr < frame_count; fr++) {
            float mx = -std::numeric_limits<float>::infinity();
            for (int tok = 0; tok < token_count; tok++) {
                mx = (std::max)(mx, heads[h_off + (size_t) tok * frame_count + fr]);
            }
            double sum = 0.0;
            for (int tok = 0; tok < token_count; tok++) {
                size_t idx    = (size_t) tok * frame_count + fr;
                float  v      = expf(heads[h_off + idx] - mx);
                col_prob[idx] = v;
                sum += v;
            }
            float inv = sum > 0.0 ? (float) (1.0 / sum) : 0.0f;
            for (int tok = 0; tok < token_count; tok++) {
                col_prob[(size_t) tok * frame_count + fr] *= inv;
            }
        }

        for (int tok = 0; tok < token_count; tok++) {
            for (int fr = 0; fr < frame_count; fr++) {
                size_t idx             = (size_t) tok * frame_count + fr;
                processed[h_off + idx] = row_prob[idx] * col_prob[idx];
            }
        }
    }

    std::vector<float> tmp;
    tmp.reserve((size_t) (std::max)(token_count, frame_count));

    for (int h = 0; h < head_count; h++) {
        const size_t h_off = (size_t) h * token_count * frame_count;
        for (int tok = 0; tok < token_count; tok++) {
            tmp.clear();
            for (int fr = 0; fr < frame_count; fr++) {
                tmp.push_back(processed[h_off + (size_t) tok * frame_count + fr]);
            }
            float med = lyric_timing_median(tmp);
            for (int fr = 0; fr < frame_count; fr++) {
                size_t idx = h_off + (size_t) tok * frame_count + fr;
                processed[idx] = (std::max)(0.0f, processed[idx] - violence_level * med);
            }
        }

        for (int fr = 0; fr < frame_count; fr++) {
            tmp.clear();
            for (int tok = 0; tok < token_count; tok++) {
                tmp.push_back(processed[h_off + (size_t) tok * frame_count + fr]);
            }
            float med = lyric_timing_median(tmp);
            for (int tok = 0; tok < token_count; tok++) {
                size_t idx = h_off + (size_t) tok * frame_count + fr;
                processed[idx] = (std::max)(0.0f, processed[idx] - violence_level * med);
            }
        }
    }

    double sum = 0.0;
    for (float & v : processed) {
        v = v * v;
        sum += v;
    }
    double mean = sum / (double) processed.size();
    double var  = 0.0;
    for (float v : processed) {
        double d = (double) v - mean;
        var += d * d;
    }
    double stddev = sqrt(var / (double) processed.size());
    double denom  = stddev + 1e-9;

    calc.assign((size_t) token_count * frame_count, 0.0f);
    for (int h = 0; h < head_count; h++) {
        const size_t h_off = (size_t) h * token_count * frame_count;
        for (int tok = 0; tok < token_count; tok++) {
            for (int fr = 0; fr < frame_count; fr++) {
                size_t idx = (size_t) tok * frame_count + fr;
                calc[idx] += (float) (((double) processed[h_off + idx] - mean) / denom);
            }
        }
    }
    float inv_heads = 1.0f / (float) head_count;
    for (float & v : calc) {
        v *= inv_heads;
    }
    return true;
}

static void lyric_timing_dtw(const std::vector<float> & calc,
                             int                        token_count,
                             int                        frame_count,
                             std::vector<int> &         text_path,
                             std::vector<int> &         time_path) {
    const float INF = std::numeric_limits<float>::infinity();
    const int   W   = frame_count + 1;
    std::vector<float> cost((size_t) (token_count + 1) * (frame_count + 1), INF);
    std::vector<int8_t> trace((size_t) (token_count + 1) * (frame_count + 1), -1);
    cost[0] = 0.0f;

    for (int j = 1; j <= frame_count; j++) {
        for (int i = 1; i <= token_count; i++) {
            float c0 = cost[(size_t) (i - 1) * W + (j - 1)];
            float c1 = cost[(size_t) (i - 1) * W + j];
            float c2 = cost[(size_t) i * W + (j - 1)];
            float c;
            int8_t t;
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
            cost[(size_t) i * W + j] = -calc[(size_t) (i - 1) * frame_count + (j - 1)] + c;
            trace[(size_t) i * W + j] = t;
        }
    }

    text_path.clear();
    time_path.clear();
    text_path.reserve(token_count + frame_count);
    time_path.reserve(token_count + frame_count);

    int i = token_count;
    int j = frame_count;
    while (i > 0 || j > 0) {
        text_path.push_back(i - 1);
        time_path.push_back(j - 1);
        int8_t t = (i >= 0 && j >= 0) ? trace[(size_t) i * W + j] : -1;
        if (i == 0) {
            t = 2;
        } else if (j == 0) {
            t = 1;
        }
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
    std::reverse(text_path.begin(), text_path.end());
    std::reverse(time_path.begin(), time_path.end());
}

static bool lyric_timing_build_json(const BPETokenizer &      bpe,
                                    const std::vector<int> &  token_ids,
                                    const std::vector<float> & selected_heads,
                                    int                        head_count,
                                    int                        frame_count,
                                    float                      total_duration,
                                    std::string &              json,
                                    std::string &              error) {
    int token_count = (int) token_ids.size();
    std::vector<float> calc;
    if (!lyric_timing_calc_matrix(selected_heads, head_count, token_count, frame_count, calc, error)) {
        return false;
    }

    std::vector<int> text_path;
    std::vector<int> time_path;
    lyric_timing_dtw(calc, token_count, frame_count, text_path, time_path);

    float seconds_per_frame = frame_count > 0 ? total_duration / (float) frame_count : 0.0f;
    std::vector<float> tok_start(token_count, 0.0f);
    std::vector<float> tok_end(token_count, 0.0f);
    std::vector<int>   tok_seen(token_count, 0);
    for (size_t p = 0; p < text_path.size(); p++) {
        int tok = text_path[p];
        int fr  = time_path[p];
        if (tok < 0 || tok >= token_count || fr < 0 || fr >= frame_count) {
            continue;
        }
        float sec = (float) fr * seconds_per_frame;
        if (!tok_seen[tok]) {
            tok_start[tok] = sec;
            tok_end[tok]   = sec;
            tok_seen[tok]  = 1;
        } else {
            tok_end[tok] = sec;
        }
    }
    for (int i = 0; i < token_count; i++) {
        if (!tok_seen[i]) {
            tok_start[i] = i > 0 ? tok_end[i - 1] : 0.0f;
            tok_end[i]   = tok_start[i];
        }
        if (tok_end[i] < tok_start[i]) {
            tok_end[i] = tok_start[i];
        }
    }

    std::vector<std::string> token_texts;
    lyric_timing_decode_incremental(bpe, token_ids, token_texts);

    struct LineRange {
        int         a;
        int         b;
        std::string text;
        float       start;
        float       end;
    };
    std::vector<LineRange> lines;
    int                    line_start = 0;
    for (int i = 0; i < token_count; i++) {
        if (token_texts[i].find('\n') != std::string::npos) {
            std::vector<int> slice(token_ids.begin() + line_start, token_ids.begin() + i + 1);
            std::string      text = lyric_timing_trim(bpe_decode(bpe, slice));
            if (!text.empty()) {
                lines.push_back({ line_start, i, text, tok_start[line_start], tok_end[i] });
            }
            line_start = i + 1;
        }
    }
    if (line_start < token_count) {
        std::vector<int> slice(token_ids.begin() + line_start, token_ids.end());
        std::string      text = lyric_timing_trim(bpe_decode(bpe, slice));
        if (!text.empty()) {
            lines.push_back({ line_start, token_count - 1, text, tok_start[line_start], tok_end[token_count - 1] });
        }
    }

    std::ostringstream lrc;
    for (size_t i = 0; i < lines.size(); i++) {
        if (i) {
            lrc << "\n";
        }
        lrc << lyric_timing_ts(lines[i].start) << lines[i].text;
    }

    std::ostringstream out;
    out << std::fixed << std::setprecision(3);
    out << "{";
    out << "\"version\":1,";
    out << "\"source\":\"ace_step_dit_attention\",";
    out << "\"duration\":" << total_duration << ",";
    out << "\"frame_count\":" << frame_count << ",";
    out << "\"seconds_per_frame\":" << seconds_per_frame << ",";
    out << "\"token_count\":" << token_count << ",";
    out << "\"head_count\":" << head_count << ",";
    out << "\"lrc\":\"" << lyric_timing_json_escape(lrc.str()) << "\",";
    out << "\"lines\":[";
    for (size_t li = 0; li < lines.size(); li++) {
        if (li) {
            out << ",";
        }
        const LineRange & line = lines[li];
        out << "{";
        out << "\"text\":\"" << lyric_timing_json_escape(line.text) << "\",";
        out << "\"start\":" << line.start << ",";
        out << "\"end\":" << line.end << ",";
        out << "\"tokens\":[";
        for (int ti = line.a; ti <= line.b; ti++) {
            if (ti > line.a) {
                out << ",";
            }
            out << "{\"index\":" << ti << ",\"token_id\":" << token_ids[ti] << ",\"start\":" << tok_start[ti]
                << ",\"end\":" << tok_end[ti] << "}";
        }
        out << "]}";
    }
    out << "]}";

    json = out.str();
    return true;
}

static std::string lyric_timing_error_json(const std::string & error) {
    return std::string("{\"version\":1,\"source\":\"ace_step_dit_attention\",\"error\":\"") +
           lyric_timing_json_escape(error) + "\"}";
}
