//
// request.cpp - AceStep request JSON read/write (yyjson)
//

#include "request.h"

#include "yyjson.h"

#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const yyjson_write_flag WRITE_FLAGS =
    YYJSON_WRITE_PRETTY | YYJSON_WRITE_PRETTY_TWO_SPACES | YYJSON_WRITE_FP_TO_FIXED(2);

// Defaults (aligned with Python GenerationParams)
void request_init(AceRequest * r) {
    r->caption = "";
    r->lyrics  = "";

    r->bpm                  = 0;
    r->duration             = 0.0f;
    r->keyscale             = "";
    r->timesignature        = "";
    r->vocal_language       = "";
    r->lm_batch_size        = 1;
    r->synth_batch_size     = 1;
    r->seed                 = -1;
    r->lm_temperature       = 0.85f;
    r->lm_cfg_scale         = 2.0f;
    r->lm_top_p             = 0.9f;
    r->lm_top_k             = 0;
    r->lm_negative_prompt   = "";
    r->use_cot_caption      = true;
    r->audio_codes          = "";
    r->inference_steps      = 0;     // 0 = auto (turbo: 8, base/sft: 50)
    r->guidance_scale       = 0.0f;  // 0 = auto (1.0 for all models)
    r->shift                = 0.0f;  // 0 = auto (turbo: 3.0, base/sft: 1.0)
    r->audio_cover_strength = 0.5f;
    r->cover_noise_strength = 0.0f;
    r->repainting_start     = -1.0f;
    r->repainting_end       = -1.0f;
    r->task_type            = "";
    r->track                = "";
}

// helper: get yyjson string as std::string
static inline std::string yy_str(yyjson_val * v) {
    return std::string(yyjson_get_str(v), yyjson_get_len(v));
}

// populate AceRequest fields from a yyjson object (must be pre-initialized)
static void request_parse_obj(yyjson_val * obj, AceRequest * r) {
    yyjson_val * v;

    // strings
    if ((v = yyjson_obj_get(obj, "caption")) && yyjson_is_str(v)) {
        r->caption = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lyrics")) && yyjson_is_str(v)) {
        r->lyrics = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "keyscale")) && yyjson_is_str(v)) {
        r->keyscale = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "timesignature")) && yyjson_is_str(v)) {
        r->timesignature = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "vocal_language")) && yyjson_is_str(v)) {
        r->vocal_language = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "audio_codes")) && yyjson_is_str(v)) {
        r->audio_codes = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_negative_prompt")) && yyjson_is_str(v)) {
        r->lm_negative_prompt = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "task_type")) && yyjson_is_str(v)) {
        r->task_type = yy_str(v);
    }
    if ((v = yyjson_obj_get(obj, "track")) && yyjson_is_str(v)) {
        r->track = yy_str(v);
    }

    // ints
    if ((v = yyjson_obj_get(obj, "bpm")) && yyjson_is_num(v)) {
        r->bpm = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_batch_size")) && yyjson_is_num(v)) {
        r->lm_batch_size = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "synth_batch_size")) && yyjson_is_num(v)) {
        r->synth_batch_size = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "seed")) && yyjson_is_num(v)) {
        r->seed = (int64_t) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_top_k")) && yyjson_is_num(v)) {
        r->lm_top_k = (int) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "inference_steps")) && yyjson_is_num(v)) {
        r->inference_steps = (int) yyjson_get_num(v);
    }

    // floats
    if ((v = yyjson_obj_get(obj, "duration")) && yyjson_is_num(v)) {
        r->duration = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_temperature")) && yyjson_is_num(v)) {
        r->lm_temperature = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_cfg_scale")) && yyjson_is_num(v)) {
        r->lm_cfg_scale = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "lm_top_p")) && yyjson_is_num(v)) {
        r->lm_top_p = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "guidance_scale")) && yyjson_is_num(v)) {
        r->guidance_scale = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "shift")) && yyjson_is_num(v)) {
        r->shift = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "audio_cover_strength")) && yyjson_is_num(v)) {
        r->audio_cover_strength = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "cover_noise_strength")) && yyjson_is_num(v)) {
        r->cover_noise_strength = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "repainting_start")) && yyjson_is_num(v)) {
        r->repainting_start = (float) yyjson_get_num(v);
    }
    if ((v = yyjson_obj_get(obj, "repainting_end")) && yyjson_is_num(v)) {
        r->repainting_end = (float) yyjson_get_num(v);
    }

    // bool
    if ((v = yyjson_obj_get(obj, "use_cot_caption"))) {
        if (yyjson_is_bool(v)) {
            r->use_cot_caption = yyjson_get_bool(v);
        } else if (yyjson_is_str(v)) {
            const char * s     = yyjson_get_str(v);
            r->use_cot_caption = (strcmp(s, "true") == 0 || strcmp(s, "1") == 0);
        }
    }
}

// Core parser: takes a raw JSON string. Used by the server directly.
bool request_parse_json(AceRequest * r, const char * json) {
    request_init(r);

    yyjson_doc * doc = yyjson_read(json, strlen(json), 0);
    if (!doc) {
        return false;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);
    if (!yyjson_is_obj(root)) {
        yyjson_doc_free(doc);
        return false;
    }

    request_parse_obj(root, r);
    yyjson_doc_free(doc);
    return true;
}

// Parse JSON: single object {} or array [{}, ...] into a vector.
bool request_parse_json_array(const char * json, std::vector<AceRequest> * out) {
    yyjson_doc * doc = yyjson_read(json, strlen(json), 0);
    if (!doc) {
        return false;
    }

    yyjson_val * root = yyjson_doc_get_root(doc);

    if (yyjson_is_obj(root)) {
        AceRequest r;
        request_init(&r);
        request_parse_obj(root, &r);
        out->push_back(r);
    } else if (yyjson_is_arr(root)) {
        size_t       idx, max;
        yyjson_val * val;
        yyjson_arr_foreach(root, idx, max, val) {
            if (!yyjson_is_obj(val)) {
                yyjson_doc_free(doc);
                return false;
            }
            AceRequest r;
            request_init(&r);
            request_parse_obj(val, &r);
            out->push_back(r);
        }
    } else {
        yyjson_doc_free(doc);
        return false;
    }

    yyjson_doc_free(doc);
    return !out->empty();
}

// File I/O
static std::string read_file(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return "";
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::string buf((size_t) sz, '\0');
    size_t      nr = fread(&buf[0], 1, (size_t) sz, f);
    fclose(f);
    if ((long) nr != sz) {
        buf.resize(nr);
    }
    return buf;
}

// File wrapper: reads JSON from disk, then delegates to request_parse_json.
bool request_parse(AceRequest * r, const char * path) {
    std::string json = read_file(path);
    if (json.empty()) {
        fprintf(stderr, "[Request] ERROR: cannot read %s\n", path);
        return false;
    }

    if (!request_parse_json(r, json.c_str())) {
        fprintf(stderr, "[Request] ERROR: malformed JSON in %s\n", path);
        return false;
    }

    fprintf(stderr, "[Request] Parsed %s\n", path);
    return true;
}

// build a yyjson mutable document from an AceRequest
static yyjson_mut_doc * request_build_doc(const AceRequest * r) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);

    yyjson_mut_obj_add_str(doc, root, "caption", r->caption.c_str());
    yyjson_mut_obj_add_str(doc, root, "lyrics", r->lyrics.c_str());
    yyjson_mut_obj_add_int(doc, root, "bpm", r->bpm);
    yyjson_mut_obj_add_real(doc, root, "duration", r->duration);
    yyjson_mut_obj_add_str(doc, root, "keyscale", r->keyscale.c_str());
    yyjson_mut_obj_add_str(doc, root, "timesignature", r->timesignature.c_str());
    yyjson_mut_obj_add_str(doc, root, "vocal_language", r->vocal_language.c_str());
    yyjson_mut_obj_add_sint(doc, root, "seed", r->seed);
    yyjson_mut_obj_add_int(doc, root, "lm_batch_size", r->lm_batch_size);
    yyjson_mut_obj_add_int(doc, root, "synth_batch_size", r->synth_batch_size);
    yyjson_mut_obj_add_real(doc, root, "lm_temperature", r->lm_temperature);
    yyjson_mut_obj_add_real(doc, root, "lm_cfg_scale", r->lm_cfg_scale);
    yyjson_mut_obj_add_real(doc, root, "lm_top_p", r->lm_top_p);
    yyjson_mut_obj_add_int(doc, root, "lm_top_k", r->lm_top_k);
    yyjson_mut_obj_add_str(doc, root, "lm_negative_prompt", r->lm_negative_prompt.c_str());
    yyjson_mut_obj_add_bool(doc, root, "use_cot_caption", r->use_cot_caption);
    yyjson_mut_obj_add_int(doc, root, "inference_steps", r->inference_steps);
    yyjson_mut_obj_add_real(doc, root, "guidance_scale", r->guidance_scale);
    yyjson_mut_obj_add_real(doc, root, "shift", r->shift);
    yyjson_mut_obj_add_real(doc, root, "audio_cover_strength", r->audio_cover_strength);
    yyjson_mut_obj_add_real(doc, root, "cover_noise_strength", r->cover_noise_strength);
    yyjson_mut_obj_add_real(doc, root, "repainting_start", r->repainting_start);
    yyjson_mut_obj_add_real(doc, root, "repainting_end", r->repainting_end);
    if (!r->task_type.empty()) {
        yyjson_mut_obj_add_str(doc, root, "task_type", r->task_type.c_str());
    }
    if (!r->track.empty()) {
        yyjson_mut_obj_add_str(doc, root, "track", r->track.c_str());
    }
    yyjson_mut_obj_add_str(doc, root, "audio_codes", r->audio_codes.c_str());

    return doc;
}

bool request_write(const AceRequest * r, const char * path) {
    FILE * f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[Request] ERROR: cannot write %s\n", path);
        return false;
    }

    yyjson_mut_doc * doc = request_build_doc(r);
    size_t           len;
    char *           json = yyjson_mut_write(doc, WRITE_FLAGS | YYJSON_WRITE_NEWLINE_AT_END, &len);
    yyjson_mut_doc_free(doc);

    fwrite(json, 1, len, f);
    free(json);
    fclose(f);

    fprintf(stderr, "[Request] Wrote %s\n", path);
    return true;
}

std::string request_to_json(const AceRequest * r) {
    yyjson_mut_doc * doc = request_build_doc(r);
    size_t           len;
    char *           json = yyjson_mut_write(doc, WRITE_FLAGS, &len);
    yyjson_mut_doc_free(doc);

    std::string result(json, len);
    free(json);
    return result;
}

void request_dump(const AceRequest * r, FILE * f) {
    fprintf(f, "[Request] seed=%lld lm_batch=%d synth_batch=%d\n", (long long) r->seed, r->lm_batch_size,
            r->synth_batch_size);
    fprintf(f, "[Request] caption: %.60s%s\n", r->caption.c_str(), r->caption.size() > 60 ? "..." : "");
    fprintf(f, "[Request] lyrics: %zu bytes\n", r->lyrics.size());
    fprintf(f, "[Request] bpm=%d dur=%.0f key=%s ts=%s lang=%s\n", r->bpm, r->duration, r->keyscale.c_str(),
            r->timesignature.c_str(), r->vocal_language.c_str());
    fprintf(f, "[Request] lm: temp=%.2f cfg=%.1f top_p=%.2f top_k=%d\n", r->lm_temperature, r->lm_cfg_scale,
            r->lm_top_p, r->lm_top_k);
    fprintf(f, "[Request] dit: steps=%d guidance=%.1f shift=%.1f\n", r->inference_steps, r->guidance_scale, r->shift);
    if (r->audio_cover_strength != 0.5f || r->cover_noise_strength != 0.0f) {
        fprintf(f, "[Request] cover: strength=%.2f noise_strength=%.2f\n", r->audio_cover_strength,
                r->cover_noise_strength);
    }
    if (r->repainting_start >= 0.0f || r->repainting_end >= 0.0f) {
        fprintf(f, "[Request] repaint: start=%.1f end=%.1f\n", r->repainting_start, r->repainting_end);
    }
    if (!r->task_type.empty()) {
        fprintf(f, "[Request] task_type: %s\n", r->task_type.c_str());
    }
    if (!r->track.empty()) {
        fprintf(f, "[Request] track: %s\n", r->track.c_str());
    }
    fprintf(f, "[Request] audio_codes: %s\n", r->audio_codes.empty() ? "(none)" : "(present)");
}

void request_resolve_seed(AceRequest * r) {
    if (r->seed < 0) {
        std::random_device rd;
        r->seed = (int64_t) rd();
    }
}
