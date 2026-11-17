// test-progress: the AceProgress report/cancel contract (#18).
//
// Header-only -- exercises ace_progress() and the pre-loop + per-step reporting
// pattern the pipelines use, without loading any model. It checks that a run
// emits ordered (stage, step, total) reports for each stage, that step 0 is
// reported before the first unit of work, and that returning true cancels.
#include "progress.h"

#include <cstdio>
#include <vector>

static int failures = 0;

#define CHECK(cond)                                                         \
    do {                                                                    \
        if (!(cond)) {                                                      \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            failures++;                                                     \
        }                                                                   \
    } while (0)

struct Report {
    AceStage stage;
    int      step;
    int      total;
};

// Records every call; cancels when the level-triggered flag is set or when step
// reaches cancel_at (-1 = never).
struct Recorder {
    std::vector<Report> reports;
    int                 cancel_at = -1;
    bool                flag      = false;
};

static bool record_fn(void * data, AceStage stage, int step, int total) {
    auto * r = static_cast<Recorder *>(data);
    r->reports.push_back({ stage, step, total });
    return r->flag || (r->cancel_at >= 0 && step >= r->cancel_at);
}

// Mirrors what every pipeline loop does: a bare cancel poll before the loop (so
// an already-cancelled stage aborts even when total == 0), then poll+report at
// the top of each iteration -- iteration 0 (step 0) sizes the bar. Returns the
// step it cancelled at, or total if it ran fully.
static int run_stage(const AceProgress & p, AceStage stage, int total) {
    if (ace_cancelled(p, stage)) {  // no progress report; honours a cancel before the loop
        return -1;
    }
    for (int step = 0; step < total; step++) {
        if (ace_progress(p, stage, step, total)) {
            return step;  // cancelled
        }
        // ... work ...
    }
    return total;
}

int main() {
    // 1. A default (empty) AceProgress never cancels and never calls anything.
    {
        AceProgress none;
        CHECK(!ace_progress(none, ACE_STAGE_DIT, 3, 10));
        CHECK(run_stage(none, ACE_STAGE_DIT, 5) == 5);  // ran fully, no callback
    }

    // 2. A run reports ordered (stage, step, total) covering all four stages, with
    //    step 0 emitted before the first unit of work.
    {
        Recorder    rec;
        AceProgress p{ record_fn, &rec };

        const struct {
            AceStage stage;
            int      total;
        } stages[] = {
            { ACE_STAGE_LM,         4 },
            { ACE_STAGE_DIT,        3 },
            { ACE_STAGE_VAE_ENCODE, 2 },
            { ACE_STAGE_VAE_DECODE, 3 },
        };

        for (auto & s : stages) {
            CHECK(run_stage(p, s.stage, s.total) == s.total);
        }

        // The progress reports (step >= 0; the bare cancel polls at step -1 are
        // not progress) are, per stage, exactly 0..total-1: step 0 first (the
        // sizing report, before any work), then non-decreasing, total constant.
        std::vector<Report> prog;
        for (auto & r : rec.reports) {
            if (r.step >= 0) {
                prog.push_back(r);
            }
        }
        size_t i = 0;
        for (auto & s : stages) {
            for (int step = 0; step < s.total; step++, i++) {
                CHECK(i < prog.size());
                CHECK(prog[i].stage == s.stage);
                CHECK(prog[i].step == step);
                CHECK(prog[i].total == s.total);
            }
        }
        CHECK(i == prog.size());  // no extra progress reports
    }

    // 3. Returning true cancels at that step, and the loop stops there.
    {
        Recorder rec;
        rec.cancel_at = 2;
        AceProgress p{ record_fn, &rec };
        CHECK(run_stage(p, ACE_STAGE_DIT, 10) == 2);  // stopped at step 2, not 10
        // sizing(0) + poll(0),poll(1),poll(2)=cancel -> last recorded step is 2.
        CHECK(!rec.reports.empty());
        CHECK(rec.reports.back().step == 2);
    }

    // 4. ace_cancelled is a bare cancel poll: it honours the cancel return but
    //    marks itself with step < 0 so a consumer can skip it for progress.
    {
        Recorder    rec;
        AceProgress p{ record_fn, &rec };
        CHECK(!ace_cancelled(p, ACE_STAGE_MP3));  // not cancelled
        CHECK(rec.reports.size() == 1);
        CHECK(rec.reports.back().stage == ACE_STAGE_MP3);
        CHECK(rec.reports.back().step < 0);  // distinguishable from a real report

        rec.flag = true;                     // now the (level-triggered) callback cancels
        CHECK(ace_cancelled(p, ACE_STAGE_MP3));

        AceProgress none;
        CHECK(!ace_cancelled(none, ACE_STAGE_MP3));  // null fn: never cancels
    }

    if (failures == 0) {
        printf("test-progress: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-progress: %d check(s) failed\n", failures);
    return 1;
}
