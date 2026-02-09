// test_bpe.cpp - Test BPE tokenizer against Python reference
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "bpe.h"

int main(int argc, char **argv) {
    const char *dir = argc > 1 ? argv[1] : "checkpoints/Qwen3-Embedding-0.6B";

    BPETokenizer tok;
    if (!load_bpe_tokenizer(&tok, dir)) {
        fprintf(stderr, "Failed to load tokenizer from %s\n", dir);
        return 1;
    }

    // Test cases: text -> expected ids (from Python tokenizer)
    struct TestCase {
        const char *name;
        const char *text;
        std::vector<int> expected;
        bool add_eos;
    };

    std::vector<TestCase> tests = {
        {"hello", "hello", {14990, 151643}, true},
        {"hello_no_eos", "hello", {14990}, false},
        {"# Instruction", "# Instruction\n",
         {2, 29051, 198}, false},
        {"120 BPM", "120 BPM",
         {16, 17, 15, 88219}, false},
        {"French", "Victoire du code natif",
         {36125, 41727, 3845, 2038, 17588, 333}, false},
        {"brackets", "[Couplet 1]\n",
         {43504, 283, 10819, 220, 16, 921}, false},
        {"endoftext_in_text", "hello<|endoftext|>\n",
         {14990, 151643, 198}, false},
    };

    int pass = 0, fail = 0;
    for (auto &tc : tests) {
        auto ids = bpe_encode(&tok, tc.text, tc.add_eos);
        bool ok = (ids == tc.expected);
        if (ok) {
            printf("  OK %s\n", tc.name);
            pass++;
        } else {
            printf("  FAIL %s\n", tc.name);
            printf("     got:      ");
            for (int id : ids) printf("%d,", id);
            printf("\n     expected: ");
            for (int id : tc.expected) printf("%d,", id);
            printf("\n");
            fail++;
        }
    }

    // Test the full SFT_GEN_PROMPT (74 tokens expected)
    std::string sft_prompt =
        "# Instruction\n"
        "Fill the audio semantic mask based on the given conditions:\n\n"
        "# Caption\n"
        "A dreamy electronic ambient track with soft synth pads and gentle beats, 120 BPM\n\n"
        "# Metas\n"
        "- bpm: N/A\n"
        "- timesignature: N/A\n"
        "- keyscale: N/A\n"
        "- duration: 120 seconds\n"
        "<|endoftext|>\n";

    auto text_ids = bpe_encode(&tok, sft_prompt, true);
    printf("  SFT_GEN_PROMPT: %zu tokens (expected 74)\n", text_ids.size());
    if (text_ids.size() == 74) {
        // Check first few and last few
        std::vector<int> expected_start = {2, 29051, 198, 14449, 279, 7699};
        std::vector<int> expected_end = {151643, 198, 151643};
        bool start_ok = true, end_ok = true;
        for (int i = 0; i < (int)expected_start.size(); i++) {
            if (text_ids[i] != expected_start[i]) start_ok = false;
        }
        for (int i = 0; i < (int)expected_end.size(); i++) {
            if (text_ids[text_ids.size() - expected_end.size() + i] != expected_end[i]) end_ok = false;
        }
        if (start_ok && end_ok) {
            printf("  OK SFT_GEN_PROMPT (74 tokens, start+end match)\n");
            pass++;
        } else {
            printf("  FAIL SFT_GEN_PROMPT start/end mismatch\n");
            printf("     first6: "); for (int i=0;i<6;i++) printf("%d,",text_ids[i]); printf("\n");
            printf("     last3:  "); for (int i=text_ids.size()-3;i<(int)text_ids.size();i++) printf("%d,",text_ids[i]); printf("\n");
            fail++;
        }
    } else {
        printf("  FAIL SFT_GEN_PROMPT: got %zu tokens\n", text_ids.size());
        printf("     all ids: "); for (int id : text_ids) printf("%d,", id); printf("\n");
        fail++;
    }

    // Test lyric template (15 tokens expected)
    std::string lyric_prompt = "# Languages\nen\n\n# Lyric\n[Instrumental]<|endoftext|>";
    auto lyric_ids = bpe_encode(&tok, lyric_prompt, true);
    printf("  Lyric template: %zu tokens (expected 15)\n", lyric_ids.size());
    std::vector<int> lyric_expected = {2, 54964, 198, 268, 271, 2, 15953, 2216, 198, 58, 56324, 278, 60, 151643, 151643};
    if (lyric_ids == lyric_expected) {
        printf("  OK Lyric template (15 tokens, exact match)\n");
        pass++;
    } else {
        printf("  FAIL Lyric template mismatch\n");
        printf("     got:      "); for (int id : lyric_ids) printf("%d,", id); printf("\n");
        printf("     expected: "); for (int id : lyric_expected) printf("%d,", id); printf("\n");
        fail++;
    }

    printf("[Test] %d/%d passed\n", pass, pass + fail);
    return fail > 0 ? 1 : 0;
}
