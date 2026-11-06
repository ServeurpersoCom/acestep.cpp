// test-utf8.cpp: utf8_valid keeps well-formed text and repairs what byte-level
// BPE decoding can leave: a character cut at the end, stray bytes inside.
//
// Usage:
//   ./test-utf8

#include "bpe.h"

#include <cstdio>
#include <string>

static int failures = 0;

static void expect(const char * label, const std::string & got, const std::string & want) {
    if (got == want) {
        return;
    }
    fprintf(stderr, "[FAIL] %s: got %zu bytes, want %zu bytes\n", label, got.size(), want.size());
    failures++;
}

int main() {
    const std::string fffd = "\xEF\xBF\xBD";

    expect("ascii", utf8_valid("[Verse 1] hello"), "[Verse 1] hello");
    const std::string words = "\xD0\x9F\xD1\x80\xD0\xB8 \xE6\x9C\x88";
    expect("cyrillic and cjk", utf8_valid(words), words);
    expect("emoji", utf8_valid("\xF0\x9F\x8E\xB5"), "\xF0\x9F\x8E\xB5");

    // generation stopped inside a character: the incomplete tail is dropped
    expect("cut cyrillic", utf8_valid("\xD0\x9F\xD1"), "\xD0\x9F");
    expect("cut cjk after one byte", utf8_valid("ab\xE6"), "ab");
    expect("cut cjk after two bytes", utf8_valid("ab\xE6\x9C"), "ab");
    expect("cut emoji", utf8_valid("x\xF0\x9F\x8E"), "x");

    // bytes that cannot be UTF-8 become U+FFFD and the text goes on
    expect("stray continuation", utf8_valid("a\x80" "b"), "a" + fffd + "b");
    expect("broken sequence inside", utf8_valid("a\xE6\x9C" "b"), "a" + fffd + "b");
    expect("overlong", utf8_valid("\xC0\xAF"), fffd + fffd);
    expect("surrogate", utf8_valid("\xED\xA0\x80"), fffd + fffd + fffd);
    expect("above U+10FFFF", utf8_valid("\xF4\x90\x80\x80"), fffd + fffd + fffd + fffd);
    expect("invalid lead byte", utf8_valid("\xFF" "z"), fffd + "z");

    if (failures) {
        fprintf(stderr, "[ERROR] %d utf8_valid cases failed\n", failures);
        return 1;
    }
    fprintf(stderr, "[OK] utf8_valid\n");
    return 0;
}
