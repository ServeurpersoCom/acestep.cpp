// test-ace-fatal: the throw-instead-of-exit path (#17).
//
// Defines ACESTEP_FATAL_THROWS for this TU only, so it exercises the throwing
// build regardless of how the rest of the engine is compiled. Verifies that a
// fatal turns into a catchable ace_fatal_error carrying the formatted message
// and the exit code, that stack unwinding runs destructors (nothing leaks), and
// that a message longer than the on-stack format buffer is reproduced in full.
#define ACESTEP_FATAL_THROWS
#include "ace-fatal.h"

#include <cstdio>
#include <string>

static int failures = 0;

#define CHECK(cond)                                                                   \
    do {                                                                              \
        if (!(cond)) {                                                                \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);           \
            failures++;                                                               \
        }                                                                             \
    } while (0)

// Flips a flag in its destructor, so we can prove unwinding ran it.
struct Sentinel {
    bool * ran;
    explicit Sentinel(bool * r) : ran(r) {}
    ~Sentinel() { *ran = true; }
};

int main() {
    // 1. Message and code ride with the exception.
    {
        bool caught = false;
        try {
            ace_fatal(7, "[X] FATAL: tensor '%s' not found (%d)\n", "decoder.proj_out", 42);
        } catch (const ace_fatal_error & e) {
            caught = true;
            CHECK(e.code == 7);
            CHECK(std::string(e.what()) == "[X] FATAL: tensor 'decoder.proj_out' not found (42)\n");
        }
        CHECK(caught);
    }

    // 2. Stack unwinding runs destructors between the throw and the catch, so a
    //    half-built module's RAII members are released -- "leaks nothing".
    {
        bool sentinel_ran = false;
        bool caught       = false;
        try {
            Sentinel s(&sentinel_ran);
            ace_fatal(1, "boom\n");
            CHECK(false);  // unreachable: ace_fatal is [[noreturn]]
        } catch (const ace_fatal_error &) {
            caught = true;
        }
        CHECK(caught);
        CHECK(sentinel_ran);
    }

    // 3. A message longer than the 1024-byte stack buffer is reproduced whole
    //    (exercises the vsnprintf heap-resize branch).
    {
        std::string big(5000, 'x');
        bool        caught = false;
        try {
            ace_fatal(2, "%s", big.c_str());
        } catch (const ace_fatal_error & e) {
            caught = true;
            CHECK(std::string(e.what()) == big);
            CHECK(e.code == 2);
        }
        CHECK(caught);
    }

    // 4. A generic handler catches it too. (ModelStore's LoadGuard sites catch
    //    the specific `const ace_fatal_error &`, not `...`, so a std::bad_alloc
    //    is not masked; this only checks the base-class catch still works.)
    {
        bool caught = false;
        try {
            ace_fatal(1, "generic\n");
        } catch (...) {
            caught = true;
        }
        CHECK(caught);
    }

    if (failures == 0) {
        printf("test-ace-fatal: all checks passed\n");
        return 0;
    }
    fprintf(stderr, "test-ace-fatal: %d check(s) failed\n", failures);
    return 1;
}
