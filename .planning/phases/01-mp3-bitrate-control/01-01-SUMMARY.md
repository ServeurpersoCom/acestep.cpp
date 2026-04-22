---
phase: 01-mp3-bitrate-control
plan: 01
subsystem: api
tags: [cpp, mp3, bitrate, server, validation]

# Dependency graph
requires: []
provides:
  - ServerFields.mp3_bitrate field with 14-value allowlist validation
  - HTTP 400 response for invalid mp3_bitrate at all parse sites
  - Per-request bitrate forwarding to audio_encode_mp3 in synth_worker
  - Startup guard for --mp3-bitrate flag with exit code 1 on invalid value
affects: [01-02, 01-03, 01-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Sentinel -1 pattern: parse_server_fields sets -1 on validation failure; callers check before proceeding"
    - "Static allowlist validation: inline const int array with linear scan, same list reused at parse and startup"

key-files:
  created: []
  modified:
    - tools/ace-server.cpp

key-decisions:
  - "Used -1 sentinel (not bool/enum) to signal invalid mp3_bitrate — avoids changing parse_server_fields signature and is consistent with 0-means-default convention already established by lora_scale"
  - "Applied HTTP 400 guard at all 6 parse_server_fields call sites including handle_understand — even though understand does not use mp3_bitrate, an invalid value in a request should be rejected for consistency and correct client error signaling"

patterns-established:
  - "Sentinel -1 in ServerFields int fields: 0 = omitted/use-default, >0 = valid value, -1 = invalid (caller returns 400)"

requirements-completed: [SRV-01, SRV-02, SRV-03]

# Metrics
duration: 12min
completed: 2026-04-22
---

# Phase 01 Plan 01: MP3 Bitrate — C++ ServerFields and synth_worker Summary

**Per-request MP3 bitrate control wired end-to-end in ace-server.cpp: validated parse with HTTP 400 on invalid values, per-request kbps forwarding to audio_encode_mp3, and startup guard that exits on invalid --mp3-bitrate**

## Performance

- **Duration:** ~12 min
- **Started:** 2026-04-22T04:11:00Z
- **Completed:** 2026-04-22T04:23:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `int mp3_bitrate` to `ServerFields` struct with 0-means-default semantics
- Validated `mp3_bitrate` in `parse_server_fields` against the full 14-value allowlist {32,40,48,56,64,80,96,112,128,160,192,224,256,320}; -1 sentinel triggers HTTP 400 at 6 call sites
- Replaced hardcoded `g_mp3_kbps` in `synth_worker` with `(sf.mp3_bitrate > 0) ? sf.mp3_bitrate : g_mp3_kbps`
- Added post-atoi startup validation of `--mp3-bitrate`; invalid value prints error to stderr and returns exit code 1

## Task Commits

Each task was committed atomically:

1. **Task 1: Add mp3_bitrate to ServerFields and implement validated parsing** - `db97c54` (feat)
2. **Task 2: Wire per-request bitrate into synth_worker and guard --mp3-bitrate startup flag** - `b119d8d` (feat)

## Files Created/Modified

- `tools/ace-server.cpp` — ServerFields struct, parse_server_fields validation block, 6 sentinel guard sites, synth_worker kbps expression, --mp3-bitrate startup validation

## Decisions Made

- Used -1 sentinel in ServerFields int field rather than a separate bool — keeps parse_server_fields signature unchanged and is readable at call sites (`if (sf.mp3_bitrate == -1)`)
- Applied the HTTP 400 guard to handle_understand call sites even though understand does not encode MP3 — an invalid mp3_bitrate in any request body should be rejected early for consistent client error behavior

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- C++ server is ready to accept `mp3_bitrate` in `/synth` request JSON
- Plans 02-04 (TypeScript types, Svelte state/UI, server.sh) can proceed in parallel (wave 2)
- Manual integration test requires a built server binary; see plan verification section for curl test cases

---
*Phase: 01-mp3-bitrate-control*
*Completed: 2026-04-22*
