---
phase: 01-mp3-bitrate-control
plan: "03"
subsystem: ui
tags: [svelte, typescript, mp3-bitrate, webui, state]

# Dependency graph
requires:
  - phase: 01-01
    provides: C++ server accepting mp3_bitrate in /synth request JSON
  - phase: 01-02
    provides: app.mp3Bitrate reactive state field and mp3Bitrate parameter on both API submit functions
provides:
  - Bitrate <select> in RequestForm format row, visible only when app.format === 'mp3'
  - props-seeding $effect: seeds app.mp3Bitrate from app.props.cli.mp3_bitrate on first load, fallback 128
  - app.mp3Bitrate passed to synthSubmit (3rd arg) and synthSubmitWithAudio (5th arg) in synthesize()
affects: [01-04]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Conditional Svelte template block: {#if app.format === 'mp3'} gates bitrate selector visibility"
    - "Dynamic fallback option: {#if ![96,128,160,192,256,320].includes(app.mp3Bitrate)} renders non-preset server value as selectable option"
    - "Props-seeding $effect pattern: app.mp3Bitrate === 0 guard prevents overwriting user selection on re-render"

key-files:
  created: []
  modified:
    - tools/webui/src/components/RequestForm.svelte

key-decisions:
  - "Placed bitrate <select> before format <select> in document order — consistent left-to-right reading, bitrate is subordinate to format choice"
  - "Dynamic 7th option uses Number(app.props.cli.mp3_bitrate) || 128 coercion — matches T-03-01 threat mitigation, non-numeric props value safely falls back to 128"

patterns-established:
  - "Non-preset server value pattern: conditional extra <option> at top of list ensures any valid but non-curated server bitrate is always visible"

requirements-completed: [BITUI-01, BITUI-02, BITUI-03]

# Metrics
duration: ~4min
completed: 2026-04-22
---

# Phase 01 Plan 03: MP3 Bitrate Selector UI Summary

**Bitrate <select> wired into RequestForm format row: visible only for MP3, bound to app.mp3Bitrate, seeded from server props, and passed to both synthesize() submit paths**

## Performance

- **Duration:** ~4 min
- **Started:** 2026-04-22T04:26:00Z
- **Completed:** 2026-04-22T04:30:00Z
- **Tasks:** 2
- **Files modified:** 1

## Accomplishments

- Added `$effect` seeding `app.mp3Bitrate` from `app.props.cli.mp3_bitrate` on first load (fallback 128), guarded by `app.mp3Bitrate === 0` to avoid overwriting user selections
- Inserted bitrate `<select bind:value={app.mp3Bitrate}>` into the format row, conditionally rendered only when `app.format === 'mp3'`, with six preset options (96/128/160/192/256/320) plus a dynamic 7th option for non-preset server-configured values
- Passed `app.mp3Bitrate` as the final argument to both `synthSubmit` and `synthSubmitWithAudio` in `synthesize()`, completing the end-to-end data flow for both the plain-JSON and multipart submission paths

## Task Commits

Each task was committed atomically:

1. **Task 1: Add props-seeding $effect and bitrate selector to the format row** - `6027df4` (feat)
2. **Task 2: Pass app.mp3Bitrate to both submit call sites in synthesize()** - `e7686c8` (feat)

## Files Created/Modified

- `tools/webui/src/components/RequestForm.svelte` — props-seeding $effect in script block, conditional bitrate `<select>` in format row template, `app.mp3Bitrate` argument at both synthesize() call sites

## Decisions Made

- Bitrate `<select>` placed before the format `<select>` in document order — the user first encounters bitrate options then format, which matches the left-to-right reading flow and the fact that bitrate is only relevant once MP3 is chosen
- The dynamic 7th option (`{app.mp3Bitrate} kbps (server default)`) is suppressed when `app.mp3Bitrate` is already one of the six presets, keeping the list clean for the common case while ensuring any non-standard server default is always visible and selectable

## Deviations from Plan

None - plan executed exactly as written.

## Known Stubs

None — `app.mp3Bitrate` is seeded from server props and bound to the select; no hardcoded placeholders remain.

## Threat Flags

None — threat mitigations from plan applied as specified:
- T-03-01 (props coercion tampering): `Number(...) || 128` fallback applied in seeding $effect
- T-03-02 (select spoofing): hard-coded option values, no free-text input
- T-03-03 (console mutation): accepted; server-side validation (Plan 01) is the authoritative guard

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- All three client-side requirements (BITUI-01, BITUI-02, BITUI-03) are complete
- Plan 04 (`buildwebui.sh` + C++ rebuild) can now proceed to compile and embed the updated WebUI
- Manual integration test requires a built WebUI and running server; see plan verification section in 01-03-PLAN.md for test cases

---
*Phase: 01-mp3-bitrate-control*
*Completed: 2026-04-22*
