---
phase: 01-mp3-bitrate-control
plan: "02"
subsystem: webui-state
tags: [typescript, svelte, state, api, mp3-bitrate, localstorage]
dependency_graph:
  requires: [01-01]
  provides: [app.mp3Bitrate reactive state, localStorage persistence, synthSubmit mp3Bitrate param, synthSubmitWithAudio mp3Bitrate param]
  affects: [tools/webui/src/lib/state.svelte.ts, tools/webui/src/lib/api.ts]
tech_stack:
  added: []
  patterns: [app-level reactive state field, localStorage validation against preset allowlist, serialization-time injection of server fields]
key_files:
  created: []
  modified:
    - tools/webui/src/lib/state.svelte.ts
    - tools/webui/src/lib/api.ts
decisions:
  - mp3Bitrate stored as top-level app state (not inside app.request) matching app.format pattern
  - mp3_bitrate injected at serialization time only; kept out of AceRequest type to prevent leakage into IndexedDB/batch/LM contexts
  - Persisted mp3Bitrate validated against curated preset list [96,128,160,192,256,320]; invalid values fall back to 0
  - Default parameter value of 0 preserves backward compatibility for existing callers
metrics:
  duration: ~2 min
  completed: "2026-04-22"
  tasks_completed: 2
  files_modified: 2
---

# Phase 01 Plan 02: TypeScript State and API Foundation Summary

Added `app.mp3Bitrate` as a top-level reactive Svelte state field with localStorage persistence and preset validation, and threaded an optional `mp3Bitrate` parameter through both API submit functions so `mp3_bitrate` is injected into the JSON body for MP3 synthesis requests.

## Tasks Completed

| Task | Name | Commit | Files |
|------|------|--------|-------|
| 1 | Add mp3Bitrate to app state and localStorage persistence | 6c3cd88 | tools/webui/src/lib/state.svelte.ts |
| 2 | Thread mp3Bitrate through both API submit functions | ece1886 | tools/webui/src/lib/api.ts |

## Changes Made

### state.svelte.ts

- Added `mp3Bitrate: number` to the `Saved` interface
- Added `validBitrates` preset validation in `load()` try-branch: values outside `[96, 128, 160, 192, 256, 320]` fall back to `0`
- Added `mp3Bitrate: 0` to the fallback `load()` return block
- Added `mp3Bitrate: saved.mp3Bitrate` to the `app $state` object
- Added `mp3Bitrate: app.mp3Bitrate` to the localStorage persistence `$effect` data object

### api.ts

- Updated `synthSubmit` signature: added `mp3Bitrate = 0` optional parameter
- Updated `synthSubmitWithAudio` signature: added `mp3Bitrate = 0` optional parameter
- Both functions compute `inject = (format === 'mp3' && mp3Bitrate > 0) ? { mp3_bitrate: mp3Bitrate } : {}`
- `inject` is spread into serialized request object(s) at call time — `mp3_bitrate` never touches `AceRequest` type

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — `app.mp3Bitrate` defaults to `0` intentionally. Plan 03 will seed it from `app.props.cli.mp3_bitrate` on mount and bind the UI select element to it.

## Threat Flags

None — threat mitigations from plan applied as specified:
- T-02-01 (localStorage tampering): validated against preset list in `load()` ✓
- T-02-03 (AceRequest type spoofing): `mp3_bitrate` excluded from `AceRequest`, injected at serialization only ✓

## Deferred Items (Pre-existing, Out of Scope)

The following TypeScript errors existed before this plan and are unrelated to these changes:
- `src/lib/dice.ts:6` — `Property 'glob' does not exist on type 'ImportMeta'`
- `@lucide/svelte` module not found (4 files: RequestForm.svelte, SongCard.svelte, LogCard.svelte, App.svelte)

## Self-Check: PASSED

Files exist:
- tools/webui/src/lib/state.svelte.ts — FOUND
- tools/webui/src/lib/api.ts — FOUND

Commits exist:
- 6c3cd88 — feat(01-02): add mp3Bitrate to app state and localStorage persistence
- ece1886 — feat(01-02): thread mp3Bitrate through both API submit functions
