# Project State

## Current Position

Phase: 1 — MP3 Bitrate Control
Plan: —
Status: Ready for planning
Last activity: 2026-04-18 — Roadmap created for Milestone v1.0

Progress: [----------] 0% (Phase 1 of 1 not started)

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Generates high-quality AI music locally from a text prompt — fast, private, no external services required.
**Current focus:** Milestone v1.0 — MP3 Bitrate Control → Phase 1

## Performance Metrics

- Phases defined: 1
- Phases complete: 0
- Requirements mapped: 7/7

## Accumulated Context

- Open source project; all work on feature branches submitted via PR
- No Claude/Anthropic co-author attribution in commits
- WebUI build: `./buildwebui.sh` regenerates `tools/public/index.html.gz`; C++ rebuild re-embeds it into `ace-server`
- `mp3_bitrate` belongs in `ServerFields` (not `AceRequest`) — same pattern as `synth_model` and `lora`
- Valid MP3 bitrates (full allowlist): 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps
- WebUI curated preset list (6 values): 96 / 128 / 160 / 192 / 256 / 320 kbps
- Implementation order: (1) C++ ServerFields + parse + encode call, (2) TypeScript types, (3) Svelte state, (4) Svelte UI select, (5) server.sh comment, (6) buildwebui.sh + C++ rebuild
- Both plain JSON and multipart submit paths must include `mp3_bitrate` in the request body
- `app.mp3Bitrate` must be seeded from `app.props.cli.mp3_bitrate` on mount — not hardcoded to 128

## Session Continuity

Next action: `/gsd-plan-phase 1`
