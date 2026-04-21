# Project State

## Current Position

Phase: Not started (defining requirements)
Plan: —
Status: Defining requirements
Last activity: 2026-04-21 — Milestone v1.0 started

## Project Reference

See: .planning/PROJECT.md (updated 2026-04-21)

**Core value:** Generates high-quality AI music locally from a text prompt — fast, private, no external services required.
**Current focus:** Milestone v1.0 — MP3 Bitrate Control

## Accumulated Context

- Open source project; all work on feature branches submitted via PR
- No Claude/Anthropic co-author attribution in commits
- WebUI build: `./buildwebui.sh` regenerates `tools/public/index.html.gz`; C++ rebuild re-embeds it into `ace-server`
- `mp3_bitrate` belongs in `ServerFields` (not `AceRequest`) — same pattern as `synth_model` and `lora`
- Valid MP3 bitrates: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps
