# acestep.cpp

## What This Is

A portable C++17 inference engine for AI music generation implementing the ACE-Step 1.5 model architecture. Users run it as a local HTTP server with an embedded Svelte web UI; they submit a text caption (and optional lyrics, metadata, or source audio) and receive stereo 48 kHz audio back as MP3 or WAV. Runs on CPU, CUDA, Metal, Vulkan, and ROCm with no cloud dependencies.

## Core Value

Generates high-quality AI music locally from a text prompt — fast, private, no external services required.

## Requirements

### Validated

<!-- Shipped and confirmed working in existing codebase. -->

- ✓ Text-to-music generation: text caption → stereo 48 kHz audio (MP3/WAV)
- ✓ Task types: text2music, cover, cover-nofsq, repaint, lego, extract, complete
- ✓ HTTP server with FIFO job queue, async polling, job cancel
- ✓ Embedded WebUI (Svelte/TypeScript, single-file build, served inline)
- ✓ Multi-backend: CPU, CUDA, Metal (Apple Silicon), Vulkan, ROCm
- ✓ LoRA adapter support (ComfyUI .safetensors + PEFT directory format)
- ✓ Understand pipeline: audio → metadata + lyrics + audio codes
- ✓ Batch generation (up to 9 tracks per request)
- ✓ WAV output variants: wav16, wav24, wav32
- ✓ MP3 output at server-configured bitrate (default 128 kbps)
- ✓ Model hot-swap: load different GGUF models per request
- ✓ LoRA scale control per request

### Active

<!-- Current milestone scope. -->

- [ ] User can select MP3 bitrate per synthesis request via the WebUI
- [ ] WebUI pre-populates bitrate control from server's current default (`GET /props`)
- [ ] Server accepts `mp3_bitrate` in `/synth` request JSON and uses it for MP3 encoding
- [ ] `server.sh` exposes `--mp3-bitrate` startup flag for configuring the server default

### Out of Scope

| Feature | Reason |
|---------|--------|
| Per-request WAV bit depth | WAV format already selectable via `?format=` query param; not a user pain point |
| Bitrate auto-selection based on duration | Adds complexity; user control is sufficient |
| Persisting bitrate preference across sessions | No persistence layer in this project; browser reload resets all UI state |

## Context

- Open source project; all changes contributed via **feature branch + pull request** — never commit directly to master
- No Claude/Anthropic co-author attribution in commit messages
- WebUI is pre-built and committed (`tools/public/index.html.gz`); must run `./buildwebui.sh` after any `.svelte` changes, then rebuild `ace-server` to re-embed
- `ServerFields` (parsed separately from `AceRequest`) is the established pattern for per-request server-side output settings (model selection, LoRA) — MP3 bitrate follows this pattern

## Constraints

- **Workflow**: Feature branch + PR — no direct pushes to master
- **Attribution**: No AI co-author lines in commits
- **Build**: WebUI changes require `./buildwebui.sh` + C++ rebuild to take effect
- **MP3 codec**: Custom in-tree encoder (`mp3/mp3enc.h`); valid bitrates are those in `mp3enc_bitrate_kbps[]`: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320 kbps

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Add `mp3_bitrate` to `ServerFields`, not `AceRequest` | Output format is a server concern, not a generation parameter; matches existing `synth_model`/`lora` pattern | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-04-21 — Milestone v1.0 started*
