# Requirements: acestep.cpp

**Defined:** 2026-04-21
**Core Value:** Generates high-quality AI music locally from a text prompt — fast, private, no external services required.

## Milestone v1.0 Requirements

### Bitrate UI

- [ ] **BITUI-01**: User can select MP3 bitrate (96 / 128 / 160 / 192 / 256 / 320 kbps) in the WebUI when MP3 is the selected output format
- [ ] **BITUI-02**: Bitrate selector is hidden when output format is WAV (wav16 / wav24 / wav32)
- [ ] **BITUI-03**: Bitrate selector initializes to the server's current default on first load (from `GET /props` → `cli.mp3_bitrate`)

### Server

- [ ] **SRV-01**: Server accepts `mp3_bitrate` in `/synth` request JSON body and uses it for MP3 encoding that request
- [ ] **SRV-02**: Server returns HTTP 400 when `mp3_bitrate` is present but not a valid value (96/128/160/192/224/256/320 kbps)
- [ ] **SRV-03**: Server falls back to `g_mp3_kbps` when `mp3_bitrate` is absent or zero in the request

### Config

- [ ] **CFG-01**: `server.sh` includes a commented `--mp3-bitrate <N>` example so operators can override the 128 kbps startup default

## Future Requirements

*(None deferred — milestone scope is complete)*

## Out of Scope

| Feature | Reason |
|---------|--------|
| Persisting bitrate across browser sessions | No persistence layer in project; server default via `/props` is sufficient initialization |
| Exposing all 14 MPEG-1 Layer III bitrate values | Values like 40/48/56/80/112/224 kbps have no user mental model; 6 curated presets cover all real use cases |
| Auto-selecting bitrate based on duration or quality | Adds complexity; explicit user control is the goal |
| Bitrate control for WAV output | WAV bit depth is a separate concern already handled by the `?format=` query param |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| BITUI-01 | Phase 1 | Pending |
| BITUI-02 | Phase 1 | Pending |
| BITUI-03 | Phase 1 | Pending |
| SRV-01 | Phase 1 | Pending |
| SRV-02 | Phase 1 | Pending |
| SRV-03 | Phase 1 | Pending |
| CFG-01 | Phase 1 | Pending |

**Coverage:**
- v1.0 requirements: 7 total
- Mapped to phases: 7
- Unmapped: 0 ✓

---
*Requirements defined: 2026-04-21*
*Last updated: 2026-04-18 — Roadmap created, all requirements confirmed mapped to Phase 1*
