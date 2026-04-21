# Roadmap — acestep.cpp

**Milestone:** v1.0 — MP3 Bitrate Control
**Created:** 2026-04-18
**Granularity:** Standard
**Coverage:** 7/7 requirements mapped

---

## Phases

- [ ] **Phase 1: MP3 Bitrate Control** - Wire per-request bitrate selection end-to-end: C++ server validation and encoding, WebUI selector seeded from server default, operator startup flag

---

## Phase Details

### Phase 1: MP3 Bitrate Control
**Goal**: Users can select the MP3 encoding bitrate per synthesis request in the WebUI, the server validates and applies the chosen bitrate, and the selector initializes from the server's configured default.
**Depends on**: Nothing (first phase)
**Requirements**: BITUI-01, BITUI-02, BITUI-03, SRV-01, SRV-02, SRV-03, CFG-01
**Success Criteria** (what must be TRUE):
  1. User submits a synthesis request with 320 kbps selected and the resulting MP3 file encodes at 320 kbps (verifiable with `ffprobe`)
  2. Bitrate selector is visible when output format is MP3 and hidden when any WAV variant is selected
  3. On first page load against a server started with `--mp3-bitrate 64`, the bitrate selector shows 64 kbps as the selected value
  4. A `/synth` request with `"mp3_bitrate": 200` (not in the valid preset list) returns HTTP 400
  5. A `/synth` request that omits `mp3_bitrate` encodes at the server's `g_mp3_kbps` default without error
**Plans**: TBD
**UI hint**: yes

---

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. MP3 Bitrate Control | 0/0 | Not started | - |
