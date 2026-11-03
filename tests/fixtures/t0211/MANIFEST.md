# tests/fixtures/t0211

- `v3_live_session.jsonl`: one real live session log (v3 guidance with the wait sentence, web mode, loopback only)
  for `scripts/replay_live_guidance.py` (T-0211 B7) and `tests/unit/test_t0211_replay_guidance.py`. Six lines, each
  copied byte for byte from the source (no line reordered or re-serialised; one field of line one edited, see the path scrub below): `log.started` (seq 0), four
  `llm.request` records (seq 7 `prefill`, seq 14 `generate`, seq 167 `tool_step`, seq 201 `completion_check`) and
  `log.stopped` (seq 294; its `records`/`bytes` counts describe the source, not this file). Every other record kind
  (`machine.step`, `wire.*`, `llm.commit`, `tts.*`, `tool.*`, `note`, `turn.record`, ...) was dropped.
- Source: `logs/t0208-handoff-9a079c3/sessions/20261030-174655-2788165.jsonl` (T-0208 C GPU run at commit 9a079c3d,
  2026-10-30, task H2 repeat 2, typed arm, turn `t1`, session `9f26965abeb7`; regime in that run's `manifest.json`:
  `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev 2ea83723, vLLM v0.29.0, profile lean, canvas 256, 48 steps, thinking
  on, `native:05de281e7bd2:live:loop20`, headless Chromium 156, loopback origin, STT down).
  - source sha256: `da4519c08c2936509d0f4aba4beb21f1c36208e39e4cec9d56c4e919b4e38446` (118944 bytes)
  - as cut, before the path scrub: `6d669d5283fb86e9b93727803b9739f8036c2a053396de205e0c983ec37218b0` (31844 bytes)
  - this file's sha256 (after the B-R6 path scrub): `a0119946f31010e5ea46db1196dd67744f37f12b93dc8ca2924cef5bd4765aee`
    (31813 bytes)
- Images: already redacted by the session log (`{"redacted": "image", "bytes", "digest", "phash", "mime"}`); no
  base64 payload is in the file. Path scrub (T-0211 B-R6, release path scrub, 2026-10-30): the only change from the cut is the `log.started` record's `data.path`, rewritten from the source's absolute path to the repo-relative `logs/t0208-handoff-9a079c3/sessions/20261030-174655-2788165.jsonl` (the prefix up to `logs/` removed, byte for byte, no re-serialisation); the other five lines and every other byte of line one are unchanged. No other `/home/` path is in the file.
- What the four requests carry (checked when the fixture was cut, 2026-10-30): one system text of 3562 characters,
  identical in all four, containing `LIVE_TOOL_GUIDANCE_V3` once and ending with `"\n" + v3 + " " +
  WAIT_TOOL_GUIDANCE`. Content: a loopback store scenario (the T-0208 H2 stove page), no real user, page or URL.
- Extraction: a scratch script kept the three kinds by `kind` and wrote the matching lines unchanged; regenerate with
  any filter that keeps those lines byte for byte and compare the sha above.
- Licence: project-internal (a log of this project's own test run).
