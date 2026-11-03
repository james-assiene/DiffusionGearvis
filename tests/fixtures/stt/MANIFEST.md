# tests/fixtures/stt — provenance

One fixture, for T-0051 group A criterion 3: the turn-1 slice of the human's own desk session of
2026-09-16. Generator: `extract_desk_turn1.py` in this directory, committed beside the data it
produced (`.claude/rules/tests.md`). Re-create it with

```
uv run python tests/fixtures/stt/extract_desk_turn1.py logs/sessions/20260916-170934-3380576.jsonl
```

| file | sha256 |
|---|---|
| `desk-2026-09-16-turn1.json` | `3c4106997c256ba9a73e10a0de5ed3796ec6196fe8ad6aa2d466e05ab1550b0a` |
| `extract_desk_turn1.py` | `c221a5b53e6b8d7654a5d28575b2655ff83caf94675029d1e42ebce53159e652` |

## `desk-2026-09-16-turn1.json`

**Recording conditions (the regime, docs/tasks/T-0051.md).** Session 2026-09-16 17:09:34 → 17:22:42
local, `logs/sessions/20260916-170934-3380576.jsonl`, 12451 records, 0 dropped, session id
`00ee8b9a4058`, git 28d7080. Profile `lean`; vLLM v0.29.0,
`nvidia/diffusiongemma-26B-A4B-it-NVFP4`, canvas 256, steps 48, thinking on; Kyutai
`kyutai/stt-1b-en_fr` in a 0.6.4 server **on the GPU** (the launch script's default — the session's
`Regime` does not record the device, which is follow-up ambiguity 4), `asr_delay_in_tokens` 6,
`batch_size` 2; Pocket TTS 3.1.0 on CPU; `audio_path=speakers`, `aec=none`, webcam microphone;
assists off (pixels only). **n = 1 session, 1 turn, no seed.** Capture health on this turn:
`capture_xruns` 2081 (≈ 8 % of blocks — ruling A6; it may be degrading recognition and is another
task's).

**Contents.** 29 consecutive steps (session-clock stamps 242874 →
245116) with their real four-entry `prs` vectors verbatim; 3 word
rows with their real `start_ms`/`end_ms` and the index of the step after which each was published;
and the four `stt.decision` rows the session logged, kept as *observed behaviour* and asserted
nowhere — they are what the defect did.

**The human's words are not here.** The texts are `w1`, `w2`, `w3`. The timings and the
pause-head vectors are the evidence; the sentence is not needed by any row and is not checked in
(docs/tasks/T-0051.md, *Fixture*).

**Two clocks, and the one free parameter.** A step's `t_ms` in this file is the *session* stamp the
log carries. The fed-audio clock the endpointer actually saw runs behind it by a constant this
session could only bound, **330–395 ms**, derived from the three steps at which the backstop did
and did not fire (T-0051 *Regime*); the file stores the midpoint, `audio_clock_offset_ms = 360`,
and its band. Word `start_ms`/`end_ms` are audio times and are **not** shifted. No rule of the
endpointer can depend on the value: it shifts every step by the same amount, and
`test_the_desk_stream_of_2026_09_16_ends_the_turn_on_the_heads_not_the_backstop` replays the stream
at both edges of the band and asserts the same decisions. The midpoint was chosen because it
reproduces the session's own two turn-1 decisions exactly under the *unfixed* module (backstop at
session 243914 and 244474) — a validation of the offset, not an assertion of the defect.

Licence: extracted here from this repository's own session log, Apache-2.0 with the repo.
