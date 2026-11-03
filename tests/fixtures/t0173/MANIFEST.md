# tests/fixtures/t0173 — W2-2 turn t1, machine.step records (T-0173 criterion 8)

**Source.** `logs/t0160-web-smoke-abcf17d/turns/W2-2/sessions/20261027-160309-468608.jsonl` (gitignored), the T-0160
web smoke at abcf17dd, trial W2-2.
sha256 of the source log: `e12b553deb5e9f589605c2f9e9539dfb2783dd9e920b44f9de106f9a926d1b32` (729 lines).

**Regime of the source (from its `turn.record`, seq 726).** Profile lean, vLLM v0.29.0,
nvidia/diffusiongemma-26B-A4B-it-NVFP4, canvas 256, 48 steps, thinking on, typed input, audio path `silent`
(`NullAudioDevice`), tools `native:bbfc53ccdd80:live:search:loop8`, seed none, n = 1.

**Extraction (verbatim lines, nothing re-serialised).**

1. Every record with `kind == "machine.step"`, in file order: 397 lines, seq 3 to 720, 110 019 bytes, sha256
   `f3fcb26088be5ca8d96c2f59854cc8f1eb4d4f8a664338fa34f0706a1fd0bc47`. These are all of turn t1: seq 3 and 4 (the
   typed `user.text` and the session's onset) carry `turn_id "t0"` because the log labels a step with the turn current
   *before* it; the onset opens t1.
2. Of those, kept: every non-`tick` record, every `tick` record with commands or a state change, and every record with
   seq >= 715. Dropped: 387 `tick` records with `commands == []` and `state_before == state_after` (ENDPOINTING,
   THINKING and SPEAKING no-op ticks). Result: 10 lines, seq 3, 4, 11, 607, 608, 612, 617, 715, 718, 720; 4 885 bytes;
   sha256 `0c179bfcadbb326b6c468f9ec26412a9ea13615146a4203abf00bba5af06899d`
   (asserted by `tests/eval/test_t0173_w2_2_replay.py`).

Step 2 is a deviation from "all t1 records": the agents may only create repo files with the Write tool, and 110 KB of
microsecond timestamps cannot be transcribed reliably by hand. The dropped ticks are no-ops by their own record; replaying
the 397-line and the 10-line set through `TurnMachine` gives identical commands at every kept step and the same result
at seq 720 (checked with a scratch driver, both sets).

Done with a scratch Python script (read the log, keep lines whose JSON `kind` is `machine.step`, then apply the step-2
filter, write bytes unchanged); the 10 lines were then written into the repo with the Write tool and the sha256 above
was checked against the script's output.

**Privacy.** Only the smoke's own typed prompt ("Go to example.com and read me the page heading."), the model's
thought and answer about example.com and the public start page (en.wikipedia.org in the address bar), the call's
arguments, and timings. No paths, no user data.

**Replay rules the eval uses.**

- Policy: `TurnPolicy(thinking=True, tool_loop=True, max_tool_steps=8)`, read from the regime above (thinking on,
  `loop8`). These are also `--tools live`'s defaults.
- Segmenter: a `SpeechGate` behind the machine's segmenter seam, one gate for the turn, as the session's
  `_segment` does. With it every replayed step reproduces the logged commands exactly.
- The typed-turn seed: after seq 4 opens the turn, the session sets the machine state directly (no machine.step;
  `Session._open_pending_transcript`, T-0039): `words = (Word(text, start_ms=onset, end_ms=onset),)`,
  `state = ENDPOINTING`, `t_eot_ms = onset`, `flush_t_ms = session_ms` of seq 4. The replay applies the same update;
  seq 11's logged `state_before` (ENDPOINTING) is the check.
