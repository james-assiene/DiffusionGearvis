# tests/fixtures/audio — provenance

Fixtures for T-0006 (playback path, the played clock, the audio rig). Generator:
`make_fixtures.py` in this directory, committed alongside the data it produced
(`.claude/rules/tests.md`: fixtures state recording conditions, generator script, licence).

Box: `gpu-box`, Linux 7.0.0-31-generic, Python 3.11 (`.python-version`),
numpy 2.4.6, PipeWire / libpipewire 1.0.5. Generated 2026-09-13.

| file | sha256 |
|---|---|
| `chirp_5s_24k.wav` | `2f582c79aeab8f8f5653a480704769b6cc765fac1ee999701061ba140671aded` |
| `pocket_tts_sentence.wav` | `75817c33dbe9842759f44c69dcdd23d82423098ccb016624e7102d0f7d887cd4` |
| `pocket_tts_sentence.words.json` | `3a895ad39a38367c1a669704ec4c3a81b77a2eb2aa6d4c975a02deb6b4424ddf` |
| `make_fixtures.py` | `7152d80234848474eacaa690e5d73229c50264cfc2659fe6f0e4c809ae8f3147` |

## `chirp_5s_24k.wav`

24 kHz mono s16le (`TTS_OUTPUT_FORMAT`), 5.000 s, linear chirp 200 Hz → 6000 Hz, amplitude 0.5,
zero-phase at t = 0, no dither, no RNG — bit-reproducible from the generator:

```
uv run python tests/fixtures/audio/make_fixtures.py chirp
```

`x(t) = 0.5 · sin(2π (200 t + 0.5 · 1160 t²))`, so the instantaneous frequency is
`f(t) = 200 + 1160 · t` Hz and `t(f) = (f − 200) / 1160` s. That inverse is why this file is the
source material for the null-sink tests: a recording of it tells you *how much of it was
rendered* without asking the module under test (an independent clock, not a re-implementation).
Licence: generated here, Apache-2.0 with the repo.

## `pocket_tts_sentence.wav` + `pocket_tts_sentence.words.json`

24 kHz mono s16le, 4.040 s, sentence `"one, two, three, four, five."`, **default voice** (no
`voice_url`/`voice_wav` sent), Pocket TTS **3.1.0** serving on CPU at `127.0.0.1:8081`
(`services/tts.sh`; the running instance was started as `pocket-tts serve --port 8081`).
`services/versions.lock` records `pocket_tts = "TBD"` — the pin is still open, so the version above
is the one that actually produced this file.

```
bash services/tts.sh &                                          # or an already-running instance
uv run python tests/fixtures/audio/make_fixtures.py tts
```

**The wav is committed because the service is not deterministic:** re-running the command produces
a different rendering (a second run during authoring moved every word boundary by 20–110 ms). Treat
this file as a recording, not as something reproducible; regenerating it means regenerating
`pocket_tts_sentence.words.json` with it, in the same command.

**Word timings are derived, not reported by the TTS.** The HTTP API (`POST /tts`, `openapi.json`
paths `/`, `/health`, `/tts`) returns audio only. `make_fixtures.py tts` recovers word boundaries
by energy segmentation of *this* rendering: 10 ms frames, peak |x| > 0.02, segments closer than
80 ms merged; the script exits non-zero unless the number of segments equals the number of words.
The sentence was chosen for that reason — the five words are separated by 340–450 ms of silence in
this rendering, an order of magnitude more than the 100 ms guard band of docs/02 §4, so no test
result depends on the segmentation being accurate to the millisecond. Boundaries found:

| word | start_ms | end_ms |
|---|---|---|
| `one,` | 50 | 570 |
| `two,` | 880 | 1330 |
| `three,` | 1780 | 2170 |
| `four,` | 2510 | 2980 |
| `five.` | 3310 | 3790 |

Licence: Pocket TTS weights are CC-BY-4.0 (`services/versions.lock`); this is a rendering of a
public-domain word list produced locally on CPU. No human voice was recorded.
