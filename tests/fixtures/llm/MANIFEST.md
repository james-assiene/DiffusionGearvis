# tests/fixtures/llm

| file | provenance | licence |
|---|---|---|
| `flight_canvases.json` | The three committed canvases of turn `t6` of the human's desk session of **2026-09-16 18:19–18:21 local**, `logs/sessions/20260916-181929-3900425.jsonl` (3678 records, session `970ae08ab809`, git `40957df`), records seq 2151 / 2178 / 2179 (`llm.commit`), 2181 (`ui.gate`), 2221 (`llm.commit`), 2223 / 2228 (`tts.utterance`). The request was a flight-booking interface, spoken. The surface did not fit in one canvas and the second half was classified `answer`: Pocket TTS read 807 characters of A2UI JSON aloud (T-0052). | Apache-2.0 (this repo); the recorded text is the user's own session, kept because it *is* the bug |
| `make_flight_canvases.py` | The generator. Reads the session log (under `logs/`, which is gitignored, so it documents the derivation rather than reproducing it elsewhere) and asserts every length it writes. | Apache-2.0 |

**Regime** (recorded in the session's own turn records, and repeated in the fixture's `regime`
field): profile **lean**, canvas length **256**, max denoising steps **48**, thinking **on**,
`vllm/vllm-openai:v0.29.0`, `nvidia/diffusiongemma-26B-A4B-it-NVFP4`, STT `kyutai/stt-1b-en_fr`,
TTS `pocket-tts@3.1.0`, audio path **speakers**, assists **pixels only**, seed **null**, voice
input, **n = 1** (one session, not a benchmark).

**Reconstruction rule** (also in the fixture's own `note` field, which is the normative copy): the
session log stores *post-split* pieces, not raw SSE deltas. So the raw content delta of canvas 1
is `"<a2ui>"` + the 409-character `ui` piece, the raw content delta of canvas 2 is the
902-character `answer` piece verbatim (`_strip_markup` removed nothing: the text contains neither
`<a2ui>` nor `</a2ui>`), and reasoning deltas are the `thought` pieces. **The model never wrote a
closing fence.** `json_tail_canvas2` / `prose_tail_canvas2` are the hand-measured split of canvas 2
at the structural end of the JSON — character **750** of canvas 2, character **1159** of the fenced
body — leaving **152** characters of ordinary prose.

`tts_utterance_text` (807 characters) and `tts_utterance_source` (810) are what Pocket TTS
actually said and what the speech gate handed it. They are the **negative oracle** for the voice
guard: no test may ever allow that string to be spoken.


## What the real gate makes of it (measured 2026-09-16, `UiGate` over this fixture)

| input | components rendered | repairs | resample |
|---|---|---|---|
| canvas 1 alone, 409 chars — what the human saw | `root`, `header` | 4 | `('row1',)` |
| the reassembled body, 1159 chars — what the fix renders | `root`, `header`, `row1`, `row2`, `row3`, `divider`, `results` | 22 | `('src', 'dest', 'date1')` |

Two components to seven is the measured win, and it is what
`tests/contract/test_fence_reassembly.py` pins. `date1`, `date2`, `passengers` and `search` render
in **neither** run: they are leaf inputs written without the properties the A2UI v0.9.1 catalog
*requires* — `date1` is `{"id":"date1","component":"DateTimeInput"}` with no properties at all, so
the cause is a **missing required** property, not a rejected one — and they leave through the
resample path rather than the render path. Reassembly does not change them; T-0052 must not touch
`ui_gate.py`.

Recorded separately as **T-0052 ruling A7** (not fixed here): six leaf components fail the schema
and `UiGate.max_resamples` is 3, so the gate asked for three of them and silently did not ask for
the other three.

## `vocab_subset_cc8d3a0c.json` (T-0088a)

| file | provenance | licence |
|---|---|---|
| `vocab_subset_cc8d3a0c.json` | 562 pieces of `tokenizer.json` (sha256 `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`) of `nvidia/diffusiongemma-26B-A4B-it-NVFP4` rev `2ea837236295d617ac27f8c17d61228081932c40`, read from the local HF cache by the path rule of `local_astra.decoding.vocab.snapshot_dir`: the 24 `special: true` added tokens, the 256 `<0xNN>` byte-fallback pieces, every id of the 24 T-0087 arm-(b)/(e) streams (276 distinct ids) and a few `▁`/fragment pieces (`extra_pieces`). `version`, `truncation`, `padding`, `added_tokens`, `normalizer`, `pre_tokenizer`, `post_processor`, `decoder` and the model config are verbatim; `model.vocab` is cut to the subset and `model.merges` is emptied. `streams` holds each stream's ids (every `choices[0].token_ids` of every `data:` chunk, from `logs/t0087-probe-724c0c4/raw.jsonl`, sha256 `f871d037fde4414abd53f8ce6011b3d7efb76cbc79bf9817bd805bbe58a848b0`, commit `724c0c4`) and `text`, the oracle decode `tokenizers` 0.23.2 `decode(ids, skip_special_tokens=False)`. Built 2026-10-12. The source of truth is the pinned file: `tests/cd/unit/test_host_detok_parity.py` re-checks every entry against it. | the tokenizer is the model's (NVIDIA / Gemma terms); the recorded ids are this repo's own probe run |
| `build_vocab_subset_cc8d3a0c.py` | The generator, a one-off build step in the `cd` group (it needs `tokenizers` for the oracle text): `uv run --all-extras --group cd python tests/fixtures/llm/build_vocab_subset_cc8d3a0c.py --raw <main checkout>/logs/t0087-probe-724c0c4/raw.jsonl`. Never imported by a test. | Apache-2.0 |

**Regime** of the recorded streams: lean profile, vLLM 0.29.0, CL 256, 48 steps, thinking on, RTX 5090, gpu-mem 0.86, n = 1 per
case and arm, no seed (docs/cd/toolcall-live-probe.md).

## `t0087_streams.jsonl` (T-0088)

| file | provenance | licence |
|---|---|---|
| `t0087_streams.jsonl` | The 24 arm-(b)/(e) streams of T-0087's probe (`logs/t0087-probe-724c0c4/raw.jsonl`, sha256 `f871d037fde4414abd53f8ce6011b3d7efb76cbc79bf9817bd805bbe58a848b0`, commit `724c0c4`; `docs/cd/toolcall-live-probe.md`). Line 1 is `{"provenance": …}`; each later line is one stream: `case`, `arm`, `status`, `request` (key order, scalar fields verbatim, `messages`/`tools` as `{len, sha256}` of their compact JSON) and `lines`, the `data:` lines **verbatim** except the role chunk, whose `prompt_token_ids` is `{len, sha256}` (re-serialised compactly; the builder asserts every recorded line round-trips byte for byte, so nothing else changes). `prompt_text` was `null` on every stream and is kept `null`. The recorded `tools` sha (`8e5271bb…`) equals `native.tools_field(tools.load())` at tools_v1 `bbfc53cc…`. Built 2026-10-12. `logs/` is not tracked, so this is the only committed copy of the wire. | the recorded text is this repo's own probe run (Apache-2.0) |
| `build_t0087_streams.py` | The generator (stdlib): `uv run python tests/fixtures/llm/build_t0087_streams.py --raw <main checkout>/logs/t0087-probe-724c0c4/raw.jsonl`. Never imported by a test. | Apache-2.0 |
| `request_bodies_9669258.json` | Golden tools-off request bodies captured from `main`'s `VllmClient` at the T-0088 base SHA `9669258` before any T-0088 implementation existed, by `capture_request_bodies_9669258.py` over the prompt battery in `tests/unit/_llm_id_support.py` (`request_body` JSON and the exact bytes sent by `generate` and `prefill` over `MockTransport`). | Apache-2.0 |

**Regime** of the recorded streams: lean profile, vLLM 0.29.0, NVFP4 rev `2ea8372`, CL 256, 48 steps, thinking on, prefix caching on, RTX 5090, gpu-mem 0.86, n = 1 per case and arm, no seed, `max_tokens` 1024.
| `t0087_q10_tokenize.json` | T-0087 q10's ten `/tokenize` requests (`prompt` = `draft.rstrip()` of the arm-(d) route, `add_special_tokens: false`) and the engine's `tokens`, from the same `raw.jsonl` (sha256 `f871d037…`, commit `724c0c4`), written by `build_t0087_streams.py`. The oracle of `tests/contract/test_llm_tools_wire_gpu.py` row 4. | this repo's own probe run |

## `t0095_row2_live.jsonl` (T-0095 part B)

| file | provenance | licence |
|---|---|---|
| `t0095_row2_live.jsonl` | T-0088 gpu row 2's `row2_live.jsonl` (`tests/contract/test_llm_tools_wire_gpu.py`, written under `ASTRA_T0088_RECORD_DIR`), copied **verbatim** from the T-0093 `/launch` at commit `efadf45`, 2026-10-13 15:29 local (`logs/t0093-launch-efadf45/row2_live.jsonl` in the launch worktree; sha256 `37737a2ab6b9833cc33f7f6c3dc197d54236f9c7be81343c8e5267cff7b0c66a`, pinned in the test). Every SSE `data:` line of one live tools-on response to row 2's `MULTI_CANVAS_USER` prompt ("Write the numbers from 1 to 300 separated by spaces."), `[DONE]` last: the role chunk (with the full `prompt_token_ids`) and five content chunks. Only the last content chunk (`finish_reason: "stop"`, `stop_reason: 106`) carries `system_fingerprint`. The oracle of `tests/contract/test_vllm_sse_live_shape.py`. | this repo's own launch run (Apache-2.0) |

**Regime:** vLLM v0.29.0, NVFP4 rev `2ea8372`, RTX 5090, gpu-mem 0.86, CL 256, 48 steps, prefix caching on, thinking on,
tools_v1 `bbfc53ccdd80`, n = 1, no seed.
