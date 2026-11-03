# tests/fixtures/canvas

Fixtures for T-0011 (the vLLM plugin's per-step canvas tap). Hand-written in this repo, no
third-party content, Apache-2.0 (this repo).

| file | provenance |
|---|---|
| `stable_history.json` | Hand-built `accepted_canvas_history` slabs of shape `[ST, CL]` with the `stable` bitmask each must produce, written on 2026-09-13 for criterion 7. **Not generated from the implementation** (`.claude/rules/tests.md`: no re-implementing the function under test, no capturing current behaviour); every `expected_bits`/`expected_hex` is derived by hand from the definition below and cross-checked by a brute-force loop over positions. |

| `trajectory_806da12.msgpack` | **A real recorded trajectory** (T-0128c group A, copied 2026-10-26 from the untracked `artifacts/canvas/20261003T003342-lean-cl256-s48-806da12.msgpack` of the main checkout, byte for byte with `cp`). sha256 `d837ef36755db7151af53d7b7830b96cefa598c85f686c2445ba9b50c424d914`, 104 675 bytes. Written by T-0011 criterion 14's recorder under `/launch` on 2026-10-03. See below. |

## `trajectory_806da12.msgpack`

msgpack of `{"regime": str, "request": dict, "frames": [dict, ...]}`. `frames` are the 51 tap payloads
exactly as received off the engine's PUB (`astra.canvas.step` dicts: 46 `denoise`, 5 `commit`), one
request `chatcmpl-b8d75dac2b8f054f-b19b91f7`, canvases 0-4 committed at steps 16, 18, 18, 19, 20 of 48,
CL 256; every frame validates as `AstraCanvasStep`. The frames carry no timestamps; the request
record says 24.65 frames/s over 2069 ms, which is why the T-0128c rows replay it at 40 ms.

`request` (verbatim): `{"arm": "T", "ms": 2068.98, "id": "chatcmpl-b8d75dac2b8f054f",
"completion_tokens": 1280, "prompt_tokens": 92, "engine_request_ids":
["chatcmpl-b8d75dac2b8f054f-b19b91f7"], "frames": 51, "frame_hz": 24.65, "commit_frames": 5,
"committed_canvases": 5, "commit_idx": [0, 1, 2, 3, 4], "idx_gaps": [], "warmup": false, "round": 33}`.

`regime` (verbatim): profile lean (engine only), commit 806da12 clean,
nvidia/diffusiongemma-26B-A4B-it-NVFP4 rev 2ea8372, vllm v0.29.0, driver 595.84, canvas 256, steps 48,
TRITON_ATTN, bf16 KV 4 GiB, max_model_len 16384, max_num_seqs 3, plugin+tap on, hz 30, prompt 518
tokens text-only, thinking on, max_tokens 768, concurrency 1, arms A=xargs astra_tap false / B=xargs
astra_tap true / D=no xargs, 3 warm-up rounds + 30 interleaved rounds (n=30 per arm), arm order rotated
per round (round r runs ARMS[r % 3:] + ARMS[:r % 3]: ABD, BDA, DAB, ...; each arm follows each other
arm 10 times over the measured rounds; A_r and B_r paired by round), trajectory: up to 3 tap-on
requests, prompt 92 tokens, max_tokens 1280; no seed, host load 1.43/0.67/0.29, python 3.11.16.

Licence: model output of the engine on a prompt written in this repo; Apache-2.0 (this repo). Used by
`tests/unit/test_canvas_relay.py` and `tests/contract/test_canvas_relay_wire.py` (via
`tests/fakes/canvas_tap.py`, which pins the sha256 above).

## The definition the expected values come from

Upstream reduces the same quantity into `mismatch`/`stable` inside `_compiled_sample_step`
(`vllm/model_executor/models/diffusion_gemma.py` l.606-612, `vllm/vllm-openai:v0.29.0`, read on
2026-09-13 — see docs/tasks/T-0011.md criterion 7):

> bit *p* of `stable` is set **iff** `history[h][p] == history[0][p]` for every row `h < ST`
> **and** `history_len >= ST`.

The mask is MSB first (bit *p* is bit `7 - (p mod 8)` of byte `p // 8`), zero-padded to whole
bytes, so `len(hex) * 4 >= canvas_length` — the invariant `AstraCanvasStep` validates.

Zero when the history is not full: the ring buffer starts zero-filled, so an unguarded
comparison reports a brand-new request as *completely* settled. That is the trap
`history_not_full_yet` is built for — its four rows are identical, and the only reason its mask
is zero is `history_len = 3 < ST = 4`.

## Derivation of each case, by hand

- **`one_unstable_head_and_a_stale_tail`** (CL 12, ST 4, len 4). Row 1 differs from row 0 at
  positions 8-11; row 3 differs at position 1. Everything else agrees in all four rows.
  Bits `p0..p11 = 1 0 1 1 1 1 1 1 0 0 0 0` → byte 0 = `1011 1111` = `bf`, byte 1 =
  `0000` + 4 pad zeros = `00` → **`bf00`** (4 hex chars ≥ 12 bits, whole bytes).
- **`three_row_window`** (CL 8, ST 3, len 3). Row 1 differs at position 1, row 2 at position 4.
  Bits `1 0 1 1 0 1 1 1` = `b7`.
- **`fully_settled`** (CL 8, ST 2, len 2). Both rows identical → `ff`.
- **`four_positions_padded_to_a_byte`** (CL 4, ST 2, len 2). Row 1 differs at positions 1 and 3 →
  bits `1 0 1 0` → padded `1010 0000` = `a0`. This is the padding case: 4 bits, one byte.
- **`history_not_full_yet`** (CL 12, ST 4, len 3). All four rows identical, but the history is
  one row short → **`0000`**, not `fff0`.

## Recordings versus constructions

`trajectory_806da12.msgpack` is the only recording here. Everything else in this directory (and
every hand-built frame in `tests/fakes/canvas_tap.py`) is a construction, and a test must never
present one as a trajectory (CLAUDE.md, "never a fake trajectory presented as real").
