# tests/fixtures/cd/bfcl

BFCL v4 live_simple, the data behind the T-0062 tree's BFCL-Live simple evaluation (T-0063, T-0070).

| file | provenance | licence |
|---|---|---|
| `BFCL_v4_live_simple.json` | ShishirPatil/gorilla at commit `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`, path `berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_live_simple.json`, verbatim (JSON lines, 258 records). SHA-256 `1af2ac87dca47556db7b7e37e51e28b459a38b594e3c7b3c792b4903598ca0c4`. | Apache-2.0 (gorilla) |
| `possible_answer/BFCL_v4_live_simple.json` | Same commit, path `berkeley-function-call-leaderboard/bfcl_eval/data/possible_answer/BFCL_v4_live_simple.json`, verbatim (258 records). SHA-256 `fec9cfa9744a936f9126981e85a2023da1e63e273eafebc81923a1162fad70ce`. | Apache-2.0 (gorilla) |

Fetched 2026-09-29 (gorilla HEAD at that time); commit pinned by the human the same day. diffgemma-fa's
`bfcl_data.py` recorded no gorilla commit, so equality with the data the JAX baseline ran on is shown by ids,
not by hash: all 130 ids of diffgemma-fa `preserved/exp_e5_grammar130_map.json` (the test cut) are present.
Question text is not compared anywhere: the preserved rows carry none. T-0063's parity tests compare ids and
function names.
