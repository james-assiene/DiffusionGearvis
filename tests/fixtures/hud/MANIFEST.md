# tests/fixtures/hud

Fixtures for T-0008 (HUD page + `BrowserFrameSource`). All hand-generated in this repo, no
third-party content, Apache-2.0 (this repo).

| file | provenance |
|---|---|
| `solid_16x16_q85.jpg` | 16x16 solid RGB(32,64,128), JPEG quality 85 (`policy.JPEG_QUALITY`). 633 bytes, starts `ffd8`, ends `ffd9`. Generated 2026-09-13 with the command below. |
| `gradient_32x24_q85.jpg` | 32x24 horizontal greyscale gradient (`x*8` per column), JPEG quality 85. 695 bytes, starts `ffd8`, ends `ffd9`. Generated 2026-09-13 with the command below. |
| `decode_corpus.jsonl` | Hand-written corpus of 52 inbound wire messages (33 until T-0128c added nineteen `astra.canvas.step` rows, see below; 27 until T-0090 added six `astra.tool.call` rows: three that decode — a dry-run search, a proposal whose query is markup and must stay data, a replayed cancelled repair — and three that are refused: an unknown status, list arguments, an undeclared `url` field) with the decode outcome each must reach (criterion 2's four outcomes: `astra` / `unknown` / `realtime` / `malformed`). Shared by `tests/hud/protocol.test.mjs` (the page's mirror) and `tests/unit/test_hud_constants_parity.py` (which checks every declared outcome against the **real** `local_astra.contracts.codec.decode`, so the corpus can never drift from the contract). The `oversized` case is built at read time on both sides as `'{"type":"astra.transcript","turn_id":"t","text":"' + "x" * (MAX_MESSAGE_BYTES + 64) + '"}'` because an 8 MiB fixture file does not belong in git. T-0027 generalises that to the `build` object described below. |

| `session_20261020_t3_t11.jsonl` | T-0128 T5. The `wire.out` `data.msg` objects of turns **t3–t11 only** (H-T128-1: t1 holds an email address and t2 a password; they are never committed), in log order, one compact JSON object per line (312 lines, 53,263 bytes), from `logs/sessions/20261020-165855-1581291.jsonl` (the 2026-10-20 desk session). A message's turn is its `turn_id`, else the `tN` of a `surface_id` of the form `turn-tN`; the session's one turn-less `astra.error` is excluded. SHA-256 `71ed804b2c0e0c889ed79f94c8fc510daca89646647836fe5019026f324f1e79`. Checked before commit with count-only greps: 0 email-shaped strings, and 0 lines that contain either of the two strings typed at t1/t2 (the same check finds 69 lines in the source log). Every line decodes with `local_astra.contracts.codec.decode` (`tests/integration/test_hud_redesign.py`). |

Extraction command for `session_20261020_t3_t11.jsonl` (run from the repo root; it filters by turn id, so t1 and t2 are never written to disk):

```bash
uv run python -c '
import json
SRC = "/home/user/demos/local_astra/logs/sessions/20261020-165855-1581291.jsonl"
DST = "tests/fixtures/hud/session_20261020_t3_t11.jsonl"
KEEP = {f"t{i}" for i in range(3, 12)}
def turn_of(m):
    t = m.get("turn_id")
    if t is None and str(m.get("surface_id") or "").startswith("turn-"):
        t = m["surface_id"][len("turn-"):]
    return t
with open(SRC, encoding="utf-8") as src, open(DST, "w", encoding="utf-8") as out:
    for line in src:
        r = json.loads(line)
        if r["kind"] != "wire.out":
            continue
        m = r["data"]["msg"]
        if turn_of(m) in KEEP:
            out.write(json.dumps(m, ensure_ascii=False, separators=(",", ":")) + "\n")
'
```

Generator command for the two JPEGs (pillow 12.3.0 from the `vision` extra; it is **not** a test
dependency — the tests only read the bytes):

```bash
uv run python - <<'PY'
from PIL import Image
import pathlib
d = pathlib.Path("tests/fixtures/hud")
im = Image.new("RGB", (16, 16), (32, 64, 128))
im.save(d / "solid_16x16_q85.jpg", "JPEG", quality=85)
g = Image.new("RGB", (32, 24))
g.putdata([(x * 8, x * 8, x * 8) for _y in range(24) for x in range(32)])
g.save(d / "gradient_32x24_q85.jpg", "JPEG", quality=85)
PY
```

These are deliberately tiny: `BrowserFrameSource` never decodes the pixels (no pillow on the
server side, docs/tasks/T-0008.md implementation notes), it only base64-decodes the bytes and
keeps them, so the fixtures exist to prove the bytes survive the round trip intact.

## The `build` object (T-0027)

Four rows of `decode_corpus.jsonl` carry a `build` object instead of `raw`, because their
payloads are 8-10 MB. Both loaders — `build_raw()` in `tests/unit/test_hud_constants_parity.py`
and `buildRaw()` in `tests/hud/protocol.test.mjs` — implement the same contract, so the page and
`local_astra.contracts.codec` are held to one fixture rather than to two constructions:

```
raw = prefix + char * count + suffix
```

- exactly one of `count` / `fill_to_bytes` is present; `delta_bytes` only with `fill_to_bytes`;
- with `fill_to_bytes`, the count is
  `(fill_to_bytes + delta_bytes − utf8len(prefix) − utf8len(suffix)) / utf8len(char)` and **must
  divide exactly** — the loaders raise rather than floor, because a floored count builds a
  payload a byte or two off the boundary the row claims to sit on;
- `fill_to_bytes` **must equal `codec.MAX_MESSAGE_BYTES`** (8 MiB): if the constant ever moves,
  the boundary rows fail instead of quietly testing some other boundary;
- any other shape (an unknown key, a missing field, neither `raw` nor `build`) makes **both**
  loaders raise/throw. `decode(undefined)` is `{kind:"malformed"}` and most rows declare
  `malformed`, so a builder that silently returned nothing would make them pass for the wrong
  reason (docs/tasks/T-0027.md criterion 12).

| row | payload | UTF-8 bytes | code points | expect |
|---|---|---|---|---|
| `multibyte_oversized_euro` | prefix + `"€" × 3,000,000` + suffix | 9,000,051 | 3,000,051 | `malformed` |
| `astral_oversized_defeats_a_times_three_shortcut` | prefix + `"😀" × 2,500,000` + suffix | 10,000,051 | 2,500,051 | `malformed` |
| `ascii_exactly_at_the_limit` | prefix + `"x" × 8,388,557` + suffix | 8,388,608 | 8,388,608 | `astra` / `astra.transcript` |
| `ascii_one_byte_over_the_limit` | prefix + `"x" × 8,388,558` + suffix | 8,388,609 | 8,388,609 | `malformed` |

`prefix` is `{"type":"astra.transcript","turn_id":"t","text":"` (49 bytes) and `suffix` is `"}`
(2 bytes). The astral row exists to kill a `length * 3 <= MAX` short-circuit copied from
`hud/src/protocol.js`: 2,500,051 × 3 = 7,500,153 ≤ 8,388,608, yet the message is 10,000,051 bytes
(the page's constant is right for *its* unit, a UTF-16 unit; Python counts code points, where the
bound is 4 bytes — docs/tasks/T-0027.md ruling R2).

A row that is red until its task lands carries `"pending": "T-NNNN"`; the Python loader turns it
into `pytest.mark.pending` (strict xfail, `tests/conftest.py`) and the JS side ignores the key.

### The T-0128c canvas-step rows and `"pending_js"` (T-0128c B1)

Rows 34–52 (19 rows, all `"type": "astra.canvas.step"`) were added for T-0128c B1, when the page's
mirror of `AstraCanvasStep` was found to refuse every real tap frame (Finding 1). They are
hand-written; the denoise frame is canvas length 8 in the plugin tap's shape (`canvas`, `argmax`,
`stable` hex MSB first, no `accepted`/`entropy`), and the `request_id`
`chatcmpl-b8d75dac2b8f054f-b19b91f7` and the five committed ids are copied from the shape of the
recorded trajectory `artifacts/canvas/20261003T003342-lean-cl256-s48-806da12.msgpack` (frame 0's
request id; the ids are arbitrary vocabulary ids, not that trajectory's tokens). Seven decode
(a real-shape denoise frame, the same with `phase` omitted, a denoise frame at `step == max_steps`
(R15), one with all four optional fields, a real-shape commit, a commit at step 0 (O6), a commit
with explicit `null` rows) and twelve are refused (denoise missing `stable`, denoise with committed
ids, denoise at `max_steps + 1`, `stable` too short / odd hex digits / not hex, a `canvas` row one
too long, an unknown `phase`, a commit with rows, a commit with empty ids, a commit at
`max_steps + 1`, more committed ids than `canvas_length`). Every verdict was checked against the
real `local_astra.contracts.codec.decode` when written, and is re-checked on every run by
`tests/unit/test_hud_constants_parity.py`.

They carry `"pending_js": "T-0128c"` rather than `"pending"`: the Python side is the spec and
already gives these verdicts, so a `pending` marker there would XPASS. `pending_js` is read only
by `tests/hud/protocol.test.mjs`, which holds those rows out of its all-corpus row and runs them
through its T-0128c group B gate (strict xfail until `docs/tasks/T-0128c.md`'s status records
"group B: tests …, activated <sha>"). The Python loader ignores the key. The older row
`canvas_step_lengths_disagree` stays malformed under the new mirror and carries no gate.

### The T-0159 piece rows (T-0159 A1/A6)

Rows 53–61 (nine rows, all `"type": "astra.canvas.step"`) were added for T-0159 A6, when the frame
gained the optional `argmax_pieces` (denoise only, `canvas_length` strings) and `committed_pieces`
(commit only, `len(committed_tokens)` strings). They are hand-written on the same CL 8 frame and
request id as the T-0128c rows. Three decode (a denoise frame with pieces, a commit frame with
pieces, a denoise frame whose pieces are `<|channel>`, a newline and `</script><b>`: text, never
markup); they carried `"pending": "T-0159"` until group A's activation removed the key. Six are refused on both sides, before and after A1: argmax
pieces whose length disagrees, committed pieces whose length disagrees, `committed_pieces` on a
denoise frame, `argmax_pieces` on a commit frame, a non-string item, a `null` item. The corpus
header line above still says 52; it now holds 64 rows.

Rows 62–64 (T-0159 A reviewer 1, T1/T2) are hand-written on the same frames: a commit frame with
an **empty** `argmax_pieces` (malformed: an empty list is still carried), and two frames with no
`phase` key, which the contract reads as denoise (row 35's rule): with valid `argmax_pieces`
(astra) and with `committed_pieces` (malformed). Verdicts were checked against the real
`codec.decode` of the group A implementation (9fe8a58a) and are re-checked on every run by
`tests/unit/test_hud_constants_parity.py`.

### The T-0180 thought rows (T-0180 A2)

Rows 65–77 (thirteen rows, all `"type": "astra.thought"`, names starting `thought_`) were added for
T-0180 A2, hand-written from the A1 field list (turn `t4`, request ids `chatcmpl-s1-t4-r<n>`,
attempt ids `t4-a<n>`). They carried `"pending": "T-0180"` until group A's activation (2026-10-28)
removed the key. The corpus now holds 77 rows.

- Decode (`astra`): an `open` step-1 thought; a `done` re-ask with `replay: true`; a thought whose
  text holds `<|channel>`, a newline and `</script><b>` (data, never markup); and two `build` rows
  at the cap: `"x" × 16000` with `omitted_chars: 4000`, and `U+1F600 × 16000` (16,000 code points,
  32,000 UTF-16 units: the cap is counted in code points on both sides).
- Refused (`malformed`): `"x" × 16001` (over the cap, a `build` row); `omitted_chars > 0` with
  text below the cap; an unknown `end`; an unknown `kind` (`pass2`); `step: 0`; an empty
  `attempt_id`; an extra field.
- `thought_from_the_client_is_the_wrong_direction` decodes as `astra` (the codec is
  direction-blind) and carries the extra key `"direction_verdict": "wrong_direction"`: the
  server's verdict for that message arriving *from* a page, pinned by
  `tests/contract/test_t0180_thought_wire.py` and read by `tests/hud/protocol.test.mjs` (the page
  never emits it). Loaders that do not know the key ignore it.

The thought `build` rows sit on the 16,000-code-point thought cap, not on the byte limit, so the
JS size table skips names starting `thought_`. Verdicts are checked against the real
`codec.decode` on every run by `tests/unit/test_hud_constants_parity.py`.

## `mockup_20261021/` — the approved HUD and card mockup (T-0154 A1)

The mockup the human approved on 2026-10-21 (T-0128; claude.ai artifact
https://claude.ai/artifact/TbSJM8AuFP8vwP7NfPDcbZ), committed byte for byte so that
`tests/integration/test_hud_mockup_parity.py` can render it in the same Chrome as the HUD and
measure the reference live. Source: the session scratchpad
`/tmp/claude-1000/-home-user-demos-local-astra/dfaad220-dee3-540f-b9f5-97bb9790a010/scratchpad/mock/`
(files dated 2026-10-20 17:22-17:23), copied with `cp` on 2026-10-25; the SHA-256 of each copy
equals its source's (checked with `sha256sum` on both sides). Hand-written HTML/CSS and its
screenshots, no third-party content, Apache-2.0 (this repo). The mockup's `index.html` (a gallery
page linking the two) is not part of the reference and is not committed.

Nothing under `hud/` references this directory and `astra hud` never serves it (the parity module
asserts both). The mockup names `"Inter"` first in its font stack; it is a family name, never a
download (no `@font-face`), and on this box it resolves to the same face as `system-ui`.

| file | SHA-256 |
|---|---|
| `mockup_20261021/hud.html` | `25cb81b28d2aa8936764c697185bd08e0da4f70b6cab262cb700f13b600f8e7a` |
| `mockup_20261021/cards.html` | `42c3fc3de4b301d0ab32edee0149b40c1e0520722eb5776971b1df9d659c0016` |
| `mockup_20261021/tokens.css` | `e94917f395e59235a46dc86eb0d23e18fb9564c1dedf7de890ad60d9d384c535` |
| `mockup_20261021/hud-dark.png` | `a2bc0fea9cb0400dceca987c938129ff0d97d36431c239adbf3c01105c4e6a08` |
| `mockup_20261021/hud-light.png` | `8b159848f1278c855e89051a2ad795b78b200e9408be5b84d8caacf44855558c` |
| `mockup_20261021/cards-dark.png` | `d20bbc1f0cab54d31294a560ce7e18ef66c73159065c9935860a41ae53dbcd3a` |
| `mockup_20261021/cards-light.png` | `9b2a11621c68f69ad34731bca01f4fa33299929e7bfd32fa3d5fdfa377747719` |

## `genui_20261027/` — the approved GenUI card look, option A (T-0164 C1)

The reference for T-0164 group C's role table (`tests/integration/test_hud_card_parity.py`) and
for the C5 composites. Source: the GenUI mock James was shown on 2026-10-27, in the session
scratchpad `/tmp/claude-1000/-home-user-demos-local-astra/dfaad220-dee3-540f-b9f5-97bb9790a010/scratchpad/genui-mock/`
(files dated 2026-10-27 00:23-00:27). Hand-written HTML/CSS/JS, no third-party content,
Apache-2.0 (this repo).

- `mock-dark.png`, `mock-light.png`: the screenshots James saw (1240x2548), copied with `cp`,
  byte for byte. They show both columns (today's vendored render and option A); only option A is
  the reference.
- `cards.html`: **the option-A column only**, built by the T-0164 C test-author on 2026-10-27 with
  `scratchpad/t0164c/build_cards.py` (sha256 `78bce94acb62e0cfd6816236aee9a01bf6e7b3b1544a106c5143ea278af5c177`;
  kept in the scratchpad, like the census generator) from `genui-mock/mock.src.html`:
  - the option-A renderer and sheet verbatim (the mock's own two `:root[data-theme="light"]`
    fill swaps included: they are the mock, not `card.css`);
  - the HUD token block **frozen** from the built `mock.html` James saw (so a later edit of
    `hud/styles.css` does not move the reference);
  - the four payloads of `genui-mock/build.py` (trial-omelette, desk-t8, desk-t9, trial-flight;
    their final canvases, gated by `UiGate` at the census gate commit
    `06a7acf7027de6242524c63ba3b58d1fd57dde93`) embedded as JSON;
  - dropped: the "today" column, its embedded screenshots, the pair headings and the lede;
  - added, as test hooks only: each card's pane carries `data-payload="<key>"`, and the renderer
    is exposed as `window.A2` / `window.A2mount(key, messages)`, so the parity module renders its
    own role-probe surface (labelled synthetic there) with the mock's renderer and sheet.

  `?theme=dark|light` picks the theme, as in the mock. The page makes no request: no image, font
  or script is loaded from anywhere (the SVG namespace string is an identifier, never fetched).
  Nothing under `hud/` references this directory.

| file | SHA-256 |
|---|---|
| `genui_20261027/cards.html` | `b3387d6560707d8ee2dd54d411161343fef81479fa096f00d261023759736435` |
| `genui_20261027/mock-dark.png` | `9275523d4fa59a981863a8c2e7f5bebd74e142f49d3eaac227776776346f91a1` |
| `genui_20261027/mock-light.png` | `f861326a758b4c7f4d185f079fea3a9953d31f1ba8936a7b2adfb9df1bb9fa4b` |

## `genui_v2_20261028/` — the approved GenUI v2 boards (T-0184 B, H-T184-8)

The three boards of the GenUI v2 mockup James approved on 2026-10-28 (claude.ai artifact
https://claude.ai/artifact/8wgfXZq9gCne6YuwMcp5KH; "looks good enough for now, go ahead"): **Main**
(the weather answer, its card and the thoughts disclosure), **Compare** (the comparison card and
its two buttons) and **Chart** (the bar-chart card). The fourth board, Calculator, is out of scope
(docs/tasks/T-0184.md Status) and is not committed. Source: the session scratchpad
`/tmp/claude-1000/-home-user-demos-local-astra/dfaad220-dee3-540f-b9f5-97bb9790a010/scratchpad/genui-v2/project/`
(files dated 2026-10-28 02:45-02:47 UTC), copied with `cp` by the T-0184 B test-author on
2026-10-28; the SHA-256 of each copy equals its source's (checked with `sha256sum` on both sides).
Hand-written HTML with inline styles, Apache-2.0 (this repo).

**Reference only.** These are Design Component pages, not standalone HTML: each loads
`./support.js` (the design tool's runtime, not committed) and links the Inter webfont from
`fonts.googleapis.com` (H-T184-3: the HUD does not bundle Inter). No test and no script opens them
in a browser, nothing under `hud/` references this directory, and `astra hud` never serves it.
They are read by people: their inline styles are the look values B3 names (frame gradient
`#1b2030` -> `#171a22`, border `rgb(255 255 255 / 0.08)`, radius 20 px, padding 22-24 px; the stat
label 12 px letter-spaced, the 56 px value; the tile, pill, bar and button values), and they sit
beside `genui-v2-{dark,light}.png` (B10, `scripts/demo/hud_screens.py --genui-v2`), which rebuilds
the three boards from the v2 payloads `board-main`, `board-compare` and `board-chart` of
`tests/fixtures/a2ui/cards_v2_exemplars.jsonl`. The bracketed values in the boards (`[17°]`,
`[PRICE]`, …) are the mockup's placeholders. `tests/integration/test_t0184_genui_v2.py` checks the
hashes below.

| file | SHA-256 |
|---|---|
| `genui_v2_20261028/Main.dc.html` | `d0041ee9d7c09c657b378dfc47409314703652fda34410c9870f31a4b34f9f49` |
| `genui_v2_20261028/Compare.dc.html` | `92e3a556726d71b077f2495fa12dba277f86e563612af99827704cff72a5a870` |
| `genui_v2_20261028/Chart.dc.html` | `eac9651753c1133ebecbe39543b90d3ef77ac462bc731a9fb00c5449eb483995` |
