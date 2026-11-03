# hud/ — display-only page (T-0008, T-0010, T-0011, T-0012)

Plain web page (no Tauri). It never handles audio (the Conductor owns PipeWire capture, playback
and any AEC — docs/02 ADR-013). It:

- captures the shared tab / camera with `getDisplayMedia` / `getUserMedia` and posts JPEG frames
  at ≤ 1 fps as `astra.frame` (v1 perception source; the page-text assist payload arrives with
  T-0012, behind its checkbox);
- renders the *spoken* channel (`astra.transcript`, `astra.spoken`), the turn-state chip and the
  thinking meter from `astra.turn.state` (T-0008);
- renders A2UI v0.9.1 surfaces with its own light-DOM renderer (`src/ui/engine.js`,
  `src/ui/render.js`; T-0164); actions go back as UI actions (T-0010, `src/ui/`);
- shows the denoising view in the Thinking pane from the real per-step `astra.canvas.step`
  frames the conductor relays from the plugin tap; an `astra.canvas.synthetic` fallback is never
  drawn, only named (T-0011, T-0128c, `src/denoise/`, see below);
- shows the assist checkboxes and the badge echoed in `astra.assist.state` (T-0012,
  `src/assists/`), behind the header's Settings gear since T-0162;
- lists every tool call per turn in the right rail's Tools pane (T-0162, `src/tools/`).

Protocol: `local_astra/contracts/protocol.py`. The page mirrors it; the Python side is the spec.

## No build step

T-0008 ships **plain ES modules, no bundler and no `node_modules`**: node 18.19.1 is what this box
has and is EOL, current bundlers need node ≥ 20, and a buildless page is directly screenshot- and
DOM-testable. The A2UI cards are drawn by the page's own modules in `src/ui/` (T-0164), so
there is no third-party renderer and nothing to install. The only manifest, `hud/package.json`,
is exactly `{"type": "module"}` so that node loads `src/` as ES modules in place; it declares no
dependency. (`node` is only a *dev* prerequisite, for `node --test tests/hud`.)

ES modules do not load over `file://` — cross-file imports are CORS-blocked — so the directory has
to be served. One command does it and prints the URL to paste, query string and all:

```bash
uv run astra hud
# http://127.0.0.1:1420/?ws=ws://127.0.0.1:8000/v1/realtime               real capture (one click on "Share")
# http://127.0.0.1:1420/?source=canvas&ws=ws://127.0.0.1:8000/v1/realtime synthetic capture, headless runs
```

`astra session --serve-hud` runs the same server beside the session, so one command gives both the
endpoint and the page. Both bind `127.0.0.1` only and refuse a taken port rather than moving to
another one.

Fallback, if you would rather not start the CLI (a box without the package installed still has to
serve the page):

```bash
python3 -m http.server --bind 127.0.0.1 -d hud 1420
```

`?ws=` is honoured only for `127.0.0.1`, `localhost` and `[::1]`; anything else is refused with a
visible error and no connection (CLAUDE.md: all services bind to loopback only).

## `astra.frame.ts_ms` is the browser's `Date.now()`

Decided here, written down because getting it wrong is invisible: `ts_ms` is the *page's* clock at
capture. The Conductor stores frames by **receive** time and answers `get_frame(at=speech_onset)`
against the audio clock, because the page's clock is not the audio clock and may even step
backwards mid-stream. The client `ts_ms` is kept only as a jitter figure
(`local_astra/perception/browser_source.py`).

## Theme: System / Light / Dark (T-0128)

The page follows the system's `prefers-color-scheme` until you choose otherwise in the header's
**Theme** control:

- **System** (the default): light or dark as the desktop says, followed live — no reload;
- **Light** / **Dark**: fixed, whatever the desktop does.

The choice is stored per browser under the `localStorage` key `astra.hud.theme` (`light` or
`dark`; choosing System removes it). Any other stored value is ignored and the page follows the
system. To reset by hand, run `localStorage.removeItem("astra.hud.theme")` in the console.

If storage is unavailable (a privacy mode, a full quota, a blocked origin), the control still
switches the theme for the life of the page, nothing is stored, and the page logs one console
line saying so.

How it works: `src/theme.js` is the one implementation of the resolution order. It writes
`<html data-theme="light|dark">`; `styles.css` declares the dark tokens at `:root` and re-binds
them in exactly one `:root[data-theme="light"]` block, which also sets `color-scheme` so native
controls follow. The A2UI cards are light DOM (each under a `[data-a2ui-root]` element, no
shadow root), themed by one page-level sheet, `src/ui/card.css` (T-0164 B5/C), bound to the same
tokens. `src/ui/index.js` links it once from the page's own origin inside `mount()` (a
`<link rel="stylesheet">`, not a `fetch`: the panel opens no transport of its own). Its rules use
tokens as the only colours and style only: no generated content, no `text-transform`, nothing the
model did not write (H-T154-6). The theme module loads as its own
module script, so a stored Light choice shows one dark frame first; if the module fails to load,
the page renders dark and everything else works.

## The frame (T-0154 group A)

The page's frame follows the approved 2026-10-21 mockup (committed with the tests' fixtures,
see their MANIFEST; never served by `astra hud`): a 1480 px app frame over the
mockup's background gradient, a transparent one-row header (brand, the state pill, the
`SYNTHETIC CAPTURE` notice while a synthetic capture runs, Share, Mute, the theme select), and the
right rail's panes with the mockup's 1 px hairline and shadow. **Share** reads "Sharing screen"
(or "Sharing camera") in green only while a *real* capture runs; a synthetic capture never says
it is sharing. Mute keeps its `m` shortcut (`aria-keyshortcuts`, and the button's title). The assist
switches have human labels, "Needs the desktop helper" on an unavailable row (the full reason is
its tooltip), and are drawn as switches; since T-0162 they live behind the Settings gear (below).
The mockup's data-less parts ("Mic · Headset", ⋯, timestamps) are deliberately not built; its
noise grid is the denoising view below, drawn only from real frames.

## Settings: the context helpers (T-0162 group B)

The header's gear (`aria-label="Settings"`, an inline SVG, no icon font) opens a popover holding
the assist panel: the same `#assist-panel` element, `data-assist-*` hooks and gate as before,
headed **Context helpers** and explained in one sentence, which is also the gear's tooltip:

> Gearvis reads your screen from pixels only; these optional helpers also send it the page's
> text, a PDF's text layer, spreadsheet cells or the accessibility tree. They are all off by
> default, and every turn records which ones were on.

`src/main.js` hands `src/assists/index.js` the popover's `[data-settings-slot]` as
`host.settingsSlot`; without one the panel appends to `<main>` as it used to. The popover closes
on Escape (focus returns to the gear) and on a click outside it, and keys typed inside it never
reach the page's `m` (mute) shortcut. Hiding the switches never hides that one is on (D12a):
while the server's badge is not `pixels`, the header shows it verbatim beside the gear
(`#assist-header-badge`); with every assist off, or before any `astra.assist.state`, it shows
nothing.

## Interrupt Gearvis (T-0193 group B)

The header's **Interrupt Gearvis** button, right after the turn chip, appears once the first
`astra.turn.state` arrives and is enabled only while Gearvis is thinking, speaking or paused. A
click sends one `astra.turn.stop` for the turn the server last echoed; the Conductor cuts the
speech, aborts generation and ends a computer-use loop, and the turn goes back to idle. The button
then stays disabled until the next `astra.turn.state`; a late or replayed state for an earlier
turn, a Restart, or a dropped socket never re-arms it. There is no keyboard shortcut (Escape stays
with Settings and the Restart confirm). `src/stop.js` holds that logic without the DOM; `main.js`
wires it.

## The Tools pane (T-0162 group B)

The right rail's Tools pane (`#tools-panel`, mounted through `PANEL_SPECS` where the Context pane
was, between Thinking and the session log) lists every `astra.tool.call`, grouped per turn
(`[data-tools-turn]`), newest turn first, calls in arrival order within a turn. Each call_id is
one native `<details>` (`[data-tools-call]`) whose summary is the tool and its arguments in short
form (`[data-tools-line]`) and the wire status verbatim (`[data-tools-status]`: proposed,
repairing, dry run, executed, refused, cancelled). Opened, it shows every argument as
`key: value`, the source (from the draft / repaired), the action, reason and query problem when
present, and the statuses seen, in order.

`src/tools/model.js` is pure: `shortArgs(name, args)` gives `click_at (x, y)`,
`type_at "…" at (x, y)` (+ ` + Enter`; the text cut at 40 code points), `scroll down`,
`press_key ctrl+l`, `google_search "…"` (cut at 60), `page_overview`, and for any other tool its
name and compact JSON cut at 80 code points; a cut ends in `…`. The pane reads only
`astra.tool.call` and `astra.turn.state` (for the turn order), so no thought or canvas text can
reach it, and every model string is set with `textContent`. The inline action cards in the thread
are unchanged.

## Model thoughts (T-0180 group C)

Each turn in the thread that had an `astra.thought` gets one native
`<details data-model-thoughts>`, the assistant body's last child (under the answer, the card and
the action cards), closed by default. The page never sets `open`: only the viewer's click, Enter or
Space on the summary toggles it, updates keep whatever state the viewer left, and nothing persists
across a reload. The summary reads `Model thoughts`, plus ` · N steps` once there is more than one
block. Each attempt (`attempt_id`, first-seen order) is one block (`[data-thought-attempt]`) with a
label from the session's `step` and `kind` (`Step 2 · re-ask`, `Step 1 · retry`, `Step 3 · repair`,
`Step 2 · ReGround`; `[data-thought-label]`), the committed text set with `textContent`
(`[data-thought-text]`, `pre-wrap`, scrolling inside 24 rem), a status line (`thinking…`,
`stopped: repeating`, `interrupted`, `stopped: error`; none when done; `[data-thought-status]`) and,
past the 16000-character cap, `… N more characters not shown (full text in the session log)`
(`[data-thought-omitted]`). The latest message per attempt wins, and a final block ignores a later
`open`. `src/panels/thoughts.js` reads only `astra.thought`, never canvas frames, so a draft is never
shown as a thought; the Thinking pane's denoising view is unchanged. A reset removes the
disclosures with their articles.

## The denoising view (T-0128c group B, T-0159 group B)

The Thinking pane draws the canvas the model is denoising, one `astra.canvas.step` frame at a
time, as the plugin tap (≤ 30 Hz per request) and the conductor's relay deliver them.
`src/denoise/model.js` is pure (no DOM, no clock): `cellStates(frame, previous)` gives each position
one state, `cellPieces(frame)` the text piece at each position, `displayPiece(piece)` its display
form, `label(frame)` the heading-row text, `digest(states)` an FNV-1a (32-bit, 8 hex) of the
state names joined by `,`.

The layout follows James's reference image (the DiffusionGemma page's denoising figure, T-0159):
`src/denoise/index.js` gives each `canvas_idx` of the shown request its own native `<details>`
section in `[data-denoise-root]`, whose `<summary>` reads that canvas's label. Inside it is a grid
of rounded token cells, one per position in position order, each filled with its state's colour and
showing the token at that position. The grid takes as many fixed-width columns as fit the pane
(8 in the rail) and scrolls inside a bounded height; the page never scrolls sideways. A frame of a
new canvas appends its section open and closes the previous newest one; sections the viewer opened
stay open; a new request's first drawn frame replaces all sections. A section whose newest frame is
a denoise frame with every argmax piece `<eos>` (`isEndOfSequence`, the trailing canvas after the
answer commits) closes with the summary `canvas k · end of sequence`, keeps its real cells, and the
newest other section opens once at that transition (undone if a later frame of it is not all
`<eos>`; T-0159 B2). Each paint touches only the
cells whose state or text changed, with no transition.

| state | from the frame | fill | text |
|---|---|---|---|
| stable | its `stable` bit is set (MSB first, whole bytes) | `--accent` | `--dn-ink-settled` |
| kept | else `canvas[p] == argmax[p]` | `--accent` at 45 % | `--dn-ink-kept` |
| noise | else | `--accent` at 12 % | `--dn-ink-noise` (dimmed, ≥ 3:1) |
| revised | its argmax moved where the previous frame of the same request and canvas had it stable | `--accent-2` | `--dn-ink-revised` |
| committed | commit frame, `p < len(committed_tokens)` | `--accent` | `--dn-ink-settled` |
| empty | commit frame, the rest | `--accent` at 6 % | none |

The text comes from the frame, never from the ids: a denoise cell shows `argmax_pieces[p]`, a
commit cell `committed_pieces[p]`, both decoded in the conductor by the pinned stdlib detokeniser
(T-0159 A). The page holds no tokenizer and no id → text table. A frame without pieces draws its
cells with no text. Pieces are set with `textContent` (and the cell's `title`), never as markup.
Display form: leading and trailing spaces trimmed, a piece of only spaces shows `␣`, a newline `↵`,
a tab `⇥`; an added or special token (`<eos>`, `<|channel>`, a `<0xE2>` byte) shows verbatim in
the monospace face at a smaller size; a piece wider than its cell ends in `…`, and the cell's title
holds the whole piece with its whitespace made visible. The state is still computed from the ids and
masks only: the text never changes a cell's state or the digest. Thought text appears only here.

The label right of the heading reads `canvas k · step s/N` (or `canvas k · committed at step
s/N`), verbatim from the newest drawn frame. The view follows the request whose first frame arrived most
recently, at once (the anticipatory prefill is shown until the generate request's first frame,
then the generate); frames of any other request, including an older one, are counted but never
drawn. The last frame stays up between turns, and a reload starts empty. At most one paint per
animation frame; each paint writes every canvas's latest waiting frame, so a canvas that the next
one superseded inside a slot still gets its section and its last frame. Under `prefers-reduced-motion: reduce` the grid repaints at most twice a
second, and the label instead follows the newest received frame, so it can be
ahead of the grid. Colours and sizes are CSS only, so a theme switch or a resize restyles the cells
without a repaint and never draws a newer frame early.

What it never does: draw anything before the first real frame (the root stays empty and the line
under it says "no denoising data yet — the model's real canvases appear here while it thinks"),
interpolate between frames, invent a frame or a token, or keep a requestAnimationFrame loop or a
timer running with no frame waiting. `astra.error` `denoise_no_tap` and `denoise_no_pieces` (the
conductor could not load the detokeniser, so frames arrive without text) put their message on that
line, and `astra.canvas.synthetic` draws nothing and the line says "synthetic canvas refused:
{reason}". A notice stays visible under the grid only while it is true: `denoise_no_tap` and the
synthetic refusal leave at the next drawn frame, `denoise_no_pieces` at the next drawn frame that
carries pieces.
The root carries `data-denoise-frames` (frames received), `data-denoise-draws` (paints) and
`data-denoise-digest` (the digest of the state array last painted), for the tests and the smoke.

`scripts/demo/hud_screens.py --compare <mockup dir>` writes `compare-{dark,light}.png` (mockup
left, HUD right, same Chromium) and `fonts.txt`; `test_hud_mockup_parity.py` measures the frame
against the mockup (both named in docs/tasks/T-0154.md A8).

## M1 — live capture checklist (one human click, criterion 19)

No automated test in this repo covers `getDisplayMedia`/`getUserMedia`: a mocked test that claims
the live path is a false report. Run this by hand and record the result, the date, the commit and
the session type (X11/Wayland) in `PROGRESS.md` under Runs:

1. serve `hud/` as above and open it in Chrome;
2. click **Share**, pick a tab → the browser's consent dialog appears;
3. the `frames` counter advances at about one per second and `capture` shows the tab's size;
4. "What I see" shows only the images actually sent to the model (T-0177 C3): with no live session,
   or before the first request, it keeps its empty line; in a live session with "Shared screen"
   chosen, ask a question and the pane shows the shared tab as it was sent;
5. press **m** (or click Mute) → frames stop *and* the browser's sharing indicator disappears;
6. re-sharing requires a new click on **Share** — mute is not a pause.
