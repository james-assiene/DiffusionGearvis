# tests/fixtures/browser — T-0111c browser executor fixtures

Hand-written HTML/JS by the T-0111c test-author, 2026-10-20; project licence. Served only by a loopback
`http.server` on 127.0.0.1 (`tests/integration/test_computer_browser_e2e.py::server`). No page loads anything from
off the box; `egress.html` *tries* to, on purpose.

| file | used by | what it is |
|---|---|---|
| `recorder.js` | every page | appends every event the page receives to `window.__log` (capture phase): pointer/mouse (x, y, button, buttons), keyboard (key, code, modifiers), `input` (value), `submit`, `scroll` (offsets), copy/cut/paste (clipboard text); `window.__state()` snapshots focus, scroll, selection, input values, open dialogs |
| `grid.html` | criteria 3, 4, 5, 10, 13 | one fixed element covering the viewport, so every CSS pixel a pointer lands on is `#grid`'s |
| `form.html` | criteria 4, 6, 11 | `#q` (x 100..499, y 100..139, value `pre-`) in a form whose submit is logged; `#other` has focus on load |
| `color.html` | criteria 2, 9 | a 400x400 red block at (440, 200), centre (640, 400); a click toggles it blue/red two animation frames later |
| `scroll.html`, `scroll_wide.html` | criterion 7 | three viewports tall / wide; a non-periodic gradient so no two offsets look alike |
| `keys.html` | criterion 8 | `#inp` in a form, `#ta` (`line one\nline two\nline three`, 28 chars), `#b`, a button, a modal `<dialog>`, a 4000 px body; a Ctrl+Enter binding that writes `#status` |
| `busy.html` | criteria 10, 11 | a pointerdown spins the renderer for `?ms=` (default 3000), so an action is in flight while the row closes the page, kills the browser or cancels |
| `sandbox.html` | criterion 13 | a download link (`t0111c-payload.txt`), a file input, alert/confirm/prompt buttons, a link guarded by `beforeunload` (armed by the row) |
| `t0111c-payload.txt` | criterion 13 | the download payload; it must never be saved |
| `egress.html`, `sw.js`, `ok.svg`, `ok.txt`, `hints.html` | criterion 14 | off-loopback attempts (img, fetch, sendBeacon, iframe, WebSocket, resource hints, a service worker that fetches on install, a top-level navigation) with a secret path and query; loopback controls `ok.svg`/`ok.txt` |
| `popup.html` | criterion 15 | page n is one `target=_blank` link to page n+1 in colour `PALETTE[n]` |
| `late_popup.html` | criterion 15 (ruling a918d74) | a red page whose click opens `popup.html?n=2` 300 ms later, after the settle |
| `late_popup_misdirect.html` | criterion 15 (ruling 683cc1e) | `#open` (x 0..639, red) opens `popup.html?n=2` 400 ms after a click; `#second` (x 640..1279, green) is the opener's second target, which a stale-frame click must not reach |
| `self_close_popup.html` | criteria 4, 15 (ruling on the round-3 review of e345ef2, mutant S9) | the opener (no query, red, whole viewport `#open`) opens `self_close_popup.html?popup=1` (blue) synchronously on click; the popup calls `window.close()` on itself when the row posts `"close"` on the BroadcastChannel `t0111c-self-close` |
| `webrtc.html` | criterion 14 (ruling a918d74) | `__rtcTypes()` reports the four WebRTC constructors in the top frame, a same-origin child and a script-made about:blank child; `__rtcAttempt(stun, turn)` tries STUN and TURN-UDP (through a blank child's constructor if the top frame has none) |
| (server) `/t0111c-redirect?to=` | criterion 14 (ruling a918d74) | the e2e module's fixture server answers a 302 to any target: the redirect vector |
| `t0134_hostile_extent.html` | T-0134 A4a (`tests/integration/test_t0134_extent_browser.py`; hand-written by the T-0134 test-author, 2026-10-22) | a 2800 px tall page (true extent at 1280x800, scroll 0: 0 above, 2000 below) whose script redefines `window.scrollY`/`pageYOffset` (4242), `innerHeight`/`outerHeight` (100), `document.scrollingElement` (a fake object) and the root element's `scrollTop`/`scrollHeight`/`clientHeight`: any reading from inside the page is wrong |
| `t0151_two_screens.html` | T-0151 A1-1/A1-4/A1-7 (`tests/integration/test_t0151_overview_frame_browser.py`; hand-written by the T-0151 test-author, 2026-10-24) | a fixed 1600 px page (2.0 screens at 1280x800, no `vh`, no smooth scroll) with four buttons at x 140..280 and page y 150..198 (`#top`), 420..468 (`#decoy`), 860..908 (`#mid`), 1500..1548 (`#low`); every click pushes the button's id to `window.__clicks`; `?wide=1` makes the body 3840 px wide |
| `t0189_fields.html` | T-0189 criteria 1-3 (`tests/integration/test_t0189_type_replaces_browser.py`; hand-written by the T-0189 test-author, 2026-10-28) | prefilled single-line inputs at x 100..499, 40 px tall, one every 60 px from y 40: `#in-text` (`old-text`), `#in-search`, `#in-url`, `#in-email`, `#in-tel`, `#in-number` (`12345`), `#in-notype` (no `type`), `#in-pw` (password, `old-secret`); `#ta` (textarea, `old textarea`) and `#ce` (contenteditable `div`, `old editable`) at x 600..1099, 200 px tall, one short line, so a click at the centre puts the caret at the end of the text (checked against a prototype, Chromium 156.0.8078.4 headless) |
| `t0194_slow_focus.html` | T-0194 criterion 9 (`tests/integration/test_t0194_slow_focus_browser.py`; hand-written by the T-0194 test-author, 2026-10-29) | a page whose main thread is busy right after the click: `#q` (text, `old`, x 100..499, y 100..139) and `#pw` (password, `old-secret`, y 200..239); each field's `mouseup` arms one 1.0 s synchronous busy-wait, run by a `MessageChannel` task or by the page's next `Element.prototype.getAttribute` call (which the focus read calls), whichever comes first, so the read always waits about 1.0 s; `window.__busyMs` lists the busy-waits that ran. The second trigger exists because a DevTools `Runtime.evaluate` runs ahead of a queued page task: with the task alone 5 of 6 reads answered in < 15 ms; with both, 3 of 3 BASE runs read for 0.50 s (the bound) and typed into `#pw` (`old-secretzz`) (Playwright 1.64.0, Chromium 156.0.8078.4 headless, test-author probe) |
| `t0200_late_paint.html` | T-0200 criterion 10 (`tests/integration/test_t0200_blank_recapture_browser.py`; hand-written by the T-0200 test-author, 2026-10-29) | a plain white page that paints nothing until `?ms=` (default 1500) has passed, then a 600x300 `rgb(0, 90, 200)` block with the text `painted` centred on (640, 400); the block is a link to `t0200_late_paint.html` (default delay), so `?ms=0` is a click target that navigates to a late-painting page |
| `t0200_token_reload.html` | T-0200 criterion 10 (same module and author) | WAF-shaped (the AWS WAF token reload of James's 2026-10-29 Amazon session): `?start=1` paints a 600x300 `rgb(0, 150, 60)` link to the page itself; without the `sessionStorage` key `t0200-token` the document stays white, stores the key and calls `location.reload()` after 800 ms; with the key it paints the green block at once |
| `t0200_white.html` | T-0200 criterion 10 (same module and author) | an empty white page: every capture is uniform, so a capture of it is re-taken until the budget is spent |
| `t0202_slow_results.html` | T-0202 B10 (`tests/integration/test_t0202_wait_browser.py`; hand-written by the T-0202 test-author, 2026-10-29) | a white page with a search box at x 340..939, y 40..79 (centre (640, 60)); Enter in it paints a `rgb(50, 50, 50)` header bar (y 100..159) at once and, 800 ms later, a 1000x400 `rgb(0, 90, 200)` results block centred on (640, 450); nothing else changes |
| `t0202_never_loads.html` | T-0202 B10 (same module; same author and date) | a white page with a `rgb(50, 50, 50)` header bar and a static "loading results" line that never changes (results that never arrive) |
| `t0188_beforeunload.html` | T-0188 criteria 1, 4, 5 (`tests/integration/test_t0188_beforeunload_browser.py`; hand-written by the T-0188 test-author, 2026-10-28) | a page that refuses to unload, as photopea.com does once a document is open: a `beforeunload` handler that calls `preventDefault()` and sets `returnValue`, armed unless `?arm=0` (the control); `#field` (text input, x 100..499, y 100..139) for the user gesture Chromium needs before it asks; the dismissed dialog cancels the navigation and `page.goto` raises `Error` (net::ERR_ABORTED), as the planner's probe saw (Playwright 1.64.0, Chromium 156.0.8078.4 headless, n=1) |
| `probe_press_key.py` | criterion 8 | plain-Playwright probe of every `press_key` combo (below) |
| `probe_sandbox.py` | criteria 9, 10, 13, 14 | plain-Playwright probe of the facts the e2e oracles rest on (below) |

## Probes (recorded 2026-10-20)

Regime: Playwright 1.64.0, Chromium 156.0.8078.4 (`chromium_headless_shell-1248` for headless; versions.lock
`[browser]`), headless, a fresh context per row, 1280x800 at DSF 1, no executor; this box (hostname
`gpu-box`, /etc/hosts -> 127.0.1.1), n = 1 unless stated.

### `uv run python tests/fixtures/browser/probe_press_key.py`

The rows the e2e table (`EXECUTED` / `REFUSED_COMBOS`) is read from: combo, setup, keydown events
`(key, code, ctrlKey, altKey)`, the effect.

| combo | setup | keydown | effect |
|---|---|---|---|
| Enter | `#inp` typed `abc` | Enter/Enter | `submit` once |
| Escape | modal dialog open | Escape/Escape | dialog closed |
| Tab | `#inp` focused | Tab/Tab | focus `#inp` -> `#ta` |
| Backspace / Delete / Space | `#inp` = `abc`, caret 2 | as named (`" "`/`Space`) | `ac` / `ab` / `ab c` |
| ArrowUp / ArrowDown / ArrowLeft / ArrowRight | `#ta` caret 12 | as named | caret 3 / 21 / 11 / 13 |
| Home / End | `#ta` caret 12 | as named | caret 9 / 17 |
| PageUp / PageDown | body scrolled to 1000 | as named | scrollY 948 / 1048 after two frames (smooth scrolling on; it keeps moving) |
| Ctrl+A | `#ta` caret 12 | Control, then `A`/KeyA ctrl | selection 0..28 |
| Ctrl+C | `#ta` selection 0..8 | `C`/KeyC ctrl | `copy` fires; a later Ctrl+V pastes the text (setup `copied`) |
| Ctrl+V | text copied, `#b` focused | `V`/KeyV ctrl | `#b` = the copied text |
| Ctrl+X | `#ta` selection 0..8 | `X`/KeyX ctrl | `cut`; `#ta` loses the selection's text |
| Ctrl+Z | `#inp` typed `abc` | `Z`/KeyZ ctrl | `#inp` = `` |
| Ctrl+Enter | any | Control, then Enter ctrl | **no default action in any setup**; only a page binding sees it (`keys.html` has one) |
| Ctrl+L / Ctrl+T / Ctrl+W | any | delivered to the page | nothing: no page opened, none closed (refused by H-T111c-7) |
| Alt+Tab | any | Alt, then Tab alt | nothing (refused by H-T111c-7) |

Notes: the clipboard is shared by every context of one browser (a Ctrl+V in a later context pasted text copied in an
earlier one), so the e2e row uses a fresh token per combo. Chromium synthesises `pointermove` and `scroll` events
after layout changes, so "dispatched nothing" comparisons leave those two out.

### `uv run python tests/fixtures/browser/probe_sandbox.py` (two runs; the settle line from a third, n = 10)

```
hostname gpu-box -> 127.0.1.1
settle wait_two_frames= False centre rgb (255, 0, 0)        # run 1; run 2 gave (0, 0, 255): 3 of 10 red (n = 10)
settle wait_two_frames= True centre rgb (0, 0, 255)
canary unfenced attempt connections 5 page {'img': {'ok': False, ...}, 'fetch': {'ok': True}, 'beacon': {'queued': True}, 'iframe': {'ok': False}, 'websocket': {'ok': False, ...}}
canary unfenced hints connections 0 page True
canary unfenced sw connections 1 page {'ok': True}
canary sw service_workers=allow register {'ok': True} registration {'installing': False, 'waiting': False, 'active': True} connections 1
canary sw service_workers=block register {'ok': True} registration None connections 0
canary unfenced static hints connections 0
loopback controls {'img': {'ok': True}, 'fetch': {'ok': True}}
download accept_downloads=False events [('download', 't0111c-payload.txt')]
filechooser files.length 0
dialog handled= False seen [] page [alert, confirm False, prompt None]
dialog handled= True seen ['alert', 'confirm', 'prompt'] page [alert, confirm False, prompt None]
beforeunload seen ['beforeunload'] url ends sandbox.html
permissions default {'geolocation': 'prompt', 'notifications': 'prompt', 'camera': 'prompt', 'microphone': 'prompt', 'Notification': 'denied'}
cancel keyboard.type inputs at cancel 1897 after 500 ms 1897
kill: pids [...] raised playwright._impl._errors TargetClosedError
close (busy.html, ms=3000) action raised TargetClosedError after 0.9 s; kill raised TargetClosedError after 0.41 s
```

What follows for the oracles: a screenshot straight after a click misses `color.html`'s change about 3 times in 10,
so the post-action row clicks eight times; unfenced Chromium reaches the canary for the five attempts and the service
worker, so a zero count is evidence; unfenced *headless* Chromium makes no connection for dns-prefetch/preconnect, so
the hint assertions have no power on this build; Playwright's defaults already dismiss dialogs, so the rows test the
*report*; permission states default to `prompt`, so `denied` is the executor's doing; `keyboard.type` typed ~1 900
characters in 300 ms, so the cancel row cancels on the first `input` event and checks it was mid-type.
